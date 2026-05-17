"""
models/llm_backbone.py

LLM Backbone推理封装。
- 管理多个Prefill静态图（按seq_len分bucket）
- 管理多个Decode静态图（按kv_len分bucket，AR模式用）
- 负责Padding输入到对应bucket
- KV Cache写入由外部KVCacheManager管理
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from aria.core.executor import NPUExecutor, GraphMeta
from aria.core.kv_cache import KVCacheManager
from aria.models.base import FrameworkConfig

logger = logging.getLogger(__name__)

PAD_TOKEN_ID = 0


class LLMBackbone:
    """
    LLM Backbone推理封装。

    Prefill图：每个seq_len bucket一张，输入token序列+视觉特征，输出hidden state + KV Cache
    Decode图： 每个kv_len bucket一张，输入单token + 当前KV Cache长度，输出logits + 新KV
    """

    def __init__(self,
                 config:    FrameworkConfig,
                 executor:  NPUExecutor,
                 kv_cache:  KVCacheManager):
        self.config   = config
        self.executor = executor
        self.kv_cache = kv_cache
        self.lcfg     = config.llm
        self.vcfg     = config.vision

        self._register_prefill_graphs()
        self._register_decode_graphs()

    # ------------------------------------------------------------------
    # 图注册
    # ------------------------------------------------------------------

    def _register_prefill_graphs(self) -> None:
        for seq_len in self.lcfg.prefill_buckets:
            name = f"prefill_{seq_len}"
            meta = GraphMeta(
                name  = name,
                path  = f"{self.config.graph_dir}/{name}.bin",
                input_shapes = {
                    "input_ids":      (self.config.max_batch, seq_len),
                    "vision_feat":    (self.config.max_batch,
                                       self.vcfg.total_vision_tokens,
                                       self.vcfg.feat_dim),
                    "attention_mask": (self.config.max_batch, seq_len),
                    "position_ids":   (self.config.max_batch, seq_len),
                    # kv_start_pos：告诉模型从哪个位置写KV Cache（多轮用）
                    "kv_start_pos":   (1,),
                },
                output_shapes = {
                    # last_hidden：最后一个真实token的隐状态，给动作头/文本头用
                    "last_hidden": (self.config.max_batch, self.lcfg.hidden_dim),
                    # 每层KV Cache输出（Prefill后写入KVCacheManager）
                    # 展平为 [num_layers * 2, batch, heads, seq_len, head_dim]
                    "kv_out": (
                        self.lcfg.num_layers * 2,
                        self.config.max_batch,
                        self.lcfg.num_heads,
                        seq_len,
                        self.lcfg.head_dim,
                    ),
                },
                output_dtypes = {
                    "last_hidden": np.float16,
                    "kv_out":      np.float16,
                },
            )
            self.executor.register_graph(meta)
        logger.info(f"[ARIA/LLM] 注册Prefill图: {self.lcfg.prefill_buckets}")

    def _register_decode_graphs(self) -> None:
        """AR模式专用：每个KV Cache长度bucket一张Decode图"""
        for kv_len in self.lcfg.decode_buckets:
            name = f"decode_{kv_len}"
            meta = GraphMeta(
                name  = name,
                path  = f"{self.config.graph_dir}/{name}.bin",
                input_shapes = {
                    "input_id":   (self.config.max_batch, 1),
                    "position_id":(self.config.max_batch, 1),
                    # 当前有效KV Cache（只传有效部分的shape对应的最大bucket）
                    "kv_in": (
                        self.lcfg.num_layers * 2,
                        self.config.max_batch,
                        self.lcfg.num_heads,
                        kv_len,
                        self.lcfg.head_dim,
                    ),
                },
                output_shapes = {
                    "logits": (self.config.max_batch, self.lcfg.vocab_size),
                    "kv_new": (
                        self.lcfg.num_layers * 2,
                        self.config.max_batch,
                        self.lcfg.num_heads,
                        1,
                        self.lcfg.head_dim,
                    ),
                },
                output_dtypes = {
                    "logits": np.float32,
                    "kv_new": np.float16,
                },
            )
            self.executor.register_graph(meta)
        logger.info(f"[ARIA/LLM] 注册Decode图: {self.lcfg.decode_buckets}")

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(self,
                token_ids:   np.ndarray,
                vision_feat: np.ndarray,
                kv_start_pos: int = 0) -> np.ndarray:
        """
        执行Prefill。

        token_ids:    [seq_len] int32，文本部分的token（视觉token由vision_feat提供）
        vision_feat:  [1, total_vision_tokens, feat_dim] float16
        kv_start_pos: 多轮对话时，历史KV的末尾位置

        返回: last_hidden [1, hidden_dim] float16
        """
        actual_len = len(token_ids) + self.vcfg.total_vision_tokens
        total_len  = kv_start_pos + actual_len

        bucket, padded_ids, attn_mask, pos_ids = self._pad_prefill_input(
            token_ids, actual_len, kv_start_pos
        )

        out = self.executor.run(
            f"prefill_{bucket}",
            {
                "input_ids":      padded_ids[np.newaxis, :],        # [1, bucket]
                "vision_feat":    vision_feat,                       # [1, vis_tok, feat]
                "attention_mask": attn_mask[np.newaxis, :],         # [1, bucket]
                "position_ids":   pos_ids[np.newaxis, :],           # [1, bucket]
                "kv_start_pos":   np.array([kv_start_pos], dtype=np.int32),
            }
        )

        # 将Prefill输出的KV写入KVCacheManager
        # kv_out shape: [num_layers*2, batch, heads, seq_len, head_dim]
        kv_out    = out["kv_out"]
        num_layers = self.lcfg.num_layers
        for layer_idx in range(num_layers):
            k = kv_out[layer_idx * 2,     :, :, :actual_len, :]
            v = kv_out[layer_idx * 2 + 1, :, :, :actual_len, :]
            self.kv_cache.write_prefill(layer_idx, k, v, start_pos=kv_start_pos)

        logger.debug(
            f"[ARIA/LLM] Prefill完成: bucket={bucket} "
            f"actual_len={actual_len} kv_start={kv_start_pos}"
        )
        return out["last_hidden"]  # [1, hidden_dim]

    # ------------------------------------------------------------------
    # Decode（AR模式）
    # ------------------------------------------------------------------

    def decode_step(self, token_id: int) -> Tuple[np.ndarray, int]:
        """
        AR Decode单步。

        token_id: 上一步生成的token（或BOS）
        返回: (logits [vocab_size], next_kv_len)
        """
        cur_kv_len = self.kv_cache.valid_len
        bucket     = self._select_decode_bucket(cur_kv_len)

        # 将当前有效KV Cache（pad到bucket长度）作为图输入
        kv_padded = self._pad_kv_for_decode(cur_kv_len, bucket)

        out = self.executor.run(
            f"decode_{bucket}",
            {
                "input_id":    np.array([[token_id]], dtype=np.int32),
                "position_id": np.array([[cur_kv_len]], dtype=np.int32),
                "kv_in":       kv_padded,
            }
        )

        # 将新生成的KV写入Cache
        kv_new = out["kv_new"]  # [num_layers*2, batch, heads, 1, head_dim]
        for layer_idx in range(self.lcfg.num_layers):
            k = kv_new[layer_idx * 2    ]
            v = kv_new[layer_idx * 2 + 1]
            self.kv_cache.write_decode(layer_idx, k, v)
        self.kv_cache.step_forward()

        logits = out["logits"][0]  # [vocab_size]
        logger.debug(f"[ARIA/LLM] Decode step kv_len={cur_kv_len}→{self.kv_cache.valid_len}")
        return logits

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _pad_prefill_input(self,
                           token_ids:    np.ndarray,
                           actual_len:   int,
                           kv_start_pos: int
                           ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        选Prefill bucket并做padding。
        返回 (bucket, padded_ids, attention_mask, position_ids)
        """
        total_len = kv_start_pos + actual_len
        bucket    = self._select_prefill_bucket(total_len)
        pad_len   = bucket - actual_len  # 只pad新增部分

        padded_ids = np.pad(
            token_ids, (0, pad_len),
            constant_values=PAD_TOKEN_ID
        ).astype(np.int32)

        # attention_mask:
        #   历史KV位置（kv_start_pos个）在KV Cache里，attention_mask对应bucket中
        #   这里简化：bucket对应当前新增序列的mask
        #   真实场景中Attention实现需要cross-attend到KV Cache，mask逻辑在模型内部
        attention_mask = np.zeros(bucket, dtype=np.int32)
        attention_mask[:actual_len] = 1

        # position_ids：历史位置 + 当前位置
        position_ids = np.zeros(bucket, dtype=np.int32)
        position_ids[:actual_len] = np.arange(
            kv_start_pos, kv_start_pos + actual_len
        )

        return bucket, padded_ids, attention_mask, position_ids

    def _pad_kv_for_decode(self, cur_len: int, bucket: int) -> np.ndarray:
        """
        将KV Cache pad到bucket长度作为Decode图输入。
        [num_layers*2, batch, heads, bucket, head_dim]
        """
        full_kv = self.kv_cache.get_all_kv()
        # full_kv: [num_layers, 2, batch, heads, valid_len, head_dim]
        nl, _, b, h, vl, d = full_kv.shape
        # 展平layers×2，pad seq维度到bucket
        flat = full_kv.reshape(nl * 2, b, h, vl, d)
        if vl < bucket:
            pad = np.zeros((nl * 2, b, h, bucket - vl, d), dtype=flat.dtype)
            flat = np.concatenate([flat, pad], axis=3)
        return flat[:, :, :, :bucket, :]

    def _select_prefill_bucket(self, length: int) -> int:
        for b in sorted(self.lcfg.prefill_buckets):
            if length <= b:
                return b
        raise ValueError(
            f"序列长度 {length} 超过最大Prefill bucket {max(self.lcfg.prefill_buckets)}"
        )

    def _select_decode_bucket(self, kv_len: int) -> int:
        for b in sorted(self.lcfg.decode_buckets):
            if kv_len <= b:
                return b
        raise ValueError(
            f"KV Cache长度 {kv_len} 超过最大Decode bucket {max(self.lcfg.decode_buckets)}"
        )
