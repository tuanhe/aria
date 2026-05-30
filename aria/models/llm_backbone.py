"""
models/llm_backbone.py

LLM Backbone推理封装。
- 管理多个Prefill静态图（按seq_len分bucket）
- 管理单张Decode静态图（固定 max buffer + 偏移，无 bucket）
- 负责Padding prefill输入到对应bucket
- KV Cache由 device-resident 的 KVCacheManager 管理，decode buffer 被 bind 到 decode 图
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
    Decode图： 单张静态图，输入单token + position + attention_mask（kv_cache 常驻 bind），
              输出logits + 新增一步KV
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

        # 把本 session 的常驻 KV buffer 绑到 decode 单图的 kv_cache 输入。
        # 绑定后自回归每步不再 H2D 重传整块 KV，只写回新增的一行。
        # （harvest / 纯导出场景 kv_cache=None，跳过绑定）
        if self.kv_cache is not None:
            self.executor.bind_input("decode", "kv_cache", self.kv_cache.addr)

    # ------------------------------------------------------------------
    # 图注册
    # ------------------------------------------------------------------

    def _register_prefill_graphs(self) -> None:
        is_llm = self.config.mode == "llm"
        for seq_len in self.lcfg.prefill_buckets:
            name = f"prefill_{seq_len}"

            input_shapes: Dict = {
                "input_ids":      (self.config.max_batch, seq_len),
                "attention_mask": (self.config.max_batch, seq_len),
                "position_ids":   (self.config.max_batch, seq_len),
                "kv_start_pos":   (1,),
            }
            if not is_llm:
                input_shapes["vision_feat"] = (
                    self.config.max_batch,
                    self.vcfg.total_vision_tokens,
                    self.vcfg.feat_dim,
                )

            kv_shape = (
                self.lcfg.num_layers * 2,
                self.config.max_batch,
                self.lcfg.num_heads,
                seq_len,
                self.lcfg.head_dim,
            )
            if is_llm:
                output_shapes  = {"logits": (self.config.max_batch, self.lcfg.vocab_size),
                                   "kv_out": kv_shape}
                output_dtypes  = {"logits": np.float32, "kv_out": np.float16}
            else:
                output_shapes  = {"last_hidden": (self.config.max_batch, self.lcfg.hidden_dim),
                                   "kv_out": kv_shape}
                output_dtypes  = {"last_hidden": np.float16, "kv_out": np.float16}

            meta = GraphMeta(
                name          = name,
                path          = f"{self.config.graph_dir}/{name}.bin",
                input_shapes  = input_shapes,
                output_shapes = output_shapes,
                output_dtypes = output_dtypes,
            )
            self.executor.register_graph(meta)
        logger.info(f"[ARIA/LLM] 注册Prefill图: {self.lcfg.prefill_buckets} (mode={self.config.mode})")

    def _register_decode_graphs(self) -> None:
        """
        AR / 文本 decode：单张静态图（固定 max buffer + 偏移），无 bucket。

        kv_cache 输入恒为 max_seq_len 长度，作为常驻 buffer 由 executor 绑定；
        自回归每步只变 position_id / attention_mask（数据），图本身唯一。
        """
        max_seq = self.lcfg.max_seq_len
        meta = GraphMeta(
            name  = "decode",
            path  = f"{self.config.graph_dir}/decode.bin",
            input_shapes = {
                "input_id":       (self.config.max_batch, 1),
                "position_id":    (self.config.max_batch, 1),
                # [0,pos)=1 有效历史, [pos,MAX)=0 垃圾尾, [MAX]=1 当前 token 自身
                "attention_mask": (self.config.max_batch, max_seq + 1),
                # 常驻 KV buffer（max_seq_len 全长），executor 绑定，不逐步重传
                "kv_cache": (
                    self.lcfg.num_layers * 2,
                    self.config.max_batch,
                    self.lcfg.num_heads,
                    max_seq,
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
        logger.info(f"[ARIA/LLM] 注册Decode单图: kv_cache max_seq={max_seq}")

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(self,
                token_ids:    np.ndarray,
                vision_feat:  Optional[np.ndarray] = None,
                kv_start_pos: int = 0) -> np.ndarray:
        """
        执行Prefill。

        token_ids:    [seq_len] int32
        vision_feat:  [1, total_vision_tokens, feat_dim] float16，llm 模式传 None
        kv_start_pos: 多轮对话时历史 KV 的末尾位置

        返回:
          llm 模式  → logits [vocab_size] float32
          vlm/vla   → last_hidden [1, hidden_dim] float16
        """
        is_llm     = self.config.mode == "llm"
        vis_tokens = 0 if is_llm else self.vcfg.total_vision_tokens
        actual_len = len(token_ids) + vis_tokens

        bucket, padded_ids, attn_mask, pos_ids = self._pad_prefill_input(
            token_ids, actual_len, kv_start_pos
        )

        inputs = {
            "input_ids":      padded_ids[np.newaxis, :],
            "attention_mask": attn_mask[np.newaxis, :],
            "position_ids":   pos_ids[np.newaxis, :],
            "kv_start_pos":   np.array([kv_start_pos], dtype=np.int32),
        }
        if not is_llm:
            inputs["vision_feat"] = vision_feat

        out = self.executor.run(f"prefill_{bucket}", inputs)

        kv_out = out["kv_out"]
        for layer_idx in range(self.lcfg.num_layers):
            k = kv_out[layer_idx * 2,     :, :, :actual_len, :]
            v = kv_out[layer_idx * 2 + 1, :, :, :actual_len, :]
            self.kv_cache.write_prefill(layer_idx, k, v, start_pos=kv_start_pos)

        logger.debug(
            "[ARIA/LLM] Prefill完成: bucket=%d actual_len=%d kv_start=%d",
            bucket, actual_len, kv_start_pos,
        )
        if is_llm:
            return out["logits"][0]    # [vocab_size] float32
        return out["last_hidden"]      # [1, hidden_dim] float16

    # ------------------------------------------------------------------
    # Decode（AR模式）
    # ------------------------------------------------------------------

    def decode_step(self, token_id: int) -> Tuple[np.ndarray, int]:
        """
        AR Decode单步。

        token_id: 上一步生成的token（或BOS）
        返回: logits [vocab_size]
        """
        cur_kv_len = self.kv_cache.valid_len
        max_seq    = self.lcfg.max_seq_len
        assert cur_kv_len < max_seq, \
            f"KV Cache已满: valid_len={cur_kv_len} max={max_seq}"

        # attention_mask: [0,pos)=1 有效历史, [pos,MAX)=0 垃圾尾, [MAX]=1 当前 token
        attn_mask = np.zeros((self.config.max_batch, max_seq + 1), dtype=np.int32)
        attn_mask[:, :cur_kv_len] = 1
        attn_mask[:, max_seq]     = 1

        # kv_cache 输入已 bind 为常驻 buffer，这里不再传（executor.run 自动复用绑定地址）
        out = self.executor.run(
            "decode",
            {
                "input_id":       np.array([[token_id]], dtype=np.int32),
                "position_id":    np.array([[cur_kv_len]], dtype=np.int32),
                "attention_mask": attn_mask,
            }
        )

        # 将新生成的KV写回常驻 buffer 第 cur_kv_len 行
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

    def _select_prefill_bucket(self, length: int) -> int:
        for b in sorted(self.lcfg.prefill_buckets):
            if length <= b:
                return b
        raise ValueError(
            f"序列长度 {length} 超过最大Prefill bucket {max(self.lcfg.prefill_buckets)}"
        )
