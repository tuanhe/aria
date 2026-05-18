"""
core/kv_cache.py

KV Cache管理器。
- 静态预分配，避免运行时malloc
- 支持Prefill写入 / Decode增量更新
- 支持多轮对话跨轮复用（VLM模式）
- 提供当前有效长度管理
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KVCacheManager:
    """
    KV Cache静态管理器。

    内存布局：
      cache[layer_idx, kv_idx, batch, head, seq, dim]
      kv_idx: 0=K, 1=V

    使用方式：
      Prefill后：write_prefill(layer_idx, k, v)
      每步Decode后：write_decode(layer_idx, k, v)
      新对话：reset()
      跨轮追加：append_turn()（VLM多轮模式）
    """

    def __init__(self,
                 num_layers:   int,
                 num_heads:    int,
                 head_dim:     int,
                 max_seq_len:  int,
                 max_batch:    int  = 1,
                 dtype:        np.dtype = np.float16):

        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.max_batch   = max_batch
        self.dtype       = dtype

        # [num_layers, 2(K/V), batch, heads, max_seq, head_dim]
        self._cache = np.zeros(
            (num_layers, 2, max_batch, num_heads, max_seq_len, head_dim),
            dtype=dtype
        )

        self._valid_len:    int = 0   # 当前有效的KV Cache长度
        self._history_len:  int = 0   # 多轮对话中，历史轮次的长度

        nbytes = self._cache.nbytes
        logger.info(
            f"[ARIA/KVCache] 初始化: layers={num_layers} heads={num_heads} "
            f"head_dim={head_dim} max_seq={max_seq_len} "
            f"大小={nbytes/1024**3:.3f}GB"
        )

    # ------------------------------------------------------------------
    # 写入接口
    # ------------------------------------------------------------------

    def write_prefill(self,
                      layer_idx: int,
                      k: np.ndarray,
                      v: np.ndarray,
                      start_pos: int = 0) -> None:
        """
        Prefill阶段批量写入KV。
        k/v shape: [batch, heads, seq_len, head_dim]
        start_pos: 多轮对话时从历史末尾开始写
        """
        seq_len = k.shape[2]
        end_pos = start_pos + seq_len
        assert end_pos <= self.max_seq_len, \
            f"KV Cache溢出: {end_pos} > {self.max_seq_len}"

        self._cache[layer_idx, 0, :, :, start_pos:end_pos, :] = k
        self._cache[layer_idx, 1, :, :, start_pos:end_pos, :] = v
        self._valid_len = end_pos

    def write_decode(self,
                     layer_idx: int,
                     k: np.ndarray,
                     v: np.ndarray) -> None:
        """
        Decode阶段写入单步KV（每次写一个token位置）。
        k/v shape: [batch, heads, 1, head_dim]
        """
        pos = self._valid_len
        assert pos < self.max_seq_len, \
            f"KV Cache已满: valid_len={pos} max={self.max_seq_len}"

        self._cache[layer_idx, 0, :, :, pos:pos+1, :] = k
        self._cache[layer_idx, 1, :, :, pos:pos+1, :] = v
        # valid_len由step_forward()统一推进，避免多层重复计数

    def step_forward(self) -> None:
        """Decode一步完成后，推进valid_len"""
        self._valid_len += 1

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------

    def get_kv(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """读取指定层的完整有效KV，用于Attention计算"""
        k = self._cache[layer_idx, 0, :, :, :self._valid_len, :]
        v = self._cache[layer_idx, 1, :, :, :self._valid_len, :]
        return k, v

    def get_all_kv(self) -> np.ndarray:
        """返回所有层的有效KV（用于将整块KV作为图输入）"""
        return self._cache[:, :, :, :, :self._valid_len, :]

    def read_range(self, start: int, end: int) -> np.ndarray:
        """
        读取 [start, end) 范围的 KV（单 batch 切片），用于写入前缀缓存。
        返回: [num_layers, 2, num_heads, end-start, head_dim]
        """
        assert 0 <= start <= end <= self._valid_len, \
            f"read_range 越界: [{start},{end}) vs valid_len={self._valid_len}"
        return self._cache[:, :, 0, :, start:end, :].copy()

    def bulk_load_prefix(self, kv_data: np.ndarray) -> None:
        """
        把一段连续的 KV 写入工作区开头（用于前缀缓存命中后回灌）。

        kv_data shape: [num_layers, 2, num_heads, M, head_dim]
                       M 必须 <= max_seq_len

        写入后 valid_len = history_len = M
        （前缀视为已成既往的历史，后续 prefill 从 M 起继续写）
        """
        assert self._valid_len == 0, \
            f"bulk_load_prefix 只能在空 Cache 上调用（当前 valid_len={self._valid_len}）"
        L, two, H, M, D = kv_data.shape
        assert L  == self.num_layers, f"num_layers 不匹配: {L} vs {self.num_layers}"
        assert two == 2,              f"K/V 维必须是 2, got {two}"
        assert H  == self.num_heads,  f"num_heads 不匹配: {H} vs {self.num_heads}"
        assert D  == self.head_dim,   f"head_dim 不匹配: {D} vs {self.head_dim}"
        assert M  <= self.max_seq_len, f"前缀长度 {M} > max_seq_len {self.max_seq_len}"

        # 端侧 batch=1：广播到 batch 维（单元素 slice）
        self._cache[:, :, 0, :, :M, :] = kv_data
        self._valid_len   = M
        self._history_len = M
        logger.debug(f"[ARIA/KVCache] 前缀回灌 M={M}")

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        重置Cache（新对话开始）。
        不清零内存，下次写入时会覆盖，节省时间。
        """
        self._valid_len   = 0
        self._history_len = 0
        logger.debug("[ARIA/KVCache] 已重置")

    def save_turn(self) -> None:
        """
        VLM多轮模式：保存当前轮的KV为历史。
        下一轮Prefill从_history_len位置开始写。
        """
        self._history_len = self._valid_len
        logger.debug(f"[ARIA/KVCache] 保存轮次，history_len={self._history_len}")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def valid_len(self) -> int:
        return self._valid_len

    @property
    def history_len(self) -> int:
        return self._history_len

    @property
    def remaining(self) -> int:
        return self.max_seq_len - self._valid_len

    def __repr__(self) -> str:
        return (
            f"KVCacheManager(valid={self._valid_len}/{self.max_seq_len}, "
            f"history={self._history_len}, "
            f"layers={self.num_layers})"
        )
