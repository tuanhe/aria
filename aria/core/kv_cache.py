"""
core/kv_cache.py

KV Cache 管理器（device-resident，单图 decode 设计）。

- 按 max_seq_len 一次性在 device 上预分配一块常驻 buffer（per-session）
  布局：buffer[layer*2 + kv, batch, head, seq, dim]，kv: 0=K, 1=V
       即 shape = [num_layers*2, max_batch, num_heads, max_seq_len, head_dim]
- decode 阶段这块 buffer 被 backbone bind 成 decode 图的 kv_cache 输入，
  自回归每步只把新一行写回（write_kv_seq 跨步写），不再整块重传。
- 读接口（get_kv / read_range）走 device→host 回读，只用于前缀缓存回写等
  冷路径，不在 decode 热路径上。
- 无 executor 时自带一个私有 MockNPUExecutor，保证脱离运行时也能单测。

使用方式：
  Prefill 后：       write_prefill(layer_idx, k, v, start_pos)
  每步 Decode 后：    write_decode(layer_idx, k, v) + step_forward()
  新对话：           reset()
  多轮：             save_turn()
  前缀缓存命中回灌：  bulk_load_prefix(kv_data)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KVCacheManager:
    """
    KV Cache 静态管理器（device-resident）。

    device buffer 由本管理器持有（per-session 生命周期）；backbone 负责把
    self.addr bind 到 decode 图的 kv_cache 输入。
    """

    def __init__(self,
                 num_layers:   int,
                 num_heads:    int,
                 head_dim:     int,
                 max_seq_len:  int,
                 max_batch:    int  = 1,
                 dtype:        np.dtype = np.float16,
                 executor=None):

        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.max_batch   = max_batch
        self.dtype       = np.dtype(dtype)

        # device buffer 布局：[L*2, batch, heads, max_seq, head_dim]
        self.buffer_shape = (
            num_layers * 2, max_batch, num_heads, max_seq_len, head_dim,
        )

        # 无 executor（脱离运行时的单测场景）时退化用私有 Mock
        if executor is None:
            from aria.core.executor import MockNPUExecutor
            executor = MockNPUExecutor(latency_ms=0.0)
        self.executor = executor

        nbytes = int(np.prod(self.buffer_shape)) * self.dtype.itemsize
        self._addr = self.executor.alloc_persistent(nbytes)
        self.executor.init_persistent(
            self._addr, np.zeros(self.buffer_shape, dtype=self.dtype)
        )

        self._valid_len:    int = 0   # 当前有效的 KV Cache 长度
        self._history_len:  int = 0   # 多轮对话中，历史轮次的长度

        logger.info(
            f"[ARIA/KVCache] 初始化(device): layers={num_layers} heads={num_heads} "
            f"head_dim={head_dim} max_seq={max_seq_len} "
            f"buffer={nbytes/1024**3:.3f}GB @0x{self._addr:x}"
        )

    # ------------------------------------------------------------------
    # device buffer 句柄
    # ------------------------------------------------------------------

    @property
    def addr(self) -> int:
        """常驻 device buffer 地址（backbone bind 给 decode 图）。"""
        return self._addr

    # ------------------------------------------------------------------
    # 写入接口（落到 device 跨步写）
    # ------------------------------------------------------------------

    def write_prefill(self,
                      layer_idx: int,
                      k: np.ndarray,
                      v: np.ndarray,
                      start_pos: int = 0) -> None:
        """
        Prefill 阶段批量写入某层 KV。
        k/v shape: [batch, heads, seq_len, head_dim]
        start_pos: 多轮对话时从历史末尾开始写
        """
        seq_len = k.shape[2]
        end_pos = start_pos + seq_len
        assert end_pos <= self.max_seq_len, \
            f"KV Cache溢出: {end_pos} > {self.max_seq_len}"

        block = np.stack([k, v], axis=0)   # [2, batch, heads, seq_len, head_dim]
        self.executor.write_kv_seq(
            self._addr, self.buffer_shape, self.dtype,
            start=start_pos, block=block, plane0=layer_idx * 2,
        )
        self._valid_len = end_pos

    def write_decode(self,
                     layer_idx: int,
                     k: np.ndarray,
                     v: np.ndarray) -> None:
        """
        Decode 阶段写入单步 KV（每次写一个 token 位置）。
        k/v shape: [batch, heads, 1, head_dim]
        valid_len 由 step_forward() 统一推进，避免多层重复计数。
        """
        pos = self._valid_len
        assert pos < self.max_seq_len, \
            f"KV Cache已满: valid_len={pos} max={self.max_seq_len}"

        block = np.stack([k, v], axis=0)   # [2, batch, heads, 1, head_dim]
        self.executor.write_kv_seq(
            self._addr, self.buffer_shape, self.dtype,
            start=pos, block=block, plane0=layer_idx * 2,
        )

    def step_forward(self) -> None:
        """Decode 一步完成后，推进 valid_len。"""
        self._valid_len += 1

    # ------------------------------------------------------------------
    # 读取接口（device→host 回读，冷路径）
    # ------------------------------------------------------------------

    def _read_buffer(self) -> np.ndarray:
        """整块回读 device buffer → [L*2, batch, heads, max_seq, head_dim]。"""
        return self.executor.read_persistent(
            self._addr, self.buffer_shape, self.dtype
        )

    def get_kv(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """读取指定层的完整有效 KV，用于（host 侧）Attention 计算或校验。"""
        buf = self._read_buffer()
        k = buf[layer_idx * 2,     :, :, :self._valid_len, :]
        v = buf[layer_idx * 2 + 1, :, :, :self._valid_len, :]
        return k, v

    def read_range(self, start: int, end: int) -> np.ndarray:
        """
        读取 [start, end) 范围的 KV（单 batch 切片），用于写入前缀缓存。
        返回: [num_layers, 2, num_heads, end-start, head_dim]
        """
        assert 0 <= start <= end <= self._valid_len, \
            f"read_range 越界: [{start},{end}) vs valid_len={self._valid_len}"
        buf = self._read_buffer()   # [L*2, batch, heads, max_seq, head_dim]
        buf = buf.reshape(self.num_layers, 2, self.max_batch,
                          self.num_heads, self.max_seq_len, self.head_dim)
        return buf[:, :, 0, :, start:end, :].copy()

    def bulk_load_prefix(self, kv_data: np.ndarray) -> None:
        """
        把一段连续的 KV 写入工作区开头（前缀缓存命中后回灌）。

        kv_data shape: [num_layers, 2, num_heads, M, head_dim]，M <= max_seq_len
        写入后 valid_len = history_len = M
        """
        assert self._valid_len == 0, \
            f"bulk_load_prefix 只能在空 Cache 上调用（当前 valid_len={self._valid_len}）"
        L, two, H, M, D = kv_data.shape
        assert L  == self.num_layers, f"num_layers 不匹配: {L} vs {self.num_layers}"
        assert two == 2,              f"K/V 维必须是 2, got {two}"
        assert H  == self.num_heads,  f"num_heads 不匹配: {H} vs {self.num_heads}"
        assert D  == self.head_dim,   f"head_dim 不匹配: {D} vs {self.head_dim}"
        assert M  <= self.max_seq_len, f"前缀长度 {M} > max_seq_len {self.max_seq_len}"

        # [L, 2, H, M, D] → [L*2, 1, H, M, D]（端侧单 batch）
        block = kv_data.reshape(L * 2, H, M, D)[:, np.newaxis, :, :, :]
        self.executor.write_kv_seq(
            self._addr, self.buffer_shape, self.dtype,
            start=0, block=block, plane0=0,
        )
        self._valid_len   = M
        self._history_len = M
        logger.debug(f"[ARIA/KVCache] 前缀回灌 M={M}")

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置 Cache（新对话开始）。不清零 buffer，下次写入会覆盖。"""
        self._valid_len   = 0
        self._history_len = 0
        logger.debug("[ARIA/KVCache] 已重置")

    def save_turn(self) -> None:
        """VLM 多轮：保存当前轮 KV 为历史，下一轮 Prefill 从 history_len 续写。"""
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
            f"layers={self.num_layers}, device@0x{self._addr:x})"
        )
