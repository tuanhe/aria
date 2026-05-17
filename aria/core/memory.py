"""
core/memory.py

静态内存规划器。
推理框架启动时一次性规划所有buffer，运行时零malloc。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferSpec:
    name:  str
    shape: Tuple[int, ...]
    dtype: np.dtype
    persistent: bool = True   # True: 整个生命周期保留（权重/KV Cache）
                               # False: 可被其他buffer复用（激活值）


class StaticMemoryPool:
    """
    静态内存规划器（Host侧模拟，真实部署时对应NPU DDR规划）。

    设计原则：
    - 权重区：persistent，所有图共享同一份
    - KV Cache区：persistent，Prefill写 / Decode读
    - Workspace区：non-persistent，所有图复用最大的那块
    - IO Buffer区：输入输出的staging区
    """

    def __init__(self):
        self._buffers:    Dict[str, np.ndarray] = {}
        self._specs:      Dict[str, BufferSpec] = {}
        self._total_bytes = 0

    def register(self, spec: BufferSpec) -> None:
        """注册一个buffer（规划阶段调用，不真正分配）"""
        if spec.name in self._specs:
            existing = self._specs[spec.name]
            # 取shape更大的那个（workspace复用场景）
            if not spec.persistent:
                new_size = int(np.prod(spec.shape)) * np.dtype(spec.dtype).itemsize
                old_size = int(np.prod(existing.shape)) * np.dtype(existing.dtype).itemsize
                if new_size <= old_size:
                    return  # 已有更大的，不需要更新
            self._specs[spec.name] = spec
        else:
            self._specs[spec.name] = spec

    def allocate_all(self) -> None:
        """一次性分配所有已注册的buffer"""
        total = 0
        for name, spec in self._specs.items():
            arr = np.zeros(spec.shape, dtype=spec.dtype)
            self._buffers[name] = arr
            size = arr.nbytes
            total += size
            logger.debug(f"[ARIA/Memory] 分配 {name}: shape={spec.shape} dtype={spec.dtype} size={size/1024:.1f}KB")

        self._total_bytes = total
        logger.info(f"[ARIA/Memory] 总分配: {total / 1024**3:.3f} GB ({len(self._buffers)} 个buffer)")

    def get(self, name: str) -> np.ndarray:
        assert name in self._buffers, f"Buffer '{name}' 未分配，请先调用allocate_all()"
        return self._buffers[name]

    def get_view(self, name: str, shape: Tuple[int, ...]) -> np.ndarray:
        """以指定shape取buffer的视图（零拷贝）"""
        buf = self.get(name)
        return buf.reshape(shape)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def summary(self) -> str:
        lines = ["[Memory Pool Summary]"]
        lines.append(f"  Total: {self._total_bytes / 1024**3:.3f} GB")
        for name, spec in sorted(self._specs.items()):
            size = int(np.prod(spec.shape)) * np.dtype(spec.dtype).itemsize
            tag  = "持久" if spec.persistent else "复用"
            lines.append(f"  [{tag}] {name}: {spec.shape} {spec.dtype} ({size/1024:.1f} KB)")
        return "\n".join(lines)
