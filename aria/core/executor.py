"""
core/executor.py

NPU执行器抽象层。
- NPUExecutor: 抽象基类，定义所有后端必须实现的接口
- MockNPUExecutor: 基于numpy的Mock实现，用于开发/测试阶段

对接真实NPU时，继承NPUExecutor并实现5个抽象方法即可。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class GraphMeta:
    """编译产物元数据"""
    name:          str
    path:          str
    input_shapes:  Dict[str, tuple]
    output_shapes: Dict[str, tuple]
    input_dtypes:  Dict[str, np.dtype] = field(default_factory=dict)
    output_dtypes: Dict[str, np.dtype] = field(default_factory=dict)
    handle:        Any = None          # 厂商SDK返回的图句柄


@dataclass
class ExecutionResult:
    outputs:      Dict[str, np.ndarray]
    latency_ms:   float = 0.0


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class NPUExecutor(ABC):
    """
    NPU执行器抽象基类。

    子类需要实现的5个方法：
      _load_graph   : 加载编译产物，返回图句柄
      _execute      : 执行一次推理
      _alloc_device : 在NPU DDR上分配内存，返回地址
      _h2d          : Host → Device 数据拷贝
      _d2h          : Device → Host 数据拷贝
    """

    def __init__(self):
        self._graphs:       Dict[str, GraphMeta] = {}
        self._weight_addrs: Dict[str, int]       = {}  # 权重名 → DDR地址
        self._profiling:    bool                 = False
        self._stats:        Dict[str, list]      = {}

    # ------------------------------------------------------------------
    # 公开接口（框架层调用）
    # ------------------------------------------------------------------

    def register_graph(self, meta: GraphMeta) -> None:
        """加载并注册一张编译好的NPU图"""
        logger.info(f"[ARIA/Executor] 加载图: {meta.name} <- {meta.path}")
        meta.handle = self._load_graph(meta.path, meta)
        self._graphs[meta.name] = meta
        logger.info(f"[ARIA/Executor] 图 {meta.name} 加载完成")

    def run(self, graph_name: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行指定图的推理。
        自动处理 Host→Device 数据搬运，执行，Device→Host 结果取回。
        本次调用 alloc 出来的所有 device 内存在 D2H 完成后通过
        _free_device 归还（后端可基于此实现池化）。
        """
        assert graph_name in self._graphs, \
            f"图 '{graph_name}' 未注册，已注册: {list(self._graphs.keys())}"

        meta = self._graphs[graph_name]
        t0   = time.perf_counter()
        transient_addrs: list = []   # 本次 run 全部 alloc 的 addr，最后统一释放

        # 输入上传到Device（权重已经常驻Device，不在inputs里）
        device_inputs = {}
        for k, v in inputs.items():
            addr = self._alloc_device(v.nbytes)
            transient_addrs.append(addr)
            self._h2d(v, addr)
            device_inputs[k] = addr

        # NPU执行（输出 addr 通常由子类 _execute 内部 alloc）
        device_outputs = self._execute(meta.handle, device_inputs, meta)
        transient_addrs.extend(device_outputs.values())

        # 结果取回Host
        host_outputs = {}
        for k, addr in device_outputs.items():
            shape = meta.output_shapes[k]
            dtype = meta.output_dtypes.get(k, np.float16)
            host_outputs[k] = self._d2h(addr, shape, dtype)

        # 归还 transient 内存（权重 / 持久 KV cache 不走这里）
        for addr in transient_addrs:
            self._free_device(addr)

        latency_ms = (time.perf_counter() - t0) * 1000
        if self._profiling:
            self._stats.setdefault(graph_name, []).append(latency_ms)
        logger.debug(f"[ARIA/Executor] {graph_name} 耗时 {latency_ms:.2f}ms")

        return host_outputs

    def load_weights(self, weight_dict: Dict[str, np.ndarray]) -> None:
        """
        将权重一次性上传到NPU DDR并记录地址。
        所有图共享这份权重，不重复拷贝。
        """
        logger.info(f"[ARIA/Executor] 上传权重共 {len(weight_dict)} 个张量")
        for name, tensor in weight_dict.items():
            addr = self._alloc_device(tensor.nbytes)
            self._h2d(tensor, addr)
            self._weight_addrs[name] = addr
        logger.info("[ARIA/Executor] 权重上传完成")

    def get_weight_addr(self, name: str) -> int:
        assert name in self._weight_addrs, f"权重 '{name}' 未加载"
        return self._weight_addrs[name]

    def enable_profiling(self, enable: bool = True) -> None:
        self._profiling = enable

    def get_profiling_stats(self) -> Dict[str, dict]:
        result = {}
        for name, latencies in self._stats.items():
            arr = np.array(latencies)
            result[name] = {
                "count":  len(arr),
                "mean":   float(arr.mean()),
                "p50":    float(np.percentile(arr, 50)),
                "p95":    float(np.percentile(arr, 95)),
                "p99":    float(np.percentile(arr, 99)),
                "max":    float(arr.max()),
            }
        return result

    # ------------------------------------------------------------------
    # 子类必须实现的抽象方法
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_graph(self, path: str, meta: GraphMeta) -> Any:
        """加载编译产物，返回图句柄（供_execute使用）"""

    @abstractmethod
    def _execute(self,
                 graph_handle: Any,
                 device_inputs: Dict[str, int],
                 meta: GraphMeta) -> Dict[str, int]:
        """
        在NPU上执行推理。
        输入/输出均为Device侧DDR地址。
        返回 {output_name: device_addr}
        """

    @abstractmethod
    def _alloc_device(self, size: int) -> int:
        """在NPU DDR上分配size字节，返回地址"""

    @abstractmethod
    def _h2d(self, data: np.ndarray, device_addr: int) -> None:
        """将numpy数组拷贝到Device"""

    @abstractmethod
    def _d2h(self, device_addr: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """从Device拷贝数据到Host，返回numpy数组"""

    # ------------------------------------------------------------------
    # 可选钩子：归还 _alloc_device 拿到的地址（默认 no-op）
    # ------------------------------------------------------------------

    def _free_device(self, device_addr: int) -> None:
        """
        把 addr 归还给后端的显存管理。默认 no-op——对 Mock 这种
        无所谓的实现没影响；对 TRT/QNN 这种真后端，应该重写为
        释放或归还到池里，避免每步推理都 cudaMalloc 泄漏。
        """
        return None


# ---------------------------------------------------------------------------
# Mock实现（基于numpy，用于开发/测试）
# ---------------------------------------------------------------------------

class MockNPUExecutor(NPUExecutor):
    """
    基于numpy的Mock执行器。
    不依赖任何NPU SDK，输出随机张量，用于验证框架流程的正确性。
    对接真实NPU时替换这个类即可，上层代码完全不变。
    """

    def __init__(self, latency_ms: float = 5.0, seed: int = 42):
        super().__init__()
        self._latency_ms = latency_ms          # 模拟NPU延迟
        self._device_mem: Dict[int, np.ndarray] = {}  # 模拟Device内存
        self._next_addr  = 0x10000000
        self._rng        = np.random.default_rng(seed)

    def _load_graph(self, path: str, meta: GraphMeta) -> Any:
        # Mock: 不真正加载文件，直接返回meta作为句柄
        logger.debug(f"[ARIA/Mock] 模拟加载图文件: {path}")
        return meta

    def _execute(self,
                 graph_handle: Any,
                 device_inputs: Dict[str, int],
                 meta: GraphMeta) -> Dict[str, int]:
        # 模拟NPU计算延迟
        time.sleep(self._latency_ms / 1000.0)

        # 为每个输出生成随机张量并存入模拟Device内存
        output_addrs = {}
        for out_name, out_shape in meta.output_shapes.items():
            dtype = meta.output_dtypes.get(out_name, np.float16)
            data  = self._rng.standard_normal(out_shape).astype(dtype)
            addr  = self._alloc_device(data.nbytes)
            self._device_mem[addr] = data
            output_addrs[out_name] = addr

        return output_addrs

    def _alloc_device(self, size: int) -> int:
        addr = self._next_addr
        self._next_addr += size + 64  # 64字节对齐
        return addr

    def _h2d(self, data: np.ndarray, device_addr: int) -> None:
        self._device_mem[device_addr] = data.copy()

    def _d2h(self, device_addr: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
        if device_addr in self._device_mem:
            return self._device_mem[device_addr].reshape(shape).astype(dtype)
        # 如果地址不存在，返回零张量（防止测试崩溃）
        return np.zeros(shape, dtype=dtype)

    def _free_device(self, device_addr: int) -> None:
        # Mock 没有真实 device，简单把对应 host 副本丢掉，避免长跑时
        # _device_mem dict 无限增长
        self._device_mem.pop(device_addr, None)
