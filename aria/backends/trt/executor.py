"""
aria/backends/trt/executor.py

TensorRT 后端的 NPU 执行器。在 NVIDIA Jetson / 桌面 CUDA 上把
预编译的 .engine（TRT plan）当作"NPU 编译产物"来跑，对 ARIA
框架而言行为跟真 NPU 一致：静态 shape、独立 device 内存、异步 stream。

通过 BuilderFlag.FP16 + DeviceType.DLA + allowGPUFallback 构建出来的
engine，会优先把支持的层下到 Orin 的 DLA 硬件 NPU 上，未支持算子
走 GPU fallback——这是目前最贴近真 NPU 行为的"模拟器"。
"""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Any, Dict, List

import numpy as np

from aria.core.executor import GraphMeta, NPUExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 显存池：按"精确字节大小"做 free-list，避免热路径反复 cudaMalloc/cudaFree
# ---------------------------------------------------------------------------

class _DevicePool:
    """
    每个 size class 一份 free-list；acquire() 优先复用，否则 cudaMalloc。
    release() 把 addr 退回对应 size 的 free-list。destroy() 时统一 cudaFree。

    ARIA 里每张图的 I/O 形状是静态的——同一 graph 反复跑会落到同一
    bucket，命中率接近 100%，cold path 只在第一次 run 时发生。
    """

    def __init__(self, cudart_module):
        self._cudart = cudart_module
        self._free: Dict[int, List[int]] = {}   # size → 闲置 addr 列表
        self._addr_size: Dict[int, int] = {}    # addr → size（addr 永远活在这张表里直到 destroy）

    def acquire(self, size: int) -> int:
        if size <= 0:
            size = 1
        free_list = self._free.get(size)
        if free_list:
            return free_list.pop()
        err, ptr = self._cudart.cudaMalloc(size)
        if int(err) != 0:
            raise RuntimeError(f"[ARIA/TRT] cudaMalloc({size}) 失败 err={int(err)}")
        addr = int(ptr)
        self._addr_size[addr] = size
        return addr

    def release(self, addr: int) -> None:
        size = self._addr_size.get(addr)
        if size is None:
            # 不是池子发出去的，直接 free，避免悬挂
            self._cudart.cudaFree(addr)
            return
        self._free.setdefault(size, []).append(addr)

    def destroy(self) -> None:
        for addr in list(self._addr_size.keys()):
            self._cudart.cudaFree(addr)
        self._free.clear()
        self._addr_size.clear()

    def stats(self) -> Dict[str, Any]:
        total_bytes = sum(self._addr_size.values())
        free_bytes  = sum(sz * len(lst) for sz, lst in self._free.items())
        return {
            "buffers_owned":   len(self._addr_size),
            "buffers_free":    sum(len(v) for v in self._free.values()),
            "bytes_owned":     total_bytes,
            "bytes_free":      free_bytes,
            "bytes_in_use":    total_bytes - free_bytes,
            "size_classes":    sorted(self._free.keys()),
        }


def _check(err, ctx: str = "") -> None:
    """cuda-python 调用统一错误检查。"""
    from cuda import cudart
    if isinstance(err, tuple):
        err = err[0]
    if int(err) != int(cudart.cudaError_t.cudaSuccess):
        raise RuntimeError(f"[ARIA/TRT] CUDA error {int(err)} at {ctx}")


_TRT_TO_NP = {
    # 延后建表 —— trt.DataType 是运行时才可用
}


def _trt_to_np_dtype(trt_dtype) -> np.dtype:
    import tensorrt as trt
    global _TRT_TO_NP
    if not _TRT_TO_NP:
        _TRT_TO_NP = {
            trt.DataType.FLOAT:  np.float32,
            trt.DataType.HALF:   np.float16,
            trt.DataType.INT8:   np.int8,
            trt.DataType.INT32:  np.int32,
            trt.DataType.INT64:  np.int64,
            trt.DataType.BOOL:   np.bool_,
            trt.DataType.UINT8:  np.uint8,
            trt.DataType.BF16:   np.float16,  # numpy 没有 bf16，记录为 fp16 占位
        }
    return np.dtype(_TRT_TO_NP[trt_dtype])


class TensorRTExecutor(NPUExecutor):
    """
    TensorRT 后端。每张图一份 ICudaEngine + IExecutionContext，
    所有图共享一条 CUDA stream，输入/输出全部走显存指针。
    """

    def __init__(self, verbose: bool = False):
        super().__init__()

        import tensorrt as trt
        from cuda import cudart

        self._trt = trt
        self._cudart = cudart

        self._logger = trt.Logger(
            trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        )
        self._runtime = trt.Runtime(self._logger)

        err, self._stream = cudart.cudaStreamCreate()
        _check(err, "cudaStreamCreate")

        self._engines: Dict[str, Any] = {}    # graph_name → engine
        self._contexts: Dict[str, Any] = {}   # graph_name → execution context
        self._pool = _DevicePool(cudart)       # 复用 device 内存，避免每步 cudaMalloc

        logger.info("[ARIA/TRT] TensorRT 执行器已初始化 (TRT %s)", trt.__version__)

    # ------------------------------------------------------------------
    # NPUExecutor 抽象方法实现
    # ------------------------------------------------------------------

    def _load_graph(self, path: str, meta: GraphMeta) -> Any:
        with open(path, "rb") as f:
            blob = f.read()
        engine = self._runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError(f"[ARIA/TRT] 反序列化失败: {path}")
        ctx = engine.create_execution_context()
        if ctx is None:
            raise RuntimeError(f"[ARIA/TRT] 无法创建 execution context: {meta.name}")

        # 用 engine 实际声明的 dtype 回填 meta（如果用户没写）
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            dt    = _trt_to_np_dtype(engine.get_tensor_dtype(tname))
            mode  = engine.get_tensor_mode(tname)
            if mode == self._trt.TensorIOMode.INPUT:
                meta.input_dtypes.setdefault(tname, dt)
            else:
                meta.output_dtypes.setdefault(tname, dt)

        self._engines[meta.name]  = engine
        self._contexts[meta.name] = ctx
        return meta.name

    def _execute(self,
                 graph_handle: Any,
                 device_inputs: Dict[str, int],
                 meta: GraphMeta) -> Dict[str, int]:
        cudart = self._cudart
        trt    = self._trt
        engine = self._engines[graph_handle]
        ctx    = self._contexts[graph_handle]

        # 为每个 output 预分配 device 内存
        output_addrs: Dict[str, int] = {}
        for out_name, out_shape in meta.output_shapes.items():
            dtype = meta.output_dtypes.get(out_name, np.float16)
            nbytes = int(np.prod(out_shape)) * np.dtype(dtype).itemsize
            addr = self._alloc_device(nbytes)
            output_addrs[out_name] = addr

        # 绑定所有 I/O 张量地址
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            mode  = engine.get_tensor_mode(tname)
            if mode == trt.TensorIOMode.INPUT:
                if tname not in device_inputs:
                    raise RuntimeError(
                        f"[ARIA/TRT] 图 {meta.name} 缺少输入 '{tname}'，"
                        f"实际提供: {list(device_inputs.keys())}"
                    )
                ctx.set_tensor_address(tname, int(device_inputs[tname]))
            else:
                if tname not in output_addrs:
                    raise RuntimeError(
                        f"[ARIA/TRT] meta.output_shapes 未声明输出 '{tname}'"
                    )
                ctx.set_tensor_address(tname, int(output_addrs[tname]))

        # 异步执行 + 同步
        ok = ctx.execute_async_v3(stream_handle=int(self._stream))
        if not ok:
            raise RuntimeError(f"[ARIA/TRT] 执行失败: {meta.name}")
        err = cudart.cudaStreamSynchronize(self._stream)
        _check(err, f"cudaStreamSynchronize({meta.name})")

        return output_addrs

    def _alloc_device(self, size: int) -> int:
        return self._pool.acquire(size)

    def _free_device(self, device_addr: int) -> None:
        self._pool.release(device_addr)

    def get_pool_stats(self) -> Dict[str, Any]:
        """显存池状态，方便诊断热路径是否仍在 cudaMalloc。"""
        return self._pool.stats()

    def _h2d(self, data: np.ndarray, device_addr: int) -> None:
        cudart = self._cudart
        data = np.ascontiguousarray(data)
        err = cudart.cudaMemcpyAsync(
            device_addr,
            data.ctypes.data,
            data.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self._stream,
        )
        _check(err, "cudaMemcpyAsync H2D")
        err = cudart.cudaStreamSynchronize(self._stream)
        _check(err, "cudaStreamSynchronize H2D")

    def _d2h(self, device_addr: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
        cudart = self._cudart
        dtype = np.dtype(dtype)
        host = np.empty(shape, dtype=dtype)
        err = cudart.cudaMemcpyAsync(
            host.ctypes.data,
            device_addr,
            host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self._stream,
        )
        _check(err, "cudaMemcpyAsync D2H")
        err = cudart.cudaStreamSynchronize(self._stream)
        _check(err, "cudaStreamSynchronize D2H")
        return host

    # ------------------------------------------------------------------
    # 资源清理
    # ------------------------------------------------------------------

    def close(self) -> None:
        cudart = self._cudart
        self._pool.destroy()
        if self._stream is not None:
            cudart.cudaStreamDestroy(self._stream)
            self._stream = None
        self._contexts.clear()
        self._engines.clear()
        logger.info("[ARIA/TRT] 资源已释放")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
