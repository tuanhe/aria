"""
aria/backends/ort/executor.py

ONNXRuntime 后端。设计目的不是"最快"，而是把 ARIA"权重一份 +
多图共享"这件事情清晰地演示出来——这一点是 TRT 后端做不到的
（TRT engine 把权重烘进 kernel 了，每个 .engine 必然带一份）。

机制：
  1) build 阶段把跨图同名 initializer 剥到 shared_weights.npz
  2) executor 启动 / 首次 _load_graph 时把 npz 整盘加载成 dict[name → ndarray]
  3) 对每个 name 做 OrtValue.ortvalue_from_numpy(arr) —— OrtValue 不拷贝，
     直接引用那块 numpy memory
  4) 同一份 SessionOptions.add_initializer(name, ortvalue) 注入所有 session

物理上 N 张图共享同一块权重 buffer，对应真 NPU 的 "权重 DDR 常驻 +
多图引用 device 地址"。

get_sharing_stats() 给出 "如果每个 session 各自带一份 = N×B 字节，
实际只占 B 字节" 的对照，方便直观验证。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from aria.core.executor import GraphMeta, NPUExecutor

logger = logging.getLogger(__name__)


class ORTExecutor(NPUExecutor):
    """
    用 ONNXRuntime 跑剥过权重的 ONNX，权重通过 add_initializer 共享。

    构造参数：
        weights_npz: shared_weights.npz 路径；为 None 时在首次 _load_graph
                     时尝试从该图所在目录自动发现（约定优于配置）。
        providers:   ORT EP 列表；默认按可用情况优选 CUDA → CPU。
        latency_ms:  Mock 风格的额外延迟（用作对比测试），默认 0。
    """

    def __init__(self,
                 weights_npz: Optional[str] = None,
                 providers: Optional[List[str]] = None,
                 verbose: bool = False,
                 latency_ms: float = 0.0):
        super().__init__()

        import onnxruntime as ort
        self._ort = ort

        self._verbose    = verbose
        self._latency_ms = latency_ms
        self._weights_npz_arg = weights_npz
        self._weights_loaded  = False

        self._shared_weights:   Dict[str, np.ndarray] = {}    # name → ndarray（持有以防 GC）
        self._shared_ortvalues: Dict[str, Any]        = {}    # name → OrtValue
        # OrtValue 实例在所有 session 间共享（add_external_initializers 拿走的是
        # 同一个引用，不复制底层内存）；SessionOptions 必须每个 session 一份，
        # 因为 add_external_initializers 要求传入的 name 必须真存在于该 session
        # 的 model.graph.initializer 里，跨 session 名字不一样就会校验失败。

        avail = set(ort.get_available_providers())
        if providers is None:
            preferred = ["CUDAExecutionProvider",
                         "TensorrtExecutionProvider",
                         "CPUExecutionProvider"]
            providers = [p for p in preferred if p in avail] or ["CPUExecutionProvider"]
        self._providers = providers

        self._sessions: Dict[str, Any] = {}    # graph_name → InferenceSession

        # 用 Mock 风格的 fake device addr 让 base.run() 流程跑通
        self._fake_addr = 0x10000000
        self._mem: Dict[int, np.ndarray] = {}

        logger.info(
            "[ARIA/ORT] 已初始化 (ort=%s providers=%s)", ort.__version__, providers
        )

    # ------------------------------------------------------------------
    # 共享权重加载
    # ------------------------------------------------------------------

    def _ensure_weights_loaded(self, hint_dir: Optional[str]) -> None:
        if self._weights_loaded:
            return
        npz = self._weights_npz_arg
        if npz is None and hint_dir:
            candidate = os.path.join(hint_dir, "shared_weights.npz")
            if os.path.exists(candidate):
                npz = candidate
        if npz and os.path.exists(npz):
            self._load_shared_weights(npz)
        else:
            logger.warning(
                "[ARIA/ORT] 未找到 shared_weights.npz；session 将自带各自权重副本，"
                "无法演示共享语义"
            )
        self._weights_loaded = True

    def _load_shared_weights(self, npz_path: str) -> None:
        ort = self._ort
        logger.info("[ARIA/ORT] 加载共享权重: %s", npz_path)
        with np.load(npz_path) as data:
            for k in data.files:
                arr = np.ascontiguousarray(data[k])
                self._shared_weights[k] = arr
                # OrtValue 直接引用 numpy memory（不复制）
                self._shared_ortvalues[k] = ort.OrtValue.ortvalue_from_numpy(arr)
        total = sum(a.nbytes for a in self._shared_weights.values())
        logger.info(
            "[ARIA/ORT] 共享权重 %d 个，单份占用 %.2f KiB（所有 session 复用同一段 numpy memory）",
            len(self._shared_weights), total / 1024
        )

    def _make_sess_opts(self):
        ort = self._ort
        so = ort.SessionOptions()
        so.log_severity_level = 1 if self._verbose else 3
        # 容器里 ORT 默认 thread pool 会 pthread_setaffinity_np 后失败；
        # 显式指定线程数就不走 affinity 那条路径
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        return so

    # ------------------------------------------------------------------
    # NPUExecutor 抽象方法实现
    # ------------------------------------------------------------------

    def _load_graph(self, path: str, meta: GraphMeta) -> Any:
        import onnx
        self._ensure_weights_loaded(os.path.dirname(path))

        # 偷看一眼模型里哪些 initializer 是共享的，构造这张 session 专属的
        # SessionOptions 并只注册需要的那几个 OrtValue。OrtValue 实例本身
        # 跨 session 共享，所以底层 numpy memory 仍然只有一份。
        # load_external_data=False：模型里有 placeholder external 指向不存在
        # 的文件（数据由 add_external_initializers 提供），不要去尝试读它
        peek = onnx.load(path, load_external_data=False)
        needed_names, needed_ovs = [], []
        for init in peek.graph.initializer:
            ov = self._shared_ortvalues.get(init.name)
            if ov is not None:
                needed_names.append(init.name)
                needed_ovs.append(ov)

        so = self._make_sess_opts()
        if needed_names:
            so.add_external_initializers(needed_names, needed_ovs)

        sess = self._ort.InferenceSession(
            path,
            sess_options = so,
            providers    = self._providers,
        )

        # 用 session 的真实输入 dtype 回填 meta，让 _execute 不用猜
        for inp in sess.get_inputs():
            meta.input_dtypes.setdefault(inp.name, _ort_to_np_dtype(inp.type))
        for out in sess.get_outputs():
            meta.output_dtypes.setdefault(out.name, _ort_to_np_dtype(out.type))

        self._sessions[meta.name] = sess
        return meta.name

    def _execute(self,
                 graph_handle: Any,
                 device_inputs: Dict[str, int],
                 meta: GraphMeta) -> Dict[str, int]:
        sess = self._sessions[graph_handle]

        ort_input_types = {i.name: i.type for i in sess.get_inputs()}
        feeds: Dict[str, np.ndarray] = {}
        for inp in sess.get_inputs():
            if inp.name not in device_inputs:
                # 不在 device_inputs 里的：理论上是共享权重，已由
                # add_external_initializers 提供，应该不会被 ORT 要求 feed；
                # 如果走到这里说明 build/runtime 哪里没对齐
                raise RuntimeError(
                    f"[ARIA/ORT] 图 {meta.name} 输入 '{inp.name}' 既不在 "
                    f"runtime feeds 里也不是共享权重"
                )
            addr = device_inputs[inp.name]
            host = self._mem[addr]
            shape = meta.input_shapes.get(inp.name)
            if shape is None:
                shape = host.shape
            dt = meta.input_dtypes.get(inp.name) or _ort_to_np_dtype(inp.type)
            feeds[inp.name] = host.reshape(shape).astype(dt, copy=False)

        if self._latency_ms > 0:
            import time as _t; _t.sleep(self._latency_ms / 1000.0)

        raw_outputs = sess.run(None, feeds)

        out_addrs: Dict[str, int] = {}
        for ort_out, val in zip(sess.get_outputs(), raw_outputs):
            addr = self._alloc_device(val.nbytes)
            self._mem[addr] = val
            out_addrs[ort_out.name] = addr
        return out_addrs

    def _alloc_device(self, size: int) -> int:
        addr = self._fake_addr
        self._fake_addr += max(size, 1) + 64
        return addr

    def _h2d(self, data: np.ndarray, device_addr: int) -> None:
        # ORT 自管 memory；这里只是把 host 副本挂在 fake addr 上让 base 流程能跑
        self._mem[device_addr] = np.ascontiguousarray(data)

    def _d2h(self, device_addr: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
        if device_addr in self._mem:
            arr = self._mem[device_addr]
            return arr.reshape(shape).astype(dtype, copy=False)
        return np.zeros(shape, dtype=dtype)

    def _free_device(self, device_addr: int) -> None:
        self._mem.pop(device_addr, None)

    # ------------------------------------------------------------------
    # 共享统计
    # ------------------------------------------------------------------

    def get_sharing_stats(self) -> Dict[str, Any]:
        """N 张图、共享 M 个权重，对照'naive 各带一份'省了多少。"""
        n_sessions = len(self._sessions)
        unique_bytes = sum(a.nbytes for a in self._shared_weights.values())
        naive_bytes  = unique_bytes * n_sessions
        return {
            "n_sessions":     n_sessions,
            "shared_tensors": len(self._shared_weights),
            "unique_bytes":   unique_bytes,
            "naive_bytes":    naive_bytes,
            "saved_bytes":    max(0, naive_bytes - unique_bytes),
            "saved_ratio":    (naive_bytes - unique_bytes) / max(naive_bytes, 1) if n_sessions > 0 else 0.0,
            "providers":      self._providers,
        }

    def close(self) -> None:
        self._sessions.clear()
        self._mem.clear()
        self._shared_ortvalues.clear()
        self._shared_weights.clear()
        logger.info("[ARIA/ORT] 资源已释放")


# ---------------------------------------------------------------------------
# dtype 映射
# ---------------------------------------------------------------------------

_ORT_TYPE_TO_NP = {
    "tensor(float)":   np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)":  np.float64,
    "tensor(int8)":    np.int8,
    "tensor(int16)":   np.int16,
    "tensor(int32)":   np.int32,
    "tensor(int64)":   np.int64,
    "tensor(uint8)":   np.uint8,
    "tensor(bool)":    np.bool_,
}


def _ort_to_np_dtype(ort_type: str) -> np.dtype:
    return np.dtype(_ORT_TYPE_TO_NP.get(ort_type, np.float32))
