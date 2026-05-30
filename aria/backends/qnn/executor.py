"""
aria/backends/qnn/executor.py

QNN 后端推理执行器。

## 背景

QNN context binary 包含多张 graph，所有 graph 共用同一份权重 blob。
加载一次 context binary 后，按 graph 名字索引执行对应的 bucket 图。

## 预期的初始化流程

    QnnBackend_create()          初始化 HTP backend
    QnnContext_createFromBinary()  加载 model.serialized
    QnnGraph_retrieve("prefill_512")   按名字拿到 graph 句柄
    QnnGraph_retrieve("decode_512")
    ...

## 预期的推理流程（每次 _execute）

    QnnTensor_createContextTensor()  创建输入 / 输出 tensor
    memcpy  host → QNN tensor buffer（_h2d）
    QnnGraph_execute(graph_handle)
    memcpy  QNN tensor buffer → host（_d2h）

## QNN SDK 参考

    头文件：QNN/include/QNN/QnnGraph.h
             QNN/include/QNN/QnnContext.h
             QNN/include/QNN/QnnTensor.h
    Python 绑定：qnn_wrapper_api（QNN SDK 附带）

## 依赖

    pip install qnn-sdk-python   # 或按 QNN SDK 文档安装 Python wrapper
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from aria.core.executor import NPUExecutor, GraphMeta, ExecutionResult

logger = logging.getLogger(__name__)


class QnnExecutor(NPUExecutor):
    """
    QNN HTP 后端推理执行器。

    加载由 tools/backends/qnn/build.py 生成的 context binary，
    按 graph_name 索引执行对应的 bucket 图。

    参数
    ----
    context_path : str
        context binary 路径（model.serialized）。
        各图的 GraphMeta.path 可以都指向同一个文件。
    backend_lib : str
        libQnnHtp.so 的路径，默认 "libQnnHtp.so"。
    """

    def __init__(self,
                 context_path: Optional[str] = None,
                 backend_lib:  str = "libQnnHtp.so",
                 **kwargs):
        super().__init__(**kwargs)
        self._context_path = context_path
        self._backend_lib  = backend_lib
        self._context      = None   # QNN context 句柄
        self._backend      = None   # QNN backend 句柄

    # ------------------------------------------------------------------
    # 抽象方法实现
    # ------------------------------------------------------------------

    def _load_graph(self, path: str, meta: GraphMeta) -> Any:
        """
        从已加载的 context 中按 graph 名字取句柄。

        TODO:
          1. 首次调用时初始化 QNN backend + 加载 context binary
             （context_path 可从 path 或 meta 推断）
          2. 调用 QnnGraph_retrieve(meta.name) 返回 graph 句柄
        """
        raise NotImplementedError(
            "QNN executor 尚未实现。\n"
            "实现步骤：\n"
            "  1. QnnBackend_create(backend_lib)\n"
            "  2. QnnContext_createFromBinary(context_path)\n"
            "  3. QnnGraph_retrieve(meta.name) → 返回 graph handle\n"
            "参考 aria/backends/qnn/executor.py 顶部文档注释。"
        )

    def _execute(self,
                 graph_handle: Any,
                 inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行一次 QNN graph 推理。

        TODO:
          1. 为每个输入创建 QnnTensor，memcpy host → HTP buffer
          2. QnnGraph_execute(graph_handle)
          3. memcpy HTP buffer → host，构建输出 dict
        """
        raise NotImplementedError("QNN _execute 尚未实现")

    def _alloc_device(self, size: int) -> int:
        """在 HTP DDR 上分配 size 字节，返回设备地址。"""
        raise NotImplementedError("QNN _alloc_device 尚未实现")

    def _h2d(self, data: np.ndarray, addr: int) -> None:
        """Host → HTP 数据拷贝。"""
        raise NotImplementedError("QNN _h2d 尚未实现")

    def _d2h(self, addr: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """HTP → Host 数据拷贝。"""
        raise NotImplementedError("QNN _d2h 尚未实现")

    def _write_kv_seq(self, addr, buffer_shape, dtype, start, block, plane0=0) -> None:
        """
        把 decode 单步的 KV 跨步写回常驻 buffer 第 start 行（方案 B）。

        实现要点：
          - buffer 在 alloc_persistent 时拿到一块 HTP DDR 地址，并 bind 为
            decode 图的 kv_cache 输入；
          - seq 维在倒数第二轴，写一行 = L*2*heads 个 head_dim 小块的跨步拷贝，
            用 QnnMem / memcpy 按 stride 搬即可（总量几十 KiB）。

        将来升级方案 A（in-place）时改为：把 decode 图的 kv_new 输出地址也
        setTensorAddress 到 buffer 第 start 行，由图内 scatter 直接写，
        本函数可空转。参见 TensorRT-Edge-LLM linearKVCache 的 commitSequenceLength。
        """
        raise NotImplementedError("QNN _write_kv_seq 尚未实现")
