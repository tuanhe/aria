"""
aria.backends —— 运行时 executor 注册表。

每个后端目录（trt/、ort/、torch/ ...）需提供：

  aria.backends.<name>.executor:<ClassName>
      继承 NPUExecutor，实现 5 个抽象钩子。

编译期的 builder 注册表（ONNX → NPU binary）已迁移到
tools/backends/__init__.py，与运行时代码分离。
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

from aria.core.executor import NPUExecutor

_EXECUTORS: Dict[str, Tuple[str, str]] = {
    "trt":   ("aria.backends.trt.executor",   "TensorRTExecutor"),
    "ort":   ("aria.backends.ort.executor",   "ORTExecutor"),
    "torch": ("aria.backends.torch.executor", "TorchExecutor"),
    "qnn":  ("aria.backends.qnn.executor",  "QnnExecutor"),
    # "rknn": ("aria.backends.rknn.executor", "RKNNExecutor"),
    # "cann": ("aria.backends.cann.executor", "CANNExecutor"),
}


def list_executors() -> List[str]:
    return ["mock"] + list(_EXECUTORS.keys())


def build_executor(name: str, **kwargs) -> NPUExecutor:
    if name == "mock":
        from aria.core.executor import MockNPUExecutor
        return MockNPUExecutor(**kwargs)
    if name not in _EXECUTORS:
        raise ValueError(
            f"未知 executor '{name}'，可选: {list_executors()}"
        )
    mod_path, cls_name = _EXECUTORS[name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(**kwargs)
