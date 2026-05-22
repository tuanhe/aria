"""
tools/backends — 编译期后端注册表。

每个后端目录（trt/、ort/、qnn/、rknn/ ...）需提供：

    tools.backends.<name>.build:build(onnx_path, out_path, meta, opts)
        把 ONNX 编译成该后端可执行的产物。

运行时的 executor 注册表在 aria/backends/__init__.py，
两张表分开：这里只管离线编译，不涉及推理。
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Tuple

_BUILDERS: Dict[str, Tuple[str, str]] = {
    "trt": ("tools.backends.trt.build", "build"),
    "ort": ("tools.backends.ort.build", "build"),
    # "qnn":  ("tools.backends.qnn.build",  "build"),
    # "rknn": ("tools.backends.rknn.build", "build"),
}


def list_builders() -> List[str]:
    return list(_BUILDERS.keys())


def get_builder(name: str) -> Callable:
    if name not in _BUILDERS:
        raise ValueError(
            f"未知 backend builder '{name}'，可选: {list_builders()}"
        )
    mod_path, fn_name = _BUILDERS[name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)
