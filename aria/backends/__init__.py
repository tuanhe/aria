"""
aria.backends —— NPU 后端注册表。

每个后端目录（trt/、qnn/、rknn/ ...）需要满足：

  aria.backends.<name>.executor:<ClassName>
      继承 NPUExecutor，实现 5 个抽象钩子。

  aria.backends.<name>.build:build(onnx_path, out_path, meta, opts)
      把 ONNX 编译成后端可执行的产物，可选；仅当该后端支持
      "本地 dummy engine 构建"时提供。

新增后端时只改这两张表 + 实现对应模块，CLI 会自动出现新的
--executor / --backend 选项；mock 之外的后端均懒加载，没装
SDK 的环境不会被 import 卡住。
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Tuple

from aria.core.executor import NPUExecutor

# name -> (module_path, class_name)
_EXECUTORS: Dict[str, Tuple[str, str]] = {
    "trt":   ("aria.backends.trt.executor",   "TensorRTExecutor"),
    "ort":   ("aria.backends.ort.executor",   "ORTExecutor"),
    "torch": ("aria.backends.torch.executor", "TorchExecutor"),
    # "qnn":  ("aria.backends.qnn.executor",  "QnnExecutor"),
    # "rknn": ("aria.backends.rknn.executor", "RKNNExecutor"),
    # "cann": ("aria.backends.cann.executor", "CANNExecutor"),
}

# name -> (module_path, func_name) —— 编译 ONNX 到后端产物
_BUILDERS: Dict[str, Tuple[str, str]] = {
    "trt": ("aria.backends.trt.build", "build"),
    "ort": ("aria.backends.ort.build", "build"),
}


def list_executors() -> List[str]:
    """所有可选 executor 名（含 mock）。"""
    return ["mock"] + list(_EXECUTORS.keys())


def list_builders() -> List[str]:
    """所有可选 backend builder 名。"""
    return list(_BUILDERS.keys())


def build_executor(name: str, **kwargs) -> NPUExecutor:
    """根据名字实例化 executor。后端 SDK 在此处才被 import。"""
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


def get_builder(name: str) -> Callable:
    """获取后端的 build(onnx_path, out_path, meta, opts) 函数。"""
    if name not in _BUILDERS:
        raise ValueError(
            f"未知 backend builder '{name}'，可选: {list_builders()}"
        )
    mod_path, fn_name = _BUILDERS[name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)
