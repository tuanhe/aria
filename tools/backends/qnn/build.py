"""
tools/backends/qnn/build.py

QNN 后端编译：把多张 ONNX 图编译成一个共享权重的 context binary。

## 背景

QNN（Qualcomm AI Engine Direct）的部署单元是 context binary（.serialized）。
与 TRT 每个 engine 自包含权重不同，一个 context binary 可以包含多张 graph，
所有 graph 共用同一份权重 blob——这正好对应 aria 的 bucket 设计
（prefill_512 / prefill_1024 / decode_512 / … 权重完全相同，只有 shape 不同）。

## 预期的编译流程

    Stage A  ONNX → QNN graph（per graph，用 qnn-onnx-converter）
    Stage B  N 张 QNN graph → 一个 context binary（用 qnn-context-binary-generator）

    aria-build 调用本模块时：
      - onnx_path  当前图的 fat ONNX（已由 _rehydrated_onnx 还原权重）
      - out_path   约定输出路径，实际产物是 out_dir/model.serialized
                   本模块在第一张图时创建/清空，后续图追加进去

## QNN SDK 工具参考

    qnn-onnx-converter \
        --input_network   <graph>.onnx \
        --output_path     <graph>.cpp \
        --float_bitwidth  16

    qnn-model-lib-generator \
        -t aarch64-android \
        -l <graph>.cpp \
        -o <graph>.so

    qnn-context-binary-generator \
        --model          <graph>.so \
        --backend        libQnnHtp.so \
        --binary_file    model \
        --output_dir     <out_dir>

    # 多图合并进同一 context 需要用 QNN SDK 的 C API：
    # QnnContext_create() + 多次 QnnGraph_create() + QnnContext_getBinarySize()
    # 参考 QNN SDK: docs/QNN_SDK_API_Guide.html

## opts 字段（供 aria-build 透传）

    backend_lib:  str   libQnnHtp.so 路径，默认 "libQnnHtp.so"
    target:       str   编译目标，默认 "aarch64-android"
    float_bw:     int   浮点精度位宽，默认 16
    quantize:     bool  是否量化（需要校准数据），默认 False
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from aria.core.executor import GraphMeta

logger = logging.getLogger(__name__)


def build(onnx_path: str,
          out_path: str,
          meta: Optional[GraphMeta] = None,
          opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    把单张 ONNX 图编译并追加进 QNN context binary。

    out_path 是该图对应的占位路径（如 compiled/prefill_512.bin），
    实际产物写到同目录下的 model.serialized。
    aria 运行时加载时也应指向 model.serialized，而非各自的 .bin。

    TODO: 实现步骤
      1. 调用 qnn-onnx-converter 把 onnx_path 转成 QNN model lib（.so）
      2. 调用 QNN C API 或 qnn-context-binary-generator 把所有 .so 打进
         同一个 context binary（需要跨多次 build() 调用共享 context 句柄，
         或者在 aria-build 里收集所有 .so 再统一生成）
      3. 写出 model.serialized，返回 meta dict
    """
    raise NotImplementedError(
        "QNN build 尚未实现。\n"
        "实现路径：\n"
        "  1. qnn-onnx-converter  ONNX → QNN model lib (.so)\n"
        "  2. qnn-context-binary-generator  多个 .so → context binary\n"
        "参考 tools/backends/qnn/build.py 顶部的文档注释。"
    )
