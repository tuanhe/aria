"""
aria/backends/ort/build.py

ORT 后端的"编译"步骤其实是 *剥权重*：把 dummy ONNX 里属于共享族
（vision_proj.* / llm_proj.* / flow_proj.*）的 initializer 全部
摘出来追加到 <out_dir>/shared_weights.npz；剩下的 .onnx 不再包含
这些权重，模型变得很小。

跑推理时，ORTExecutor 把 shared_weights.npz 加载成一份 OrtValue，
通过 SessionOptions.add_initializer 注入每个 InferenceSession——
所有 session 物理上引用同一份 numpy memory，真正实现"一份权重 +
N 张图"的 NPU 语义模拟。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from aria.core.executor import GraphMeta

logger = logging.getLogger(__name__)


# 凡是 initializer 名字以这些前缀开头的，都被认为是"跨图共享"权重
_SHARED_PREFIXES = ("vision_proj.", "llm_proj.", "flow_proj.")


def build(onnx_path: str,
          out_path: str,
          meta: Optional[GraphMeta] = None,
          opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import onnx
    from onnx import TensorProto, external_data_helper, numpy_helper

    opts = opts or {}
    weights_npz = opts.get("weights_npz") or str(
        Path(out_path).parent / "shared_weights.npz"
    )
    prefixes = tuple(opts.get("shared_prefixes", _SHARED_PREFIXES))

    model = onnx.load(onnx_path)

    bank: Dict[str, np.ndarray] = {}
    if os.path.exists(weights_npz):
        with np.load(weights_npz) as data:
            bank = {k: data[k] for k in data.files}

    stripped_names = []
    for init in model.graph.initializer:
        if not any(init.name.startswith(p) for p in prefixes):
            continue

        # 抽出值放到共享 bank
        arr = numpy_helper.to_array(init)
        bank[init.name] = np.ascontiguousarray(arr)
        stripped_names.append(init.name)

        # 把 initializer 标记为 EXTERNAL，data 字段全清空：
        # 模型结构上仍然有这个 initializer（不破坏 graph 合法性），
        # 但磁盘上不再带权重数据，运行时由 add_external_initializers 提供。
        external_data_helper.set_external_data(
            init,
            location="_aria_external_placeholder.bin",
            offset=0,
            length=arr.nbytes,
        )
        init.data_location = TensorProto.EXTERNAL
        for f in ("raw_data", "float_data", "int32_data", "int64_data",
                  "double_data", "uint64_data", "string_data"):
            init.ClearField(f)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)
    np.savez(weights_npz, **bank)

    nbytes = os.path.getsize(out_path)
    logger.info(
        "[ARIA/ORT-build] %s (%.1f KiB, stripped=%s, bank=%d tensors)",
        out_path, nbytes / 1024, stripped_names or "-", len(bank)
    )
    return {
        "backend":  "ORT",
        "bytes":    nbytes,
        "stripped": stripped_names,
        "bank":     weights_npz,
    }
