"""
tools/backends/trt/build.py

把 ONNX 编译成 TensorRT .engine。

opts:
    fp16:          bool  默认 True
    use_dla:       bool  默认 False
    dla_core:      int   默认 0
    workspace_mib: int   默认 1024
    verbose:       bool  默认 False
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from aria.core.executor import GraphMeta

logger = logging.getLogger(__name__)


def _build_one(onnx_path: str,
               fp16: bool,
               use_dla: bool,
               dla_core: int,
               workspace_mib: int,
               verbose: bool):
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.ERROR)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    parser  = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"[ARIA/TRT-build] ONNX parse 失败:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * (1 << 20))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    used_backend = "GPU"
    if use_dla and builder.num_DLA_cores > 0:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = min(dla_core, builder.num_DLA_cores - 1)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        used_backend = f"DLA{config.DLA_core}+GPU"

    serialized = builder.build_serialized_network(network, config)
    return serialized, used_backend


def build(onnx_path: str,
          out_path: str,
          meta: Optional[GraphMeta] = None,
          opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    opts = opts or {}
    fp16          = opts.get("fp16", True)
    use_dla       = opts.get("use_dla", False)
    dla_core      = opts.get("dla_core", 0)
    workspace_mib = opts.get("workspace_mib", 1024)
    verbose       = opts.get("verbose", False)

    serialized, backend = None, "GPU"

    if use_dla:
        try:
            serialized, backend = _build_one(
                onnx_path, fp16, True, dla_core, workspace_mib, verbose
            )
        except Exception as e:
            logger.warning("[ARIA/TRT-build] DLA 路径失败 (%s)，回退 GPU", e)
            serialized = None

    if serialized is None:
        serialized, backend = _build_one(
            onnx_path, fp16, False, dla_core, workspace_mib, verbose
        )

    if serialized is None:
        raise RuntimeError(f"[ARIA/TRT-build] 引擎构建彻底失败: {out_path}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(serialized)
    nbytes = os.path.getsize(out_path)
    logger.info("[ARIA/TRT-build] %s (%.1f MiB, backend=%s)",
                out_path, nbytes / (1 << 20), backend)
    return {"backend": backend, "bytes": nbytes}
