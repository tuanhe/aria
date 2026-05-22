"""
tools/build_engines.py

把 ONNX 编译为 NPU 后端可执行文件（aria-build CLI 入口）。

支持两种模式：
  真实模式   --onnx <dir>  接收 aria-export 产出的 .onnx 文件
  Dummy 模式 （不加 --onnx）生成随机权重 ONNX 再编译，产物不可推理

用法：
    # 真实模式（接 aria-export 输出）
    aria-build --config configs/vlm_qwen3.yaml --out compiled/qwen3 \\
               --onnx onnx_exports/qwen3 --backend trt

    # Dummy 模式（测试用）
    aria-build --config configs/vla_demo_orin.yaml --out compiled/demo \\
               --backend trt [--use-dla] [--no-fp16]

后端专属编译逻辑在 tools/backends/<name>/build.py。
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from tools.backends import get_builder, list_builders
from aria.core.executor import GraphMeta, NPUExecutor
from aria.models.base import FrameworkConfig

logger = logging.getLogger("aria.build")


# ---------------------------------------------------------------------------
# 输入 dtype 推断（GraphMeta.input_dtypes 通常没填，按张量名硬编码）
# ---------------------------------------------------------------------------

_INT_INPUTS = {
    "input_ids", "input_id",
    "position_ids", "position_id",
    "attention_mask",
    "kv_start_pos",
}
_FP32_INPUTS = {"timestep", "tiles"}   # tiles 在 vision_encoder 预处理后是 fp32
# 其余默认 fp16


def _infer_input_dtype(name: str, declared: Dict[str, np.dtype]) -> np.dtype:
    if name in declared:
        return np.dtype(declared[name])
    if name in _INT_INPUTS:
        return np.dtype(np.int32)
    if name in _FP32_INPUTS:
        return np.dtype(np.float32)
    return np.dtype(np.float16)


_NP_TO_TORCH = None


def _np_to_torch_dtype(npd: np.dtype):
    import torch
    global _NP_TO_TORCH
    if _NP_TO_TORCH is None:
        _NP_TO_TORCH = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int32):   torch.int32,
            np.dtype(np.int64):   torch.int64,
            np.dtype(np.uint8):   torch.uint8,
            np.dtype(np.bool_):   torch.bool,
        }
    return _NP_TO_TORCH[np.dtype(npd)]


# ---------------------------------------------------------------------------
# 图收集 —— 一个只记录 register_graph 调用的"哑"执行器
# ---------------------------------------------------------------------------

class _HarvestExecutor(NPUExecutor):
    """实例化各个模型构件时，把 register_graph 的所有 GraphMeta 拢起来。"""

    def __init__(self):
        super().__init__()
        self.collected: List[GraphMeta] = []

    def _load_graph(self, path, meta):
        return None

    def register_graph(self, meta: GraphMeta) -> None:
        # 不真正 load，只收集
        self.collected.append(meta)
        self._graphs[meta.name] = meta

    def _execute(self, *_a, **_k):
        raise RuntimeError("harvest executor 不应被 run()")

    def _alloc_device(self, size):
        return 0

    def _h2d(self, data, addr):
        pass

    def _d2h(self, addr, shape, dtype):
        return np.zeros(shape, dtype=dtype)


def harvest_graphs(cfg: FrameworkConfig) -> List[GraphMeta]:
    """跑一遍模型构造函数，把所有 register 的图捞出来。"""
    from aria.models.vision_encoder import VisionEncoder
    from aria.models.llm_backbone   import LLMBackbone
    from aria.models.flow_decoder   import FlowDecoder

    harvester = _HarvestExecutor()

    VisionEncoder(cfg, harvester)
    LLMBackbone(cfg, harvester, kv_cache=None)

    if cfg.mode == "vla" and cfg.action.head_type == "flow_matching":
        FlowDecoder(cfg, harvester)
    # AR / Text decoder 复用 LLMBackbone 的 decode bucket，不额外注册图

    return harvester.collected


# ---------------------------------------------------------------------------
# 共享权重族 —— 决定每张图挂哪些 nn.Linear，名字跨图保持一致
# 这样 ORT 后端 build 阶段就能把同名 initializer 剥出来到一份 npz，
# 真正模拟 "权重一份 + 多图引用" 的 NPU 行为。
# ---------------------------------------------------------------------------

def _shared_params_for(graph_name: str,
                       cfg: FrameworkConfig
                       ) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
    """
    返回 {full_param_name: (shape, np_dtype)}。
    full_param_name 形如 "llm_proj.weight"，导出 ONNX 后正是 initializer 名。
    """
    if graph_name == "vision_encoder":
        d = cfg.vision.feat_dim
        return {"vision_proj.weight": ((d, d), np.dtype(np.float16))}
    if graph_name.startswith("prefill_") or graph_name.startswith("decode_"):
        d = cfg.llm.hidden_dim
        # 所有 prefill_* / decode_* 共用同一份 llm_proj.weight
        return {"llm_proj.weight": ((d, d), np.dtype(np.float16))}
    if graph_name == "flow_head":
        d = cfg.action.action_dim
        return {"flow_proj.weight": ((d, d), np.dtype(np.float16))}
    return {}


# ---------------------------------------------------------------------------
# Dummy nn.Module —— 输出依赖输入 + 引用共享权重（保证 export 进 ONNX initializer）
# ---------------------------------------------------------------------------

def _make_dummy_module(meta: GraphMeta,
                       shared_params: Dict[str, Tuple[Tuple[int, ...], np.dtype]]):
    import torch
    import torch.nn as nn

    out_specs: List[Tuple[Tuple[int, ...], np.dtype]] = []
    for k, shape in meta.output_shapes.items():
        dt = meta.output_dtypes.get(k, np.float16)
        out_specs.append((tuple(shape), np.dtype(dt)))

    # 把 "llm_proj.weight" 切成 submodule 名 "llm_proj"，
    # 用 nn.Linear 注册以后导出的 initializer 名字就是 "llm_proj.weight"
    submodules: Dict[str, Tuple[int, int]] = {}
    for full_name, (shape, _np_dtype) in shared_params.items():
        sub, attr = full_name.rsplit(".", 1)
        assert attr == "weight", f"暂只支持 .weight 子参数: {full_name}"
        assert len(shape) == 2, f"只支持 2D 共享权重: {shape}"
        out_f, in_f = shape
        submodules[sub] = (out_f, in_f)

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            for sub_name, (out_f, in_f) in submodules.items():
                setattr(self, sub_name, nn.Linear(in_f, out_f, bias=False))

        def forward(self, *inputs):
            seed = None
            for x in inputs:
                if torch.is_floating_point(x):
                    seed = x.to(torch.float32).reshape(-1)[:1].sum() * 0.0
                    break
            if seed is None:
                seed = inputs[0].to(torch.float32).reshape(-1)[:1].sum() * 0.0

            # 让共享权重参与 forward 一次，保证 do_constant_folding=False 下
            # 它们以 initializer 形式落进 ONNX
            for sub_name in submodules:
                linear = getattr(self, sub_name)
                seed = seed + linear.weight.to(torch.float32).reshape(-1)[:1].sum() * 0.0

            outs = []
            for shape, np_dtype in out_specs:
                t_dtype = _np_to_torch_dtype(np_dtype)
                base = torch.ones(shape, dtype=torch.float32) * 0.01 + seed
                outs.append(base.to(t_dtype))
            return tuple(outs) if len(outs) > 1 else outs[0]

    return DummyNet()


def export_onnx(meta: GraphMeta,
                onnx_path: str,
                shared_params: Dict[str, Tuple[Tuple[int, ...], np.dtype]]) -> List[str]:
    """把 dummy 模型导出成 ONNX，返回输入名顺序。"""
    import torch

    input_names  = list(meta.input_shapes.keys())
    output_names = list(meta.output_shapes.keys())

    example_inputs = []
    for name in input_names:
        shape    = meta.input_shapes[name]
        np_dtype = _infer_input_dtype(name, meta.input_dtypes)
        torch_dtype = _np_to_torch_dtype(np_dtype)
        ex = torch.zeros(shape, dtype=torch_dtype)
        example_inputs.append(ex)

    model = _make_dummy_module(meta, shared_params).eval()

    # 旧版 torchscript 导出器不依赖 onnxscript，对 dummy 模型够用了
    torch.onnx.export(
        model,
        tuple(example_inputs),
        onnx_path,
        input_names    = input_names,
        output_names   = output_names,
        opset_version  = 17,
        do_constant_folding = False,
        dynamo         = False,
    )
    return input_names


# ---------------------------------------------------------------------------
# 两条编译路径
# ---------------------------------------------------------------------------

def _build_from_onnx(
    metas:         List[GraphMeta],
    onnx_dir:      str,
    out_dir:       Path,
    backend_build,
    opts:          Dict[str, Any],
) -> None:
    """
    真实 ONNX 模式：把 aria-export 产出的 lean ONNX 编译为 NPU 产物。

    aria-export 产出的 .onnx 是"瘦身版"：权重数据在 weights.bin 里，
    ONNX initializer 只有外部引用。NPU 编译器（TRT / QNN / RKNN 等）的
    ONNX 解析器默认不加载 external data，直接传过去会得到空权重。

    因此每张图编译前先 rehydrate：把 weights.bin 重新嵌回 ONNX，
    写到临时文件交给编译器，编译完即删除。
    峰值磁盘 = weights.bin（一份）+ 当前图临时 fat ONNX（一份）。
    """
    missing = []
    for meta in metas:
        lean_path = os.path.join(onnx_dir, f"{meta.name}.onnx")
        if not os.path.exists(lean_path):
            logger.warning("找不到 %s，跳过", lean_path)
            missing.append(meta.name)
            continue

        out_path = str(out_dir / f"{meta.name}.bin")
        logger.info("== %s ==", meta.name)

        with _rehydrated_onnx(lean_path, onnx_dir) as fat_path:
            logger.info("  onnx: %s  (%.1f MiB)",
                        fat_path, os.path.getsize(fat_path) / 1024 / 1024)
            backend_build(fat_path, out_path, meta, opts)

    if missing:
        logger.warning("以下图在 --onnx 目录中缺失，已跳过: %s", missing)


@contextlib.contextmanager
def _rehydrated_onnx(lean_path: str, data_dir: str):
    """
    context manager：把 lean ONNX（external data 引用）还原成带完整权重的
    临时 ONNX 文件，退出时自动删除。

    如果该 ONNX 本身已经是自包含（无 external data），直接 yield 原路径，
    不产生临时文件。
    """
    try:
        import onnx
    except ImportError:
        # onnx 未装，只能直接传原路径，让编译器自己处理
        logger.warning("[rehydrate] onnx 未安装，直接传入 lean ONNX（可能报错）")
        yield lean_path
        return

    model = onnx.load(lean_path, load_external_data=False)

    has_external = any(
        init.data_location == onnx.TensorProto.EXTERNAL
        for init in model.graph.initializer
    )

    if not has_external:
        yield lean_path
        return

    # 加载时让 onnx 自动解析 external data（需要 data_dir 在同一目录）
    model = onnx.load(lean_path, load_external_data=True)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".onnx", prefix="aria_fat_")
    os.close(tmp_fd)
    try:
        onnx.save(model, tmp_path)
        logger.info("  rehydrate → %s (%.1f MiB)",
                    os.path.basename(tmp_path),
                    os.path.getsize(tmp_path) / 1024 / 1024)
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug("  删除临时文件 %s", tmp_path)


def _build_from_dummy(
    metas:         List[GraphMeta],
    cfg:           "FrameworkConfig",
    out_dir:       Path,
    backend_build,
    opts:          Dict[str, Any],
) -> None:
    """Dummy 模式：用随机权重生成 ONNX 再编译（用于测试，产物不可推理）。"""
    with tempfile.TemporaryDirectory(prefix="aria_onnx_") as tmp:
        for meta in metas:
            onnx_path = os.path.join(tmp, f"{meta.name}.onnx")
            out_path  = str(out_dir / f"{meta.name}.bin")
            shared    = _shared_params_for(meta.name, cfg)

            logger.info("== %s ==  (dummy 权重)", meta.name)
            logger.info("  inputs:  %s", dict(meta.input_shapes))
            logger.info("  outputs: %s", dict(meta.output_shapes))
            if shared:
                logger.info("  shared:  %s", list(shared.keys()))

            export_onnx(meta, onnx_path, shared)
            backend_build(onnx_path, out_path, meta, opts)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog        = "aria-build",
        description = "把 ONNX 编译为 NPU 后端可执行文件（.bin）",
    )
    parser.add_argument("--config",  required=True,
                        help="yaml 配置（决定 bucket / 维度等）")
    parser.add_argument("--out",     required=True,
                        help="engine 输出目录（每张图一个 .bin 文件）")
    parser.add_argument("--onnx",    default=None, metavar="DIR",
                        help="真实 ONNX 目录（来自 aria-export）；"
                             "省略则自动生成随机权重的 dummy ONNX")
    parser.add_argument("--backend", default="trt",
                        choices=list_builders(),
                        help="编译后端")
    parser.add_argument("--use-dla", action="store_true",
                        help="(TRT) 尝试把支持的层下到 Orin DLA core")
    parser.add_argument("--dla-core",      type=int, default=0)
    parser.add_argument("--no-fp16",       action="store_true",
                        help="禁用 FP16（默认开）")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    cfg     = FrameworkConfig.from_yaml(args.config)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = harvest_graphs(cfg)
    mode  = f"真实 ONNX ({args.onnx})" if args.onnx else "dummy"
    logger.info("待构建图数量: %d   backend=%s   模式=%s",
                len(metas), args.backend, mode)

    backend_build = get_builder(args.backend)
    opts = {
        "fp16":          not args.no_fp16,
        "use_dla":       args.use_dla,
        "dla_core":      args.dla_core,
        "workspace_mib": args.workspace_mib,
        "verbose":       args.verbose,
        "weights_npz":   str(out_dir / "shared_weights.npz"),
    }

    if args.onnx:
        _build_from_onnx(metas, args.onnx, out_dir, backend_build, opts)
    else:
        _build_from_dummy(metas, cfg, out_dir, backend_build, opts)

    logger.info("全部产物已写入 %s", out_dir)


if __name__ == "__main__":
    main()
