"""
tools/export_hf.py

CLI 入口：aria-export

把 HuggingFace safetensors 模型按框架的静态图约定导出为一组 ONNX 文件。
导出的 .onnx 文件可直接交给 aria-build 做后端编译（TRT / QNN / RKNN 等）。

用法示例：
    aria-export --model Qwen/Qwen3-7B \\
                --config configs/vlm_qwen3.yaml \\
                --exporter qwen3 \\
                --out onnx_exports/qwen3

    # 只导出 prefill 图（调试用）
    aria-export --model ./local_model \\
                --config configs/vlm_qwen3.yaml \\
                --exporter qwen3 \\
                --out onnx_exports/qwen3 \\
                --only prefill

    # 只导出某个 bucket
    aria-export ... --only prefill_1024
"""

from __future__ import annotations

import argparse
import logging
import sys

from aria.models.base import FrameworkConfig
from tools.exporters import build_exporter, list_exporters


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog        = "aria-export",
        description = "HuggingFace safetensors → ONNX（框架静态图格式）",
    )
    parser.add_argument(
        "--model", required=True,
        help="HF Hub model id 或本地模型目录路径",
    )
    parser.add_argument(
        "--config", required=True,
        help="框架 yaml 配置文件（决定 bucket / shape）",
    )
    parser.add_argument(
        "--exporter", required=True, choices=list_exporters(),
        help="使用的 exporter（如 qwen3）",
    )
    parser.add_argument(
        "--out", required=True,
        help="ONNX 输出目录",
    )
    parser.add_argument(
        "--only", default=None,
        metavar="GRAPH",
        help=(
            "只导出指定图，其余跳过。"
            "可以是 'prefill'（全部 prefill bucket）、"
            "'decode'（全部 decode bucket）、"
            "或精确图名如 'prefill_1024'。"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level   = logging.DEBUG if args.verbose else logging.INFO,
        format  = "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        stream  = sys.stdout,
    )
    logger = logging.getLogger("aria.export")

    cfg      = FrameworkConfig.from_yaml(args.config)
    exporter = build_exporter(args.exporter, cfg=cfg, model_path=args.model)

    logger.info("模型路径  : %s", args.model)
    logger.info("配置文件  : %s", args.config)
    logger.info("输出目录  : %s", args.out)
    logger.info("Exporter  : %s", args.exporter)

    exporter.load_model()

    only = args.only

    if only is None:
        paths = exporter.export_all(args.out)
    else:
        paths = _export_subset(exporter, cfg, args.out, only)

    logger.info("导出完成，共 %d 张图：", len(paths))
    for p in paths:
        logger.info("  %s", p)


def _export_subset(exporter, cfg, out_dir: str, only: str):
    """按 --only 参数筛选要导出的图。"""
    import os
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []

    for seq_len in cfg.llm.prefill_buckets:
        name = f"prefill_{seq_len}"
        if only in ("prefill", name):
            paths.append(exporter.export_prefill(out_dir, seq_len))

    for kv_len in cfg.llm.decode_buckets:
        name = f"decode_{kv_len}"
        if only in ("decode", name):
            paths.append(exporter.export_decode(out_dir, kv_len))

    if not paths:
        raise ValueError(
            f"--only '{only}' 没有匹配任何图。"
            f"可选: prefill, decode, "
            f"{[f'prefill_{b}' for b in cfg.llm.prefill_buckets]}, "
            f"{[f'decode_{b}' for b in cfg.llm.decode_buckets]}"
        )
    return paths


if __name__ == "__main__":
    main()
