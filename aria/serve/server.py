"""
aria/serve/server.py

aria-serve CLI 入口：启动 OpenAI 兼容推理服务。

用法:
    aria-serve \\
        --config   configs/llm_qwen3.yaml \\
        --tokenizer /path/to/Qwen3-7B \\
        --model-name qwen3-7b \\
        --host 0.0.0.0 \\
        --port 8080 \\
        --backend mock
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from aria.backends import build_executor
from aria.models.base import FrameworkConfig
from aria.runtime.llm_runtime import LLMRuntime
from aria.serve.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aria-serve",
        description="aria OpenAI-compatible inference server",
    )
    parser.add_argument("--config",      required=True,
                        help="FrameworkConfig YAML 路径")
    parser.add_argument("--tokenizer",   required=True,
                        help="HuggingFace tokenizer 本地路径或 Hub 名称")
    parser.add_argument("--model-name",  default="aria-llm",
                        help="对外暴露的模型名称（默认 aria-llm）")
    parser.add_argument("--host",        default="0.0.0.0")
    parser.add_argument("--port",        type=int, default=8080)
    parser.add_argument("--backend",     default="mock",
                        help="executor 后端：mock / ort / trt / qnn（默认 mock）")
    parser.add_argument("--log-level",   default="info",
                        choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config   = FrameworkConfig.from_yaml(args.config)
    executor = build_executor(args.backend)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )

    runtime = LLMRuntime.from_config(config, executor, tokenizer=tokenizer)
    app     = create_app(runtime, tokenizer, args.model_name)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
