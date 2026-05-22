"""
examples/serve_client.py

aria-serve 客户端示例，使用 openai SDK。

依赖：
    pip install openai

用法：
    # 先启动服务
    aria-serve --config configs/llm_qwen3.yaml --tokenizer /path/to/Qwen3-7B --backend mock

    # 非流式
    python examples/serve_client.py

    # 流式
    python examples/serve_client.py --stream
"""

from __future__ import annotations

import argparse
import sys

from openai import OpenAI


MESSAGES = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user",   "content": "用一句话解释什么是 KV Cache。"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="aria-serve 客户端示例")
    parser.add_argument("--base-url", default="http://localhost:8080",
                        help="服务地址（默认 http://localhost:8080）")
    parser.add_argument("--model",    default="aria-llm",
                        help="模型名称（需与 aria-serve --model-name 一致）")
    parser.add_argument("--stream",   action="store_true",
                        help="启用流式输出")
    args = parser.parse_args()

    client = OpenAI(base_url=f"{args.base_url}/v1", api_key="aria")

    if args.stream:
        stream = client.chat.completions.create(
            model=args.model,
            messages=MESSAGES,
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            print(text, end="", flush=True)
        print()
    else:
        resp = client.chat.completions.create(
            model=args.model,
            messages=MESSAGES,
        )
        print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
