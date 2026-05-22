"""
aria/serve/app.py

OpenAI 兼容 FastAPI 应用。

路由:
    GET  /v1/models
    POST /v1/chat/completions   支持 stream=true（SSE）和非流式两种模式
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from aria.serve.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ModelCard,
    ModelList,
    UsageInfo,
)

logger = logging.getLogger(__name__)


def create_app(runtime: Any, tokenizer: Any, model_name: str) -> FastAPI:
    """
    创建并返回 FastAPI 应用实例。

    runtime:    LLMRuntime 实例
    tokenizer:  transformers AutoTokenizer 实例
    model_name: 对外暴露的模型名称（/v1/models 返回）
    """
    app = FastAPI(title="aria OpenAI-compatible API", version="0.1.0")
    app.state.runtime    = runtime
    app.state.tokenizer  = tokenizer
    app.state.model_name = model_name

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(data=[ModelCard(id=app.state.model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, req: Request):
        _runtime    = req.app.state.runtime
        _tokenizer  = req.app.state.tokenizer
        _model_name = req.app.state.model_name

        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        prompt   = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        req_id  = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        if body.stream:
            return StreamingResponse(
                _sse_generator(_runtime, prompt, req_id, created, _model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        reply = _runtime.generate(prompt)
        return ChatCompletionResponse(
            id=req_id,
            created=created,
            model=_model_name,
            choices=[ChatCompletionResponseChoice(
                message=ChatMessage(role="assistant", content=reply),
            )],
            usage=UsageInfo(),
        )

    return app


def _sse_generator(runtime: Any,
                   prompt:      str,
                   req_id:      str,
                   created:     int,
                   model_name:  str) -> Iterator[str]:
    """同步 SSE 生成器，FastAPI StreamingResponse 接受同步可迭代。"""

    def _chunk(delta: DeltaMessage, finish_reason=None) -> str:
        obj = ChatCompletionStreamResponse(
            id=req_id, created=created, model=model_name,
            choices=[ChatCompletionStreamChoice(delta=delta, finish_reason=finish_reason)],
        )
        return f"data: {obj.model_dump_json()}\n\n"

    # 首包：发送 role
    yield _chunk(DeltaMessage(role="assistant"))

    for token_text in runtime.generate_stream(prompt):
        if token_text:
            yield _chunk(DeltaMessage(content=token_text))

    # 末包：finish_reason=stop
    yield _chunk(DeltaMessage(), finish_reason="stop")
    yield "data: [DONE]\n\n"
