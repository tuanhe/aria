# aria serve — OpenAI 兼容推理服务

`aria-serve` 启动一个兼容 OpenAI Chat Completions API 的 HTTP 服务，
可直接对接任何支持 OpenAI 协议的客户端（openai SDK、curl、LangChain 等）。

---

## 安装

```bash
pip install -e ".[serve]"
# 依赖：fastapi, uvicorn[standard], pydantic>=2, transformers
```

---

## 快速启动

```bash
aria-serve \
    --config    configs/llm_qwen3.yaml \
    --tokenizer /path/to/Qwen3-7B \
    --model-name qwen3-7b \
    --host 0.0.0.0 \
    --port 8080 \
    --backend mock        # mock | ort | trt | qnn
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | FrameworkConfig YAML 路径 | 必填 |
| `--tokenizer` | HuggingFace tokenizer 本地路径或 Hub 名称 | 必填 |
| `--model-name` | 对外暴露的模型名称 | `aria-llm` |
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | 监听端口 | `8080` |
| `--backend` | executor 后端 | `mock` |
| `--log-level` | 日志级别 | `info` |

---

## API 端点

### `GET /v1/models`

返回当前加载的模型列表。

```bash
curl http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [{"id": "qwen3-7b", "object": "model", "owned_by": "aria"}]
}
```

---

### `POST /v1/chat/completions`

**请求体：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | string | 模型名称（需与 `--model-name` 一致） |
| `messages` | array | 对话历史，每条包含 `role` 和 `content` |
| `stream` | bool | `true` 启用 SSE 流式输出，默认 `false` |
| `max_tokens` | int | 最大生成 token 数（可选） |
| `temperature` | float | 采样温度，默认 `1.0` |
| `top_p` | float | nucleus 采样，默认 `1.0` |

**非流式示例：**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-7b",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user",   "content": "用一句话介绍量子计算。"}
    ]
  }'
```

**流式示例：**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-7b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

流式响应格式（SSE）：

```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"你"}}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## 用 openai SDK 调用

```bash
pip install openai

# 非流式
python examples/serve_client.py --model qwen3-7b

# 流式
python examples/serve_client.py --model qwen3-7b --stream
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="aria")

# 非流式
resp = client.chat.completions.create(
    model="qwen3-7b",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="qwen3-7b",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## 架构说明

```
aria-serve CLI
    └── LLMRuntime.from_config()    加载配置 + executor
    └── AutoTokenizer.from_pretrained()
    └── create_app(runtime, tokenizer, model_name)
            ├── GET  /v1/models
            └── POST /v1/chat/completions
                    ├── apply_chat_template()   messages → prompt
                    ├── runtime.generate()      非流式
                    └── runtime.generate_stream() → StreamingResponse (SSE)
```

多轮对话的 KV Cache 状态保存在 `LLMRuntime` 的 `Session` 对象里，
每次请求通过 `session_id` 关联（当前 serve 层每次请求新建临时 session，
多轮支持后续可通过 HTTP header 或请求体扩展传入 `session_id`）。
