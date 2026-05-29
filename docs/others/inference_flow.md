# TensorRT-Edge-LLM 推理流程

## 入口层

### Python 高层 API
文件：`experimental/server/engine.py`

```python
LLM.generate(request)          # 批量生成
LLM.generate_stream(request)   # 流式生成
LLM.chat(messages)             # 对话接口
```

### C++ PyBind 接口
文件：`experimental/pybind/edgellm_pybind.cpp`

```cpp
// 普通解码
PyLLMRuntime(engineDir, multimodalEngineDir, loraWeightsMap)

// Eagle 投机解码
PyLLMRuntime(engineDir, multimodalEngineDir, loraWeightsMap, draftTopK, draftStep, verifyTreeSize)

PyLLMRuntime.handleRequest(request)         // 实际推理入口
PyLLMRuntime.captureDecodingCudaGraph()     // CUDA Graph 优化
```

---

## Phase 0：初始化

文件：`cpp/runtime/llmInferenceSpecDecodeRuntime.cpp`

```
LLMInferenceSpecDecodeRuntime()
  ├─ loadEmbeddingTable(embeddingPath, stream)
  │    → 加载 Embedding 表 [vocabSize, hiddenSize]
  │
  ├─ LLMEngineRunner(enginePath, configPath, loraWeightsMap, stream)
  │    ├─ mRuntime->deserializeCudaEngine()       // 反序列化 .engine 文件
  │    ├─ mEngine->createExecutionContext()        // 创建 TRT 执行上下文
  │    ├─ InitRoPE(LongRope / MRope / Persistent) // 预计算 RoPE cos/sin 缓存
  │    ├─ HybridCacheManager::init()              // KV Cache + Mamba 状态管理
  │    └─ bindLoRAWeights(loraWeightsMap)         // 加载 LoRA 权重（若有）
  │
  └─ EagleDraftEngineRunner(draftEnginePath, ...)  // 仅 Eagle 投机解码时加载
       └─ 加载草稿词表映射表 d2t.safetensors
```

### Engine 文件结构说明

一个 `.engine` 文件内嵌**两个 Optimization Profile**（TRT Multi-Profile 机制）：

| Profile | 索引 | 用途 | seq_len 范围 |
|---------|------|------|-------------|
| `kPREFILL_PROFILE_INDEX` | 0 | Prefill 阶段，处理完整输入 | `1 ~ max_input_len` |
| `kGENERATION_PROFILE_INDEX` | 1 | Decode 阶段，逐 token 生成 | 固定为 `1` |

运行时通过 `setOptimizationProfileAsync(index, stream)` 按阶段切换，无需加载两个文件。
decode 阶段 seq_len 固定为 1 是为了让 TRT 做专项内核优化，也是 CUDA Graph 能在 decode 阶段生效的前提。

### 各输入 Shape 范围

| 输入张量 | Prefill Profile | Generation Profile |
|----------|----------------|--------------------|
| `inputs_embeds` | `[1~B, 1~L, H]` | `[1~B, 1, H]` |
| `context_lengths` | `[1~B]` | `[1~B]` |
| KV Cache | `[1~B, heads, 0~max_kv, head_dim]` | 同左 |
| `rope_rotary_cos_sin` | `[B, max_seq, rot_dim]` | 同左 |

> `B` = max_batch_size，`L` = max_input_len，`H` = hidden_size，均由 `config.json` 定义。

---

## Phase 1：请求预处理

文件：`cpp/runtime/llmInferenceSpecDecodeRuntime.cpp`，`handleRequest()` 约第 525 行

```
validateRequestConfig(request)
validateStreamingSubmission(request)
  ↓
mTokenizer->applyChatTemplate(
    request.requests[i],
    request.formattedRequests[i],
    applyChatTemplate,
    addGenerationPrompt,
    enableThinking
)
// 按模型对话模板拼装 System / User / Assistant 轮次
  ↓
multiModalRuntimePreprocess(request, context, stream)
// 处理图片 / 音频 / 视频 Embedding（多模态模型时）
```

---

## Phase 2：Tokenization

文件：`cpp/tokenizer/tokenizer.h` | `cpp/tokenizer/tokenizer.cpp`

```
mTokenizer->encode(formattedText, addBos=false)
  ├─ Pre-tokenization：正则分词
  ├─ BPE merge ranks 编码
  └─ → std::vector<int32_t> token_ids
```

---

## Phase 3：Prefill 准备

```
setUpForPrefillExecution(context)
  ├─ 将 token IDs 打包为 batch 格式
  ├─ 计算 KV Cache 容量需求
  ├─ 复用系统提示的 KV Cache（若有缓存）
  └─ 重置新序列的 KV Cache
```

---

## Phase 4：Prefill（全量前向）

文件：`cpp/runtime/llmEngineRunner.cpp`，`executePrefillStep()`

```
runBaseModelPrefill(context)
  ↓
embeddingLookup(tokenIds, embeddingTable, scales, inputsEmbeds, stream)
  // cpp/kernels/embeddingKernels/embeddingKernels.h
  // token_ids → [B, L, H] 浮点 Embedding（支持 FP16 / FP8）
  ↓
LLMEngineRunner::executePrefillStep(
    inputsEmbeds,        // [B, L, H]
    contextLengths,      // [B]
    outputLogits,        // [B, L, V]
    outputHiddenStates,  // [B, L, H]
    stream
)
  // setOptimizationProfileAsync(kPREFILL_PROFILE_INDEX, stream)
  // setInputShape("inputs_embeds", {B, L, H})
  // IExecutionContext::enqueueV3(stream)
  // 同时填充所有 Transformer 层的 KV Cache
```

---

## Phase 5：流式输出准备

```
attachStreamChannel(request.streamChannels[i], batchIdx)
// 注册 StreamChannel，供后续 decode 循环异步推送 token
```

---

## Phase 6：Decode 循环（自回归生成）

文件：`cpp/runtime/llmInferenceSpecDecodeRuntime.cpp`，约第 814 行

```
while (!checkAllFinished(context)):
  │
  ├─ applyCancellationToFinishStates(context)
  │
  ├─【普通解码】runVanillaDecodingStep(context)
  │    ├─ 将上一步采样 token 做 embeddingLookup → [B, 1, H]
  │    ├─ LLMEngineRunner::executeVanillaDecodingStep([B, 1, H])
  │    │    // setOptimizationProfileAsync(kGENERATION_PROFILE_INDEX, stream)
  │    │    // 单步 TRT 前向，增量更新 KV Cache
  │    │    // → logits [B, V]
  │    └─ topKtopPSamplingFromLogits(logits, selectedIndices, params, workspace, stream)
  │         // cpp/sampler/sampling.cu
  │         // Temperature scaling → Top-K 过滤 → Top-P 过滤 → Softmax → 采样
  │         // → sampled token_id
  │
  ├─【Eagle 投机解码】（若开启）
  │    ├─ runDraftProposal(context)
  │    │    // EagleDraftEngineRunner::executeDraftProposal()
  │    │    // selectAllTopK() → 生成候选 token 树 [B, draftTopK]
  │    ├─ buildDraftTree(context)
  │    │    // 组装用于验证的 token 树结构
  │    └─ runBaseModelTreeVerification(context)
  │         // LLMEngineRunner::executeEagleBaseTreeDecodingStep()
  │         // 批量验证草稿 token，Accept / Reject
  │
  ├─ 追加 token：context.tokenIds[i].push_back(sampledToken)
  ├─ 更新 context.currentGenerateLengths[i]++
  │
  ├─ 检查结束条件：
  │    ├─ sampledToken == EOS token？
  │    └─ currentGenerateLength >= maxGenerateLength？
  │         → context.finishedStates[i] = 1
  │
  ├─ 流式输出（若开启 streaming）：
  │    └─ emitChunks(context, tokenizer)
  │         └─ mTokenizer->decode(newTokens, skipSpecialTokens)
  │              → text 片段 → StreamChannel 异步推送
  │
  └─ performBatchEvict(context)
       // 移除已完成序列，压缩 KV Cache，更新 batchIndexMapping
```

---

## Phase 7：结果输出

```
mTokenizer->decode(tokenIds[i], skipSpecialTokens)
  → std::string outputText

response.outputTexts[i] = outputText
response.outputIds[i]   = tokenIds[i]
// StreamChannel 最终 flush，关闭流
```

---

## 关键类与接口

| 类 / 文件 | 职责 | 关键方法 |
|-----------|------|----------|
| `LLMInferenceSpecDecodeRuntime` | 顶层编排 | `handleRequest()` |
| `LLMEngineRunner` | TRT 引擎封装 | `executePrefillStep()` `executeVanillaDecodingStep()` |
| `EagleDraftEngineRunner` | Eagle 草稿模型 | `executeDraftProposal()` |
| `HybridCacheManager` | KV Cache + Mamba 状态 | `allocateKVCacheForNewBatch()` `advanceKVCachePointer()` |
| `Tokenizer` | 分词 / 解码 | `encode()` `decode()` `applyChatTemplate()` |
| `sampling.cu` | CUDA 采样内核 | `topKtopPSamplingFromLogits()` `selectAllTopK()` |
| `embeddingKernels.h` | GPU Embedding 查表 | `embeddingLookup()` |
| `StreamChannel` | 流式 token 推送 | `push()` `flush()` |

---

## 完整数据流

```
原始文本
  ↓  applyChatTemplate() + multiModalRuntimePreprocess()
格式化 Prompt + 多模态 Embedding
  ↓  mTokenizer->encode()
token_ids [L]
  ↓  embeddingLookup()
inputs_embeds [B, L, H]
  ↓  executePrefillStep()              (Profile 0)
KV Cache 填充完毕 + logits [B, L, V]
  ↓  topKtopPSamplingFromLogits()
首个生成 token
  ↓  executeVanillaDecodingStep() × N  (Profile 1，seq_len 固定=1)
逐步生成 token，增量更新 KV Cache
  ↓  mTokenizer->decode()
最终文本输出
```
