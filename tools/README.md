# tools — 模型转换工具链

离线的模型转换流水线，把 HuggingFace safetensors 模型变成框架可用的 NPU 可执行文件。**这里的代码永远是 Python，不随 aria 运行时（将来是 C++）一起部署到端侧。**

## 目录结构

```
tools/
  README.md
  build_engines.py       # aria-build CLI 入口
  export_hf.py           # aria-export CLI 入口
  backends/
    __init__.py          # builder 注册表（编译期，与运行时 executor 注册表分离）
    ort/build.py         # ORT 后端：剥共享权重
    trt/build.py         # TensorRT 后端：编译 .engine
  exporters/
    __init__.py          # exporter 注册表
    base.py              # BaseExporter 抽象基类（含权重去重逻辑）
    qwen3.py             # Qwen3 LLM / VLM exporter
```

---

## 完整流程

```
HuggingFace safetensors
        │
        │  aria-export
        ▼
  onnx_exports/
    weights.bin          ← 所有 bucket 共享，只有一份
    prefill_512.onnx     ← 纯 graph 拓扑，无权重（几百 KiB）
    prefill_1024.onnx
    decode_512.onnx
    ...
        │
        │  aria-build
        ▼
  compiled/
    prefill_512.bin      ← NPU 可执行（TRT engine / QNN .so / RKNN bin …）
    prefill_1024.bin
    decode_512.bin
    ...
```

Stage 1（`aria-export`）把模型按框架的静态图约定切成多张 ONNX，权重共享一份；
Stage 2（`aria-build`）把 ONNX 编译成目标 NPU 的可执行格式。

---

## 安装依赖

```bash
pip install -e ".[export]"   # torch + transformers + onnx + safetensors
```

---

## Stage 1：HF safetensors → ONNX

### LLM 模式（纯文本，用于前期调试）

```bash
aria-export \
  --model   Qwen/Qwen3-7B \
  --config  configs/llm_qwen3.yaml \
  --exporter qwen3 \
  --out     onnx_exports/qwen3_llm
```

### VLM 模式（图文多模态）

```bash
aria-export \
  --model   Qwen/Qwen3-VL-7B \
  --config  configs/vlm_qwen3.yaml \
  --exporter qwen3 \
  --out     onnx_exports/qwen3_vlm
```

### 使用本地模型（无网络）

```bash
aria-export \
  --model  /data/models/Qwen3-7B \
  --config configs/llm_qwen3.yaml \
  --exporter qwen3 \
  --out    onnx_exports/qwen3_llm
```

### 只导出部分图（调试用）

```bash
aria-export ... --only prefill        # 只导出全部 prefill 图
aria-export ... --only decode         # 只导出全部 decode 图
aria-export ... --only prefill_1024   # 只导出一张
```

> **注意**：`--only` 导出单张图时不触发权重去重，该 .onnx 仍包含完整权重。
> 调试完成后用完整 `export_all` 流程生成正式产物。

### 输出文件

```
onnx_exports/qwen3_llm/
  weights.bin          ← Qwen3-7B fp16 ≈ 14 GiB，所有图共用
  prefill_512.onnx     ← graph 拓扑，外部引用 weights.bin（< 1 MiB）
  prefill_1024.onnx
  prefill_2048.onnx
  decode_512.onnx
  decode_1024.onnx
  decode_2048.onnx
```

**权重剥离是即时的**：每张图导出后立刻剥离，不等所有图完成。  
峰值磁盘 ≈ `weights.bin`（14 GiB）＋ 当前图的原始 ONNX（14 GiB）≈ **28 GiB**，  
而非所有图同时落盘的 N × 14 GiB。

---

## Stage 2：ONNX → NPU 可执行

```bash
aria-build \
  --onnx    onnx_exports/qwen3_llm \
  --config  configs/llm_qwen3.yaml \
  --backend trt \
  --out     compiled/qwen3_llm
```

`--onnx` 省略时退回 **dummy 模式**（随机权重，仅测试编译流程，产物不可推理）：

```bash
aria-build \
  --config  configs/llm_qwen3.yaml \
  --backend ort \
  --out     compiled/demo
```

---

## 导出的图 I/O 约定

所有图均为**全静态 shape**，无动态维度。

### prefill\_{seq\_len}

**LLM 模式**（`mode: llm`）

| 名称 | Shape | dtype | 说明 |
|---|---|---|---|
| `input_ids` | `[1, seq_len]` | int32 | 文本 token，不足 seq_len 补 0 |
| `attention_mask` | `[1, seq_len]` | int32 | 有效位置为 1，padding 为 0 |
| `position_ids` | `[1, seq_len]` | int32 | 绝对位置编号 |
| `kv_start_pos` | `[1]` | int32 | 多轮时历史 KV 末尾位置，首轮传 0 |
| → `logits` | `[1, vocab_size]` | fp32 | 最后有效 token 的 logits |
| → `kv_out` | `[L×2, 1, kv_heads, seq_len, head_dim]` | fp16 | 本次 prefill 产生的全部 KV |

**VLM 模式**（`mode: vlm`）

| 名称 | Shape | dtype | 说明 |
|---|---|---|---|
| `input_ids` | `[1, seq_len]` | int32 | 文本 token |
| `vision_feat` | `[1, vis_tokens, feat_dim]` | fp16 | 视觉特征，纯文本时传全零 |
| `attention_mask` | `[1, seq_len]` | int32 | |
| `position_ids` | `[1, seq_len]` | int32 | |
| `kv_start_pos` | `[1]` | int32 | |
| → `last_hidden` | `[1, hidden_dim]` | fp16 | 最后有效 token 的隐状态（给动作头 / lm_head 用）|
| → `kv_out` | `[L×2, 1, kv_heads, seq_len, head_dim]` | fp16 | |

### decode\_{kv\_len}（LLM / VLM 相同）

| 名称 | Shape | dtype | 说明 |
|---|---|---|---|
| `input_id` | `[1, 1]` | int32 | 当前步输入 token |
| `position_id` | `[1, 1]` | int32 | 当前步位置编号（= 当前 KV 长度） |
| `kv_in` | `[L×2, 1, kv_heads, kv_len, head_dim]` | fp16 | 历史 KV cache，不足 kv_len 补 0 |
| → `logits` | `[1, vocab_size]` | fp32 | 当前步 logits |
| → `kv_new` | `[L×2, 1, kv_heads, 1, head_dim]` | fp16 | 本步新增的 KV（1 个 token） |

**KV 排列规则**：`kv[i*2]` = 第 i 层的 K，`kv[i*2+1]` = 第 i 层的 V。

---

## 配置文件

| 文件 | mode | 用途 |
|---|---|---|
| `configs/llm_qwen3.yaml` | `llm` | Qwen3 纯文本，无 vision 配置，适合前期调试 |
| `configs/vlm_qwen3.yaml` | `vlm` | Qwen3-VL 图文多模态 |
| `configs/vla_demo_orin.yaml` | `vla` | Orin 板子上的 dummy 演示 |

### GQA 模型的 num\_heads

Qwen3 使用 GQA，yaml 里的 `llm.num_heads` 填 **KV heads**，不是 Q heads：

| 模型 | Q heads | KV heads（填这个）|
|---|---|---|
| Qwen3-7B | 28 | **8** |
| Qwen3-14B | 40 | **8** |

```yaml
llm:
  num_heads: 8      # KV heads
  head_dim:  128
```

---

## 添加新的 exporter

1. 在 `tools/exporters/` 下新建 `your_model.py`，继承 `BaseExporter`：

```python
from tools.exporters.base import BaseExporter

class YourModelExporter(BaseExporter):
    def load_model(self):
        ...

    def export_prefill(self, out_dir, seq_len) -> str:
        # 导出 prefill_{seq_len}.onnx，返回路径
        # 权重去重由基类 export_all() 自动处理
        ...

    def export_decode(self, out_dir, kv_len) -> str:
        ...
```

2. 在 `tools/exporters/__init__.py` 的 `_EXPORTERS` 表里注册：

```python
_EXPORTERS = {
    "qwen3":      ("tools.exporters.qwen3",      "Qwen3Exporter"),
    "your_model": ("tools.exporters.your_model", "YourModelExporter"),
}
```

3. 用 `--exporter your_model` 调用。

## 添加新的编译后端

1. 在 `tools/backends/` 下新建 `your_backend/build.py`，实现：

```python
def build(onnx_path: str, out_path: str, meta, opts) -> dict:
    ...
    return {"backend": "YOUR_BACKEND", "bytes": nbytes}
```

2. 在 `tools/backends/__init__.py` 的 `_BUILDERS` 表里注册：

```python
_BUILDERS = {
    "trt":          ("tools.backends.trt.build",          "build"),
    "ort":          ("tools.backends.ort.build",          "build"),
    "your_backend": ("tools.backends.your_backend.build", "build"),
}
```

3. 用 `--backend your_backend` 调用。
