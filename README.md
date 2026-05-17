# ARIA
**Action Reasoning Inference Accelerator**

端侧NPU推理框架，支持 VLA（π0 / OpenVLA）和 VLM（Qwen3 VL）两种模型。

---

## 特性

- 静态多图 + 权重共享，适配不支持动态shape的端侧NPU
- Prefill / Decode 分离，AR / Flow Matching 两条动作解码路径
- 多轮对话 KV Cache 跨轮复用（VLM模式）
- 三级流水线：视觉预处理 → LLM推理 → 输出执行
- 配置驱动，切换模型只改yaml
- NPU后端抽象层，可对接 CANN / RKNN / QNN 等厂商SDK

---

## 项目结构

```
aria/                            # 仓库根目录
├── aria/                        # 可安装的 Python 包
│   ├── __init__.py
│   ├── __main__.py              # 支持 python -m aria
│   ├── cli.py                   # CLI 入口（aria 命令）
│   ├── core/                    # 后端无关的核心抽象
│   │   ├── executor.py          # NPUExecutor 抽象基类 + MockNPUExecutor
│   │   ├── memory.py            # 静态内存池管理
│   │   ├── kv_cache.py          # KV Cache管理
│   │   └── scheduler.py         # 三级流水线调度器
│   ├── backends/                # NPU 后端实现（按厂商一目一包）
│   │   ├── __init__.py          # 注册表：build_executor() / get_builder()
│   │   └── trt/                 # TensorRT 后端（含 Orin DLA）
│   │       ├── executor.py      # TensorRTExecutor
│   │       └── build.py         # ONNX → .engine 编译
│   ├── models/
│   │   ├── base.py              # 模型配置数据类
│   │   ├── vision_encoder.py    # 视觉编码器（固定分辨率）
│   │   ├── llm_backbone.py      # LLM Backbone（多bucket图管理）
│   │   ├── ar_decoder.py        # 自回归动作解码头（OpenVLA / RT-2）
│   │   ├── flow_decoder.py      # Flow Matching动作解码头（π0）
│   │   └── text_decoder.py      # 文本解码头（Qwen3 VL）
│   ├── runtime/
│   │   ├── vla_runtime.py       # VLA推理运行时
│   │   ├── vlm_runtime.py       # VLM推理运行时（多轮对话）
│   │   └── session.py           # 多轮对话Session管理
│   └── tools/
│       └── build_dummy_engines.py  # aria-build：后端无关的 harvest + ONNX，
│                                    # 编译逻辑分派到 backends/<name>/build.py
├── configs/                     # 示例配置（不打进包，仅作为参考）
│   ├── vla_pi0.yaml
│   ├── vla_openvla.yaml
│   ├── vlm_qwen3.yaml
│   └── vla_demo_orin.yaml       # 给 Orin TRT 后端打通用的小尺寸 demo
├── tests/
│   └── test_mock.py             # 端到端测试（基于MockNPUExecutor）
└── pyproject.toml               # 安装入口（依赖在此声明）
```

> **添加新后端**：在 `aria/backends/` 下新建 `<name>/` 子包，按
> `executor.py:<ClassName>` + 可选 `build.py:build(...)` 的约定实现，
> 然后在 `aria/backends/__init__.py` 的两张表里登记一行——`aria`
> 和 `aria-build` 的 `--executor` / `--backend` 选项会自动出现。

---

## 快速开始

> 推荐在 `aria` 容器/conda 环境里执行。

```bash
# 安装（开发模式：源码改动立即生效）
pip install -e .

# 如果需要 ONNX / Torch / TensorRT / 测试依赖：
pip install -e ".[onnx,torch,trt,dev]"

# === Mock 后端（无需 NPU/GPU）===
aria --config configs/vla_pi0.yaml
aria --config configs/vla_openvla.yaml
aria --config configs/vlm_qwen3.yaml

# 等价写法
python -m aria --config configs/vla_pi0.yaml

# 运行测试
pytest tests/
```

### 用 NVIDIA Orin 模拟 NPU（TensorRT 后端）

手头没真 NPU 时，可以在 Jetson Orin（或带 CUDA 的桌面卡）上用 TensorRT
后端跑——`.engine` 文件相当于真 NPU 的 `.om`/`.rknn`，行为语义最贴近
（静态 shape、预编译图、独立 device 内存、异步 stream）。Orin 上还能
让支持的层下到 **DLA 硬件 NPU**，未支持算子 GPU fallback。

```bash
# 1) 用随机权重为给定配置生成全套 .engine（dummy 模型，只为打通管线）
aria-build --config configs/vla_demo_orin.yaml \
           --out compiled/demo_orin

# 2) 加 --use-dla 让 vision encoder / flow head 下到 DLA core
aria-build --config configs/vla_demo_orin.yaml \
           --out compiled/demo_orin_dla --use-dla

# 3) 用 TensorRT 后端跑（--graph-dir 覆盖 yaml 里的 graph_dir）
aria --config configs/vla_demo_orin.yaml --executor trt \
     --graph-dir compiled/demo_orin_dla
```

> **注**：`aria-build` 生成的是 trivial 占位模型，输出值无意义；
> 它的作用是验证 **流水线 + bucket 切图 + NPU 抽象层** 是否打通。
> 真实部署时把 ONNX 换成实际权重，重新跑 `aria-build` 即可。

---

## 对接真实NPU

新增一个后端只需两步：在 `aria/backends/` 下放一个子包实现执行器
（可选再加一个 ONNX→产物 的编译脚本），然后在 `aria/backends/__init__.py`
的注册表里登记一行。上层 `runtime/` `models/` 代码不动。

**Step 1**：实现执行器，继承 `NPUExecutor` 把 5 个钩子填上。

```python
# aria/backends/qnn/executor.py
from aria.core.executor import NPUExecutor, GraphMeta

class QnnExecutor(NPUExecutor):

    def _load_graph(self, path: str, meta: GraphMeta):
        # 调用厂商SDK加载编译产物（.om / .rknn / .dlc / .qnn ...）
        ...

    def _execute(self, graph_handle, device_inputs: dict, meta: GraphMeta) -> dict:
        # 调用厂商SDK执行推理，输入输出均为 Device 侧地址
        ...

    def _alloc_device(self, size: int) -> int:
        # 在NPU DDR上分配内存，返回地址
        ...

    def _h2d(self, data, device_addr: int) -> None:
        # Host → Device 数据拷贝
        ...

    def _d2h(self, device_addr: int, shape, dtype):
        # Device → Host 数据拷贝，返回 numpy 数组
        ...
```

**Step 2（可选）**：实现 `build()`，让 `aria-build` 能为这个后端
直接产出 dummy engine 用作打通管线。签名是所有后端共用的。

```python
# aria/backends/qnn/build.py
from aria.core.executor import GraphMeta

def build(onnx_path: str, out_path: str,
          meta: GraphMeta, opts: dict) -> dict:
    # 调用 qnn-onnx-converter / qnn-model-lib-generator 等
    # 输出写到 out_path
    return {"backend": "QNN", "bytes": ...}
```

**Step 3**：在 `aria/backends/__init__.py` 注册：

```python
_EXECUTORS = {
    "trt": ("aria.backends.trt.executor", "TensorRTExecutor"),
    "qnn": ("aria.backends.qnn.executor", "QnnExecutor"),   # 新增
}
_BUILDERS = {
    "trt": ("aria.backends.trt.build", "build"),
    "qnn": ("aria.backends.qnn.build", "build"),            # 新增
}
```

`aria --executor qnn` 和 `aria-build --backend qnn` 立刻生效；
未装 QNN SDK 的环境不会被 `import qnn` 拖崩——后端是懒加载的，
只有用到该后端时才 import。

---

## 配置说明

```yaml
mode: vla                    # vla 或 vlm

vision:
  resolution: [448, 448]     # 端侧固定分辨率（消除动态shape）
  tile_size: [224, 224]
  tokens_per_tile: 256

llm:
  num_layers: 32
  hidden_dim: 4096
  num_heads: 32
  head_dim: 128
  prefill_buckets: [512, 1024, 2048]   # 静态多图bucket
  decode_buckets:  [512, 1024, 2048]
  max_seq_len: 4096

action:                      # VLA动作头
  head_type: flow_matching   # flow_matching 或 autoregressive
  action_dim: 7
  action_horizon: 16
  num_denoise_steps: 15

text:                        # VLM文本输出
  max_new_tokens: 512
  do_sample: true
  temperature: 0.7
  top_p: 0.9
  eos_token_ids: [151645, 151643]
```

---

## 架构概览

```
┌──────────────────────────────────────────────────────┐
│         VLARuntime.infer() / VLMRuntime.chat()       │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│              视觉编码器（共享）                        │
│     固定分辨率 → 固定tile数 → 一张静态NPU图            │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│           LLM Backbone（共享）                        │
│   多bucket Prefill图 / Decode图 / KV Cache            │
│        权重只加载一次，所有图共享同一份               │
└──────────────┬───────────────────────┬───────────────┘
               │                       │
┌──────────────▼──────────┐ ┌──────────▼──────────────┐
│   Flow Matching头        │ │  AR动作头 / 文本解码头   │
│   π0：非自回归           │ │  OpenVLA / Qwen3 VL     │
└─────────────────────────┘ └────────────────────────-┘
               │                       │
┌──────────────▼───────────────────────▼───────────────┐
│               NPU执行器抽象层                         │
│    MockNPUExecutor（开发）/ 厂商SDK（部署）           │
└──────────────────────────────────────────────────────┘
```

---

## License

MIT
