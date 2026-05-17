# ARIA 架构设计文档

**Action Reasoning Inference Accelerator —— 端侧 NPU 推理框架**

> 范围：ARIA 整个项目的架构设计、模块职责、关键抽象、数据流、设计取舍、扩展方式。
>
> 适用读者：第一次接触 ARIA 想搞清楚整体设计、想给 ARIA 接新后端 / 新模型、或负责后续优化的工程师。

---

## 1. 项目背景与目标

### 1.1 解决什么问题

端侧 NPU（Ascend / RKNN / QNN / DLA 等）跟数据中心 GPU 的本质差别在于：

- **不支持动态 shape**：编译期 shape 必须固定，运行期不能改
- **编译产物预生成**：图编译需要厂商 SDK，运行时只 load + execute，不能 JIT
- **算子集受限**：常见的 dynamic indexing、复杂 mask、灵活 Reduce 等可能不支持
- **内存模型不同**：CPU 和 NPU 通常共享 DRAM 但 IOMMU 隔离（详见 `kvcache_and_memory.md`）

ARIA 的目标是用一套**与厂商 SDK 解耦**的框架，吃掉这些约束：

- 静态多 bucket 图 + 权重共享，绕开动态 shape 限制
- Prefill / Decode 分图，匹配 NPU 静态编译模型
- 配置驱动，切换 VLA / VLM / 不同动作头只改 yaml
- 后端可插拔，Mock / TRT / ORT / 厂商 SDK 平等共存

### 1.2 当前覆盖的模型路径

| 类别 | 代表模型 | 解码方式 |
|---|---|---|
| VLA | π0 | Flow Matching（非自回归，多步去噪） |
| VLA | OpenVLA / RT-2 | 自回归（token-by-token） |
| VLM | Qwen3 VL | 自回归 + 多轮对话 |

### 1.3 设计原则

1. **后端无关**：上层 runtime / models 完全不依赖任何具体 NPU SDK
2. **配置驱动**：模型超参 / bucket / 解码路径全部在 yaml 里
3. **静态优先**：所有 buffer、所有 graph 在启动时预先分配 / 加载
4. **零侵入扩展**：加一个新后端 / 新模型类型，只改注册表 + 新文件，不动既有代码

---

## 2. 仓库布局

```
aria/                                # 仓库根
├── aria/                            # 可安装的 Python 包
│   ├── __init__.py
│   ├── __main__.py                  # 支持 python -m aria
│   ├── cli.py                       # aria 命令入口
│   │
│   ├── core/                        # 后端无关的核心抽象
│   │   ├── executor.py              # NPUExecutor 抽象基类 + MockNPUExecutor + GraphMeta
│   │   ├── memory.py                # StaticMemoryPool（host 侧静态规划）
│   │   ├── kv_cache.py              # KVCacheManager
│   │   └── scheduler.py             # 三级流水线（vision / inference / output）
│   │
│   ├── backends/                    # NPU 后端实现（按厂商一目一包）
│   │   ├── __init__.py              # 注册表：build_executor / get_builder
│   │   ├── trt/                     # TensorRT 后端（含 Orin DLA）
│   │   │   ├── executor.py          # TensorRTExecutor + _DevicePool
│   │   │   └── build.py             # ONNX → .engine
│   │   └── ort/                     # ONNXRuntime 后端（演示权重共享）
│   │       ├── executor.py          # ORTExecutor
│   │       └── build.py             # ONNX → 剥共享权重的 .onnx + shared_weights.npz
│   │
│   ├── models/                      # 模型构件（注册图 + 调用 executor.run）
│   │   ├── base.py                  # FrameworkConfig / VisionConfig / LLMConfig 等
│   │   ├── vision_encoder.py        # 视觉编码器
│   │   ├── llm_backbone.py          # LLM Backbone（多 bucket 图管理）
│   │   ├── ar_decoder.py            # 自回归动作头（OpenVLA / RT-2）
│   │   ├── flow_decoder.py          # Flow Matching 动作头（π0）
│   │   └── text_decoder.py          # 文本解码头（Qwen3 VL）
│   │
│   ├── runtime/                     # 端到端运行时
│   │   ├── vla_runtime.py           # VLARuntime（单轮）
│   │   ├── vlm_runtime.py           # VLMRuntime（多轮对话）
│   │   └── session.py               # 多轮 Session
│   │
│   └── tools/
│       └── build_dummy_engines.py   # aria-build：harvest GraphMeta + 生成 dummy ONNX
│
├── configs/                         # 示例配置
│   ├── vla_pi0.yaml
│   ├── vla_openvla.yaml
│   ├── vlm_qwen3.yaml
│   └── vla_demo_orin.yaml           # Orin 上能跑通的小尺寸 demo
│
├── tests/
│   └── test_mock.py                 # 端到端测试（Mock 后端）
│
├── pyproject.toml                   # 安装入口 + extras：[trt] [ort] [dev]
└── README.md
```

---

## 3. 总体架构

### 3.1 分层视图

```
┌─────────────────────────────────────────────────────────────────────┐
│  Application                                                         │
│  CLI (aria.cli)  /  用户代码 import aria.runtime                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│  Runtime Layer                                                       │
│  VLARuntime.infer()       │       VLMRuntime.chat() + Session       │
└──────────┬──────────────────────────────┬───────────────────────────┘
           │                              │
┌──────────▼──────────────────────────────▼───────────────────────────┐
│  Model Layer                                                         │
│  VisionEncoder   LLMBackbone   ARDecoder   FlowDecoder   TextDecoder │
│        ↓             ↓             ↓             ↓            ↓     │
│        └─────────────┴─────register_graph───────┴────────────┘      │
│                          executor.run("graph_name", inputs)         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│  Core Abstraction (后端无关)                                          │
│  NPUExecutor 抽象基类 + GraphMeta + KVCacheManager + StaticMemoryPool │
└──────────────────┬──────────────────────────────────────────────────┘
                   │                                       注册表
┌──────────────────▼──────────────────────────────────────────────────┐
│  Backend Layer                                                       │
│ ┌────────────┐ ┌──────────────────┐ ┌────────────────────────────┐  │
│ │   Mock     │ │   TensorRT       │ │   ONNXRuntime              │  │
│ │ (纯 numpy) │ │ (含 Orin DLA)    │ │ (权重共享演示)             │  │
│ └────────────┘ └──────────────────┘ └────────────────────────────┘  │
│ 未来：QNN / RKNN / CANN ...（按目录约定加入）                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 关键依赖方向

- 上层只依赖下层，**下层不知道上层存在**
- `models/` 依赖 `core/`，但不直接 `import` 任何 backend
- `backends/` 依赖 `core/`，但不直接 `import` 任何 model
- `runtime/` 串起来调度 `models/` 用 `executor`，executor 是哪种类型由 CLI / 调用方决定
- **`backends/__init__.py` 是唯一已知所有后端的地方**，懒加载

---

## 4. 核心抽象层 (`aria/core/`)

### 4.1 `NPUExecutor` —— 后端契约（`executor.py`）

整个 ARIA 框架的**唯一一处后端契约**。任何 NPU 后端只要实现这 5 个钩子，上层全部代码不变。

```python
class NPUExecutor(ABC):
    @abstractmethod
    def _load_graph(self, path: str, meta: GraphMeta) -> Any:        # 加载编译产物
    @abstractmethod
    def _execute(self, handle, device_inputs: dict, meta) -> dict:    # 执行推理
    @abstractmethod
    def _alloc_device(self, size: int) -> int:                        # NPU DDR 分配
    @abstractmethod
    def _h2d(self, data: np.ndarray, addr: int) -> None:              # Host → Device
    @abstractmethod
    def _d2h(self, addr: int, shape, dtype) -> np.ndarray:            # Device → Host

    def _free_device(self, addr: int) -> None:                        # 可选：归还内存
        return None
```

公开接口（基类实现好的，子类不要重写）：

| 方法 | 作用 |
|---|---|
| `register_graph(meta)` | 调用 `_load_graph` 拿 handle，登记到 `_graphs` 字典 |
| `run(graph_name, inputs)` | 标准流程：H2D inputs → execute → D2H outputs → 释放 transient 内存 |
| `load_weights(weight_dict)` | 权重一次性上传 DDR，常驻不释放（多图共享） |
| `enable_profiling(bool)` | 开启每张图的延迟统计 |
| `get_profiling_stats()` | 返回 mean/p50/p95/p99 |

### 4.2 `GraphMeta` —— 编译产物元数据

```python
@dataclass
class GraphMeta:
    name:          str                 # 图名，如 "prefill_512"
    path:          str                 # 编译产物路径
    input_shapes:  Dict[str, tuple]    # 各输入张量形状
    output_shapes: Dict[str, tuple]    # 各输出张量形状
    input_dtypes:  Dict[str, np.dtype] # 子类 _load_graph 时回填
    output_dtypes: Dict[str, np.dtype] # 同上
    handle:        Any = None          # _load_graph 返回的句柄
```

GraphMeta 由模型构件（`VisionEncoder` / `LLMBackbone` / `FlowDecoder`）在构造时填充并 `register_graph` 到 executor。**Backend 只读这个数据结构，不关心 graph 是 .engine / .onnx / .om / .rknn**。

### 4.3 `KVCacheManager` —— KV 静态管理（`kv_cache.py`）

详见 `kvcache_and_memory.md`，要点：

- 一块 6 维 ndarray：`[layers, 2(K/V), batch, heads, max_seq_len, head_dim]`
- 双游标：`valid_len`（当前长度）+ `history_len`（多轮起点）
- 三类写入：`write_prefill` / `write_decode` / `step_forward`
- 多轮 `save_turn`，跨轮零拷贝

### 4.4 `StaticMemoryPool` —— 静态内存规划（`memory.py`）

Host 侧模拟 NPU DDR 规划，按 `BufferSpec` 注册再统一 `allocate_all()`。三类 buffer：

- **持久区**：权重、KV Cache —— 整个生命周期保留
- **复用区**：激活值、workspace —— 所有图复用最大的那块
- **IO 区**：输入输出 staging

> 注：当前实现是 host numpy 模拟，未真正接管 device 内存。真 NPU 后端实现里这部分由各厂商 SDK 内部负责，框架层不强制使用 `StaticMemoryPool`。这是一个**预留位**。

### 4.5 `PipelineScheduler` —— 三级异步流水线（`scheduler.py`）

```
┌──────────────┐  vision_q  ┌──────────────┐  prefill_q  ┌──────────────┐
│ VisionWorker │ ─────────▶ │InferenceWrkr │ ──────────▶ │ OutputWorker │
└──────────────┘            └──────────────┘             └──────────────┘
   视觉预处理                LLM Prefill+Decode             消费 result
```

`submit(image, text)` → 返回 `request_id`，三线程异步处理，掩盖各阶段延迟。**当前未被 VLARuntime / VLMRuntime 启用**（同步路径已够用），是为高吞吐场景预留。

---

## 5. 后端系统 (`aria/backends/`)

### 5.1 注册表 + 懒加载

`aria/backends/__init__.py`：

```python
_EXECUTORS: Dict[str, Tuple[str, str]] = {
    "trt": ("aria.backends.trt.executor", "TensorRTExecutor"),
    "ort": ("aria.backends.ort.executor", "ORTExecutor"),
    # 未来：
    # "qnn":  ("aria.backends.qnn.executor",  "QnnExecutor"),
    # "rknn": ("aria.backends.rknn.executor", "RKNNExecutor"),
}

_BUILDERS: Dict[str, Tuple[str, str]] = {
    "trt": ("aria.backends.trt.build", "build"),
    "ort": ("aria.backends.ort.build", "build"),
}

def build_executor(name, **kwargs) -> NPUExecutor: ...   # 用到才 import
def get_builder(name) -> Callable: ...
def list_executors() -> List[str]:  ["mock"] + 注册表
def list_builders()  -> List[str]:  注册表
```

**懒加载意味着**：没装 tensorrt 的环境，`import aria` 不会崩；只有 `--executor trt` 时才真正 `importlib.import_module`。

### 5.2 已实现的后端

#### Mock（`core/executor.py:MockNPUExecutor`）

纯 numpy 实现，模拟 device 地址（自增整数）+ 模拟内存（dict），随机张量做输出。开发 / 测试时跑通整条流水。

#### TensorRT（`backends/trt/`）

- `executor.py:TensorRTExecutor`：
  - `_load_graph` 用 `trt.Runtime.deserialize_cuda_engine` 反序列化 .engine
  - `_execute` 用 `IExecutionContext.set_tensor_address(name, addr)` + `execute_async_v3(stream)`
  - 内置 `_DevicePool` 复用 `cudaMalloc` 出来的 device memory，按精确 size 做 free-list
  - 提供 `get_pool_stats()` 观测
- `build.py`：ONNX → .engine
  - 支持 FP16
  - 支持 `--use-dla`：尝试把支持层下到 Orin DLA core，单图失败自动回退 GPU

#### ONNXRuntime（`backends/ort/`）

设计目标是**演示"权重一份 + 多图共享"语义**——这是 TRT 做不到的事（详见第 11 节）。

- `build.py`：把跨图同名 initializer（`vision_proj.` / `llm_proj.` / `flow_proj.` 前缀）剥到 `shared_weights.npz`，将原 initializer 在 .onnx 中标记为 `data_location=EXTERNAL` 占位
- `executor.py:ORTExecutor`：
  - 首次 `_load_graph` 时加载 npz → `OrtValue.ortvalue_from_numpy(arr)`（零拷贝引用 numpy）
  - 每张 session **单独** SessionOptions（API 要求名字必须在该模型里存在），但**注入同一组 OrtValue 实例**
  - 物理验证：`OrtValue.data_ptr() == np.ndarray.ctypes.data`，全部 session 走同一段内存

### 5.3 怎么加新后端（以 QNN 为例）

```
aria/backends/qnn/
├── __init__.py
├── executor.py    # class QnnExecutor(NPUExecutor): ... 实现 5 钩子
└── build.py       # def build(onnx_path, out_path, meta, opts) -> dict: ...
```

然后在 `aria/backends/__init__.py` 各加一行：

```python
_EXECUTORS["qnn"] = ("aria.backends.qnn.executor", "QnnExecutor")
_BUILDERS["qnn"]  = ("aria.backends.qnn.build",  "build")
```

`aria --executor qnn` 和 `aria-build --backend qnn` 立刻可用。其他代码一字不动。

---

## 6. 模型层 (`aria/models/`)

### 6.1 职责

模型层每个文件代表一个**模型构件**，负责：

1. 在 `__init__` 里**注册自己用到的 graph**（通过 `executor.register_graph(meta)`）
2. 提供具体功能接口（`encode` / `prefill` / `decode_step` / `decode`），内部调用 `executor.run(name, inputs)`
3. 不持有 device 内存，所有数据通过 `executor` 的 5 个钩子流转

### 6.2 各构件一览

| 构件 | 注册的 graph | 主要接口 | 输入 | 输出 |
|---|---|---|---|---|
| `VisionEncoder` | `vision_encoder` | `encode(image)` | `[H, W, C] uint8` | `[1, total_tokens, feat_dim] fp16` |
| `LLMBackbone` | `prefill_{N}` × buckets | `prefill(token_ids, vision_feat, kv_start_pos)` | tokens + vision_feat | `last_hidden [1, hidden_dim]` |
| `LLMBackbone` | `decode_{N}` × buckets | `decode_step(token_id)` | 单 token + KV | `logits [vocab_size]` |
| `ARDecoder` | （复用 LLM decode） | `decode(bos_id)` | LLM | `action [action_dim] fp32` |
| `FlowDecoder` | `flow_head` | `decode(hidden_state)` | last_hidden | `action [horizon, dim] fp32` |
| `TextDecoder` | （复用 LLM decode） | `decode(last_hidden)` | LLM | `text str` |

### 6.3 LLMBackbone 的核心：多 bucket 切图

```python
# 注册 N 张 prefill 图（每个 bucket 一张）
for seq_len in cfg.llm.prefill_buckets:        # [512, 1024, 2048]
    register_graph(GraphMeta(name=f"prefill_{seq_len}", ...))

# 注册 N 张 decode 图
for kv_len in cfg.llm.decode_buckets:
    register_graph(GraphMeta(name=f"decode_{kv_len}", ...))
```

调用时按当前长度选 bucket：

```python
def prefill(self, token_ids, vision_feat, kv_start_pos):
    bucket = self._select_prefill_bucket(kv_start_pos + actual_len)
    out = self.executor.run(f"prefill_{bucket}", { ... })
    # 写 KV Cache
    for layer_idx in range(num_layers):
        self.kv_cache.write_prefill(layer_idx, ...)
    return out["last_hidden"]
```

### 6.4 Flow Decoder 的特殊性：多步去噪

`FlowDecoder.decode(hidden_state)` 内部跑一个去噪循环：

```python
action = rng.standard_normal((..., horizon, action_dim))  # 纯噪声起点
for i in range(num_denoise_steps):
    t      = timesteps[i]
    velocity = executor.run("flow_head", {                # 同一张 flow_head 图，跑 N 次
        "hidden_state": hidden_state,
        "noisy_action": action,
        "timestep":     [t],
    })["velocity"]
    action = action + velocity * dt                       # Euler 积分
return action[0]
```

特点：**单图被反复调用**，KV Cache 不参与（动作头不用 LLM 的注意力）。

---

## 7. 运行时层 (`aria/runtime/`)

### 7.1 `VLARuntime` —— 单轮 VLA 推理

```
infer(image, instruction):
    kv_cache.reset()                          # VLA 每次单轮，KV 清零
    vision_feat = vision_encoder.encode(image)
    token_ids   = tokenize(instruction)
    last_hidden = llm.prefill(token_ids, vision_feat, kv_start_pos=0)
    if head_type == "flow_matching":
        return flow_decoder.decode(last_hidden)            # 非自回归
    else:
        return ar_decoder.decode(bos_token_id)             # 自回归走 LLM.decode_step
```

### 7.2 `VLMRuntime` + `Session` —— 多轮对话

```
new_session() → session_id:
    kv_cache = KVCacheManager(...)            # 每个 session 独立 KV
    return session_id

chat(messages, session_id):
    session = _sessions[session_id]
    # 用 history_len 作为新轮 Prefill 起点（不 reset KV）
    backbone = LLMBackbone(..., session.kv_cache)
    vision_feat = vision_encoder.encode(image_in_messages)
    last_hidden = backbone.prefill(tokens, vision_feat,
                                   kv_start_pos=session.history_kv_len)
    text = text_decoder.decode(last_hidden)   # 自回归生成
    session.add_assistant_turn(text, ...)      # 内部调 kv_cache.save_turn()
    return text
```

**关键**：每个 session 持有独立的 `KVCacheManager`（隔离不同对话历史），但**所有 session 共用同一个 `executor` + 同一组 graph**（NPU 上只 load 一次）。

### 7.3 同步 vs 异步

当前 `VLARuntime.infer()` 和 `VLMRuntime.chat()` 都是**同步阻塞**调用——按 vision → prefill → decode 顺序走，不开线程。

`PipelineScheduler` 已经实现但**尚未接入** runtime——预留给：
- 多 NPU 异步流水（vision 在 NPU0、LLM 在 NPU1）
- 高吞吐多请求服务化场景

---

## 8. 配置系统

### 8.1 yaml → `FrameworkConfig`

```yaml
mode: vla                          # vla / vlm
graph_dir: compiled/pi0            # 编译产物目录
weight_path: weights/pi0.bin

vision:
  resolution: [224, 224]
  tile_size:  [224, 224]
  tokens_per_tile: 256
  feat_dim: 4096

llm:
  num_layers: 32
  hidden_dim: 4096
  num_heads:  32
  head_dim:   128
  vocab_size: 32000
  prefill_buckets: [512, 768, 1024]
  decode_buckets:  [512, 768, 1024]
  max_seq_len: 2048

action:                            # 仅 mode=vla 用
  head_type: flow_matching         # flow_matching / autoregressive
  action_dim: 7
  action_horizon: 16
  num_denoise_steps: 15

text:                              # 仅 mode=vlm 用
  max_new_tokens: 512
  do_sample: true
  temperature: 0.7
  top_p: 0.9
  eos_token_ids: [151645, 151643]

max_batch: 1
pad_token_id: 0
bos_token_id: 1
eos_token_id: 2
```

### 8.2 数据类分层

```
FrameworkConfig
 ├─ mode, graph_dir, weight_path, max_batch, ...
 ├─ vision  : VisionConfig    (resolution, tile_size, tokens_per_tile, feat_dim)
 │            派生属性: num_tiles, total_vision_tokens
 ├─ llm     : LLMConfig       (num_layers, hidden_dim, *_buckets, max_seq_len, ...)
 ├─ action  : ActionConfig    (head_type, action_dim, horizon, num_denoise_steps)
 └─ text    : TextConfig      (max_new_tokens, sampling 参数, eos)
```

所有模型构件 / runtime 通过 `cfg.vision.feat_dim` / `cfg.llm.hidden_dim` 这种长路径访问。

---

## 9. 构建管线（`aria-build`）

### 9.1 整体流程

```
┌────────────────────────────────────────────────────────────────────────┐
│ aria-build --config X.yaml --out compiled/X --backend trt              │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
       ┌──────────────────────────▼────────────────────────────┐
       │  1. FrameworkConfig.from_yaml(X.yaml)                  │
       └──────────────────────────┬────────────────────────────┘
                                  │
       ┌──────────────────────────▼────────────────────────────┐
       │  2. harvest_graphs(cfg)：用 _HarvestExecutor 喂给       │
       │     VisionEncoder / LLMBackbone / FlowDecoder           │
       │     拢出 List[GraphMeta]                                │
       └──────────────────────────┬────────────────────────────┘
                                  │
       ┌──────────────────────────▼────────────────────────────┐
       │  3. 对每个 meta:                                       │
       │     a. _shared_params_for(name, cfg) 决定共享权重族     │
       │     b. _make_dummy_module(meta, shared) 造一个          │
       │        trivial nn.Module（含正确 shape + shared params）│
       │     c. export_onnx() → 临时 .onnx                       │
       │     d. backend_build(onnx_path, out_path, meta, opts)   │
       │        分派到 backends/<name>/build.py                   │
       └──────────────────────────┬────────────────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       ┌────────────────────┐           ┌────────────────────┐
       │  backends/trt/build │           │  backends/ort/build │
       │  → .engine          │           │  → 剥过权重的 .onnx │
       │                     │           │   + shared_weights  │
       │  (可选 DLA)         │           │       .npz          │
       └────────────────────┘           └────────────────────┘
```

### 9.2 后端无关 vs 后端专属

`tools/build_dummy_engines.py` 里这些是后端无关的：

- `harvest_graphs(cfg)`：实例化模型构件并拢 GraphMeta
- `_shared_params_for(name, cfg)`：决定每图挂哪些 nn.Linear（共享权重族）
- `_make_dummy_module(meta, shared)`：生成 trivial PyTorch 模型
- `export_onnx(meta, path, shared)`：导出标准 ONNX

后端专属逻辑在 `backends/<name>/build.py`，签名统一为：

```python
def build(onnx_path: str, out_path: str,
          meta: GraphMeta,
          opts: dict) -> dict
```

- **TRT** 实现：ONNX 解析 + Builder + 可选 DLA 配置 → 序列化 engine
- **ORT** 实现：剥 initializer 到 npz + 在 .onnx 标记 EXTERNAL → save

### 9.3 共享权重族

为了让 ORT 后端能演示"权重一份多图共享"，build 阶段会按 graph 名挂 nn.Linear：

| graph 名前缀 | 共享参数 | 形状 |
|---|---|---|
| `vision_encoder` | `vision_proj.weight` | `(feat_dim, feat_dim)` |
| `prefill_*` / `decode_*` | `llm_proj.weight` | `(hidden_dim, hidden_dim)` |
| `flow_head` | `flow_proj.weight` | `(action_dim, action_dim)` |

所有 prefill / decode bucket 图**共用同一个名字** `llm_proj.weight` —— 在 ORT 后端启动时，N 个 session 通过 `add_external_initializers` 引用同一份 numpy 内存。

---

## 10. 端到端执行流

### 10.1 VLA 推理（π0 Flow Matching）

```
用户调用 runtime.infer(image, "pick up the cube")
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ VLARuntime.infer()                                              │
│   ├─ kv_cache.reset()                          [valid_len=0]    │
│   │                                                             │
│   ├─ vision_encoder.encode(image)              [Step Vision]    │
│   │      │                                                      │
│   │      ▼ executor.run("vision_encoder", {"tiles": ...})       │
│   │      → vision_feat [1, total_tokens, feat_dim]              │
│   │                                                             │
│   ├─ tokenize(instruction) → token_ids                          │
│   │                                                             │
│   ├─ llm.prefill(token_ids, vision_feat, kv_start_pos=0)        │
│   │      │                                       [Step Prefill] │
│   │      ├─ select bucket b = 512 / 1024 / 2048                 │
│   │      ├─ pad inputs to bucket                                │
│   │      ▼ executor.run(f"prefill_{b}", {...})                  │
│   │      → kv_out [layers*2, ...] + last_hidden [1, hidden_dim] │
│   │      ├─ for layer in range(N):                              │
│   │      │   kv_cache.write_prefill(layer, k, v, start_pos=0)   │
│   │      └─ return last_hidden                  [valid_len=seq] │
│   │                                                             │
│   └─ flow_decoder.decode(last_hidden)           [Step Decode]   │
│          │                                                      │
│          ├─ noisy_action ← N(0, 1)                              │
│          ├─ for step in range(num_denoise_steps):               │
│          │   executor.run("flow_head", {                        │
│          │     hidden_state, noisy_action, timestep})           │
│          │   → velocity                                         │
│          │   noisy_action += velocity * dt                      │
│          └─ return action [horizon, action_dim]                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  action 数组（机械臂控制信号）
```

### 10.2 VLM 多轮（Qwen3 VL）

```
runtime.new_session() → session_id
         │
         ▼ 第 1 轮 chat(messages_1, session_id)
┌─────────────────────────────────────────────────────────────────┐
│ session.kv_cache: history_len=0, valid_len=0                    │
│   vision_encoder.encode(image_in_msg_1)                         │
│   llm.prefill(tokens_1, vision_feat, kv_start_pos=0)            │
│     → KV[0:L1] 写入                            [valid_len=L1]   │
│   text_decoder.decode(last_hidden):                             │
│     while not EOS:                                              │
│       llm.decode_step(token)                                    │
│         → KV[L1+i] 写入                  [valid_len=L1+i]       │
│       sample next token from logits                             │
│   session.add_assistant_turn(text):                             │
│     kv_cache.save_turn()              [history_len ← valid_len] │
└─────────────────────────────────────────────────────────────────┘

         │
         ▼ 第 2 轮 chat(messages_2, session_id)
┌─────────────────────────────────────────────────────────────────┐
│ session.kv_cache: history_len=H1, valid_len=H1                  │
│ （第 1 轮 KV 原地保留在 _cache 里！）                              │
│                                                                 │
│   llm.prefill(tokens_2, vision_feat=None, kv_start_pos=H1)      │
│     → KV[H1:H1+L2] 追加写              [valid_len=H1+L2]       │
│   text_decoder.decode(...)                                      │
│   session.add_assistant_turn(...)                               │
│     kv_cache.save_turn()                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. 关键设计决策

### 11.1 为什么静态多图 + 权重共享

**问题**：NPU 编译期 shape 必须固定，但 LLM 序列长度天然动态。

**做法**：把序列长度离散化成几个 bucket（如 512 / 1024 / 2048），每个 bucket 编一张 Prefill 图 + 一张 Decode 图。运行期按当前长度选图。

**权衡**：
- 编译产物多 N 张图（N≤5 一般够），可控
- 每张图 pad 浪费部分算力，但比"单大 bucket 永远占满"省
- 真 NPU 上**权重应当多图共享**（否则 N 张图 N 份权重，DDR 撑爆）—— TRT engine 实际做不到（详见 11.5），ORT 用 `add_external_initializers` 做到了

### 11.2 为什么 Prefill / Decode 分两套图

**问题**：Prefill 和 Decode 的计算模式差太多：
- Prefill：`[1, seq_len, hidden]` × `[hidden, hidden]` = `[1, seq_len, hidden]`，**算密集**
- Decode：`[1, 1, hidden]` × `[hidden, hidden]` = `[1, 1, hidden]`，**访存密集**（每步还得读全部 KV）

**做法**：编两套独立的图，让编译器各自针对计算模式做调度（Prefill 优化矩阵分块、Decode 优化 KV 流式读取）。

### 11.3 为什么用 `NPUExecutor` 这种 host/device 分离抽象

**问题**：不同厂商 NPU SDK 都给 `Malloc/Memcpy/Execute` 这套接口（Ascend ACL、RKNN、QNN、CUDA 都一样）。即使端侧 SoC 物理内存共享，软件层仍然要管 IOMMU 映射 + cache 一致性。

**做法**：基类强制 5 个钩子，按"完全分离"语义建模——最保守、跨厂商通用。

**取舍**：在 Jetson 这种统一内存平台上有过度抽象，每步 decode 多花约 1-2 ms 在"假 H2D / D2H"上（详见 `kvcache_and_memory.md` 第 7 节）。可以在不改抽象的前提下，让具体后端（如 TRT）的 `_alloc_device` 切到 `cudaHostAllocMapped` 做 zero-copy。

### 11.4 为什么 KV Cache 走 host 路径

**问题**：当前 KV Cache 是 numpy ndarray，每步 decode 把整段 H2D + 单 token D2H。

**原因**：让 Mock 后端能跑通。Mock 没有真 device，KV 只能放 host。框架层为了通用退化到 host 路径。

**未来优化方向**：

```
当前：                              真 NPU 应当：
KV in host numpy                    KV 常驻 device DDR
Decode 图签名 kv_in [..., bucket]   Decode 图签名只接 kv_offset 标量
每步 H2D 整段                       set_tensor_address 一次绑定 device ptr
```

要落地需要：(a) 后端层 KVCacheManager 加 device-resident 模式；(b) 重编 decode engine 改图签名。**抽象不变，只换实现**。

### 11.5 为什么三个后端共存

| 后端 | 验证的语义维度 |
|---|---|
| **Mock** | 流水线 / KV cache / scheduler 本身是否正确 |
| **TRT (+ DLA)** | 静态 shape / 预编译图 / device 内存隔离 / 真硬件延迟 / DLA 算子映射 |
| **ORT** | "一份权重 + N 图共享"（TRT 做不到——engine 把权重烘进 kernel） |

三个后端**互补**，分别压一个独立的语义维度。任何一个都不可省。

### 11.6 为什么 ORT 用 `add_external_initializers` 而非 `add_initializer`

走了一段弯路：

1. 先试 `add_initializer` —— Python wrapper 的 `_validate_input` 把 graph.input 列必填，过不去
2. 再试 strip initializer + 提到 graph.input —— validate 过去了，但 `sess.run()` 又要求 feed 这些权重名
3. 最终用 `add_external_initializers` + ONNX 把 initializer 标 EXTERNAL（占位 location）：
   - 模型结构合法（initializer 仍在 graph.initializer 列表里）
   - 数据剥到外部
   - 运行时用 `add_external_initializers` 提供 OrtValue，ORT 跳过外部文件读取

每个 session 必须用**自己的** SessionOptions（API 要求名字精确匹配该模型），但**注入同一组 OrtValue 实例**——OrtValue 引用同一段 numpy memory。

---

## 12. 扩展点

### 12.1 加新后端

参考第 5.3 节。三步：
1. 新建 `aria/backends/<name>/` 目录
2. 实现 `executor.py:YourExecutor`（继承 NPUExecutor，实现 5 钩子）
3. 可选实现 `build.py:build(...)` 让 `aria-build` 支持你的后端
4. 在 `aria/backends/__init__.py` 注册表加一行

### 12.2 加新动作头

例：扩散模型的 score-based 头。

1. 在 `aria/models/` 加 `diffusion_decoder.py`
2. 类似 `FlowDecoder`，在 `__init__` 注册图，提供 `decode(hidden_state)` 接口
3. 在 `models/__init__.py` re-export
4. 在 `VLARuntime.__init__` 加一个 `elif self.acfg.head_type == "diffusion":` 分支
5. 在 `ActionConfig` 加相关超参（去噪步数 / scheduler 等）

### 12.3 加新模型类型（如 VLA + Audio）

VLA / VLM 之外加一个 VAA（Vision-Audio-Action）？

1. 加 `models/audio_encoder.py`
2. 加 `runtime/vaa_runtime.py`，组合 vision + audio + LLM + action
3. 在 `cli.py` 的 `main()` 里加 mode 分派
4. 写一份 `configs/vaa_xxx.yaml`

模型层和 runtime 是**多对多**：可以多种 runtime 共用同一个 LLMBackbone，也可以多种动作头插同一个 runtime。

### 12.4 给执行器加优化

例：TRT 后端在 Orin 上做 zero-copy。

- 只改 `aria/backends/trt/executor.py`
- 把 `_alloc_device` 从 `cudaMalloc` 换成 `cudaHostAlloc(...Mapped)`
- `_h2d` 退化为 `cudaStreamSynchronize`
- **基类抽象不变，上层 runtime / models / build script 全部不动**

---

## 13. 当前已挂的账

整理一下设计上知道、但暂未做的事，未来对接真硬件 / 真模型时会需要：

| 项 | 当前 | 应做 |
|---|---|---|
| KV Cache 物理位置 | host numpy | device 常驻，`set_tensor_address` 绑定 |
| Decode 图签名 | 接受整段 `kv_in` 输入 | 只接 `kv_offset` 标量 |
| Tokenizer | mock 字符映射 | 接 `transformers.AutoTokenizer` / sentencepiece |
| Image preprocess | numpy resize 占位 | 真 cv2 / libjpeg-turbo / NPU 视频流接入 |
| Dynamic shape | 不支持，bucket 离散化 | TensorRTExecutor 加 `set_input_shape`，配合 GraphMeta 语义扩展 |
| Pipeline scheduler | 实现了未接入 | 高吞吐场景下接入 VLM/VLA runtime |
| StaticMemoryPool | 实现了未接入 | 用作 backend 内部内存规划的统一接口 |
| 真模型对接 | dummy ONNX | π0 / OpenVLA / Qwen3 VL 的真实 ONNX 导出脚本 |
| weight 共享在 TRT | engine 各自带 | 用 TRT refit API 或 plugin（成本高，可能不值得） |
| Jetson zero-copy | 标准 `cudaMalloc` 路径 | `cudaHostAllocMapped` 实测延迟差 |

---

## 14. 一页速查

```
入口：
  aria        命令 / python -m aria             ← cli.py
  aria-build  命令                              ← tools/build_dummy_engines.py

最重要的 5 个文件（看懂这 5 个 = 看懂 ARIA 一半）：
  aria/core/executor.py          NPUExecutor 抽象 + Mock 实现
  aria/core/kv_cache.py          KV Cache 数据结构
  aria/models/llm_backbone.py    多 bucket prefill/decode 图管理
  aria/runtime/vla_runtime.py    端到端编排
  aria/backends/__init__.py      后端注册表（懒加载）

关键概念：
  GraphMeta        编译产物元数据，跨后端契约
  bucket           离散 shape 集合，规避 NPU 动态 shape 限制
  kv_start_pos     多轮 KV 起点
  shared_weights   ORT 后端专用，N 图引用一份权重

跑通 demo：
  在 aria 容器内
  pip install -e ".[trt,ort,dev]"
  aria-build --config configs/vla_demo_orin.yaml --out compiled/demo --backend trt
  aria --config configs/vla_demo_orin.yaml --executor trt --graph-dir compiled/demo

进阶参考：
  docs/kvcache_and_memory.md     KV Cache 实现 + 端侧内存模型详解
```
