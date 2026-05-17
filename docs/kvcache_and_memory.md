# ARIA 内部参考：KV Cache 与端侧内存模型

> 范围：ARIA 当前 KV Cache 实现的全貌、端侧 SoC 的统一内存现实、`NPUExecutor` 抽象层的取舍，以及把 KV Cache 真正"零拷贝"喂到 NPU 的路径。
>
> 适用：希望理解 ARIA `KVCacheManager` 工作机制、想在 Jetson Orin 等统一内存平台上做内存优化的工程师。

---

## 1. KV Cache 实现总览

### 1.1 内存结构：一块静态预分配的大 ndarray

```
self._cache : np.ndarray
shape  = [num_layers, 2, batch, heads, max_seq_len, head_dim]
                       ^^^
                       0 = K
                       1 = V
dtype  = fp16
```

要点：

- 开机时一次性 `np.zeros(...)`，**全程没有动态 malloc**
- K/V 折在第 2 轴上，节省半量的索引和分支
- 两个标量游标管理生命周期：
  - `valid_len`：当前累计已写入多少 token
  - `history_len`：多轮对话里"上一轮收尾"的位置，下一轮 Prefill 起点

`reset()` 只把游标置零，**不真清零内存**——下次写入直接覆盖，省一次大 memset。

### 1.2 写入：三个接口分管两个阶段

| 阶段 | 接口 | 写入形状 | 游标推进 |
|---|---|---|---|
| Prefill（一次性） | `write_prefill(layer, k, v, start_pos)` | `[1, heads, seq, dim]` 整段 | 写完直接 `valid_len = start_pos + seq` |
| Decode（每步） | `write_decode(layer, k, v)` | `[1, heads, 1, dim]` 单 token | **不推进** |
| Decode step 结束 | `step_forward()` | – | `valid_len += 1` |

Decode 写入故意不在 `write_decode` 里 +1 —— 因为一步要写 N 层 KV，必须先 N 次 `write_decode(layer_i)` 全部落地，最后**一次** `step_forward()` 推进游标，避免循环里多 +N。这是个容易踩的坑，写法是刻意的。

### 1.3 读取：两种粒度

| 接口 | 返回 | 用途 |
|---|---|---|
| `get_kv(layer_idx)` | 单层 `(k, v)`，shape `[1, heads, valid_len, head_dim]` | 上层模型如果在 host 端实现 attention 时用 |
| `get_all_kv()` | 整段 `[layers, 2, batch, heads, valid_len, head_dim]` | 把整块 KV 作为 graph 输入丢给 decode 图（当前主路径） |

---

## 2. KV Cache 怎么"喂回" graph

### 2.1 Prefill 路径：图吐 KV，runtime 写入 cache

Prefill 图签名：

```
inputs:  input_ids, vision_feat, attention_mask, position_ids, kv_start_pos
outputs: last_hidden, kv_out
         kv_out shape = [num_layers*2, batch, heads, seq_len, head_dim]
```

执行后 runtime 切片写入：

```python
kv_out = out["kv_out"]
for layer_idx in range(num_layers):
    k = kv_out[layer_idx * 2,     :, :, :actual_len, :]
    v = kv_out[layer_idx * 2 + 1, :, :, :actual_len, :]
    self.kv_cache.write_prefill(layer_idx, k, v, start_pos=kv_start_pos)
```

### 2.2 Decode 路径：把整块 KV 作为图输入（核心 trade-off）

```python
kv_padded = self._pad_kv_for_decode(cur_kv_len, bucket)
out = self.executor.run(f"decode_{bucket}", {
    "input_id":    [[token_id]],
    "position_id": [[cur_kv_len]],
    "kv_in":       kv_padded,                # ← 把"全部历史 KV"作为输入传给图
})
```

每一步 Decode 都把当前累计的所有 KV 通过 input tensor 喂给图。具体动作：

1. 取当前 valid_len 段 KV
2. pad 到 bucket 大小（见下一节）
3. 整块 H2D 到 NPU
4. 图算完吐出 `kv_new [num_layers*2, batch, heads, 1, head_dim]`（单 token 的新 KV）
5. runtime 写回 `_cache` 下一格

**为什么这么"笨"**：`NPUExecutor` 抽象层假设所有有状态数据通过 inputs/outputs 流，否则 Mock 后端没法实现。**真 NPU 上不应该这么干**——KV 应该常驻 device DDR，模型用 `kv_offset` 标量定位。这是个挂账，第 5 节详细讲怎么补。

---

## 3. Bucket 机制：KV 长度涨过阈值就换图

```python
def _select_decode_bucket(self, kv_len: int) -> int:
    for b in sorted(self.lcfg.decode_buckets):
        if kv_len <= b:
            return b
```

配置示例 `decode_buckets: [512, 768, 1024]`：

| KV 当前长度 | 选用 Decode 图 | pad 到 |
|---|---|---|
| 200 | `decode_512` | 512 |
| 600 | `decode_768` | 768 |
| 900 | `decode_1024` | 1024 |

**模型本体在 NPU 上只编译这 3 张静态图**，每张接受**固定 KV 长度**的输入——这是为啥每步要 pad 到 bucket。Bucket 边界跨越时 graph handle 换一下，**KV 数据本身在 host cache 里不动**，只是切了张图来引用它。

这种"几个分立的静态 bucket"是 NPU 编译型推理框架的标准妥协方案：

- 真正动态 shape 编译产物体积爆炸 / 编不出来
- 单 bucket 上限太大浪费算力
- 几个分位 bucket 折中

---

## 4. 多轮（VLM）KV 复用：零拷贝原地续写

VLA 单轮：每次 `infer()` 进来 `kv_cache.reset()`，从 0 开始。

VLM 多轮的关键是这对：

```python
# Session 在 assistant 回复完时调
self.kv_cache.save_turn()    # 把当前 valid_len 抄给 history_len
```

下一轮新输入进来，Prefill 调用是：

```python
backbone.prefill(token_ids, vision_feat,
                 kv_start_pos=session.history_kv_len)
```

行为：

- 历史轮的 KV **原地保留在 `_cache` 里不动**（reset 才会清游标）
- 新轮 Prefill 输出从 `kv_start_pos = history_len` 开始往后追写
- valid_len 变成 `history_len + 新 prefill seq + 后续 decode 步数`
- 下一轮 `save_turn` 时 `history_len` 再前移

**跨轮 KV 是物理同一块 ndarray，零拷贝**。这是 ARIA 多轮 KV 复用的核心机制。

---

## 5. 端侧设备的内存模型：统一 DRAM 是常态

### 5.1 物理现实

端侧 SoC 几乎一律是统一 DRAM：

| SoC | CPU 与 NPU/GPU 共享 LPDDR | 备注 |
|---|---|---|
| Jetson Orin | 是，LPDDR5 | CPU / iGPU / DLA 全共享 |
| RK3588 | 是，LPDDR5 | A76 + RKNPU 共享 |
| Snapdragon 8 Gen* | 是 | CPU / Adreno / Hexagon 共享 |
| Apple M / A 系列 | 是，"Unified Memory Architecture" | 营销词都打出来了 |
| 高通 QCS / 联发科天玑 | 是 | 同上 |
| 例外 | Ascend 310 PCIe 卡这种"加卡"形态 | 不算典型端侧 |

### 5.2 那为什么软件层还要 H2D / D2H？

在 Orin 上 `cudaMemcpyHostToDevice` 是**逻辑搬运不是物理搬运**——DRAM 物理地址同一块，搬的实际上是：

1. **页表映射**：CPU 虚拟地址 ↔ NPU 设备虚拟地址（NPU 有自己的 IOMMU/SMMU）
2. **缓存一致性**：CPU L1/L2 里的脏数据 flush 回 LPDDR，让 NPU DMA 读到的是最新值
3. **driver 开销**：进 kernel、入队、stream 同步

**"区分 host / device"的本质不是物理隔离而是缓存域 + 地址空间隔离**。即使在统一内存平台上，也不能直接把 `malloc` 拿到的指针塞给 NPU，因为：

- NPU 看到的是另一套 IOMMU 映射
- CPU cache 里的脏数据 NPU DMA 引擎看不到
- NPU DMA 引擎对地址对齐 / 页边界有要求，普通 `malloc` 出来的不一定满足

### 5.3 三档拷贝优化

CUDA 提供三套不同程度的优化 API（其他厂商类似），按"拷贝量"递减：

| API | 行为 | 适用 |
|---|---|---|
| `cudaMalloc` + `cudaMemcpy`（**ARIA 当前 TRT 后端用的**） | host 一份 + device 一份，显式拷贝 | 最保守、跨平台稳，性能不极致 |
| `cudaHostAlloc(cudaHostAllocMapped)` + `cudaHostGetDevicePointer` | host 分配，但能拿到一个"device 视角的指针"，NPU 直接读写——**零拷贝** | Jetson / Tegra 最合适 |
| `cudaMallocManaged`（Unified Memory） | 一个指针 host/device 都用，driver 自动迁移页 | 易用但 demand-paging 不一定最快 |

---

## 6. ARIA `NPUExecutor` 抽象的位置

当前的 5 个钩子：

```
_alloc_device(size) → addr
_h2d(np.ndarray, addr)
_d2h(addr, shape, dtype) → np.ndarray
_load_graph(path, meta) → handle
_execute(handle, device_inputs, meta) → device_outputs
```

**这套抽象隐式假设"完全物理分离"语义**（数据中心 GPU 的世界观）。在 Orin / RK3588 上：

- **它没错**：物理共享时这套 API 仍然能正确表达"NPU 视角的指针 + 缓存一致性同步"
- **它过度抽象**：在 Jetson 上 `_h2d` 应该实现成"刷一下 CPU cache"而不是"分配新 device 内存 + memcpy"

如果想更精确反映端侧硬件，更 native 的抽象应该是：

```python
# 端侧 native 抽象（未实现，作为下一步参考）
_alloc_unified(size) → addr          # 一次分配，host/device 都能用的指针
_sync_for_device(addr, size)         # CPU 写完 → 刷 cache，让 NPU 看到
_sync_for_host(addr, size)           # NPU 写完 → invalidate CPU cache
_execute(...)
```

CUDA / RKNN / QNN 这类 SDK 底层都提供了这套语义的 API（`cudaHostAllocMapped`、RKNN 的 `rknn_create_mem`、QNN 的 ION / dmabuf 路径），只是大家用得少。

---

## 7. KV Cache 当前实现 vs 真 NPU 优化路径

| 当前状态 | 真 NPU 上该怎么做 |
|---|---|
| KV 在 host numpy 上，每步 decode H2D 整段 + D2H 单 token | KV 常驻 device DDR；图用 device 指针 + offset 引用 |
| Decode 图 input 里有完整 `kv_in [..., bucket, ...]` | 图签名里只接 `kv_offset` 标量；KV buffer 通过 `set_tensor_address` 一次绑定 |
| `_pad_kv_for_decode` 每步 concat 一段零 | 不需要，常驻 buffer 本来就 `max_seq_len` 大 |
| 多轮：history 物理零拷贝（已做对） | 同上，跨轮也不动 device buffer |

简单说：**KV 的"逻辑语义"已经对了**（静态预分配、增量更新、多轮零拷贝、bucket 切图），**"物理实现"还停留在 host 拷贝路径，是为了让 Mock 后端能跑通而做的妥协**。

要在 TRT 后端让 KV 真正常驻 device，路径：

1. `KVCacheManager` 增加 device-resident 模式：`_alloc_device(num_layers * 2 * max_batch * heads * max_seq_len * head_dim * 2)` 拿一块大 buffer
2. `write_prefill` / `write_decode` 从 host copy 改成 `cudaMemcpyAsync` slice 写入（或者在 zero-copy 路径下直接写到 mapped buffer）
3. Decode 图签名改造：去掉 `kv_in` 输入，加一个 `kv_offset` 标量（这要重新编 engine）
4. `set_tensor_address` 把 KV device buffer 绑死到所有 decode 图的 KV binding

### 7.1 Jetson 上的快赢路径：直接换 zero-copy alloc

如果不想重编 engine（保持现 graph 签名），仍然能在 Jetson 上拿到大部分收益：

- 把 TRT 后端的 `_alloc_device` 从 `cudaMalloc` 切到 `cudaHostAlloc(...Mapped)`
- `_h2d` 退化成 `cudaStreamSynchronize`（cache 由 stream 同步统一处理）
- 上层 KV cache 数据结构不动

预期效果：Decode 步耗时里那部分 `cudaMemcpyAsync × 多次` 的 driver overhead + cache 同步 ~2 ms / step（我们 demo 数据）大部分能省掉。但这条路是 Jetson 专属优化，离散 NPU 上没意义。

### 7.2 抽象不变的承诺

无论选哪条路径，**`NPUExecutor` 5 个钩子的签名都不动**，上层 `runtime/` `models/` 代码不用改。变的只是某个具体后端内部的实现细节。这是 ARIA 抽象层的核心价值。

---

## 8. 一句话总结

- **KV cache 的逻辑设计**：静态预分配 + 双游标 + bucket 切图 + 多轮原地续写——对了
- **KV cache 的物理实现**：当前走 host 拷贝路径，是为了 Mock 后端能跑——可以在不变抽象的前提下，按后端优化掉
- **端侧内存模型**：统一 DRAM 是常态，"区分 host / device" 是 IOMMU + 缓存域的事，不是物理隔离——优化的核心是消除 cache 同步开销，不是消除"拷贝"
