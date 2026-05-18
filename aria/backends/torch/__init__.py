"""
aria.backends.torch —— PyTorch 参考后端。

用途：在没有 NPU 的环境下做"实际能算"的 debug，
而不是像 MockNPUExecutor 那样产生随机张量。

- 同一份权重在 prefill / decode 之间复用
- attention 真读 past_kv，可以验证 KV cache / 前缀缓存的语义
- vision_encoder / flow_head 用桩模块满足形状契约

不追求精度或性能。
"""
