"""
tests/test_torch.py

Torch 参考后端的测试。两个层次：
  1. TinyLLM 模型层：直接 forward，对比"全程"与"prefix+suffix(带 past_kv)"
     在 last_hidden 上的数值差异 —— 验证 attention 的 past_kv 语义对齐。
  2. TorchExecutor 层：通过 executor.run() 走同样的等价检查，
     验证 numpy↔torch 转换、graph 名分派、bucket padding 等没把语义改坏。
  3. drop-in 兼容：用 TorchExecutor 替换 Mock 跑 VLM/VLA 流，
     保证现有 runtime 代码不需要任何改动。
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # 没装 torch 就跳过整个文件

from aria.core.executor              import GraphMeta
from aria.models.base                import FrameworkConfig
from aria.backends.torch.executor    import TorchExecutor
from aria.backends.torch.model       import TinyLLM
from aria.runtime.vla_runtime        import VLARuntime
from aria.runtime.vlm_runtime        import VLMRuntime

logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------
# 配置工厂
# ------------------------------------------------------------------

def _tiny_text_cfg() -> FrameworkConfig:
    """模型等价性测试用：V=0 纯文本，跑得快。"""
    cfg = FrameworkConfig()
    cfg.mode                   = "vlm"
    cfg.vision.resolution      = [32, 32]
    cfg.vision.tile_size       = [32, 32]
    cfg.vision.tokens_per_tile = 0          # V = 0
    cfg.vision.feat_dim        = 32
    cfg.llm.num_layers         = 2
    cfg.llm.hidden_dim         = 64
    cfg.llm.num_heads          = 4
    cfg.llm.head_dim           = 16
    cfg.llm.vocab_size         = 256
    cfg.llm.prefill_buckets    = [64, 128]
    cfg.llm.decode_buckets     = [64, 128]
    cfg.llm.max_seq_len        = 128
    return cfg


def _vla_cfg() -> FrameworkConfig:
    cfg = FrameworkConfig()
    cfg.mode                       = "vla"
    cfg.vision.resolution          = [224, 224]
    cfg.vision.tile_size           = [224, 224]
    cfg.vision.tokens_per_tile     = 16
    cfg.vision.feat_dim            = 64
    cfg.llm.num_layers             = 2
    cfg.llm.hidden_dim             = 64
    cfg.llm.num_heads              = 4
    cfg.llm.head_dim               = 16
    cfg.llm.vocab_size             = 256
    cfg.llm.prefill_buckets        = [128, 256]
    cfg.llm.decode_buckets         = [128, 256]
    cfg.llm.max_seq_len            = 256
    cfg.action.head_type           = "flow_matching"
    cfg.action.action_dim          = 7
    cfg.action.action_horizon      = 4
    cfg.action.num_denoise_steps   = 3
    cfg.action.num_action_tokens   = 7
    cfg.action.action_token_start  = 100
    return cfg


def _vlm_cfg() -> FrameworkConfig:
    cfg = FrameworkConfig()
    cfg.mode                   = "vlm"
    cfg.vision.resolution      = [224, 224]
    cfg.vision.tile_size       = [224, 224]
    cfg.vision.tokens_per_tile = 16
    cfg.vision.feat_dim        = 64
    cfg.llm.num_layers         = 2
    cfg.llm.hidden_dim         = 64
    cfg.llm.num_heads          = 4
    cfg.llm.head_dim           = 16
    cfg.llm.vocab_size         = 256
    cfg.llm.prefill_buckets    = [256, 512]
    cfg.llm.decode_buckets     = [256, 512]
    cfg.llm.max_seq_len        = 512
    cfg.text.max_new_tokens    = 5
    cfg.text.do_sample         = False
    cfg.text.eos_token_ids     = [255]
    # 默认 BOS/EOS 是 Qwen3 词表里的大数 (151643/151645)，
    # 这里 vocab_size=256 必须压回小范围避免 embedding 越界
    cfg.bos_token_id           = 2
    cfg.eos_token_id           = 3
    cfg.pad_token_id           = 0
    return cfg


# ------------------------------------------------------------------
# 1) 模型层等价性
# ------------------------------------------------------------------

class TestTinyLLMEquivalence:

    def test_past_kv_matches_full_forward(self):
        """全程 forward 与 prefix+suffix(带 past_kv) 的 last_hidden 应该数值等价。"""
        cfg   = _tiny_text_cfg()
        torch.manual_seed(0)
        model = TinyLLM(cfg).eval()

        full_len   = 32
        prefix_len = 12
        tokens     = torch.arange(1, full_len + 1, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            # 全程 forward
            x = (model.token_emb(tokens)
                 + model.pos_emb(torch.arange(full_len, dtype=torch.long).unsqueeze(0)))
            for layer in model.layers:
                x, _, _ = layer(x)
            last_full = model.norm_out(x)[:, -1, :]

            # 先 prefix 拿 KV
            xp = (model.token_emb(tokens[:, :prefix_len])
                  + model.pos_emb(torch.arange(prefix_len, dtype=torch.long).unsqueeze(0)))
            prefix_kvs = []
            for layer in model.layers:
                xp, k, v = layer(xp)
                prefix_kvs.append((k, v))

            # 再 suffix 带 past_kv 继续
            xs = (model.token_emb(tokens[:, prefix_len:])
                  + model.pos_emb(
                      torch.arange(prefix_len, full_len, dtype=torch.long).unsqueeze(0)))
            for layer, (pk, pv) in zip(model.layers, prefix_kvs):
                xs, _, _ = layer(xs, pk, pv)
            last_suffix = model.norm_out(xs)[:, -1, :]

        diff = (last_full - last_suffix).abs().max().item()
        print(f"\n[model-level] max |Δ last_hidden| = {diff:.2e}")
        # 纯 fp32 forward，差异应该接近浮点 round-off
        assert diff < 1e-5, f"prefix/suffix split diverges from full forward: {diff}"


# ------------------------------------------------------------------
# 2) Executor 层等价性
# ------------------------------------------------------------------

class TestTorchExecutorEquivalence:

    def _make_prefill_meta(self, cfg: FrameworkConfig, bucket: int) -> GraphMeta:
        V = cfg.vision.total_vision_tokens
        return GraphMeta(
            name  = f"prefill_{bucket}",
            path  = "dummy",
            input_shapes = {
                "input_ids":      (1, bucket),
                "vision_feat":    (1, V, cfg.vision.feat_dim),
                "attention_mask": (1, bucket),
                "position_ids":   (1, bucket),
                "kv_start_pos":   (1,),
            },
            output_shapes = {
                "last_hidden": (1, cfg.llm.hidden_dim),
                "kv_out": (
                    cfg.llm.num_layers * 2, 1,
                    cfg.llm.num_heads, bucket, cfg.llm.head_dim,
                ),
            },
            # 用 fp32 输出避免 fp16 round-trip 带来的容差膨胀
            output_dtypes = {
                "last_hidden": np.float32,
                "kv_out":      np.float32,
            },
        )

    def test_past_kv_round_trip(self):
        """
        TorchExecutor.run() 三连：
          1. prefill(full)         -> last_hidden_full
          2. prefill(prefix)       -> kv_prefix
          3. prefill(suffix, past_kv=kv_prefix) -> last_hidden_split
        应该有 last_hidden_full ≈ last_hidden_split
        """
        cfg      = _tiny_text_cfg()
        executor = TorchExecutor(cfg, seed=7)
        bucket   = 64
        executor.register_graph(self._make_prefill_meta(cfg, bucket))

        full_len   = 30
        prefix_len = 11
        rng        = np.random.default_rng(13)
        all_tokens = rng.integers(1, cfg.llm.vocab_size, full_len, dtype=np.int32)
        V          = cfg.vision.total_vision_tokens   # 0 for tiny_text_cfg

        def inputs_for(token_ids, kv_start_pos, past_kv=None):
            tl     = len(token_ids)
            padded = np.zeros(bucket, dtype=np.int32); padded[:tl] = token_ids
            mask   = np.zeros(bucket, dtype=np.int32); mask[:tl + V] = 1
            pos    = np.zeros(bucket, dtype=np.int32)
            pos[:tl + V] = np.arange(kv_start_pos, kv_start_pos + tl + V)
            d = {
                "input_ids":      padded[None, :],
                "vision_feat":    np.zeros((1, V, cfg.vision.feat_dim), dtype=np.float16),
                "attention_mask": mask[None, :],
                "position_ids":   pos[None, :],
                "kv_start_pos":   np.array([kv_start_pos], dtype=np.int32),
            }
            if past_kv is not None:
                d["past_kv"]     = past_kv
                d["past_kv_len"] = np.array([past_kv.shape[-2]], dtype=np.int32)
            return d

        out_full = executor.run(f"prefill_{bucket}",
                                inputs_for(all_tokens, kv_start_pos=0))
        last_full = out_full["last_hidden"]

        out_pref = executor.run(f"prefill_{bucket}",
                                inputs_for(all_tokens[:prefix_len], kv_start_pos=0))
        kv_pref  = out_pref["kv_out"]
        # [L*2, B, NH, bucket, D] → [L, 2, B, NH, prefix_len, D]
        kv5d = kv_pref.reshape(cfg.llm.num_layers, 2, 1,
                                cfg.llm.num_heads, bucket, cfg.llm.head_dim)
        past_kv = kv5d[:, :, :, :, :prefix_len, :].astype(np.float32)

        out_suf = executor.run(f"prefill_{bucket}",
                               inputs_for(all_tokens[prefix_len:],
                                          kv_start_pos=prefix_len,
                                          past_kv=past_kv))
        last_suf = out_suf["last_hidden"]

        diff = np.abs(last_full - last_suf).max()
        print(f"\n[executor-level] max |Δ last_hidden| = {diff:.2e}")
        assert diff < 1e-4, f"executor full vs split prefill diverges: {diff}"

    def test_past_kv_actually_used(self):
        """Sanity check：不传 past_kv 时 suffix 的输出应该跟"传了 past_kv"明显不同，
        否则说明 past_kv 路径根本没生效。"""
        cfg      = _tiny_text_cfg()
        executor = TorchExecutor(cfg, seed=7)
        bucket   = 64
        executor.register_graph(self._make_prefill_meta(cfg, bucket))

        rng    = np.random.default_rng(13)
        full   = rng.integers(1, cfg.llm.vocab_size, 30, dtype=np.int32)
        prefix_len = 11
        V      = cfg.vision.total_vision_tokens

        def inputs_for(token_ids, kv_start_pos, past_kv=None):
            tl     = len(token_ids)
            padded = np.zeros(bucket, dtype=np.int32); padded[:tl] = token_ids
            mask   = np.zeros(bucket, dtype=np.int32); mask[:tl + V] = 1
            pos    = np.zeros(bucket, dtype=np.int32)
            pos[:tl + V] = np.arange(kv_start_pos, kv_start_pos + tl + V)
            d = {
                "input_ids":      padded[None, :],
                "vision_feat":    np.zeros((1, V, cfg.vision.feat_dim), dtype=np.float16),
                "attention_mask": mask[None, :],
                "position_ids":   pos[None, :],
                "kv_start_pos":   np.array([kv_start_pos], dtype=np.int32),
            }
            if past_kv is not None:
                d["past_kv"]     = past_kv
                d["past_kv_len"] = np.array([past_kv.shape[-2]], dtype=np.int32)
            return d

        out_pref = executor.run(f"prefill_{bucket}",
                                inputs_for(full[:prefix_len], 0))
        kv5d = out_pref["kv_out"].reshape(
            cfg.llm.num_layers, 2, 1,
            cfg.llm.num_heads, bucket, cfg.llm.head_dim,
        )
        past_kv = kv5d[:, :, :, :, :prefix_len, :].astype(np.float32)

        # 带 past_kv
        out_with = executor.run(f"prefill_{bucket}",
                                inputs_for(full[prefix_len:],
                                           kv_start_pos=prefix_len,
                                           past_kv=past_kv))
        # 不带 past_kv（位置还在 prefix_len，但 model 看不到历史）
        out_without = executor.run(f"prefill_{bucket}",
                                   inputs_for(full[prefix_len:],
                                              kv_start_pos=prefix_len,
                                              past_kv=None))

        diff = np.abs(out_with["last_hidden"] - out_without["last_hidden"]).max()
        print(f"\n[sanity] |with_past − without_past| = {diff:.2e}")
        assert diff > 1e-3, "past_kv 路径似乎根本没影响输出，可能 wiring 出错了"


# ------------------------------------------------------------------
# 3) drop-in 兼容性
# ------------------------------------------------------------------

class TestTorchAsDropIn:

    def test_vla_flow(self):
        cfg      = _vla_cfg()
        executor = TorchExecutor(cfg, seed=42)
        runtime  = VLARuntime.from_config(cfg, executor)
        image    = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
        action   = runtime.infer(image, "pick up the cube")
        assert action.shape == (cfg.action.action_horizon, cfg.action.action_dim)
        assert action.dtype == np.float32

    def test_vlm_text_only(self):
        cfg      = _vlm_cfg()
        executor = TorchExecutor(cfg, seed=42)
        runtime  = VLMRuntime.from_config(cfg, executor)
        r        = runtime.chat([{"role": "user", "content": "hello"}])
        assert isinstance(r, str)

    def test_vlm_with_image(self):
        cfg      = _vlm_cfg()
        executor = TorchExecutor(cfg, seed=42)
        runtime  = VLMRuntime.from_config(cfg, executor)
        image    = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
        r        = runtime.chat([{"role": "user", "content": [
            {"type": "image", "data": image},
            {"type": "text",  "data": "what's there"},
        ]}])
        assert isinstance(r, str)
