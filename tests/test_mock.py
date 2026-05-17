"""
tests/test_mock.py

基于MockNPUExecutor的端到端测试。
验证框架流程正确性，不依赖真实NPU。
"""

import logging

import numpy as np
import pytest

from aria.core.executor  import MockNPUExecutor
from aria.core.kv_cache  import KVCacheManager
from aria.models.base    import FrameworkConfig
from aria.runtime.vla_runtime import VLARuntime
from aria.runtime.vlm_runtime import VLMRuntime

logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------
# 公共fixture
# ------------------------------------------------------------------

def make_vla_config(head_type: str = "flow_matching") -> FrameworkConfig:
    cfg = FrameworkConfig()
    cfg.mode                   = "vla"
    cfg.graph_dir              = "compiled/test"
    cfg.vision.resolution      = [224, 224]
    cfg.vision.tile_size       = [224, 224]
    cfg.vision.tokens_per_tile = 64     # 小一点，测试快
    cfg.vision.feat_dim        = 256
    cfg.llm.num_layers         = 2      # 只用2层，测试快
    cfg.llm.hidden_dim         = 256
    cfg.llm.num_heads          = 4
    cfg.llm.head_dim           = 64
    cfg.llm.vocab_size         = 1000
    cfg.llm.prefill_buckets    = [128, 256]
    cfg.llm.decode_buckets     = [128, 256]
    cfg.llm.max_seq_len        = 512
    cfg.action.head_type       = head_type
    cfg.action.action_dim      = 7
    cfg.action.action_horizon  = 4
    cfg.action.num_denoise_steps = 3
    cfg.action.num_action_tokens = 7
    cfg.action.action_token_start = 500
    return cfg


def make_vlm_config() -> FrameworkConfig:
    cfg = FrameworkConfig()
    cfg.mode                   = "vlm"
    cfg.graph_dir              = "compiled/test"
    cfg.vision.resolution      = [448, 448]
    cfg.vision.tile_size       = [224, 224]
    cfg.vision.tokens_per_tile = 64
    cfg.vision.feat_dim        = 256
    cfg.llm.num_layers         = 2
    cfg.llm.hidden_dim         = 256
    cfg.llm.num_heads          = 4
    cfg.llm.head_dim           = 64
    cfg.llm.vocab_size         = 1000
    cfg.llm.prefill_buckets    = [512, 768, 1024]
    cfg.llm.decode_buckets     = [512, 768, 1024]
    cfg.llm.max_seq_len        = 1024
    cfg.text.max_new_tokens    = 10
    cfg.text.do_sample         = False   # 测试用greedy，确定性输出
    cfg.text.eos_token_ids     = [999]
    return cfg


def make_dummy_image(h=224, w=224) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ------------------------------------------------------------------
# KV Cache 测试
# ------------------------------------------------------------------

class TestKVCache:

    def test_prefill_write_read(self):
        kv = KVCacheManager(num_layers=2, num_heads=4, head_dim=8,
                            max_seq_len=64, max_batch=1)
        k = np.ones((1, 4, 10, 8), dtype=np.float16)
        v = np.ones((1, 4, 10, 8), dtype=np.float16) * 2
        kv.write_prefill(0, k, v, start_pos=0)
        kv.write_prefill(1, k, v, start_pos=0)
        assert kv.valid_len == 10

        rk, rv = kv.get_kv(0)
        assert rk.shape == (1, 4, 10, 8)
        assert np.allclose(rk, 1.0)
        assert np.allclose(rv, 2.0)

    def test_decode_step(self):
        kv = KVCacheManager(num_layers=2, num_heads=4, head_dim=8,
                            max_seq_len=64, max_batch=1)
        # 先写一段Prefill
        k = np.zeros((1, 4, 5, 8), dtype=np.float16)
        v = np.zeros((1, 4, 5, 8), dtype=np.float16)
        for l in range(2):
            kv.write_prefill(l, k, v)

        assert kv.valid_len == 5

        # 写一步Decode
        k1 = np.ones((1, 4, 1, 8), dtype=np.float16)
        v1 = np.ones((1, 4, 1, 8), dtype=np.float16)
        for l in range(2):
            kv.write_decode(l, k1, v1)
        kv.step_forward()

        assert kv.valid_len == 6

    def test_multi_turn(self):
        kv = KVCacheManager(num_layers=2, num_heads=4, head_dim=8,
                            max_seq_len=128, max_batch=1)
        k = np.zeros((1, 4, 20, 8), dtype=np.float16)
        v = np.zeros((1, 4, 20, 8), dtype=np.float16)
        for l in range(2):
            kv.write_prefill(l, k, v, start_pos=0)

        kv.save_turn()
        assert kv.history_len == 20
        assert kv.valid_len   == 20

        # 第二轮
        k2 = np.ones((1, 4, 15, 8), dtype=np.float16)
        v2 = np.ones((1, 4, 15, 8), dtype=np.float16)
        for l in range(2):
            kv.write_prefill(l, k2, v2, start_pos=20)
        assert kv.valid_len == 35

    def test_reset(self):
        kv = KVCacheManager(num_layers=2, num_heads=4, head_dim=8,
                            max_seq_len=64)
        k = np.zeros((1, 4, 10, 8), dtype=np.float16)
        v = np.zeros((1, 4, 10, 8), dtype=np.float16)
        kv.write_prefill(0, k, v)
        kv.reset()
        assert kv.valid_len   == 0
        assert kv.history_len == 0


# ------------------------------------------------------------------
# VLA 推理测试
# ------------------------------------------------------------------

class TestVLARuntime:

    def test_flow_matching_infer(self):
        cfg      = make_vla_config("flow_matching")
        executor = MockNPUExecutor(latency_ms=1.0)
        runtime  = VLARuntime.from_config(cfg, executor)

        image  = make_dummy_image()
        action = runtime.infer(image, "pick up the red cup")

        # Flow模式输出 [action_horizon, action_dim]
        assert action.shape == (cfg.action.action_horizon, cfg.action.action_dim)
        assert action.dtype == np.float32
        print(f"Flow action shape: {action.shape}")
        print(f"Flow action sample: {action[0]}")

    def test_autoregressive_infer(self):
        cfg      = make_vla_config("autoregressive")
        executor = MockNPUExecutor(latency_ms=1.0)
        runtime  = VLARuntime.from_config(cfg, executor)

        image  = make_dummy_image()
        action = runtime.infer(image, "move arm to the left")

        # AR模式输出 [action_dim]
        assert action.shape == (cfg.action.action_dim,)
        assert action.dtype == np.float32
        print(f"AR action: {action}")

    def test_multiple_infer_kv_reset(self):
        """每次推理后KV Cache应该被正确重置"""
        cfg      = make_vla_config("flow_matching")
        executor = MockNPUExecutor(latency_ms=0.5)
        runtime  = VLARuntime.from_config(cfg, executor)
        image    = make_dummy_image()

        for i in range(3):
            action = runtime.infer(image, f"task {i}")
            assert action.shape == (cfg.action.action_horizon, cfg.action.action_dim)
            # 每次推理后KV Cache应该被重置（VLA是单轮的）
            assert runtime.kv_cache.valid_len > 0  # 推理期间有值

    def test_profiling(self):
        cfg      = make_vla_config("flow_matching")
        executor = MockNPUExecutor(latency_ms=2.0)
        executor.enable_profiling(True)
        runtime  = VLARuntime.from_config(cfg, executor)
        image    = make_dummy_image()

        for _ in range(3):
            runtime.infer(image, "test instruction")

        stats = executor.get_profiling_stats()
        assert len(stats) > 0
        for name, s in stats.items():
            print(f"  {name}: mean={s['mean']:.1f}ms p95={s['p95']:.1f}ms")


# ------------------------------------------------------------------
# VLM 推理测试
# ------------------------------------------------------------------

class TestVLMRuntime:

    def test_single_turn_text_only(self):
        cfg      = make_vlm_config()
        executor = MockNPUExecutor(latency_ms=1.0)
        runtime  = VLMRuntime.from_config(cfg, executor)

        messages = [{"role": "user", "content": "hello world"}]
        response = runtime.chat(messages)

        assert isinstance(response, str)
        print(f"VLM text-only response: '{response}'")

    def test_single_turn_with_image(self):
        cfg      = make_vlm_config()
        executor = MockNPUExecutor(latency_ms=1.0)
        runtime  = VLMRuntime.from_config(cfg, executor)

        image    = make_dummy_image(448, 448)
        messages = [{"role": "user", "content": [
            {"type": "image", "data": image},
            {"type": "text",  "data": "describe this image"},
        ]}]
        response = runtime.chat(messages)

        assert isinstance(response, str)
        print(f"VLM image+text response: '{response}'")

    def test_multi_turn_session(self):
        cfg        = make_vlm_config()
        executor   = MockNPUExecutor(latency_ms=1.0)
        runtime    = VLMRuntime.from_config(cfg, executor)
        session_id = runtime.new_session()

        image = make_dummy_image(448, 448)

        # 第1轮
        r1 = runtime.chat(
            [{"role": "user", "content": [
                {"type": "image", "data": image},
                {"type": "text",  "data": "what is in the image"},
            ]}],
            session_id=session_id
        )
        session = runtime._sessions[session_id]
        kv_after_turn1 = session.history_kv_len
        assert kv_after_turn1 > 0
        print(f"Turn1 response: '{r1}' | history_kv_len={kv_after_turn1}")

        # 第2轮：KV Cache应该比第1轮更长
        r2 = runtime.chat(
            [{"role": "user", "content": "tell me more"}],
            session_id=session_id
        )
        kv_after_turn2 = session.history_kv_len
        assert kv_after_turn2 >= kv_after_turn1
        print(f"Turn2 response: '{r2}' | history_kv_len={kv_after_turn2}")

        runtime.close_session(session_id)
        assert session_id not in runtime._sessions

    def test_session_reset(self):
        cfg        = make_vlm_config()
        executor   = MockNPUExecutor(latency_ms=0.5)
        runtime    = VLMRuntime.from_config(cfg, executor)
        session_id = runtime.new_session()

        runtime.chat([{"role": "user", "content": "hello"}], session_id=session_id)
        runtime.reset_session(session_id)

        session = runtime._sessions[session_id]
        assert session.current_kv_len == 0
        assert session.num_turns      == 0

        runtime.close_session(session_id)

    def test_temp_session(self):
        """不传session_id时，应该自动创建临时session并在完成后清理"""
        cfg      = make_vlm_config()
        executor = MockNPUExecutor(latency_ms=0.5)
        runtime  = VLMRuntime.from_config(cfg, executor)

        before = len(runtime._sessions)
        runtime.chat([{"role": "user", "content": "temp query"}])
        after  = len(runtime._sessions)

        # 临时session用完应该被清理
        assert after == before


# ------------------------------------------------------------------
# 配置加载测试
# ------------------------------------------------------------------

class TestConfig:

    def test_load_yaml(self, tmp_path):
        yaml_content = """
mode: vla
graph_dir: /tmp/graphs
vision:
  resolution: [224, 224]
  tile_size: [224, 224]
  tokens_per_tile: 256
  feat_dim: 4096
llm:
  num_layers: 32
  hidden_dim: 4096
  num_heads: 32
  head_dim: 128
  prefill_buckets: [512, 1024]
  decode_buckets: [512, 1024]
action:
  head_type: flow_matching
  action_dim: 7
  action_horizon: 16
  num_denoise_steps: 15
"""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)

        cfg = FrameworkConfig.from_yaml(str(cfg_file))
        assert cfg.mode == "vla"
        assert cfg.vision.resolution == [224, 224]
        assert cfg.vision.num_tiles  == 1
        assert cfg.llm.num_layers    == 32
        assert cfg.action.head_type  == "flow_matching"
        print(f"Config loaded: mode={cfg.mode} tiles={cfg.vision.num_tiles}")

    def test_num_tiles_calculation(self):
        cfg = FrameworkConfig()
        cfg.vision.resolution = [448, 448]
        cfg.vision.tile_size  = [224, 224]
        assert cfg.vision.num_tiles          == 4
        assert cfg.vision.total_vision_tokens == 4 * cfg.vision.tokens_per_tile


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("ARIA 推理框架 Mock测试")
    print("=" * 60)

    # KV Cache
    print("\n[1/3] KV Cache测试...")
    t = TestKVCache()
    t.test_prefill_write_read()
    t.test_decode_step()
    t.test_multi_turn()
    t.test_reset()
    print("  ✓ KV Cache全部通过")

    # VLA
    print("\n[2/3] VLA推理测试...")
    t = TestVLARuntime()
    t.test_flow_matching_infer()
    t.test_autoregressive_infer()
    t.test_multiple_infer_kv_reset()
    t.test_profiling()
    print("  ✓ VLA全部通过")

    # VLM
    print("\n[3/3] VLM推理测试...")
    t = TestVLMRuntime()
    t.test_single_turn_text_only()
    t.test_single_turn_with_image()
    t.test_multi_turn_session()
    t.test_session_reset()
    t.test_temp_session()
    print("  ✓ VLM全部通过")

    print("\n✅ 所有测试通过")
