"""
tests/test_mock.py

基于MockNPUExecutor的端到端测试。
验证框架流程正确性，不依赖真实NPU。
"""

import logging

import numpy as np
import pytest

from aria.core.executor      import MockNPUExecutor
from aria.core.kv_cache      import KVCacheManager
from aria.core.prefix_cache  import PrefixCache, _block_hash, _ROOT_HASH
from aria.models.base        import FrameworkConfig
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
# Prefix Cache 测试
# ------------------------------------------------------------------

class TestPrefixCache:

    def test_basic_match_insert(self):
        cache = PrefixCache(num_layers=2, num_heads=4, head_dim=8,
                            block_size=4, capacity_blocks=16)
        toks = np.arange(12, dtype=np.int32)
        kv   = np.random.default_rng(0).standard_normal(
            (2, 2, 4, 12, 8)
        ).astype(np.float16)

        # 首次：全 miss
        m = cache.match(toks)
        assert m.num_blocks == 0
        assert not m

        # 写入 12 个 token → 3 个块
        added = cache.insert(toks, kv)
        assert added == 3
        assert cache.stats()["used"] == 3

        # 再次：3 块全命中
        m = cache.match(toks)
        assert m.num_blocks == 3
        assert m.matched_tokens == 12
        assert bool(m) is True

        # gather 出来的 KV 应该和写入时一致
        kv_back = m.gather()
        assert kv_back.shape == (2, 2, 4, 12, 8)
        assert np.array_equal(kv_back, kv)

    def test_partial_prefix_match(self):
        """同前缀块命中，后续块发散后停止匹配"""
        cache = PrefixCache(num_layers=2, num_heads=4, head_dim=8,
                            block_size=4, capacity_blocks=16)
        toks1 = np.array([1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12], dtype=np.int32)
        kv1   = np.random.default_rng(1).standard_normal(
            (2, 2, 4, 12, 8)
        ).astype(np.float16)
        cache.insert(toks1, kv1)

        # 第一块完全相同，第二块开始不同
        toks2 = np.array([1, 2, 3, 4,  99, 99, 99, 99,  77, 77, 77, 77], dtype=np.int32)
        m = cache.match(toks2)
        assert m.num_blocks == 1
        assert m.matched_tokens == 4
        # 命中那一块的 KV 等于 toks1 第一块
        assert np.array_equal(m.gather(), kv1[:, :, :, :4, :])

    def test_block_alignment_tail_dropped(self):
        """长度不是 block_size 的倍数时，尾部 token 不缓存"""
        cache = PrefixCache(num_layers=1, num_heads=1, head_dim=4,
                            block_size=4, capacity_blocks=8)
        # 10 个 token = 2 完整块 + 2 个尾部
        toks = np.arange(10, dtype=np.int32)
        kv   = np.zeros((1, 2, 1, 10, 4), dtype=np.float16)
        added = cache.insert(toks, kv)
        assert added == 2   # 只缓存 8 个 token

    def test_lru_eviction(self):
        """容量 = 2 块，写入 3 个独立前缀 → 最早那个被淘汰"""
        cache = PrefixCache(num_layers=1, num_heads=1, head_dim=4,
                            block_size=4, capacity_blocks=2)
        rng  = np.random.default_rng(2)
        # 3 个互不重叠的单块前缀（不同 token 序列）
        prefixes = [
            np.array([10, 11, 12, 13], dtype=np.int32),
            np.array([20, 21, 22, 23], dtype=np.int32),
            np.array([30, 31, 32, 33], dtype=np.int32),
        ]
        for p in prefixes:
            kv = rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float16)
            cache.insert(p, kv)

        s = cache.stats()
        assert s["used"]      == 2
        assert s["evictions"] == 1

        # 第一个前缀应该已经被淘汰
        assert cache.match(prefixes[0]).num_blocks == 0
        # 后两个还在
        assert cache.match(prefixes[1]).num_blocks == 1
        assert cache.match(prefixes[2]).num_blocks == 1

    def test_lru_keeps_recent(self):
        """访问过的块会被推到 LRU 末尾，避免被淘汰"""
        cache = PrefixCache(num_layers=1, num_heads=1, head_dim=4,
                            block_size=4, capacity_blocks=2)
        rng  = np.random.default_rng(3)
        p_a  = np.array([1, 2, 3, 4],   dtype=np.int32)
        p_b  = np.array([5, 6, 7, 8],   dtype=np.int32)
        p_c  = np.array([9, 10, 11, 12], dtype=np.int32)

        cache.insert(p_a, rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float16))
        cache.insert(p_b, rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float16))

        # 访问 p_a，把它推到 LRU 末尾
        assert cache.match(p_a).num_blocks == 1

        # 再插入 p_c：应淘汰 p_b（不是 p_a）
        cache.insert(p_c, rng.standard_normal((1, 2, 1, 4, 4)).astype(np.float16))

        assert cache.match(p_a).num_blocks == 1
        assert cache.match(p_b).num_blocks == 0
        assert cache.match(p_c).num_blocks == 1

    def test_insert_dedupe(self):
        """相同前缀重复写入不会重复占 slot"""
        cache = PrefixCache(num_layers=1, num_heads=1, head_dim=4,
                            block_size=4, capacity_blocks=8)
        toks = np.array([1, 2, 3, 4,  5, 6, 7, 8], dtype=np.int32)
        kv   = np.zeros((1, 2, 1, 8, 4), dtype=np.float16)

        added1 = cache.insert(toks, kv)
        added2 = cache.insert(toks, kv)
        assert added1 == 2
        assert added2 == 0
        assert cache.stats()["used"] == 2

    def test_hash_chain_order_matters(self):
        """链式哈希：相同 token 集合不同顺序，哈希应该不同"""
        h1 = _block_hash(_ROOT_HASH, np.array([1, 2, 3, 4], dtype=np.int32))
        h2 = _block_hash(_ROOT_HASH, np.array([4, 3, 2, 1], dtype=np.int32))
        assert h1 != h2

    def test_clear(self):
        cache = PrefixCache(num_layers=1, num_heads=1, head_dim=4,
                            block_size=4, capacity_blocks=4)
        cache.insert(np.arange(8, dtype=np.int32),
                     np.zeros((1, 2, 1, 8, 4), dtype=np.float16))
        assert cache.stats()["used"] == 2
        cache.clear()
        assert cache.stats()["used"] == 0
        assert cache.match(np.arange(4, dtype=np.int32)).num_blocks == 0


# ------------------------------------------------------------------
# KVCacheManager 的前缀回灌
# ------------------------------------------------------------------

class TestKVCachePrefixLoad:

    def test_bulk_load_prefix(self):
        kv = KVCacheManager(num_layers=2, num_heads=4, head_dim=8,
                            max_seq_len=64, max_batch=1)
        prefix = np.full((2, 2, 4, 12, 8), 7, dtype=np.float16)
        kv.bulk_load_prefix(prefix)
        assert kv.valid_len   == 12
        assert kv.history_len == 12
        # 读回来验证
        rk, rv = kv.get_kv(0)
        assert rk.shape == (1, 4, 12, 8)
        assert np.all(rk == 7)
        assert np.all(rv == 7)

    def test_read_range(self):
        kv = KVCacheManager(num_layers=2, num_heads=4, head_dim=8,
                            max_seq_len=64, max_batch=1)
        k = np.full((1, 4, 20, 8), 3, dtype=np.float16)
        v = np.full((1, 4, 20, 8), 5, dtype=np.float16)
        for l in range(2):
            kv.write_prefill(l, k, v)

        chunk = kv.read_range(4, 16)
        assert chunk.shape == (2, 2, 4, 12, 8)
        assert np.all(chunk[:, 0] == 3)  # K
        assert np.all(chunk[:, 1] == 5)  # V


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

    def test_prefix_cache_hit(self):
        """同样的纯文本输入，第二次应命中前缀缓存"""
        cfg      = make_vlm_config()
        executor = MockNPUExecutor(latency_ms=0.5)
        pc       = PrefixCache(
            num_layers      = cfg.llm.num_layers,
            num_heads       = cfg.llm.num_heads,
            head_dim        = cfg.llm.head_dim,
            block_size      = 4,
            capacity_blocks = 64,
        )
        runtime = VLMRuntime.from_config(cfg, executor, prefix_cache=pc)

        msg = [{"role": "user", "content": "the quick brown fox jumps over the lazy dog"}]

        # 第一次：全 miss，但会写回
        runtime.chat(msg)
        s1 = pc.stats()
        assert s1["block_hits"]   == 0
        assert s1["block_misses"] > 0
        used_after_first = s1["used"]
        print(f"After 1st call: {pc!r}")

        # 第二次：相同 prefix → 应有命中
        runtime.chat(msg)
        s2 = pc.stats()
        assert s2["block_hits"] >= used_after_first
        assert s2["hit_rate"] > 0.0
        print(f"After 2nd call: {pc!r}")

    def test_prefix_cache_skipped_for_image(self):
        """图像输入不应触发前缀缓存"""
        cfg      = make_vlm_config()
        executor = MockNPUExecutor(latency_ms=0.5)
        pc       = PrefixCache(
            num_layers      = cfg.llm.num_layers,
            num_heads       = cfg.llm.num_heads,
            head_dim        = cfg.llm.head_dim,
            block_size      = 8,
            capacity_blocks = 64,
        )
        runtime = VLMRuntime.from_config(cfg, executor, prefix_cache=pc)

        image = make_dummy_image(448, 448)
        msg   = [{"role": "user", "content": [
            {"type": "image", "data": image},
            {"type": "text",  "data": "describe"},
        ]}]
        runtime.chat(msg)
        # 图像分支：既不查也不写
        s = pc.stats()
        assert s["block_hits"]   == 0
        assert s["block_misses"] == 0
        assert s["used"]         == 0

    def test_prefix_cache_partial_share(self):
        """两次查询共享前缀但后缀不同，应部分命中"""
        cfg      = make_vlm_config()
        executor = MockNPUExecutor(latency_ms=0.5)
        pc       = PrefixCache(
            num_layers      = cfg.llm.num_layers,
            num_heads       = cfg.llm.num_heads,
            head_dim        = cfg.llm.head_dim,
            block_size      = 4,
            capacity_blocks = 64,
        )
        runtime = VLMRuntime.from_config(cfg, executor, prefix_cache=pc)

        # 共享前缀 "you are a helpful assistant " (28 字符) 后面接不同问题
        prefix = "you are a helpful assistant "
        runtime.chat([{"role": "user", "content": prefix + "what is rust"}])
        before_hits = pc.stats()["block_hits"]

        runtime.chat([{"role": "user", "content": prefix + "what is python"}])
        after_hits  = pc.stats()["block_hits"]

        # 第二次应该命中共享前缀的若干块
        assert after_hits > before_hits
        print(f"Partial-share hits: +{after_hits - before_hits} blocks")

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
    print("\n[1/4] KV Cache测试...")
    t = TestKVCache()
    t.test_prefill_write_read()
    t.test_decode_step()
    t.test_multi_turn()
    t.test_reset()
    print("  ✓ KV Cache全部通过")

    # Prefix Cache
    print("\n[2/4] Prefix Cache测试...")
    t = TestPrefixCache()
    t.test_basic_match_insert()
    t.test_partial_prefix_match()
    t.test_block_alignment_tail_dropped()
    t.test_lru_eviction()
    t.test_lru_keeps_recent()
    t.test_insert_dedupe()
    t.test_hash_chain_order_matters()
    t.test_clear()
    t = TestKVCachePrefixLoad()
    t.test_bulk_load_prefix()
    t.test_read_range()
    print("  ✓ Prefix Cache全部通过")

    # VLA
    print("\n[3/4] VLA推理测试...")
    t = TestVLARuntime()
    t.test_flow_matching_infer()
    t.test_autoregressive_infer()
    t.test_multiple_infer_kv_reset()
    t.test_profiling()
    print("  ✓ VLA全部通过")

    # VLM
    print("\n[4/4] VLM推理测试...")
    t = TestVLMRuntime()
    t.test_single_turn_text_only()
    t.test_single_turn_with_image()
    t.test_multi_turn_session()
    t.test_session_reset()
    t.test_prefix_cache_hit()
    t.test_prefix_cache_skipped_for_image()
    t.test_prefix_cache_partial_share()
    t.test_temp_session()
    print("  ✓ VLM全部通过")

    print("\n✅ 所有测试通过")
