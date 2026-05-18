"""
core/prefix_cache.py

块级前缀缓存（Block-based Prefix Cache）。

- 固定块大小 B（默认 16 tokens）
- 哈希链：block_hash = blake2b(prev_block_hash || tokens_bytes)
- 预分配 KV 池，LRU 淘汰
- 操作粒度：块对齐，尾部不足 B 的部分不缓存
- 端侧单 batch 假设：池里只存 batch=1 的 KV

典型用法：
    cache = PrefixCache(num_layers, num_heads, head_dim,
                        block_size=16, capacity_blocks=256)

    # Prefill 之前查
    m = cache.match(token_ids)
    if m.num_blocks > 0:
        kv_cache.bulk_load_prefix(m.gather())   # 工作区前 M*B 个位置被覆盖
        token_ids = token_ids[m.matched_tokens:] # 后续 prefill 只处理 suffix

    # Prefill+Decode 之后写回（block 对齐部分会去重）
    full_kv = kv_cache.read_range(0, len(full_token_ids))
    cache.insert(full_token_ids, full_kv)
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 起始链哈希：空前缀
_ROOT_HASH = b"\x00" * 16


def _block_hash(prev: bytes, tokens: np.ndarray) -> bytes:
    """链式哈希：blake2b(prev || tokens.tobytes())"""
    h = hashlib.blake2b(digest_size=16)
    h.update(prev)
    h.update(np.ascontiguousarray(tokens, dtype=np.int32).tobytes())
    return h.digest()


class PrefixMatch:
    """match() 返回值，描述命中了哪些块。"""

    def __init__(self, cache: "PrefixCache", block_ids: List[int]):
        self._cache    = cache
        self.block_ids = block_ids

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def matched_tokens(self) -> int:
        return self.num_blocks * self._cache.block_size

    def gather(self) -> np.ndarray:
        """把命中的块拼成 [num_layers, 2, num_heads, M*B, head_dim]。"""
        return self._cache.gather(self.block_ids)

    def __bool__(self) -> bool:
        return self.num_blocks > 0

    def __repr__(self) -> str:
        return f"PrefixMatch(blocks={self.num_blocks}, tokens={self.matched_tokens})"


class PrefixCache:
    """
    块级前缀 KV 缓存。

    内存布局：
      _pool: [capacity_blocks, num_layers, 2, num_heads, block_size, head_dim]
             (单 batch，端侧场景够用)
      _hash_to_block: hash → slot_id
      _block_to_hash: slot_id → hash（淘汰时反查）
      _lru:           OrderedDict[hash], 末尾=最近访问
      _free_slots:    空闲 slot 栈
    """

    def __init__(self,
                 num_layers:       int,
                 num_heads:        int,
                 head_dim:         int,
                 block_size:       int      = 16,
                 capacity_blocks:  int      = 256,
                 dtype:            np.dtype = np.float16):

        assert block_size > 0
        assert capacity_blocks > 0

        self.num_layers      = num_layers
        self.num_heads       = num_heads
        self.head_dim        = head_dim
        self.block_size      = block_size
        self.capacity_blocks = capacity_blocks
        self.dtype           = dtype

        self._pool = np.zeros(
            (capacity_blocks, num_layers, 2, num_heads, block_size, head_dim),
            dtype=dtype,
        )
        self._hash_to_block: Dict[bytes, int]           = {}
        self._block_to_hash: List[Optional[bytes]]      = [None] * capacity_blocks
        self._lru:           "OrderedDict[bytes, None]" = OrderedDict()
        self._free_slots:    List[int]                  = list(range(capacity_blocks))

        # 统计
        self._lookups   = 0
        self._hits      = 0   # 命中的 block 数
        self._misses    = 0   # 写入新 block 的次数
        self._evictions = 0

        bytes_per_block = (
            num_layers * 2 * num_heads * block_size * head_dim * self._pool.itemsize
        )
        logger.info(
            f"[ARIA/PrefixCache] 初始化 block_size={block_size} "
            f"capacity={capacity_blocks} blocks "
            f"每块={bytes_per_block/1024:.1f}KB "
            f"总池={self._pool.nbytes/1024**2:.2f}MB"
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def match(self, token_ids: np.ndarray) -> PrefixMatch:
        """
        贪心地按 block_size 对齐查找最长前缀。
        token_ids: 1-D 整型数组
        """
        self._lookups += 1
        n         = int(len(token_ids))
        block_ids: List[int] = []
        prev_hash = _ROOT_HASH

        num_full = n // self.block_size
        for i in range(num_full):
            start = i * self.block_size
            chunk = token_ids[start:start + self.block_size]
            h     = _block_hash(prev_hash, chunk)
            slot  = self._hash_to_block.get(h)
            if slot is None:
                break
            block_ids.append(slot)
            self._lru.move_to_end(h)
            self._hits += 1
            prev_hash   = h

        return PrefixMatch(cache=self, block_ids=block_ids)

    def gather(self, block_ids: List[int]) -> np.ndarray:
        """
        把 block_ids 对应的 KV 拼成 [num_layers, 2, num_heads, M*B, head_dim]。
        """
        M = len(block_ids)
        if M == 0:
            return np.zeros(
                (self.num_layers, 2, self.num_heads, 0, self.head_dim),
                dtype=self.dtype,
            )
        # _pool[block_ids] : [M, L, 2, H, B, D]
        sub = self._pool[block_ids]
        # 目标: [L, 2, H, M*B, D]，M 与 B 维相邻便于 reshape
        L, _, H, B, D = sub.shape[1], sub.shape[2], sub.shape[3], sub.shape[4], sub.shape[5]
        sub = sub.transpose(1, 2, 3, 0, 4, 5)        # [L, 2, H, M, B, D]
        return np.ascontiguousarray(sub).reshape(L, 2, H, M * B, D)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def insert(self, token_ids: np.ndarray, kv_data: np.ndarray) -> int:
        """
        将 token_ids 对应的 KV 按 block_size 分块插入缓存。
        已存在的块只更新 LRU 顺序，不重复写入。

        token_ids: 1-D 整型数组，长度 N
        kv_data:   [num_layers, 2, num_heads, N, head_dim]
                   尾部 (N % block_size) 个 token 的 KV 不会被缓存
        返回:      新分配 slot 的块数
        """
        n  = int(len(token_ids))
        L, two, H, kv_n, D = kv_data.shape
        assert two == 2,                  f"kv_data 第 2 维必须是 2 (K/V), got {two}"
        assert L  == self.num_layers,     f"num_layers 不匹配: {L} vs {self.num_layers}"
        assert H  == self.num_heads,      f"num_heads 不匹配: {H} vs {self.num_heads}"
        assert D  == self.head_dim,       f"head_dim 不匹配: {D} vs {self.head_dim}"
        assert kv_n == n,                 f"token_ids 长度 {n} 与 kv 序列维 {kv_n} 不一致"

        prev_hash = _ROOT_HASH
        num_full  = n // self.block_size
        inserted  = 0

        for i in range(num_full):
            start = i * self.block_size
            end   = start + self.block_size
            chunk = token_ids[start:end]
            h     = _block_hash(prev_hash, chunk)

            if h in self._hash_to_block:
                self._lru.move_to_end(h)
                prev_hash = h
                continue

            slot = self._allocate_slot()
            self._pool[slot] = kv_data[:, :, :, start:end, :]
            self._hash_to_block[h]    = slot
            self._block_to_hash[slot] = h
            self._lru[h]              = None
            inserted     += 1
            self._misses += 1
            prev_hash     = h

        return inserted

    def _allocate_slot(self) -> int:
        """从空闲列表拿；池满则淘汰 LRU 头部并复用。"""
        if self._free_slots:
            return self._free_slots.pop()
        oldest_hash, _ = self._lru.popitem(last=False)
        slot = self._hash_to_block.pop(oldest_hash)
        self._block_to_hash[slot] = None
        self._evictions += 1
        return slot

    # ------------------------------------------------------------------
    # 维护 / 观测
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """清空所有块（重置 LRU/索引，池内存不释放）。"""
        self._hash_to_block.clear()
        self._block_to_hash = [None] * self.capacity_blocks
        self._lru.clear()
        self._free_slots = list(range(self.capacity_blocks))
        logger.info("[ARIA/PrefixCache] 已清空")

    def stats(self) -> Dict[str, float]:
        used = self.capacity_blocks - len(self._free_slots)
        attempted = self._hits + self._misses
        hit_rate = (self._hits / attempted) if attempted > 0 else 0.0
        return {
            "capacity":      self.capacity_blocks,
            "used":          used,
            "block_size":    self.block_size,
            "lookups":       self._lookups,
            "block_hits":    self._hits,
            "block_misses":  self._misses,
            "evictions":     self._evictions,
            "hit_rate":      hit_rate,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"PrefixCache(used={s['used']}/{s['capacity']} "
            f"block_size={s['block_size']} "
            f"hits={s['block_hits']} misses={s['block_misses']} "
            f"hit_rate={s['hit_rate']:.1%})"
        )
