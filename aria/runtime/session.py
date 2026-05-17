"""
runtime/session.py

多轮对话Session管理（VLM模式）。
维护对话历史、KV Cache跨轮复用状态。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aria.core.kv_cache import KVCacheManager

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    role:    str          # "user" / "assistant"
    content: Any          # 文本或message列表
    tokens:  List[int] = field(default_factory=list)
    timestamp: float   = field(default_factory=time.time)


class Session:
    """
    单次对话Session。

    VLM多轮对话场景下：
    - 每轮用户输入只做新增部分的Prefill
    - 历史轮的KV Cache保留在KVCacheManager中，通过kv_start_pos复用
    - reset()开始新对话
    """

    def __init__(self, session_id: str, kv_cache: KVCacheManager):
        self.session_id   = session_id
        self.kv_cache     = kv_cache
        self.turns:       List[Turn]   = []
        self.created_at   = time.time()
        self._total_tokens = 0

        logger.info(f"[ARIA/Session] 新建 session_id={session_id}")

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def history_kv_len(self) -> int:
        """历史轮次已占用的KV Cache长度（新一轮Prefill的起始位置）"""
        return self.kv_cache.history_len

    @property
    def current_kv_len(self) -> int:
        """当前KV Cache有效长度（包含当前轮未保存的部分）"""
        return self.kv_cache.valid_len

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def remaining_kv(self) -> int:
        return self.kv_cache.remaining

    # ------------------------------------------------------------------
    # 轮次管理
    # ------------------------------------------------------------------

    def add_user_turn(self, content: Any, tokens: List[int]) -> None:
        self.turns.append(Turn(role="user", content=content, tokens=tokens))
        self._total_tokens += len(tokens)

    def add_assistant_turn(self, content: str, tokens: List[int]) -> None:
        self.turns.append(Turn(role="assistant", content=content, tokens=tokens))
        self._total_tokens += len(tokens)

        # 将当前轮生成的KV Cache标记为历史，供下一轮复用
        self.kv_cache.save_turn()
        logger.debug(
            f"[ARIA/Session] 轮次完成 turn={self.num_turns} "
            f"history_kv_len={self.history_kv_len}"
        )

    def reset(self) -> None:
        """开始新对话，清空历史"""
        self.turns.clear()
        self.kv_cache.reset()
        self._total_tokens = 0
        logger.info(f"[ARIA/Session] 重置 session_id={self.session_id}")

    def can_accept_tokens(self, new_token_count: int) -> bool:
        """检查是否还有足够的KV Cache空间"""
        return self.current_kv_len + new_token_count <= self.kv_cache.max_seq_len

    def __repr__(self) -> str:
        return (
            f"Session(id={self.session_id}, turns={self.num_turns}, "
            f"kv={self.current_kv_len}/{self.kv_cache.max_seq_len})"
        )
