"""
runtime/llm_runtime.py

纯文本 LLM 推理运行时（Qwen3 等）。
无视觉编码器，输入文本字符串，输出生成文本。
支持多轮对话，KV Cache 跨轮复用。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional

import numpy as np

from aria.core.executor import NPUExecutor, MockNPUExecutor
from aria.core.kv_cache import KVCacheManager
from aria.models.base import FrameworkConfig
from aria.models.llm_backbone import LLMBackbone
from aria.models.text_decoder import TextDecoder
from aria.runtime.session import Session

logger = logging.getLogger(__name__)

# Mock tokenizer（真实部署替换为 transformers AutoTokenizer）
_CHAR_TO_ID = {c: i + 10 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
_ID_TO_CHAR = {v: k for k, v in _CHAR_TO_ID.items()}


def _mock_tokenize(text: str) -> List[int]:
    return [_CHAR_TO_ID.get(c.lower(), 1) for c in text[:2048]]


def _mock_detokenize(ids: List[int]) -> str:
    return "".join(_ID_TO_CHAR.get(t, "?") for t in ids)


class LLMRuntime:
    """
    纯文本 LLM 推理运行时。

    单轮用法：
        runtime = LLMRuntime.from_config(config)
        reply   = runtime.generate("你好，请介绍一下自己")

    多轮用法：
        sid   = runtime.new_session()
        r1    = runtime.generate("第一个问题", session_id=sid)
        r2    = runtime.generate("继续上面的话题", session_id=sid)
        runtime.close_session(sid)
    """

    def __init__(self,
                 config:   FrameworkConfig,
                 executor: NPUExecutor):
        assert config.mode == "llm", \
            f"LLMRuntime 需要 mode=llm，当前 mode={config.mode}"
        self.config    = config
        self.executor  = executor
        self._sessions: Dict[str, Session] = {}
        logger.info("[ARIA/LLM] LLMRuntime 初始化完成")

    @classmethod
    def from_config(cls,
                    config:   FrameworkConfig,
                    executor: Optional[NPUExecutor] = None) -> "LLMRuntime":
        if executor is None:
            executor = MockNPUExecutor()
            logger.info("[ARIA/LLM] 使用 MockNPUExecutor")
        return cls(config, executor)

    # ------------------------------------------------------------------
    # Session 管理
    # ------------------------------------------------------------------

    def new_session(self) -> str:
        sid      = str(uuid.uuid4())[:8]
        kv_cache = KVCacheManager(
            num_layers  = self.config.llm.num_layers,
            num_heads   = self.config.llm.num_heads,
            head_dim    = self.config.llm.head_dim,
            max_seq_len = self.config.llm.max_seq_len,
            max_batch   = self.config.max_batch,
        )
        self._sessions[sid] = Session(sid, kv_cache)
        logger.info("[ARIA/LLM] 新建 session: %s", sid)
        return sid

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        logger.info("[ARIA/LLM] 关闭 session: %s", session_id)

    def reset_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].reset()

    # ------------------------------------------------------------------
    # 主推理接口
    # ------------------------------------------------------------------

    def generate(self,
                 prompt:     str,
                 session_id: Optional[str] = None) -> str:
        """
        文本生成。

        prompt:     本轮输入（不含历史，历史在 KV Cache 里）
        session_id: 多轮时传入，None 则新建临时 session

        返回: 生成的文本字符串
        """
        t0      = time.perf_counter()
        is_temp = session_id is None
        if is_temp:
            session_id = self.new_session()
        session = self._sessions[session_id]

        llm = LLMBackbone(self.config, self.executor, session.kv_cache)

        # tokenize
        token_ids = _mock_tokenize(prompt)
        if not session.can_accept_tokens(len(token_ids)):
            raise RuntimeError(
                f"KV Cache 不足: 当前={session.current_kv_len} "
                f"新增={len(token_ids)} 最大={self.config.llm.max_seq_len}"
            )

        # prefill → 直接得到 logits（llm 模式）
        first_logits = llm.prefill(
            token_ids    = np.array(token_ids, dtype=np.int32),
            kv_start_pos = session.history_kv_len,
        )  # [vocab_size] float32

        # decode loop
        decoder  = TextDecoder(self.config, llm)
        gen_ids  = decoder.decode(first_logits)
        response = _mock_detokenize(gen_ids)

        session.add_user_turn({"role": "user", "content": prompt}, token_ids)
        session.add_assistant_turn(response, gen_ids)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "[ARIA/LLM] generate 完成 session=%s elapsed=%.1fms tokens=%d",
            session_id, elapsed, len(gen_ids),
        )

        if is_temp:
            self.close_session(session_id)

        return response
