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
from typing import Any, Dict, Iterator, List, Optional

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

    流式用法：
        for token_text in runtime.generate_stream("你好"):
            print(token_text, end="", flush=True)
    """

    def __init__(self,
                 config:     FrameworkConfig,
                 executor:   NPUExecutor,
                 tokenizer:  Any = None):
        assert config.mode == "llm", \
            f"LLMRuntime 需要 mode=llm，当前 mode={config.mode}"
        self.config      = config
        self.executor    = executor
        self._tokenizer  = tokenizer   # transformers AutoTokenizer，None 则用 mock
        self._sessions: Dict[str, Session] = {}
        logger.info("[ARIA/LLM] LLMRuntime 初始化完成 tokenizer=%s",
                    type(tokenizer).__name__ if tokenizer else "mock")

    @classmethod
    def from_config(cls,
                    config:    FrameworkConfig,
                    executor:  Optional[NPUExecutor] = None,
                    tokenizer: Any = None) -> "LLMRuntime":
        if executor is None:
            executor = MockNPUExecutor()
            logger.info("[ARIA/LLM] 使用 MockNPUExecutor")
        return cls(config, executor, tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[int]:
        if self._tokenizer is not None:
            return self._tokenizer.encode(text, add_special_tokens=False)
        return _mock_tokenize(text)

    def _detokenize(self, ids: List[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(ids, skip_special_tokens=True)
        return _mock_detokenize(ids)

    def _detokenize_token(self, token_id: int) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode([token_id], skip_special_tokens=True)
        return _mock_detokenize([token_id])

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
            executor    = self.executor,
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

    def generate_stream(self,
                        prompt:     str,
                        session_id: Optional[str] = None) -> Iterator[str]:
        """
        流式文本生成，逐 token yield 解码后的文本片段。

        prompt:     本轮输入（不含历史，历史在 KV Cache 里）
        session_id: 多轮时传入，None 则新建临时 session
        """
        t0      = time.perf_counter()
        is_temp = session_id is None
        if is_temp:
            session_id = self.new_session()
        session = self._sessions[session_id]

        llm = LLMBackbone(self.config, self.executor, session.kv_cache)

        token_ids = self._tokenize(prompt)
        if not session.can_accept_tokens(len(token_ids)):
            if is_temp:
                self.close_session(session_id)
            raise RuntimeError(
                f"KV Cache 不足: 当前={session.current_kv_len} "
                f"新增={len(token_ids)} 最大={self.config.llm.max_seq_len}"
            )

        first_logits = llm.prefill(
            token_ids    = np.array(token_ids, dtype=np.int32),
            kv_start_pos = session.history_kv_len,
        )

        decoder = TextDecoder(self.config, llm)
        gen_ids: List[int] = []
        for token_id in decoder.decode_stream(first_logits):
            gen_ids.append(token_id)
            yield self._detokenize_token(token_id)

        response = self._detokenize(gen_ids)
        session.add_user_turn({"role": "user", "content": prompt}, token_ids)
        session.add_assistant_turn(response, gen_ids)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "[ARIA/LLM] generate_stream 完成 session=%s elapsed=%.1fms tokens=%d",
            session_id, elapsed, len(gen_ids),
        )

        if is_temp:
            self.close_session(session_id)

    def generate(self,
                 prompt:     str,
                 session_id: Optional[str] = None) -> str:
        """
        文本生成（非流式，收集 generate_stream 的所有输出后返回）。

        prompt:     本轮输入（不含历史，历史在 KV Cache 里）
        session_id: 多轮时传入，None 则新建临时 session

        返回: 生成的文本字符串
        """
        return "".join(self.generate_stream(prompt, session_id))
