"""
runtime/vlm_runtime.py

VLM推理运行时（Qwen3 VL风格）。
支持多轮对话，KV Cache跨轮复用。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np

from aria.core.executor import NPUExecutor, MockNPUExecutor
from aria.core.kv_cache import KVCacheManager
from aria.core.prefix_cache import PrefixCache
from aria.models.base import FrameworkConfig
from aria.models.vision_encoder import VisionEncoder
from aria.models.llm_backbone import LLMBackbone
from aria.models.text_decoder import TextDecoder
from aria.runtime.session import Session

logger = logging.getLogger(__name__)

# Mock tokenizer：真实部署替换为 transformers AutoTokenizer
_CHAR_TO_ID = {c: i + 10 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
_ID_TO_CHAR = {v: k for k, v in _CHAR_TO_ID.items()}


def _mock_tokenize(text: str) -> List[int]:
    return [_CHAR_TO_ID.get(c.lower(), 1) for c in text[:256]]


def _mock_detokenize(token_ids: List[int]) -> str:
    return "".join(_ID_TO_CHAR.get(t, "?") for t in token_ids)


# Message格式：与OpenAI Chat API兼容
# content可以是字符串，或包含text/image的列表
Message = Dict[str, Any]


class VLMRuntime:
    """
    VLM推理运行时（支持多轮对话）。

    用法（单轮）：
        runtime  = VLMRuntime.from_config(config)
        response = runtime.chat([
            {"role": "user", "content": [
                {"type": "image", "data": image_array},
                {"type": "text",  "data": "描述这张图片"},
            ]}
        ])

    用法（多轮）：
        session_id = runtime.new_session()
        r1 = runtime.chat(messages1, session_id=session_id)
        r2 = runtime.chat(messages2, session_id=session_id)  # 复用KV Cache
        runtime.close_session(session_id)
    """

    def __init__(self,
                 config:       FrameworkConfig,
                 executor:     NPUExecutor,
                 prefix_cache: Optional[PrefixCache] = None):
        self.config       = config
        self.executor     = executor
        self.prefix_cache = prefix_cache

        # 每个session独立的KV Cache
        # 端侧通常单session，这里用dict支持多session扩展
        self._sessions: Dict[str, Session] = {}

        # 子模块（所有session共享）
        self.vision_encoder = VisionEncoder(config, executor)

        msg = "[ARIA/VLM] 初始化完成"
        if prefix_cache is not None:
            msg += (f" prefix_cache=on(block={prefix_cache.block_size} "
                    f"capacity={prefix_cache.capacity_blocks})")
        logger.info(msg)

    @classmethod
    def from_config(cls,
                    config:       FrameworkConfig,
                    executor:     Optional[NPUExecutor]  = None,
                    prefix_cache: Optional[PrefixCache]  = None) -> "VLMRuntime":
        if executor is None:
            executor = MockNPUExecutor()
            logger.info("[ARIA/VLM] 使用MockNPUExecutor")
        return cls(config, executor, prefix_cache=prefix_cache)

    # ------------------------------------------------------------------
    # Session管理
    # ------------------------------------------------------------------

    def new_session(self) -> str:
        """创建新的对话session，返回session_id"""
        session_id = str(uuid.uuid4())[:8]
        kv_cache   = KVCacheManager(
            num_layers  = self.config.llm.num_layers,
            num_heads   = self.config.llm.num_heads,
            head_dim    = self.config.llm.head_dim,
            max_seq_len = self.config.llm.max_seq_len,
            max_batch   = self.config.max_batch,
            executor    = self.executor,
        )
        self._sessions[session_id] = Session(session_id, kv_cache)
        logger.info(f"[ARIA/VLM] 新建session: {session_id}")
        return session_id

    def close_session(self, session_id: str) -> None:
        """关闭session，释放KV Cache"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"[ARIA/VLM] 关闭session: {session_id}")

    def reset_session(self, session_id: str) -> None:
        """重置session（清空对话历史）"""
        if session_id in self._sessions:
            self._sessions[session_id].reset()

    def _get_or_create_session(self, session_id: Optional[str]) -> Session:
        if session_id is None:
            session_id = self.new_session()
        if session_id not in self._sessions:
            raise ValueError(f"Session '{session_id}' 不存在")
        return self._sessions[session_id]

    # ------------------------------------------------------------------
    # 主推理接口
    # ------------------------------------------------------------------

    def chat(self,
             messages:    List[Message],
             session_id:  Optional[str] = None) -> str:
        """
        多模态对话推理。

        messages:   本轮新增的消息列表（不含历史，历史在session KV Cache中）
        session_id: 多轮对话时传入，None表示新建临时session

        返回: 模型回复的文本字符串
        """
        t_start    = time.perf_counter()
        is_temp    = session_id is None
        session    = self._get_or_create_session(session_id)

        # 构建LLM（绑定本session的KV Cache）
        llm = LLMBackbone(self.config, self.executor, session.kv_cache)

        # Step 1: 解析messages，提取文本和图像
        t0 = time.perf_counter()
        token_ids, vision_feat = self._parse_messages(messages)
        t_parse = (time.perf_counter() - t0) * 1000

        # 检查KV Cache空间（按未命中前缀缓存的总长度算上限）
        new_len = len(token_ids) + (
            self.config.vision.total_vision_tokens if vision_feat is not None else 0
        )
        if not session.can_accept_tokens(new_len):
            raise RuntimeError(
                f"KV Cache不足: 当前={session.current_kv_len} "
                f"新增={new_len} 最大={self.config.llm.max_seq_len}"
            )

        # Step 1.5: 前缀缓存查询（仅在新 session + 纯文本时启用）
        # 图像模式下 token_ids 里的 vision 位置是占位 BOS，跟实际 vision_feat
        # 解耦，命中错位会用错 KV，所以图像分支不走这里。
        full_user_tokens = list(token_ids)   # 留作 insert 时的 key
        prefix_hit_blocks = 0
        if (self.prefix_cache is not None
            and session.history_kv_len == 0
            and vision_feat is None
            and len(token_ids) >= self.prefix_cache.block_size):

            arr   = np.array(token_ids, dtype=np.int32)
            match = self.prefix_cache.match(arr)
            if match.num_blocks > 0:
                session.kv_cache.bulk_load_prefix(match.gather())
                prefix_hit_blocks = match.num_blocks
                token_ids = token_ids[match.matched_tokens:]
                logger.info(
                    f"[ARIA/VLM] prefix-cache 命中: "
                    f"{match.num_blocks} blocks ({match.matched_tokens} tokens)"
                )

        # Step 2: Prefill（只对新增内容做Prefill，历史复用KV Cache）
        t0 = time.perf_counter()
        last_hidden = llm.prefill(
            token_ids    = np.array(token_ids, dtype=np.int32),
            vision_feat  = vision_feat if vision_feat is not None
                           else self._empty_vision_feat(),
            kv_start_pos = session.history_kv_len,
        )
        t_prefill = (time.perf_counter() - t0) * 1000

        # Step 3: 获取第一个token的logits（由last_hidden过lm_head得到）
        # 在真实模型中，Prefill图会直接输出最后一个token的logits
        # 这里用Mock：直接生成随机logits
        first_logits = self._get_first_logits(last_hidden)

        # Step 4: 文本Decode Loop
        t0      = time.perf_counter()
        decoder = TextDecoder(self.config, llm)
        gen_ids = decoder.decode(first_logits)
        t_decode = (time.perf_counter() - t0) * 1000

        # Step 5: Detokenize
        response = _mock_detokenize(gen_ids)

        # Step 6: 更新session状态
        session.add_user_turn(messages, token_ids)
        session.add_assistant_turn(response, gen_ids)

        # Step 7: 把用户消息的 KV 写回前缀缓存（已存在的 block 会去重）
        if self.prefix_cache is not None and vision_feat is None and full_user_tokens:
            arr      = np.array(full_user_tokens, dtype=np.int32)
            user_kv  = session.kv_cache.read_range(0, len(arr))
            inserted = self.prefix_cache.insert(arr, user_kv)
            if inserted > 0:
                logger.info(f"[ARIA/VLM] prefix-cache 写入: +{inserted} blocks")

        t_total = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[ARIA/VLM] 推理完成 session={session.session_id} "
            f"parse={t_parse:.1f}ms prefill={t_prefill:.1f}ms "
            f"decode={t_decode:.1f}ms total={t_total:.1f}ms "
            f"生成={len(gen_ids)}tokens "
            f"prefix_hit={prefix_hit_blocks}blk"
        )

        # 临时session用完即关闭
        if is_temp:
            self.close_session(session.session_id)

        return response

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _parse_messages(self, messages: List[Message]):
        """
        解析messages列表，返回 (token_ids, vision_feat)。
        vision_feat为None表示纯文本输入。
        """
        all_token_ids = []
        vision_feat   = None

        for msg in messages:
            content = msg.get("content", "")

            # 纯文本消息
            if isinstance(content, str):
                all_token_ids.extend(_mock_tokenize(content))
                continue

            # 多模态消息（列表格式）
            for item in content:
                if item["type"] == "text":
                    all_token_ids.extend(_mock_tokenize(item["data"]))

                elif item["type"] == "image":
                    img         = item["data"]
                    vision_feat = self.vision_encoder.encode(img)
                    # 插入视觉占位符token
                    vis_tokens  = [self.config.bos_token_id] * \
                                   self.config.vision.total_vision_tokens
                    all_token_ids.extend(vis_tokens)

        return all_token_ids, vision_feat

    def _empty_vision_feat(self) -> np.ndarray:
        """纯文本输入时返回零视觉特征"""
        return np.zeros(
            (self.config.max_batch,
             self.config.vision.total_vision_tokens,
             self.config.vision.feat_dim),
            dtype=np.float16
        )

    def _get_first_logits(self, last_hidden: np.ndarray) -> np.ndarray:
        """
        由last_hidden经lm_head得到第一个token的logits。
        真实模型中lm_head是一个Linear层，Prefill图直接输出logits。
        这里Mock为随机logits（shape正确）。
        """
        vocab_size = self.config.llm.vocab_size
        rng        = np.random.default_rng()
        return rng.standard_normal(vocab_size).astype(np.float32)
