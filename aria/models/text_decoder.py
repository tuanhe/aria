"""
models/text_decoder.py

文本解码头（Qwen3 VL等VLM模型用）。
支持 greedy / top-p sampling，EOS停止，多轮KV Cache复用。
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import numpy as np

from aria.models.base import FrameworkConfig

logger = logging.getLogger(__name__)


class TextDecoder:
    """
    自回归文本解码头。

    解码流程：
      Prefill后得到last_hidden → lm_head → 第一个token
      → Decode Loop直到EOS或max_new_tokens
    """

    def __init__(self, config: FrameworkConfig, llm_backbone):
        self.config   = config
        self.backbone = llm_backbone
        self.tcfg     = config.text
        self._rng     = np.random.default_rng(42)

        logger.info(
            f"[ARIA/Text] max_new_tokens={self.tcfg.max_new_tokens} "
            f"do_sample={self.tcfg.do_sample} "
            f"temperature={self.tcfg.temperature} "
            f"top_p={self.tcfg.top_p}"
        )

    def decode(self, first_token_logits: np.ndarray) -> List[int]:
        """
        从Prefill输出的logits开始，自回归生成token序列直到EOS。

        first_token_logits: [vocab_size] float32
        返回: 生成的token id列表（不含EOS）
        """
        generated: List[int] = []
        logits = first_token_logits

        for step in range(self.tcfg.max_new_tokens):
            token_id = self._sample(logits)

            if token_id in self.tcfg.eos_token_ids:
                logger.debug(f"[ARIA/Text] 遇到EOS，步数={step}")
                break

            generated.append(token_id)

            # 继续Decode
            logits = self.backbone.decode_step(token_id)

            logger.debug(f"[ARIA/Text] step={step} token={token_id}")

        return generated

    def decode_stream(self, first_token_logits: np.ndarray) -> Iterator[int]:
        """
        同 decode()，但逐 token yield，供流式输出使用。
        """
        logits = first_token_logits
        for step in range(self.tcfg.max_new_tokens):
            token_id = self._sample(logits)
            if token_id in self.tcfg.eos_token_ids:
                logger.debug("[ARIA/Text] 遇到EOS，步数=%d", step)
                return
            yield token_id
            logits = self.backbone.decode_step(token_id)
            logger.debug("[ARIA/Text] step=%d token=%d", step, token_id)

    # ------------------------------------------------------------------
    # 采样策略
    # ------------------------------------------------------------------

    def _sample(self, logits: np.ndarray) -> int:
        """根据配置选择采样策略"""
        if not self.tcfg.do_sample:
            return int(logits.argmax())

        # Temperature缩放
        logits = logits.astype(np.float32) / max(self.tcfg.temperature, 1e-6)

        # 数值稳定：减去最大值防止exp溢出
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()

        # Top-p（nucleus）过滤
        if self.tcfg.top_p < 1.0:
            probs = self._top_p_filter(probs, self.tcfg.top_p)

        return int(self._rng.choice(len(probs), p=probs))

    def _top_p_filter(self, probs: np.ndarray, top_p: float) -> np.ndarray:
        """
        保留累积概率 <= top_p 的最小token集合，其余置零后重归一化。
        """
        sorted_idx    = np.argsort(probs)[::-1]
        sorted_probs  = probs[sorted_idx]
        cumsum        = np.cumsum(sorted_probs)

        # 找到第一个超过top_p的位置，保留到该位置（含）
        cutoff_idx = int(np.searchsorted(cumsum, top_p))
        cutoff_idx = min(cutoff_idx + 1, len(sorted_probs))

        # 构建mask
        mask = np.zeros_like(probs)
        mask[sorted_idx[:cutoff_idx]] = 1.0

        filtered = probs * mask
        total    = filtered.sum()
        if total <= 0:
            # 极端情况：所有概率都被过滤，fallback到greedy
            return (probs == probs.max()).astype(np.float32)
        return filtered / total
