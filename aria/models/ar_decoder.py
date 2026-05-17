"""
models/ar_decoder.py

自回归动作解码头（OpenVLA / RT-2风格）。
将连续动作离散化为token，通过LLM自回归逐token生成。
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from aria.models.base import FrameworkConfig

logger = logging.getLogger(__name__)


class ARDecoder:
    """
    自回归动作解码头。

    动作表示：
      每个动作维度 → 一个离散token（256 bin量化）
      token_id = action_token_start + bin_index

    解码流程：
      LLM Decode Loop × num_action_tokens步
      → 收集action tokens
      → 反量化到连续动作值
    """

    def __init__(self, config: FrameworkConfig, llm_backbone):
        self.config    = config
        self.backbone  = llm_backbone
        self.acfg      = config.action

        self.action_token_start = self.acfg.action_token_start
        self.num_bins           = 256
        self.action_min         = -1.0
        self.action_max         =  1.0

        logger.info(
            f"[ARIA/AR] action_dim={self.acfg.action_dim} "
            f"token_start={self.action_token_start}"
        )

    def decode(self, bos_token_id: int) -> np.ndarray:
        """
        从BOS token开始，自回归生成动作序列。

        返回: actions [action_dim] float32
        """
        action_tokens: List[int] = []
        cur_token = bos_token_id

        for step in range(self.acfg.num_action_tokens):
            logits = self.backbone.decode_step(cur_token)

            # 只在动作token范围内采样（屏蔽其他token）
            action_logits = logits[
                self.action_token_start:
                self.action_token_start + self.num_bins
            ]
            cur_token = int(action_logits.argmax()) + self.action_token_start
            action_tokens.append(cur_token)

            logger.debug(f"[ARIA/AR] step={step} token={cur_token}")

        return self._tokens_to_action(action_tokens)

    def _tokens_to_action(self, tokens: List[int]) -> np.ndarray:
        """将动作token列表反量化为连续动作值"""
        bin_indices = np.array(tokens) - self.action_token_start
        # 均匀量化反量化
        actions = (bin_indices / (self.num_bins - 1)) * \
                  (self.action_max - self.action_min) + self.action_min
        return actions.astype(np.float32)

    @staticmethod
    def action_to_tokens(actions: np.ndarray,
                         action_token_start: int,
                         num_bins: int = 256,
                         action_min: float = -1.0,
                         action_max: float = 1.0) -> List[int]:
        """工具方法：连续动作 → token（训练/评估时用）"""
        bins = np.clip(
            ((actions - action_min) / (action_max - action_min) * (num_bins - 1)),
            0, num_bins - 1
        ).astype(int)
        return (bins + action_token_start).tolist()
