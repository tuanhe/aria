"""
runtime/vla_runtime.py

VLA推理运行时。
支持 Flow Matching（π0）和自回归（OpenVLA / RT-2）两种动作解码路径。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from aria.core.executor import NPUExecutor, MockNPUExecutor
from aria.core.kv_cache import KVCacheManager
from aria.models.base import FrameworkConfig
from aria.models.vision_encoder import VisionEncoder
from aria.models.llm_backbone import LLMBackbone
from aria.models.ar_decoder import ARDecoder
from aria.models.flow_decoder import FlowDecoder

logger = logging.getLogger(__name__)

# 简单tokenizer Mock：真实部署替换为transformers / sentencepiece
_CHAR_TO_ID = {c: i + 10 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}


def _mock_tokenize(text: str) -> list:
    return [_CHAR_TO_ID.get(c.lower(), 1) for c in text[:64]]


class VLARuntime:
    """
    VLA推理运行时。

    用法：
        runtime = VLARuntime.from_config(config, executor)
        action  = runtime.infer(image, "pick up the red cup")
        # action: [action_horizon, action_dim] float32（Flow模式）
        #         [action_dim] float32（AR模式）
    """

    def __init__(self,
                 config:   FrameworkConfig,
                 executor: NPUExecutor):
        self.config   = config
        self.executor = executor
        self.acfg     = config.action

        # KV Cache（VLA单轮推理，每次reset）
        self.kv_cache = KVCacheManager(
            num_layers  = config.llm.num_layers,
            num_heads   = config.llm.num_heads,
            head_dim    = config.llm.head_dim,
            max_seq_len = config.llm.max_seq_len,
            max_batch   = config.max_batch,
        )

        # 子模块
        self.vision_encoder = VisionEncoder(config, executor)
        self.llm            = LLMBackbone(config, executor, self.kv_cache)

        # 动作头：根据配置选择路径
        if self.acfg.head_type == "flow_matching":
            self.action_decoder = FlowDecoder(config, executor)
            logger.info("[ARIA/VLA] 动作头: Flow Matching（π0风格）")
        elif self.acfg.head_type == "autoregressive":
            self.action_decoder = ARDecoder(config, self.llm)
            logger.info("[ARIA/VLA] 动作头: 自回归（OpenVLA风格）")
        else:
            raise ValueError(f"未知动作头类型: {self.acfg.head_type}")

        logger.info(f"[ARIA/VLA] 初始化完成 mode={self.acfg.head_type}")

    @classmethod
    def from_config(cls,
                    config:   FrameworkConfig,
                    executor: Optional[NPUExecutor] = None) -> "VLARuntime":
        if executor is None:
            executor = MockNPUExecutor()
            logger.info("[ARIA/VLA] 使用MockNPUExecutor")
        return cls(config, executor)

    # ------------------------------------------------------------------
    # 主推理接口
    # ------------------------------------------------------------------

    def infer(self,
              image:       np.ndarray,
              instruction: str) -> np.ndarray:
        """
        单帧推理。

        image:       原始图像 [H, W, C] uint8
        instruction: 任务指令字符串

        返回:
          Flow模式: [action_horizon, action_dim] float32
          AR模式:   [action_dim] float32
        """
        t_start = time.perf_counter()

        # 每次推理重置KV Cache（VLA是单轮的）
        self.kv_cache.reset()

        # Step 1: 视觉编码
        t0          = time.perf_counter()
        vision_feat = self.vision_encoder.encode(image)
        t_vision    = (time.perf_counter() - t0) * 1000

        # Step 2: Tokenize指令
        token_ids = np.array(_mock_tokenize(instruction), dtype=np.int32)

        # Step 3: Prefill
        t0          = time.perf_counter()
        last_hidden = self.llm.prefill(
            token_ids    = token_ids,
            vision_feat  = vision_feat,
            kv_start_pos = 0,
        )
        t_prefill = (time.perf_counter() - t0) * 1000

        # Step 4: 动作解码
        t0 = time.perf_counter()
        if self.acfg.head_type == "flow_matching":
            action = self.action_decoder.decode(last_hidden)
        else:
            action = self.action_decoder.decode(self.config.bos_token_id)
        t_decode = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[ARIA/VLA] 推理完成 "
            f"vision={t_vision:.1f}ms prefill={t_prefill:.1f}ms "
            f"decode={t_decode:.1f}ms total={t_total:.1f}ms"
        )

        return action

    def infer_batch(self,
                    images:       list,
                    instructions: list) -> list:
        """批量推理（逐个串行，NPU batch=1场景）"""
        return [self.infer(img, inst) for img, inst in zip(images, instructions)]
