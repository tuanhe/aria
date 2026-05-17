"""
models/flow_decoder.py

Flow Matching动作解码头（π0风格）。
非自回归，通过迭代去噪一次性生成完整动作序列。
"""

from __future__ import annotations

import logging

import numpy as np

from aria.core.executor import NPUExecutor, GraphMeta
from aria.models.base import FrameworkConfig

logger = logging.getLogger(__name__)

GRAPH_NAME = "flow_head"


class FlowDecoder:
    """
    Flow Matching动作解码头。

    去噪流程：
      随机噪声 action_T
      → Flow网络预测速度场 velocity
      → Euler/Heun积分 × num_denoise_steps步
      → 干净动作 action_0

    Flow网络输入：
      hidden_state（来自LLM Prefill，每步复用，不更新）
      noisy_action（当前噪声动作）
      timestep（当前时间步）
    """

    def __init__(self, config: FrameworkConfig, executor: NPUExecutor):
        self.config   = config
        self.executor = executor
        self.acfg     = config.action

        self._register_graph()
        self._build_timesteps()

        logger.info(
            f"[ARIA/Flow] action_dim={self.acfg.action_dim} "
            f"horizon={self.acfg.action_horizon} "
            f"denoise_steps={self.acfg.num_denoise_steps}"
        )

    def _register_graph(self) -> None:
        meta = GraphMeta(
            name  = GRAPH_NAME,
            path  = f"{self.config.graph_dir}/{GRAPH_NAME}.bin",
            input_shapes = {
                "hidden_state": (self.config.max_batch, self.config.llm.hidden_dim),
                "noisy_action": (self.config.max_batch,
                                  self.acfg.action_horizon,
                                  self.acfg.action_dim),
                "timestep":     (1,),
            },
            output_shapes = {
                "velocity": (self.config.max_batch,
                              self.acfg.action_horizon,
                              self.acfg.action_dim),
            },
            output_dtypes = {"velocity": np.float32},
        )
        self.executor.register_graph(meta)

    def _build_timesteps(self) -> None:
        """构建去噪时间步序列（从T→0的均匀序列）"""
        self._timesteps = np.linspace(
            1.0, 0.0,
            self.acfg.num_denoise_steps + 1,
            dtype=np.float32
        )

    def decode(self, hidden_state: np.ndarray, seed: int = None) -> np.ndarray:
        """
        从纯噪声出发，迭代去噪生成动作序列。

        hidden_state: [1, hidden_dim] float16，来自LLM Prefill
        返回: actions [action_horizon, action_dim] float32
        """
        rng = np.random.default_rng(seed)

        # 从纯高斯噪声开始
        action = rng.standard_normal(
            (self.config.max_batch, self.acfg.action_horizon, self.acfg.action_dim)
        ).astype(np.float32)

        # 迭代去噪
        for i in range(self.acfg.num_denoise_steps):
            t    = self._timesteps[i]
            dt   = self._timesteps[i] - self._timesteps[i + 1]

            out = self.executor.run(
                GRAPH_NAME,
                {
                    "hidden_state": hidden_state.astype(np.float16),
                    "noisy_action": action.astype(np.float16),
                    "timestep":     np.array([t], dtype=np.float32),
                }
            )
            velocity = out["velocity"]  # [batch, horizon, action_dim]

            # Euler积分步
            action = action + velocity * dt

            logger.debug(f"[ARIA/Flow] step={i}/{self.acfg.num_denoise_steps} t={t:.3f}")

        # 返回第一个batch的动作序列
        return action[0]  # [action_horizon, action_dim]

    def decode_with_heun(self, hidden_state: np.ndarray) -> np.ndarray:
        """
        使用Heun方法（2阶Runge-Kutta）去噪，精度更高但计算量翻倍。
        适合对精度要求高的场景。
        """
        rng    = np.random.default_rng()
        action = rng.standard_normal(
            (self.config.max_batch, self.acfg.action_horizon, self.acfg.action_dim)
        ).astype(np.float32)

        for i in range(self.acfg.num_denoise_steps):
            t  = self._timesteps[i]
            t_next = self._timesteps[i + 1]
            dt = t - t_next

            # Heun: 先Euler预测
            v1 = self.executor.run(GRAPH_NAME, {
                "hidden_state": hidden_state.astype(np.float16),
                "noisy_action": action.astype(np.float16),
                "timestep":     np.array([t], dtype=np.float32),
            })["velocity"]

            action_pred = action + v1 * dt

            # 再用预测点的速度修正
            v2 = self.executor.run(GRAPH_NAME, {
                "hidden_state": hidden_state.astype(np.float16),
                "noisy_action": action_pred.astype(np.float16),
                "timestep":     np.array([t_next], dtype=np.float32),
            })["velocity"]

            action = action + (v1 + v2) * 0.5 * dt

        return action[0]
