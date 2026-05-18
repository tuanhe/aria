"""
backends/torch/model.py

参考用的小型 LLM / Vision / Flow 模块。所有 prefill / decode bucket
共享同一份 TinyLLM 权重，attention 支持 past_kv 输入（这是验证
前缀缓存 / 多轮 KV 复用语义所必须的）。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from aria.models.base import FrameworkConfig


# ---------------------------------------------------------------------------
# LLM 子模块
# ---------------------------------------------------------------------------

class TinyAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        inner = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_dim, inner, bias=False)
        self.k_proj = nn.Linear(hidden_dim, inner, bias=False)
        self.v_proj = nn.Linear(hidden_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, hidden_dim, bias=False)

    def forward(self,
                x:      torch.Tensor,
                past_k: Optional[torch.Tensor] = None,
                past_v: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:      [B, L_new, H]
        past_k: [B, num_heads, L_past, head_dim] 或 None
        past_v: 同上
        返回:   (out [B, L_new, H], new_k, new_v)  —— new_k/v 只是本次新位置的
        """
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if past_k is not None and past_k.size(2) > 0:
            k_full = torch.cat([past_k, k], dim=2)
            v_full = torch.cat([past_v, v], dim=2)
            L_past = past_k.size(2)
        else:
            k_full, v_full = k, v
            L_past = 0
        L_total = L_past + L

        # scaled dot-product
        attn = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 因果 mask：new 位置 i 可见 [0, L_past + i + 1)
        idx_row = torch.arange(L,       device=x.device).unsqueeze(1)
        idx_col = torch.arange(L_total, device=x.device).unsqueeze(0)
        causal  = idx_col <= (L_past + idx_row)
        attn = attn.masked_fill(~causal.view(1, 1, L, L_total), float("-inf"))
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v_full).transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out), k, v


class TinyLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn  = TinyAttention(hidden_dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, past_k=None, past_v=None):
        attn_out, k, v = self.attn(self.norm1(x), past_k, past_v)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, k, v


class TinyLLM(nn.Module):
    """
    最小可用的 Transformer 主干：
    - token embedding + 学习式 position embedding（避免 RoPE 的实现复杂度）
    - N 层 attention + MLP
    - 共用 LayerNorm + lm_head 出口
    一份权重覆盖所有 prefill / decode bucket，前缀缓存等价性才有意义。
    """

    def __init__(self, cfg: FrameworkConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.llm.hidden_dim
        max_pos = (cfg.llm.max_seq_len
                   + max([0] + cfg.llm.prefill_buckets + cfg.llm.decode_buckets)
                   + 64)
        self.token_emb   = nn.Embedding(cfg.llm.vocab_size, H)
        self.pos_emb     = nn.Embedding(max_pos, H)
        self.vision_proj = nn.Linear(cfg.vision.feat_dim, H)
        self.layers      = nn.ModuleList([
            TinyLayer(H, cfg.llm.num_heads, cfg.llm.head_dim)
            for _ in range(cfg.llm.num_layers)
        ])
        self.norm_out = nn.LayerNorm(H)
        self.lm_head  = nn.Linear(H, cfg.llm.vocab_size, bias=False)


# ---------------------------------------------------------------------------
# 视觉桩
# ---------------------------------------------------------------------------

class TinyVisionEncoder(nn.Module):
    """
    AdaptiveAvgPool2d → Linear 把每个 tile 压成 tokens_per_tile × feat_dim。
    pool 到固定 4×4 是为了让 Linear 权重跟原始分辨率无关，
    不至于在 224×224 输入下产生 GB 级权重。
    """

    POOL_SIZE = 4

    def __init__(self, cfg: FrameworkConfig):
        super().__init__()
        self.cfg = cfg
        self.pool = nn.AdaptiveAvgPool2d((self.POOL_SIZE, self.POOL_SIZE))
        in_dim    = cfg.vision.channels * self.POOL_SIZE * self.POOL_SIZE
        out_dim   = cfg.vision.tokens_per_tile * cfg.vision.feat_dim
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        tiles: [B, num_tiles, C, H, W]
        返回:   [B, total_vision_tokens, feat_dim]
        """
        B, N, C, H, W = tiles.shape
        x = tiles.reshape(B * N, C, H, W)
        x = self.pool(x).reshape(B * N, -1)
        x = self.proj(x)
        return x.view(B, N * self.cfg.vision.tokens_per_tile, self.cfg.vision.feat_dim)


# ---------------------------------------------------------------------------
# Flow Matching 桩
# ---------------------------------------------------------------------------

class TinyFlowHead(nn.Module):
    """concat[hidden, action.flatten(), t] → MLP → velocity。"""

    def __init__(self, cfg: FrameworkConfig):
        super().__init__()
        self.cfg = cfg
        in_dim  = (cfg.llm.hidden_dim
                   + cfg.action.action_horizon * cfg.action.action_dim
                   + 1)
        out_dim = cfg.action.action_horizon * cfg.action.action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self,
                hidden_state: torch.Tensor,
                noisy_action: torch.Tensor,
                timestep:     torch.Tensor) -> torch.Tensor:
        """
        hidden_state: [B, hidden_dim]
        noisy_action: [B, horizon, action_dim]
        timestep:     [1] 或 [B]
        返回 velocity:[B, horizon, action_dim]
        """
        B = hidden_state.shape[0]
        flat = noisy_action.view(B, -1)
        ts   = timestep.view(1).expand(B, 1) if timestep.numel() == 1 else timestep.view(B, 1)
        x    = torch.cat([hidden_state, flat, ts], dim=1)
        v    = self.net(x).view(B, self.cfg.action.action_horizon, self.cfg.action.action_dim)
        return v
