"""
models/base.py

模型配置数据类 + 推理基类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    resolution:      List[int] = field(default_factory=lambda: [448, 448])
    tile_size:       List[int] = field(default_factory=lambda: [224, 224])
    tokens_per_tile: int       = 256
    channels:        int       = 3
    feat_dim:        int       = 4096

    @property
    def num_tiles(self) -> int:
        h = self.resolution[0] // self.tile_size[0]
        w = self.resolution[1] // self.tile_size[1]
        return h * w

    @property
    def total_vision_tokens(self) -> int:
        return self.num_tiles * self.tokens_per_tile


@dataclass
class LLMConfig:
    num_layers:       int       = 32
    hidden_dim:       int       = 4096
    num_heads:        int       = 32
    head_dim:         int       = 128
    vocab_size:       int       = 152064      # Qwen3 vocab
    prefill_buckets:  List[int] = field(default_factory=lambda: [512, 1024, 2048])
    decode_buckets:   List[int] = field(default_factory=lambda: [512, 1024, 2048])
    max_seq_len:      int       = 4096


@dataclass
class ActionConfig:
    """VLA动作头配置"""
    head_type:         str   = "flow_matching"   # flow_matching / autoregressive
    action_dim:        int   = 7                 # 机械臂自由度
    action_horizon:    int   = 16                # 预测步数
    num_denoise_steps: int   = 15                # Flow去噪步数
    num_action_tokens: int   = 7                 # AR模式：动作token数
    action_token_start: int  = 32000             # AR模式：动作token在词表中的起始id


@dataclass
class TextConfig:
    """VLM文本输出配置"""
    max_new_tokens: int       = 512
    do_sample:      bool      = True
    temperature:    float     = 0.7
    top_p:          float     = 0.9
    eos_token_ids:  List[int] = field(default_factory=lambda: [151645, 151643])


@dataclass
class FrameworkConfig:
    mode:          str           = "vla"          # vla / vlm
    graph_dir:     str           = "compiled/"
    weight_path:   str           = "weights/weights.bin"
    vision:        VisionConfig  = field(default_factory=VisionConfig)
    llm:           LLMConfig     = field(default_factory=LLMConfig)
    action:        ActionConfig  = field(default_factory=ActionConfig)
    text:          TextConfig    = field(default_factory=TextConfig)
    max_batch:     int           = 1
    pad_token_id:  int           = 0
    bos_token_id:  int           = 151643
    eos_token_id:  int           = 151645

    @classmethod
    def from_yaml(cls, path: str) -> "FrameworkConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        cfg            = cls()
        cfg.mode       = raw.get("mode", cfg.mode)
        cfg.graph_dir  = raw.get("graph_dir", cfg.graph_dir)
        cfg.weight_path= raw.get("weight_path", cfg.weight_path)

        if "vision" in raw:
            v = raw["vision"]
            cfg.vision = VisionConfig(
                resolution      = v.get("resolution",      cfg.vision.resolution),
                tile_size       = v.get("tile_size",        cfg.vision.tile_size),
                tokens_per_tile = v.get("tokens_per_tile",  cfg.vision.tokens_per_tile),
                feat_dim        = v.get("feat_dim",          cfg.vision.feat_dim),
            )

        if "llm" in raw:
            l = raw["llm"]
            cfg.llm = LLMConfig(
                num_layers      = l.get("num_layers",      cfg.llm.num_layers),
                hidden_dim      = l.get("hidden_dim",      cfg.llm.hidden_dim),
                num_heads       = l.get("num_heads",       cfg.llm.num_heads),
                head_dim        = l.get("head_dim",        cfg.llm.head_dim),
                vocab_size      = l.get("vocab_size",      cfg.llm.vocab_size),
                prefill_buckets = l.get("prefill_buckets", cfg.llm.prefill_buckets),
                decode_buckets  = l.get("decode_buckets",  cfg.llm.decode_buckets),
                max_seq_len     = l.get("max_seq_len",     cfg.llm.max_seq_len),
            )

        if "action" in raw:
            a = raw["action"]
            cfg.action = ActionConfig(
                head_type          = a.get("head_type",           cfg.action.head_type),
                action_dim         = a.get("action_dim",          cfg.action.action_dim),
                action_horizon     = a.get("action_horizon",      cfg.action.action_horizon),
                num_denoise_steps  = a.get("num_denoise_steps",   cfg.action.num_denoise_steps),
                num_action_tokens  = a.get("num_action_tokens",   cfg.action.num_action_tokens),
            )

        if "text" in raw:
            t = raw["text"]
            cfg.text = TextConfig(
                max_new_tokens = t.get("max_new_tokens", cfg.text.max_new_tokens),
                do_sample      = t.get("do_sample",      cfg.text.do_sample),
                temperature    = t.get("temperature",    cfg.text.temperature),
                top_p          = t.get("top_p",          cfg.text.top_p),
                eos_token_ids  = t.get("eos_token_ids",  cfg.text.eos_token_ids),
            )

        return cfg
