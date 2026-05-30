"""
tools/exporters/qwen3.py

Qwen3 LLM/VLM backbone → ONNX 导出器。支持两种模式（由 cfg.mode 决定）：

  mode=llm  — 纯文本，prefill 无 vision_feat 输入，直接输出 logits
  mode=vlm  — 多模态，prefill 含 vision_feat 输入，输出 last_hidden

  prefill_{seq_len}（llm 模式）
    输入 : input_ids      [1, seq_len]   int32
           attention_mask [1, seq_len]   int32
           position_ids   [1, seq_len]   int32
           kv_start_pos   [1]            int32
    输出 : logits         [1, vocab_size] fp32
           kv_out         [L*2, 1, kv_heads, seq_len, head_dim]  fp16

  prefill_{seq_len}（vlm 模式）
    输入 : input_ids      [1, seq_len]               int32
           vision_feat    [1, vis_tokens, feat_dim]   fp16
           attention_mask [1, seq_len]                int32
           position_ids   [1, seq_len]                int32
           kv_start_pos   [1]                         int32
    输出 : last_hidden    [1, hidden_dim]              fp16
           kv_out         [L*2, 1, kv_heads, seq_len, head_dim]  fp16

  decode（两种模式相同，单图、shape 固定，无 bucket）
    输入 : input_id       [1, 1]                                  int32
           position_id    [1, 1]                                  int32  当前序列长度 pos
           attention_mask [1, max_seq_len + 1]                    int32  [0,pos)=1, [pos,MAX)=0, [MAX]=1
           kv_cache       [L*2, 1, kv_heads, max_seq_len, head_dim] fp16  常驻 buffer
    输出 : logits         [1, vocab_size]                         fp32
           kv_new         [L*2, 1, kv_heads, 1, head_dim]        fp16

KV cache：奇偶交错，kv[i*2]=layer_i K，kv[i*2+1]=layer_i V。
Qwen3 GQA：kv_heads = model.config.num_key_value_heads。
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from aria.models.base import FrameworkConfig
from tools.exporters.base import BaseExporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部 wrapper modules（静态 shape，可被 torch.onnx.export 追踪）
# ---------------------------------------------------------------------------

class _PrefillWrapper(nn.Module):
    """
    覆盖 HF model.model.forward() 的追踪入口。
    - 用 inputs_embeds 注入 vision_feat（替换前 vis_tokens 个位置）
    - 用 attention_mask 的 sum 定位最后一个有效 token
    - 展平 DynamicCache 为 [L*2, 1, kv_heads, seq_len, head_dim]
    """

    def __init__(self,
                 hf_model,
                 vis_tokens: int,
                 feat_dim:   int):
        super().__init__()
        hidden_dim    = hf_model.config.hidden_size
        self._inner   = hf_model.model   # Qwen3Model（含 layers / norm）
        self._vis_tok = vis_tokens

        # vision → hidden_dim 投影（仅 feat_dim ≠ hidden_dim 时生效）
        if feat_dim != hidden_dim:
            self._vis_proj: nn.Module = nn.Linear(feat_dim, hidden_dim, bias=False)
        else:
            self._vis_proj = nn.Identity()

    def forward(
        self,
        input_ids:      torch.Tensor,   # [1, seq_len] int32
        vision_feat:    torch.Tensor,   # [1, vis_tokens, feat_dim] fp16
        attention_mask: torch.Tensor,   # [1, seq_len] int32
        position_ids:   torch.Tensor,   # [1, seq_len] int32
        kv_start_pos:   torch.Tensor,   # [1] int32  (不参与计算，仅保留为 ONNX 输入节点)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. 文本 embedding
        text_emb = self._inner.embed_tokens(input_ids)     # [1, seq_len, hidden_dim]

        # 2. vision 特征投影 + 注入前 vis_tokens 个位置
        vis_emb = self._vis_proj(vision_feat.to(text_emb.dtype))  # [1, vis_tokens, hidden_dim]
        mixed   = torch.cat([vis_emb, text_emb[:, self._vis_tok:, :]], dim=1)
        # mixed: [1, seq_len, hidden_dim]

        # 3. 整体前向（Qwen3Model 内部处理 RoPE / causal mask）
        out = self._inner(
            inputs_embeds  = mixed,
            attention_mask = attention_mask,
            position_ids   = position_ids,
            use_cache      = True,
            return_dict    = True,
        )

        hidden = out.last_hidden_state  # [1, seq_len, hidden_dim]
        pkv    = out.past_key_values    # DynamicCache

        # 4. 取最后一个有效 token 的 hidden state
        # attention_mask 的 sum 即有效长度；用 matmul 替代动态 index 以保证静态 shape
        # mask_1hot: [1, seq_len]，有效区段最后一位为 1
        seq_len  = hidden.shape[1]
        cum      = attention_mask[0].to(hidden.dtype).cumsum(0)        # [seq_len]
        valid_n  = cum[-1]                                              # 有效 token 数（标量）
        one_hot  = (cum == valid_n).to(hidden.dtype)                   # [seq_len]
        last_hidden = torch.einsum("bsh,s->bh", hidden, one_hot)       # [1, hidden_dim]

        # 5. 展平 KV cache → [L*2, 1, kv_heads, seq_len, head_dim]
        # transformers ≥4.51: DynamicCache 用 layers[i].keys / .values，不再有 key_cache/value_cache 列表
        k_list: List[torch.Tensor] = [ly.keys   for ly in pkv.layers]  # List[Tensor [1, kv_heads, S, head_dim]]
        v_list: List[torch.Tensor] = [ly.values for ly in pkv.layers]
        k_stack = torch.stack(k_list, dim=0)           # [L, 1, kv_heads, S, head_dim]
        v_stack = torch.stack(v_list, dim=0)
        # 交错排列：k0, v0, k1, v1, ...
        kv_out = torch.stack([k_stack, v_stack], dim=1).reshape(
            2 * k_stack.shape[0], *k_stack.shape[1:]
        )  # [L*2, 1, kv_heads, seq_len, head_dim]

        return last_hidden.to(torch.float16), kv_out.to(torch.float16)


class _LLMPrefillWrapper(nn.Module):
    """
    LLM（纯文本）prefill wrapper。
    无 vision_feat 输入；用 lm_head 直接输出 logits。
    """

    def __init__(self, hf_model):
        super().__init__()
        self._inner   = hf_model.model
        self._lm_head = hf_model.lm_head

    def forward(
        self,
        input_ids:      torch.Tensor,   # [1, seq_len] int32
        attention_mask: torch.Tensor,   # [1, seq_len] int32
        position_ids:   torch.Tensor,   # [1, seq_len] int32
        kv_start_pos:   torch.Tensor,   # [1] int32
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        out = self._inner(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            position_ids   = position_ids,
            use_cache      = True,
            return_dict    = True,
        )

        hidden = out.last_hidden_state   # [1, seq_len, hidden_dim]
        pkv    = out.past_key_values

        # 最后一个有效 token 的 logits（静态 matmul 选位，避免动态 index）
        seq_len = hidden.shape[1]
        cum     = attention_mask[0].to(hidden.dtype).cumsum(0)
        one_hot = (cum == cum[-1]).to(hidden.dtype)                # [seq_len]
        last_h  = torch.einsum("bsh,s->bh", hidden, one_hot)      # [1, hidden_dim]
        logits  = self._lm_head(last_h)                            # [1, vocab_size]

        k_list: List[torch.Tensor] = [ly.keys   for ly in pkv.layers]
        v_list: List[torch.Tensor] = [ly.values for ly in pkv.layers]
        k_stack = torch.stack(k_list, dim=0)
        v_stack = torch.stack(v_list, dim=0)
        kv_out  = torch.stack([k_stack, v_stack], dim=1).reshape(
            2 * k_stack.shape[0], *k_stack.shape[1:]
        )

        return logits.to(torch.float32), kv_out.to(torch.float16)


class _DecodeWrapper(nn.Module):
    """
    单步 AR decode —— 「固定 max buffer + 偏移」设计，全程单图、shape 固定。

    与旧版（按 kv_len 分 bucket）的区别：
      - kv_cache 输入恒为 max_seq_len 长度的完整 buffer，只有 [0, pos) 行有效，
        [pos, max_seq_len) 是垃圾（上一轮残留 / 未初始化）；
      - 用显式 attention_mask 门控哪些 KV 可见，垃圾尾被 mask 掉。

    当前 token 的 k/v 由 DynamicCache.update() 追加在 buffer 末尾，
    物理 index = max_seq_len（即 cache 拼接后长度 = max_seq_len + 1）。
    因此 attention_mask 长度 = max_seq_len + 1，含义：
        [0, pos)            = 1   有效历史
        [pos, max_seq_len)  = 0   垃圾尾（必须屏蔽）
        [max_seq_len]       = 1   当前 token 自身

    causal mask 在这里是「全可见」的（query 的 cache_position = max_seq_len，
    >= 所有 kv index），真正的门控完全交给上面这条 padding mask；
    RoPE 位置由显式 position_id 决定，与 cache_position 解耦。
    前向结束后 keys[i] shape = [1, kv_heads, max_seq_len+1, head_dim]，
    取最后一列即新增的 kv_new。
    """

    def __init__(self, hf_model, num_layers: int):
        super().__init__()
        self._inner    = hf_model.model
        self._lm_head  = hf_model.lm_head
        self._n_layers = num_layers

    def forward(
        self,
        input_id:       torch.Tensor,   # [1, 1] int32
        position_id:    torch.Tensor,   # [1, 1] int32  = 当前序列长度 pos
        attention_mask: torch.Tensor,   # [1, max_seq_len + 1] int32
        kv_cache:       torch.Tensor,   # [L*2, 1, kv_heads, max_seq_len, head_dim] fp16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from transformers import DynamicCache

        # 重建 DynamicCache（transformers ≥4.51 API：用 from_legacy_cache 构造）
        # 每层 K/V 长度 = max_seq_len；update 后拼接当前 token → max_seq_len + 1
        past = [(kv_cache[i * 2], kv_cache[i * 2 + 1]) for i in range(self._n_layers)]
        cache = DynamicCache.from_legacy_cache(past)

        out = self._inner(
            input_ids        = input_id,
            position_ids     = position_id,
            attention_mask   = attention_mask,
            past_key_values  = cache,
            use_cache        = True,
            return_dict      = True,
        )

        hidden  = out.last_hidden_state              # [1, 1, hidden_dim]
        logits  = self._lm_head(hidden[:, -1, :])   # [1, vocab_size]
        pkv_new = out.past_key_values                # DynamicCache, seq len = max_seq_len + 1

        # 只取新增的最后 1 个 token 的 KV
        k_list: List[torch.Tensor] = [ly.keys   for ly in pkv_new.layers]
        v_list: List[torch.Tensor] = [ly.values for ly in pkv_new.layers]
        k_new = torch.stack([k[:, :, -1:, :] for k in k_list], dim=0)  # [L, 1, kv_heads, 1, d]
        v_new = torch.stack([v[:, :, -1:, :] for v in v_list], dim=0)
        kv_new = torch.stack([k_new, v_new], dim=1).reshape(
            2 * self._n_layers, *k_new.shape[1:]
        )  # [L*2, 1, kv_heads, 1, head_dim]

        return logits.to(torch.float32), kv_new.to(torch.float16)


# ---------------------------------------------------------------------------
# 公共导出器
# ---------------------------------------------------------------------------

class Qwen3Exporter(BaseExporter):

    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM
        # attn_implementation="eager" 绕开 transformers ≥4.50 在 SDPA 路径下
        # 用 torch.vmap 构造 causal mask 的代码 —— 老 onnx tracer 无法处理 vmap，
        # 会抛 "RuntimeError: _Map_base::at"。
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype       = torch.float16,
            device_map        = "cpu",
            trust_remote_code = True,
            attn_implementation = "eager",
        ).eval()
        logger.info(
            "[Qwen3] 模型加载完毕  layers=%d  hidden=%d  kv_heads=%d",
            self._model.config.num_hidden_layers,
            self._model.config.hidden_size,
            self._model.config.num_key_value_heads,
        )

    # ------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------

    def export_prefill(self, out_dir: str, seq_len: int) -> str:
        name         = f"prefill_{seq_len}"
        onnx_path    = self._onnx_path(out_dir, name)
        lcfg         = self.cfg.llm
        vcfg         = self.cfg.vision
        is_llm       = self.cfg.mode == "llm"
        num_kv_heads = self._model.config.num_key_value_heads
        # Qwen3 把 head_dim 和 hidden_size 解耦（0.6B: hidden=1024, q_heads=16, head_dim=128），
        # 不能用 hidden_size // num_heads 推导
        head_dim     = getattr(
            self._model.config, "head_dim",
            self._model.config.hidden_size // self._model.config.num_attention_heads,
        )

        if is_llm:
            wrapper = _LLMPrefillWrapper(self._model).eval()
            dummy = {
                "input_ids":      torch.zeros(1, seq_len, dtype=torch.int32),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.int32),
                "position_ids":   torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
                "kv_start_pos":   torch.zeros(1, dtype=torch.int32),
            }
            output_names = ["logits", "kv_out"]
            expected_out = {
                "logits": (1, lcfg.vocab_size),
                "kv_out": (lcfg.num_layers * 2, 1, num_kv_heads, seq_len, head_dim),
            }
            expected_in  = {
                "input_ids":      (1, seq_len),
                "attention_mask": (1, seq_len),
                "position_ids":   (1, seq_len),
                "kv_start_pos":   (1,),
            }
        else:
            wrapper = _PrefillWrapper(
                hf_model   = self._model,
                vis_tokens = vcfg.total_vision_tokens,
                feat_dim   = vcfg.feat_dim,
            ).eval()
            dummy = {
                "input_ids":      torch.zeros(1, seq_len, dtype=torch.int32),
                "vision_feat":    torch.zeros(1, vcfg.total_vision_tokens, vcfg.feat_dim,
                                              dtype=torch.float16),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.int32),
                "position_ids":   torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
                "kv_start_pos":   torch.zeros(1, dtype=torch.int32),
            }
            output_names = ["last_hidden", "kv_out"]
            expected_out = {
                "last_hidden": (1, lcfg.hidden_dim),
                "kv_out":      (lcfg.num_layers * 2, 1, num_kv_heads, seq_len, head_dim),
            }
            expected_in  = {
                "input_ids":      (1, seq_len),
                "vision_feat":    (1, vcfg.total_vision_tokens, vcfg.feat_dim),
                "attention_mask": (1, seq_len),
                "position_ids":   (1, seq_len),
                "kv_start_pos":   (1,),
            }

        _export_onnx(
            wrapper      = wrapper,
            dummy_inputs = tuple(dummy.values()),
            onnx_path    = onnx_path,
            input_names  = list(dummy.keys()),
            output_names = output_names,
            dynamic_axes = None,
        )
        _verify_io_shapes(onnx_path, expected_in, expected_out)
        return onnx_path

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def export_decode(self, out_dir: str) -> str:
        name      = "decode"
        onnx_path = self._onnx_path(out_dir, name)
        cfg       = self.cfg
        lcfg      = cfg.llm
        max_seq   = lcfg.max_seq_len

        num_kv_heads = self._model.config.num_key_value_heads
        head_dim     = getattr(
            self._model.config, "head_dim",
            self._model.config.hidden_size // self._model.config.num_attention_heads,
        )

        wrapper = _DecodeWrapper(
            hf_model   = self._model,
            num_layers = self._model.config.num_hidden_layers,
        ).eval()

        # dummy 的 mask/position 值不影响图拓扑（都是运行期数据），
        # 取一个合法形态即可：pos=0，仅当前 token（末位）可见。
        dummy_mask = torch.zeros(1, max_seq + 1, dtype=torch.int32)
        dummy_mask[0, max_seq] = 1
        dummy = {
            "input_id":       torch.zeros(1, 1, dtype=torch.int32),
            "position_id":    torch.zeros(1, 1, dtype=torch.int32),
            "attention_mask": dummy_mask,
            "kv_cache":       torch.zeros(
                                  lcfg.num_layers * 2, 1, num_kv_heads, max_seq, head_dim,
                                  dtype=torch.float16
                              ),
        }

        input_names  = list(dummy.keys())
        output_names = ["logits", "kv_new"]

        _export_onnx(
            wrapper      = wrapper,
            dummy_inputs = tuple(dummy.values()),
            onnx_path    = onnx_path,
            input_names  = input_names,
            output_names = output_names,
            dynamic_axes = None,
        )
        _verify_io_shapes(
            onnx_path,
            expected_in  = {
                "input_id":       (1, 1),
                "position_id":    (1, 1),
                "attention_mask": (1, max_seq + 1),
                "kv_cache":       (lcfg.num_layers * 2, 1, num_kv_heads, max_seq, head_dim),
            },
            expected_out = {
                "logits":  (1, lcfg.vocab_size),
                "kv_new":  (lcfg.num_layers * 2, 1, num_kv_heads, 1, head_dim),
            },
        )
        return onnx_path


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_mask_vmap_for_tracing():
    """
    transformers ≥4.50 在 create_causal_mask 里用两条不兼容 onnx tracer 的机制：
      1. torch.vmap 构造 4D mask；
      2. `TransformGetItemToIndex` 上下文管理器把 `tensor[i, j]` hook 成
         `_trace_wrapped_higher_order_op`（dynamo HOP），用于 padding mask 索引。

    老 onnx tracer dispatch 不了 vmap / HOP，C++ 层抛 "_Map_base::at"。

    解决方案：
      - 强制走 `sdpa_mask_older_torch` 路径：它不进 TransformGetItemToIndex
        上下文，padding mask 用普通张量外乘合并；
      - 把 `_vmap_for_bhqkv` 替换成纯 broadcasting 等价版本（同时支持
        bh_indices=True/False 两种调用形态）。
    """
    try:
        import transformers.masking_utils as mu
    except ImportError:
        yield
        return

    saved = {}

    if hasattr(mu, "sdpa_mask") and hasattr(mu, "sdpa_mask_older_torch"):
        saved["sdpa_mask"] = mu.sdpa_mask
        mu.sdpa_mask = mu.sdpa_mask_older_torch

    if hasattr(mu, "_vmap_for_bhqkv"):
        saved["_vmap_for_bhqkv"] = mu._vmap_for_bhqkv

        def _broadcast_for_bhqkv(mask_function, bh_indices=True):
            def fn(batch_arange, head_arange, q_arange, kv_arange):
                if bh_indices:
                    b = batch_arange.view(-1, 1, 1, 1)
                    h = head_arange.view(1, -1, 1, 1)
                    q = q_arange.view(1, 1, -1, 1)
                    k = kv_arange.view(1, 1, 1, -1)
                else:
                    # 旧路径：不在 b/h 维度上 vmap，只返回 [Q, K]
                    b = None
                    h = None
                    q = q_arange.view(-1, 1)
                    k = kv_arange.view(1, -1)
                return mask_function(b, h, q, k)
            return fn

        mu._vmap_for_bhqkv = _broadcast_for_bhqkv

    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(mu, name, value)


def _export_onnx(
    wrapper,
    dummy_inputs,
    onnx_path:    str,
    input_names:  list,
    output_names: list,
    dynamic_axes,
) -> None:
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), _patch_mask_vmap_for_tracing():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            onnx_path,
            input_names         = input_names,
            output_names        = output_names,
            dynamic_axes        = dynamic_axes,
            opset_version       = 17,
            do_constant_folding = True,
            dynamo              = False,
        )


def _verify_io_shapes(onnx_path: str, expected_in: dict, expected_out: dict) -> None:
    """用 onnx.checker + shape_inference 校验 I/O shape 与预期一致。"""
    try:
        import onnx
        from onnx import shape_inference
    except ImportError:
        logger.warning("[verify] onnx 未安装，跳过 shape 校验")
        return

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)

    # 收集所有值的 shape（input + value_info + output）
    shape_map: dict = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = tuple(
            d.dim_value for d in vi.type.tensor_type.shape.dim
        )
        shape_map[vi.name] = dims

    all_ok = True
    for name, expected in {**expected_in, **expected_out}.items():
        actual = shape_map.get(name)
        if actual != expected:
            logger.warning("[verify] %s  expected=%s  actual=%s", name, expected, actual)
            all_ok = False

    if all_ok:
        logger.info("[verify] %s  shape 校验通过", Path(onnx_path).name)
    else:
        logger.warning("[verify] %s  shape 校验存在差异，请检查配置", Path(onnx_path).name)
