"""
backends/torch/executor.py

PyTorch 参考执行器。所有 graph 都在同一组 nn.Module 上 forward：
  - prefill_{N}  / decode（单图，kv_cache 常驻 bind） → TinyLLM
  - vision_encoder              → TinyVisionEncoder
  - flow_head                   → TinyFlowHead

特殊约定：
  - prefill_* 接受可选的 past_kv / past_kv_len 两个 input。
    GraphMeta 里不强制声明，调用方决定是否塞进 inputs。
    其它后端（Mock/TRT/ORT）见到也会照常走 _h2d → 没人读 → _free，无副作用。
  - 输入/输出 numpy 都先按 dtype 上传到 self._mem 里的 torch.Tensor，
    内部计算统一 float32，最后 _d2h 时再 astype 回去。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Torch backend 需要 torch，请 `pip install -e .[torch]`"
    ) from e

from aria.core.executor import NPUExecutor, GraphMeta
from aria.models.base    import FrameworkConfig
from aria.backends.torch.model import (
    TinyLLM, TinyVisionEncoder, TinyFlowHead,
)

logger = logging.getLogger(__name__)


class TorchExecutor(NPUExecutor):
    """
    构造时需要 FrameworkConfig；模块尺寸（hidden/num_heads/head_dim/feat_dim 等）
    都按 config 实例化。后续 register_graph 只是登记 meta，不改变模型。
    """

    def __init__(self,
                 config: FrameworkConfig,
                 seed:   int = 42,
                 device: str = "cpu"):
        super().__init__()
        torch.manual_seed(seed)
        self._cfg    = config
        self._device = torch.device(device)
        self._llm        = TinyLLM(config).to(self._device).eval()
        self._vision_enc = TinyVisionEncoder(config).to(self._device).eval()
        self._flow_head  = TinyFlowHead(config).to(self._device).eval()

        # 模拟 device 内存：addr → torch.Tensor
        self._mem: Dict[int, torch.Tensor] = {}
        self._next_addr = 0x80000000

        n_params = sum(p.numel() for p in self._llm.parameters())
        logger.info(
            f"[ARIA/Torch] 初始化 device={device} "
            f"TinyLLM={n_params/1e6:.2f}M params"
        )

    # ------------------------------------------------------------------
    # NPUExecutor 抽象钩子
    # ------------------------------------------------------------------

    def _load_graph(self, path: str, meta: GraphMeta) -> Any:
        logger.debug(f"[ARIA/Torch] 登记图: {meta.name} (no-op)")
        return meta

    @torch.no_grad()
    def _execute(self, handle, device_inputs, meta):
        name = meta.name
        if name == "vision_encoder":
            return self._exec_vision(device_inputs, meta)
        if name.startswith("prefill_"):
            return self._exec_prefill(device_inputs, meta)
        if name == "decode":
            return self._exec_decode(device_inputs, meta)
        if name == "flow_head":
            return self._exec_flow(device_inputs, meta)
        raise ValueError(f"[ARIA/Torch] 未知 graph: {name}")

    def _alloc_device(self, size: int) -> int:
        addr = self._next_addr
        self._next_addr += size + 64
        return addr

    def _h2d(self, data: np.ndarray, addr: int) -> None:
        # 上传时保持 numpy dtype 对应的 torch dtype，避免 fp16→fp32 静默放大。
        # 计算时各 _exec_* 自己做 .float()。
        self._mem[addr] = torch.from_numpy(np.ascontiguousarray(data)).to(self._device)

    def _d2h(self, addr: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
        t = self._mem.get(addr)
        if t is None:
            return np.zeros(shape, dtype=dtype)
        arr = t.detach().cpu().numpy()
        if arr.size == int(np.prod(shape)):
            arr = arr.reshape(shape)
        return arr.astype(dtype)

    def _free_device(self, addr: int) -> None:
        if addr in self._persistent_addrs:
            return
        self._mem.pop(addr, None)

    def _write_kv_seq(self, addr, buffer_shape, dtype, start, block, plane0=0) -> None:
        buf = self._mem.get(addr)
        if buf is None or tuple(buf.shape) != tuple(buffer_shape):
            buf = torch.from_numpy(
                np.zeros(buffer_shape, dtype=dtype)
            ).to(self._device)
            self._mem[addr] = buf
        blk = torch.from_numpy(np.ascontiguousarray(block)).to(buf)
        p = blk.shape[0]
        n = blk.shape[3]
        buf[plane0:plane0 + p, :, :, start:start + n, :] = blk

    # ------------------------------------------------------------------
    # 各 graph 实现
    # ------------------------------------------------------------------

    def _exec_vision(self, device_inputs, meta):
        tiles = self._mem[device_inputs["tiles"]].float()
        vf    = self._vision_enc(tiles)
        return self._emit("vision_feat", vf, meta)

    def _exec_prefill(self, device_inputs, meta):
        input_ids      = self._mem[device_inputs["input_ids"]].long()
        vision_feat    = self._mem[device_inputs["vision_feat"]].float()
        attention_mask = self._mem[device_inputs["attention_mask"]].long()
        position_ids   = self._mem[device_inputs["position_ids"]].long()

        # 可选输入：past_kv / past_kv_len（约定见模块 docstring）
        past_kv = None
        if "past_kv" in device_inputs:
            pk = self._mem[device_inputs["past_kv"]].float()
            pkl = (int(self._mem[device_inputs["past_kv_len"]][0].item())
                   if "past_kv_len" in device_inputs else pk.shape[-2])
            past_kv = pk[..., :pkl, :].contiguous() if pkl > 0 else None

        # 注意：llm_backbone 的 padded_ids 实际长度是 (bucket - vision_tokens)，
        # 跟 GraphMeta 里 input_ids 声明的 (1, bucket) 名义上不一致。
        # 这里以 meta 里 kv_out 的 seq 维为权威 bucket，input_ids 只读真实文本部分。
        B          = input_ids.shape[0]
        bucket     = int(meta.output_shapes["kv_out"][3])
        V          = vision_feat.shape[1]
        actual_len = int(attention_mask[0].sum().item())
        text_len   = max(0, actual_len - V)

        vision_proj = self._llm.vision_proj(vision_feat)             # [B, V, H]
        text_embeds = self._llm.token_emb(input_ids[:, :text_len])   # [B, T, H]
        x = torch.cat([vision_proj, text_embeds], dim=1)             # [B, actual_len, H]

        pos = position_ids[:, :actual_len].clamp(min=0)
        x = x + self._llm.pos_emb(pos)

        new_kvs = []
        for i, layer in enumerate(self._llm.layers):
            past_k = past_kv[i, 0] if past_kv is not None else None
            past_v = past_kv[i, 1] if past_kv is not None else None
            x, k, v = layer(x, past_k, past_v)
            new_kvs.append(torch.stack([k, v], dim=0))

        x = self._llm.norm_out(x)
        last_hidden = x[:, -1, :]   # [B, H]

        L  = self._cfg.llm.num_layers
        NH = self._cfg.llm.num_heads
        D  = self._cfg.llm.head_dim
        new_kv = torch.stack(new_kvs, dim=0)             # [L, 2, B, NH, actual_len, D]
        kv_out = torch.zeros(L * 2, B, NH, bucket, D,
                              dtype=new_kv.dtype, device=self._device)
        kv_out[:, :, :, :actual_len, :] = new_kv.view(L * 2, B, NH, actual_len, D)

        out = {}
        out.update(self._emit("last_hidden", last_hidden, meta))
        out.update(self._emit("kv_out",      kv_out,      meta))
        return out

    def _exec_decode(self, device_inputs, meta):
        input_id    = self._mem[device_inputs["input_id"]].long()
        position_id = self._mem[device_inputs["position_id"]].long()
        # kv_cache 是常驻 buffer（bind 输入），全长 max_seq_len，只有 [0,cur_pos) 有效
        kv_in       = self._mem[device_inputs["kv_cache"]].float()

        B = input_id.shape[0]
        L  = self._cfg.llm.num_layers
        NH = self._cfg.llm.num_heads
        D  = self._cfg.llm.head_dim
        kv_bucket = kv_in.shape[3]
        kv_in_5d  = kv_in.view(L, 2, B, NH, kv_bucket, D)

        cur_pos = int(position_id[0, 0].item())
        past    = (kv_in_5d[..., :cur_pos, :].contiguous()
                   if cur_pos > 0 else None)

        x = self._llm.token_emb(input_id)                 # [B, 1, H]
        x = x + self._llm.pos_emb(position_id.clamp(min=0))

        new_kvs = []
        for i, layer in enumerate(self._llm.layers):
            past_k = past[i, 0] if past is not None else None
            past_v = past[i, 1] if past is not None else None
            x, k, v = layer(x, past_k, past_v)
            new_kvs.append(torch.stack([k, v], dim=0))

        x      = self._llm.norm_out(x)
        logits = self._llm.lm_head(x[:, 0, :])             # [B, vocab]
        new_kv = torch.stack(new_kvs, dim=0).view(L * 2, B, NH, 1, D)

        out = {}
        out.update(self._emit("logits", logits, meta))
        out.update(self._emit("kv_new", new_kv, meta))
        return out

    def _exec_flow(self, device_inputs, meta):
        hs = self._mem[device_inputs["hidden_state"]].float()
        na = self._mem[device_inputs["noisy_action"]].float()
        ts = self._mem[device_inputs["timestep"]].float()
        v  = self._flow_head(hs, na, ts)
        return self._emit("velocity", v, meta)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _emit(self,
              name:   str,
              tensor: torch.Tensor,
              meta:   GraphMeta) -> Dict[str, int]:
        """把一个输出 tensor 写到模拟 device 内存，返回 {name: addr}。"""
        t = tensor.detach().contiguous()
        addr = self._alloc_device(t.element_size() * t.numel())
        self._mem[addr] = t
        return {name: addr}
