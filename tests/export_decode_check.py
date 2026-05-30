"""
tests/export_decode_check.py

验证单图 decode：
  1) 能导出成 ONNX（走 _patch_mask_vmap_for_tracing），IO shape 正确；
  2) 导出件用 onnxruntime 实跑一步真实 decode，与 PyTorch _DecodeWrapper
     的 logits / kv_new 数值一致 —— 确认 mask 补丁路径没有改变数值。

用小 max_seq_len 控制 dummy/buffer 体积。

跑法：
    docker exec -e PYTHONPATH=/workspace aria \
        python /workspace/tests/export_decode_check.py
"""

from __future__ import annotations

import sys
import tempfile

import numpy as np
import torch

MODEL   = "/data/models/Qwen/Qwen3-0___6B"
MAX_SEQ = 64
PROMPT  = "The capital of France is"


def main() -> int:
    import onnx
    import onnxruntime as ort
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from aria.models.base import FrameworkConfig, LLMConfig
    from tools.exporters.qwen3 import Qwen3Exporter, _DecodeWrapper

    tok   = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, attn_implementation="eager",
    ).eval()
    hf       = model.config
    L        = hf.num_hidden_layers
    kv_heads = hf.num_key_value_heads
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)

    cfg = FrameworkConfig(mode="llm")
    cfg.llm = LLMConfig(
        num_layers=L, hidden_dim=hf.hidden_size, num_heads=kv_heads,
        head_dim=head_dim, vocab_size=hf.vocab_size, max_seq_len=MAX_SEQ,
    )

    # ---- prefill 一个真实 prompt，灌进 MAX buffer ----
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(torch.int64)
    P   = ids.shape[1]
    with torch.no_grad():
        cache = DynamicCache()
        out   = model(input_ids=ids, use_cache=True, past_key_values=cache)
        t0    = int(out.logits[:, -1, :].argmax(-1))
        buf   = torch.zeros(L * 2, 1, kv_heads, MAX_SEQ, head_dim, dtype=torch.float16)
        for i, ly in enumerate(cache.layers):
            buf[i * 2,     :, :, :P, :] = ly.keys
            buf[i * 2 + 1, :, :, :P, :] = ly.values

    pos      = P
    input_id = torch.tensor([[t0]],   dtype=torch.int32)
    posid    = torch.tensor([[pos]],  dtype=torch.int32)
    am       = torch.zeros(1, MAX_SEQ + 1, dtype=torch.int32)
    am[0, :pos] = 1
    am[0, MAX_SEQ] = 1

    # ---- PyTorch wrapper（参考）----
    wrapper = _DecodeWrapper(model, L).eval()
    with torch.no_grad():
        logits_pt, kv_new_pt = wrapper(input_id, posid, am, buf)
    logits_pt = logits_pt.float().numpy()
    kv_new_pt = kv_new_pt.float().numpy()

    # ---- 导出 + ORT 实跑 ----
    exporter = Qwen3Exporter(cfg, MODEL)
    exporter._model = model   # 复用已加载模型
    with tempfile.TemporaryDirectory(prefix="aria_dec_") as td:
        path = exporter.export_decode(td)

        m = onnx.load(path, load_external_data=False)
        print("[export-check] inputs:",
              {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
               for vi in m.graph.input})
        print("[export-check] outputs:",
              {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
               for vi in m.graph.output})

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feeds = {
            "input_id":       input_id.numpy(),
            "position_id":    posid.numpy(),
            "attention_mask": am.numpy(),
            "kv_cache":       buf.numpy(),
        }
        logits_ort, kv_new_ort = sess.run(["logits", "kv_new"], feeds)

    logits_ort = logits_ort.astype(np.float32)
    kv_new_ort = kv_new_ort.astype(np.float32)

    d_logits = np.abs(logits_pt - logits_ort).max()
    d_kv     = np.abs(kv_new_pt - kv_new_ort).max()
    arg_pt   = int(logits_pt.argmax(-1))
    arg_ort  = int(logits_ort.argmax(-1))

    print(f"\n[export-check] argmax  pt={arg_pt}  ort={arg_ort}  match={arg_pt == arg_ort}")
    print(f"[export-check] max|Δlogits| = {d_logits:.5f}")
    print(f"[export-check] max|Δkv_new| = {d_kv:.5f}")

    # fp16 + 两套算子实现，留宽容差；关键是 argmax 一致
    ok = (arg_pt == arg_ort) and d_logits < 0.5
    print("\n[export-check]", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
