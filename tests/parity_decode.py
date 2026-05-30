"""
tests/parity_decode.py

数值对拍：验证「固定 max buffer + 偏移 + 显式 mask」的单图 decode 与
HF 原生带 cache 的逐步 decode 在 logits 上一致。

这是方案 B 的数值闸门——这关不过，后面所有改动都白搭。

跑法（在 aria 容器内）：
    docker exec -e PYTHONPATH=/workspace aria \
        python /workspace/tests/parity_decode.py
"""

from __future__ import annotations

import sys

import torch

MODEL   = "/data/models/Qwen/Qwen3-0___6B"
MAX_SEQ = 64      # decode KV buffer 长度（测试用小值）
N_STEPS = 8       # 对拍的 decode 步数
PROMPT  = "The capital of France is"

# logits 容差（fp16 路径，两条代码路径理论上应当逐元素相同，留点裕量）
ATOL = 5e-2


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from tools.exporters.qwen3 import _DecodeWrapper

    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype         = torch.float16,
        device_map          = "cpu",
        trust_remote_code   = True,
        attn_implementation = "eager",
    ).eval()

    cfg      = model.config
    L        = cfg.num_hidden_layers
    kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"[parity] L={L} kv_heads={kv_heads} head_dim={head_dim} MAX_SEQ={MAX_SEQ}")

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(torch.int64)   # [1, P]
    P   = ids.shape[1]
    assert P + N_STEPS < MAX_SEQ, f"P({P})+N_STEPS({N_STEPS}) 必须 < MAX_SEQ({MAX_SEQ})"
    print(f"[parity] prompt 长度 P={P}")

    # ------------------------------------------------------------------
    # 参考：HF 原生带 cache 的贪心 decode
    #   ref_tokens[k] = 第 k 步喂进去的 token
    #   ref_logits[k] = 喂进 ref_tokens[k] 之后预测的 logits
    # ------------------------------------------------------------------
    ref_tokens: list = []
    ref_logits: list = []
    with torch.no_grad():
        cache = DynamicCache()
        out   = model(input_ids=ids, use_cache=True, past_key_values=cache)
        nxt   = out.logits[:, -1, :].float()             # prefill 后预测的 next token
        for _ in range(N_STEPS):
            t = int(nxt.argmax(-1))
            ref_tokens.append(t)
            out = model(input_ids=torch.tensor([[t]]), use_cache=True, past_key_values=cache)
            nxt = out.logits[:, -1, :].float()
            ref_logits.append(nxt)

    # ------------------------------------------------------------------
    # 新路径：prefill KV 灌进 MAX buffer，逐步用 _DecodeWrapper + mask decode
    # ------------------------------------------------------------------
    with torch.no_grad():
        cache2 = DynamicCache()
        model(input_ids=ids, use_cache=True, past_key_values=cache2)
        buf = torch.zeros(L * 2, 1, kv_heads, MAX_SEQ, head_dim, dtype=torch.float16)
        for i, ly in enumerate(cache2.layers):
            buf[i * 2,     :, :, :P, :] = ly.keys      # [1, kv_heads, P, d]
            buf[i * 2 + 1, :, :, :P, :] = ly.values

    wrapper = _DecodeWrapper(model, L).eval()

    new_logits: list = []
    pos = P
    with torch.no_grad():
        for step in range(N_STEPS):
            t           = ref_tokens[step]
            input_id    = torch.tensor([[t]],   dtype=torch.int32)
            position_id = torch.tensor([[pos]], dtype=torch.int32)
            am          = torch.zeros(1, MAX_SEQ + 1, dtype=torch.int32)
            am[0, :pos]    = 1     # 有效历史 [0, pos)
            am[0, MAX_SEQ] = 1     # 当前 token 自身（拼接在末尾）
            logits, kv_new = wrapper(input_id, position_id, am, buf)
            new_logits.append(logits.float())
            # 写回：buffer 第 pos 行 = 新 token 的 KV（跨步赋值）
            buf[:, :, :, pos, :] = kv_new[:, :, :, 0, :]
            pos += 1

    # ------------------------------------------------------------------
    # 比较
    # ------------------------------------------------------------------
    print("\nstep | argmax_match | max_abs_diff | ref_tok -> ref_next | new_next")
    ok = True
    for step in range(N_STEPS):
        a, b      = ref_logits[step], new_logits[step]
        diff      = (a - b).abs().max().item()
        a_arg     = int(a.argmax(-1))
        b_arg     = int(b.argmax(-1))
        match     = a_arg == b_arg
        ok        = ok and match and (diff <= ATOL)
        print(f"  {step:2d} |     {str(match):5s}    | {diff:11.5f}  | "
              f"{ref_tokens[step]:6d} -> {a_arg:6d}      | {b_arg:6d}")

    print("\n[parity]", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
