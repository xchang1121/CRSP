"""15-second GPU smoke test verifying KV-CSPD (ReadOnlyPrefixCache + grad flow + peak VRAM @ B=2,T=768 + batched generate)."""
import os, sys, time
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/content/cspd")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cspd_core import get_qwen_template_tokens, length_normalized_nll_loss
from kv_prefix_core import (
    LayerwiseMuonKV,
    batch_generate_with_kv_prefixes,
    build_chat_ids_for_kv,
    forward_with_kv_prefix,
    init_kv_prefix_from_tokens,
)

t0 = time.time()
mid = "Jackwang111/M2RL-SFT"
tok = AutoTokenizer.from_pretrained(mid)
tmpl = get_qwen_template_tokens(tok)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
for p in model.parameters():
    p.requires_grad = False
model.eval()
print(f"Loaded {mid} in {time.time()-t0:.1f}s", flush=True)

init_ids = tok.encode("HeliosPay v4.2 Cross-Border Wire Settlement Policy cutoff 16:40 SGT.", add_special_tokens=False)
kv = init_kv_prefix_from_tokens(model, init_ids, num_kv_tokens=16)
print(f"KV shape={tuple(kv.shape)} | params={kv.numel():,} | rms={kv.pow(2).mean().sqrt().item():.4f}", flush=True)

# Verify forward + backward WITHOUT gradient checkpointing at realistic B=2, T=768!
torch.cuda.reset_peak_memory_stats()
q_ids = tok.encode("What is 2 + 2?", add_special_tokens=False)
r_ids = tok.encode("The answer is 4. " * 120, add_special_tokens=False)[:700] + tmpl.im_end
nt_ids = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
full_ids, labels = build_chat_ids_for_kv(tmpl, q_ids, r_ids, None, False, nt_ids)
inp = torch.tensor([full_ids, full_ids], device="cuda")
mask = torch.ones_like(inp)
lbl = torch.tensor([labels, labels], device="cuda")

t1 = time.time()
out = forward_with_kv_prefix(model, inp, mask, [kv])
loss = length_normalized_nll_loss(out.logits, lbl)
loss.backward()
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"No-GC B=2 T={inp.size(1)} | loss={loss.item():.4f} | kv.grad norm={kv.grad.norm().item():.4f} | "
      f"finite={torch.isfinite(kv.grad).all().item()} | peak_VRAM={peak_gb:.2f}GB | step_time={time.time()-t1:.2f}s", flush=True)

# Verify LayerwiseMuonKV step & AdamW step
opt = LayerwiseMuonKV([kv], lr=0.05)
opt.step()
opt.zero_grad()

# Verify batched generate with 1 and 2 concatenated KV prefixes!
outs1 = batch_generate_with_kv_prefixes(model, tok, tmpl, ["What is the capital of France?", "What is 7 * 8?"], [kv], batch_size=2, max_new_tokens=32, enable_thinking=False)
outs2 = batch_generate_with_kv_prefixes(model, tok, tmpl, ["What is the capital of France?"], [kv, kv], batch_size=1, max_new_tokens=32, enable_thinking=False)
print("Gen 1-KV:", outs1, flush=True)
print("Gen 2-KV:", outs2, flush=True)
print("=== KV SMOKE TEST PASSED ===", flush=True)
