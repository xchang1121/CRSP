import os, sys, time, torch
sys.path.insert(0, "/content/cspd")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from cspd_core import (
    get_qwen_template_tokens,
    init_soft_prompt_from_vocab,
    build_soft_prompt_inputs,
    compute_response_logprobs,
    length_normalized_nll_loss,
    MuonSoftPrompt,
)
from benchmarks import batch_generate_with_soft_prompts, SUPPORTED_IFEVAL_IDS

t0 = time.time()
MODEL_ID = "Jackwang111/M2RL-SFT"
print(f"Loading tokenizer & model {MODEL_ID}...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"Loaded in {time.time() - t0:.1f}s | VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 1. Test soft prompt forward + backward + Muon step with gradient checkpointing
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
tmpl = get_qwen_template_tokens(tok)
sp = init_soft_prompt_from_vocab(model.get_input_embeddings().weight, num_vectors=64, seed=42)
opt = MuonSoftPrompt([sp], lr=0.03)

q_ids = tok.encode("What is 17 * 23?", add_special_tokens=False)
r_ids = tok.encode("<think>\n17*20=340, 17*3=51, 340+51=391.\n</think>\n\\boxed{391}<|im_end|>\n", add_special_tokens=False)
emb, labels, spans = build_soft_prompt_inputs(
    model.get_input_embeddings(), tmpl, [sp], q_ids, r_ids, device=torch.device("cuda"), model_dtype=torch.bfloat16
)
out = model(inputs_embeds=emb)
loss = length_normalized_nll_loss(out.logits, labels)
loss.backward()
gnorm = sp.grad.norm().item()
opt.step()
opt.zero_grad()
print(f"Train sanity OK | loss={loss.item():.4f} | grad_norm={gnorm:.4f} | spans={spans}")

# 2. Test 4D attention pathway masking (Eq. 6)
model.gradient_checkpointing_disable()
for mode in ["full", "response_only", "input_only", "blocked"]:
    lp, _ = compute_response_logprobs(
        model, model.get_input_embeddings(), tmpl, [sp.detach()], q_ids, r_ids, attention_mode=mode
    )
    print(f"  attn_mode={mode:14s} | mean_logp={lp.max(dim=-1).values.mean().item():.4f}")

# 3. Test left-padded batched generation vs batch=1 greedy match
qs = ["What is 12 + 35? Give only the final number.", "Compute 9 * 8 and state the exact answer briefly."]
g_b1 = [batch_generate_with_soft_prompts(model, tok, tmpl, [q], [sp.detach()], batch_size=1, max_new_tokens=64, temperature=0.0, enable_thinking=False)[0] for q in qs]
g_b2 = batch_generate_with_soft_prompts(model, tok, tmpl, qs, [sp.detach()], batch_size=2, max_new_tokens=64, temperature=0.0, enable_thinking=False)
print("Batch=1 vs Batch=2 match:", [a == b for a, b in zip(g_b1, g_b2)])
for i in range(2):
    print(f"  Q{i} b1={g_b1[i][:80]!r} | b2={g_b2[i][:80]!r}")

# 4. Check public HF datasets availability
print("\n=== Checking HF datasets ===")
for name, split in [
    ("HuggingFaceH4/MATH-500", "test"),
    ("google/IFEval", "train"),
    ("DigitalLearningGmbH/MATH-lighteval", "train"),
    ("Maxwell-Jia/AIME_2024", "train"),
]:
    try:
        ds = load_dataset(name, split=split)
        print(f"  OK: {name} [{split}] -> {len(ds)} rows | cols={ds.column_names}")
    except Exception as e:
        print(f"  FAIL: {name} [{split}] -> {type(e).__name__}: {str(e)[:120]}")
