"""
Fast 90-second diagnostic on Colab A100:
1. Check M2RL-RL_IF vs M2RL-SFT truncation & KL(T_IF || S) vs KL(T_Math || S).
2. Measure held-out G(A) across training steps (0, 2, 4, 6, 10, 15, 25) to pinpoint
   exact overfitting transition on small vs larger trace sets, and verify Attention Pathway K(A).
"""
import gc, json, math, os, random, sys, time
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/cspd")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from cspd_core import (
    MuonSoftPrompt,
    build_soft_prompt_inputs,
    compute_response_logprobs,
    get_qwen_template_tokens,
    init_soft_prompt_from_vocab,
    length_normalized_nll_loss,
    mean_kl_divergence,
    pad_and_stack_train_batch,
    sample_random_prefix,
    should_add_prefix,
)

STUDENT_ID = "Jackwang111/M2RL-SFT"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"

t_data = json.load(open("/content/cspd/exp1_results/teacher_data.json"))
print(f"Cached teacher_data: math_tr={len(t_data['math_train_verified'])}, "
      f"if_tr={len(t_data['if_train_verified'])}, mech_math={len(t_data['mech_math_traces'])}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
tmpl = get_qwen_template_tokens(tokenizer)

# Check token lengths of cached verified traces
m_lens = [len(tokenizer.encode(x["response"], add_special_tokens=False)) for x in t_data["math_train_verified"]]
i_lens = [len(tokenizer.encode(x["response"], add_special_tokens=False)) for x in t_data["if_train_verified"]]
print(f"Math verified tok lens: min={min(m_lens)}, med={sorted(m_lens)[len(m_lens)//2]}, max={max(m_lens)}", flush=True)
print(f"IF   verified tok lens: min={min(i_lens)}, med={sorted(i_lens)[len(i_lens)//2]}, max={max(i_lens)}", flush=True)

# Check how chat template normally formats a user message WITHOUT system prompt
sample_no_sys = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True
)
sample_with_sys = tokenizer.apply_chat_template(
    [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True
)
print("Qwen3 template (no sys):", repr(sample_no_sys), flush=True)
print("Qwen3 template (with sys):", repr(sample_with_sys), flush=True)

# Load Student & IF Teacher to measure KL(T_IF || S) and test step-by-step G(A)!
student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
student.eval()
for p in student.parameters():
    p.requires_grad = False
s_embed = student.get_input_embeddings()

# Split IF verified traces: 35 train, 14 held-out val!
if_all = t_data["if_train_verified"]
if_val = if_all[:14]
if_tr = if_all[14:]

t_if = AutoModelForCausalLM.from_pretrained(IF_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
t_if.eval()
t_if_embed = t_if.get_input_embeddings()

# Precompute teacher logprobs on 14 held-out IF traces AND baseline student KL
val_cache = []
kl_s_list = []
with torch.no_grad():
    for tr in if_val:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:1024]
        t_lp, _ = compute_response_logprobs(t_if, t_if_embed, tmpl, [], q_ids, r_ids)
        s_lp, _ = compute_response_logprobs(student, s_embed, tmpl, [], q_ids, r_ids)
        kl_s_list.append(mean_kl_divergence(t_lp, s_lp))
        val_cache.append((q_ids, r_ids, t_lp.cpu()))

kl_S_if = sum(kl_s_list) / len(kl_s_list)
print(f"\n[IF] Held-out baseline KL(T_IF || S) = {kl_S_if:.5f} (over {len(val_cache)} traces)", flush=True)

# Also check WITH no empty system block on T_IF and S!
def raw_no_sys_logprobs(model, q_ids, r_ids):
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    pre = enc("<|im_start|>user\n") + q_ids + enc("<|im_end|>\n<|im_start|>assistant\n")
    full = pre + r_ids
    ids = torch.tensor([full], device="cuda")
    out = model(input_ids=ids)
    logits = out.logits[0, len(pre)-1 : len(full)-1, :].float()
    return F.log_softmax(logits, dim=-1)

kl_nosys_list = []
with torch.no_grad():
    for tr in if_val:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:1024]
        t_lp = raw_no_sys_logprobs(t_if, q_ids, r_ids)
        s_lp = raw_no_sys_logprobs(student, q_ids, r_ids)
        kl_nosys_list.append(mean_kl_divergence(t_lp, s_lp))
print(f"[IF] Baseline KL(T_IF || S) WITHOUT empty system block = {sum(kl_nosys_list)/len(kl_nosys_list):.5f}", flush=True)

# Also check Teacher-vs-Student KL on the 11 held-out Math traces!
del t_if
gc.collect()
torch.cuda.empty_cache()

t_math = AutoModelForCausalLM.from_pretrained(MATH_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
t_math.eval()
t_m_embed = t_math.get_input_embeddings()
m_val_cache = []
m_kl_s, m_kl_nosys = [], []
with torch.no_grad():
    for tr in t_data["mech_math_traces"][:10]:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:1024]
        t_lp, _ = compute_response_logprobs(t_math, t_m_embed, tmpl, [], q_ids, r_ids)
        s_lp, _ = compute_response_logprobs(student, s_embed, tmpl, [], q_ids, r_ids)
        m_kl_s.append(mean_kl_divergence(t_lp, s_lp))
        m_val_cache.append((q_ids, r_ids, t_lp.cpu()))
        t_lp2 = raw_no_sys_logprobs(t_math, q_ids, r_ids)
        s_lp2 = raw_no_sys_logprobs(student, q_ids, r_ids)
        m_kl_nosys.append(mean_kl_divergence(t_lp2, s_lp2))
print(f"[Math] Held-out KL(T_Math || S): empty_sys={sum(m_kl_s)/len(m_kl_s):.5f} | no_sys={sum(m_kl_nosys)/len(m_kl_nosys):.5f}", flush=True)
del t_math
gc.collect()
torch.cuda.empty_cache()

# Now run a quick 20-step training of P_if (L=256, Muon lr=0.015 and lr=0.03) evaluating held-out G(A) at steps 0,2,4,6,8,12,16,20!
def eval_val_G(sp_tensor, cache, kl_base):
    kls = []
    with torch.no_grad():
        for q_ids, r_ids, t_lp in cache:
            sp_lp, _ = compute_response_logprobs(student, s_embed, tmpl, [sp_tensor], q_ids, r_ids)
            kls.append(mean_kl_divergence(t_lp.cuda(), sp_lp))
    kl = sum(kls) / len(kls)
    G = 100.0 * (1.0 - kl / kl_base)
    return kl, G

pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
pad_vec = s_embed(torch.tensor([pad_id], device="cuda")).detach().to(torch.bfloat16)

for domain_name, tr_list, cache, kl_base in [
    ("IF_35tr", if_tr, val_cache, kl_S_if),
    ("Math_32tr", t_data["math_train_verified"], m_val_cache, sum(m_kl_s)/len(m_kl_s)),
]:
    print(f"\n--- Step-by-step Held-Out G(A) trajectory for {domain_name} ---", flush=True)
    tokenized = []
    prompt_stream = []
    for tr in tr_list:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False) + tmpl.im_end
        tokenized.append((tr["question"], q_ids, r_ids))
        prompt_stream.extend(tmpl.sys_close_user_open + q_ids + tmpl.user_close_asst_open)

    sp = init_soft_prompt_from_vocab(s_embed.weight, num_vectors=256, seed=42)
    opt = MuonSoftPrompt([sp], lr=0.01, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)
    kl0, g0 = eval_val_G(sp, cache, kl_base)
    print(f"  step  0 | heldout KL={kl0:.5f} | G={g0:+.2f}%", flush=True)

    rng = random.Random(42)
    for step in range(1, 21):
        student.config.use_cache = False
        student.train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt.zero_grad()
        batch = [tokenized[rng.randrange(len(tokenized))] for _ in range(16)]
        batch.sort(key=lambda x: len(x[1]) + len(x[2]))
        sl = 0.0
        for mb_i in range(0, 16, 2):
            sub = batch[mb_i : mb_i + 2]
            el, ll = [], []
            for qt, qi, ri in sub:
                pf = sample_random_prefix(prompt_stream, rng, 128, 256) if should_add_prefix(qt) else None
                e, l, _ = build_soft_prompt_inputs(s_embed, tmpl, [sp], qi, ri, None, pf, torch.device("cuda"), torch.bfloat16)
                el.append(e); ll.append(l)
            be, bm, bl = pad_and_stack_train_batch(el, ll, pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm)
            loss = length_normalized_nll_loss(out.logits, bl)
            sl += loss.item()
            (loss / 8.0).backward()
            del be, bm, bl, out
        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True
        if step in [1, 2, 4, 6, 8, 10, 15, 20]:
            kl_v, g_v = eval_val_G(sp, cache, kl_base)
            print(f"  step {step:2d} | train_nll={sl/8:.4f} | heldout KL={kl_v:.5f} | G={g_v:+.2f}%", flush=True)

print("\n=== DIAGNOSTIC COMPLETE ===", flush=True)
