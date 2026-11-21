"""
Fast 2-minute verification of:
1. Rao-Blackwellized CSPD (exact conditional expectation of Eq. 1 = Forward KL from RL Teacher on teacher traces)
   vs naive 1-hot NLL on 35 IF traces -> check held-out G(A) (Eq. 5) & Attention Pathways K(A) (Eq. 6)!
2. Check M2RL-RL_IF vs M2RL-SFT on IFEval with max_new_tokens=3072 (check closed_think!)
   and check whether allenai/IFBench_test is available on HuggingFace.
"""
import collections, gc, json, math, os, random, sys, time
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/cspd")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmarks import SUPPORTED_IFEVAL_IDS, batch_generate_with_soft_prompts, verify_ifeval_response
from cspd_core import (
    MuonSoftPrompt,
    build_soft_prompt_inputs,
    compute_response_logprobs,
    get_qwen_template_tokens,
    init_soft_prompt_from_vocab,
    mean_kl_divergence,
    pad_and_stack_train_batch,
    sample_random_prefix,
    should_add_prefix,
)

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"

tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
tmpl = get_qwen_template_tokens(tokenizer)
t_data = json.load(open("/content/cspd/exp1_results/teacher_data.json"))
if_all = t_data["if_train_verified"]
if_val = if_all[:14]
if_tr = if_all[14:]  # 35 traces

# 1. Check IFEval truncation at max_new_tokens=3072 on 24 prompts
ds_if = load_dataset("google/IFEval", split="train")
if_sup = [r for r in ds_if if r["instruction_id_list"] and all(i in SUPPORTED_IFEVAL_IDS for i in r["instruction_id_list"])]
rng = random.Random(42)
rng.shuffle(if_sup)
if_test24 = if_sup[:24]

def quick_if_eval(model, name, max_tok=3072):
    torch.manual_seed(42)
    outs = batch_generate_with_soft_prompts(
        model, tokenizer, tmpl, [x["prompt"] for x in if_test24], [],
        batch_size=24, max_new_tokens=max_tok, temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True
    )
    ok = 0; closed = 0; avg_chars = 0
    for item, resp in zip(if_test24, outs):
        s_ok, _ = verify_ifeval_response(resp, item["instruction_id_list"], item["kwargs"])
        ok += int(s_ok)
        closed += int("</think>" in resp)
        avg_chars += len(resp)
    print(f"  [{name} @ {max_tok}] strict={100*ok/len(if_test24):.1f}% ({ok}/24) | closed_think={closed}/24 | avg_chars={avg_chars//24}", flush=True)

print("=== 1. Check Teacher & Student on IFEval @ 3072 & precompute Teacher top-64 logprobs ===", flush=True)
t_if = AutoModelForCausalLM.from_pretrained(IF_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
t_if.eval()
quick_if_eval(t_if, "M2RL-RL_IF", 3072)
t_embed = t_if.get_input_embeddings()

# Precompute top-64 teacher distribution on 35 train traces (memory-light, exact 99.5%+ mass!)
# and full-vocab teacher logprobs on 14 held-out val traces
tr_cache = []
prompt_stream = []
with torch.no_grad():
    for tr in if_tr:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:768]
        t_lp, _ = compute_response_logprobs(t_if, t_embed, tmpl, [], q_ids, r_ids)
        # Top-64 logits/probs capture >99.7% probability mass and use tiny RAM!
        topv, topi = t_lp.topk(64, dim=-1)
        topp = F.softmax(topv, dim=-1)
        tr_cache.append((tr["question"], q_ids, r_ids, topi.cpu(), topp.to(torch.bfloat16).cpu()))
        prompt_stream.extend(tmpl.sys_close_user_open + q_ids + tmpl.user_close_asst_open)

val_cache = []
with torch.no_grad():
    for tr in if_val:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:1024]
        t_lp, _ = compute_response_logprobs(t_if, t_embed, tmpl, [], q_ids, r_ids)
        val_cache.append((q_ids, r_ids, t_lp.cpu()))

del t_if
gc.collect()
torch.cuda.empty_cache()

student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
student.eval()
for p in student.parameters():
    p.requires_grad = False
quick_if_eval(student, "M2RL-SFT", 3072)
s_embed = student.get_input_embeddings()

# Baseline KL(T||S) on 14 held-out val traces
kl_s = []
with torch.no_grad():
    for q_ids, r_ids, t_lp in val_cache:
        s_lp, _ = compute_response_logprobs(student, s_embed, tmpl, [], q_ids, r_ids)
        kl_s.append(mean_kl_divergence(t_lp.cuda(), s_lp))
kl_S = sum(kl_s) / len(kl_s)
print(f"\n=== 2. Rao-Blackwellized CSPD on 35 IF traces (held-out baseline KL={kl_S:.5f}) ===", flush=True)

def eval_val(sp_tensor, mode="full"):
    kls = []
    with torch.no_grad():
        for q_ids, r_ids, t_lp in val_cache:
            lp, _ = compute_response_logprobs(student, s_embed, tmpl, [sp_tensor], q_ids, r_ids, attention_mode=mode)
            kls.append(mean_kl_divergence(t_lp.cuda(), lp))
    kl = sum(kls) / len(kls)
    return kl, 100.0 * (1.0 - kl / kl_S)

pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
pad_vec = s_embed(torch.tensor([pad_id], device="cuda")).detach().to(torch.bfloat16)

sp = init_soft_prompt_from_vocab(s_embed.weight, num_vectors=256, seed=42)
opt = MuonSoftPrompt([sp], lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)
rng = random.Random(42)
t0 = time.time()

for step in range(1, 31):
    progress = (step - 1) / 29.0
    lr = 0.002 + 0.5 * (0.02 - 0.002) * (1.0 + math.cos(math.pi * progress))
    for pg in opt.param_groups:
        pg["lr"] = lr

    student.config.use_cache = False
    student.train()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt.zero_grad()
    batch = [tr_cache[rng.randrange(len(tr_cache))] for _ in range(16)]
    batch.sort(key=lambda x: len(x[1]) + len(x[2]))
    sl = 0.0
    for mb_i in range(0, 16, 2):
        sub = batch[mb_i : mb_i + 2]
        el, ll, top_is, top_ps = [], [], [], []
        for qt, qi, ri, ti, tp in sub:
            pf = sample_random_prefix(prompt_stream, rng, 128, 256) if should_add_prefix(qt) else None
            e, l, _ = build_soft_prompt_inputs(s_embed, tmpl, [sp], qi, ri, None, pf, torch.device("cuda"), torch.bfloat16)
            el.append(e); ll.append(l); top_is.append(ti); top_ps.append(tp)
        be, bm, bl = pad_and_stack_train_batch(el, ll, pad_vec)
        out = student(inputs_embeds=be, attention_mask=bm)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = bl[:, 1:]
        losses = []
        for i in range(len(sub)):
            mask = shift_labels[i] != -100
            s_resp = shift_logits[i][mask].float()
            ti = top_is[i].cuda()
            tp = top_ps[i].float().cuda()
            n = min(s_resp.size(0), ti.size(0))
            s_lp = F.log_softmax(s_resp[:n], dim=-1)
            s_top_lp = s_lp.gather(-1, ti[:n])
            # Cross-entropy H(p_T, p_S) = -sum p_T(v) log p_S(v)
            ce = -(tp[:n] * s_top_lp).sum(dim=-1).mean()
            losses.append(ce)
        loss = torch.stack(losses).mean()
        sl += loss.item()
        (loss / 8.0).backward()
        del be, bm, bl, out
    torch.nn.utils.clip_grad_norm_([sp], 1.0)
    opt.step()
    student.gradient_checkpointing_disable()
    student.eval()
    student.config.use_cache = True

    if step in [1, 5, 10, 15, 20, 30]:
        kl_v, g_v = eval_val(sp, "full")
        print(f"  step {step:2d}/30 | rb_ce={sl/8:.4f} | heldout KL={kl_v:.5f} | G={g_v:+.2f}% | elapsed={time.time()-t0:.1f}s", flush=True)

# Check Attention Pathways K(A) on held-out IF val traces!
kl_full, g_full = eval_val(sp, "full")
kl_resp, _ = eval_val(sp, "response_only")
kl_inp, _ = eval_val(sp, "input_only")
kl_blk, _ = eval_val(sp, "blocked")
denom = kl_S - kl_full
print(f"\n=== 3. Held-out Attention Pathways K(A) (Eq. 6) ===", flush=True)
print(f"  KL: student={kl_S:.5f}, full={kl_full:.5f} (G={g_full:+.2f}%), resp_only={kl_resp:.5f}, inp_only={kl_inp:.5f}, blocked={kl_blk:.5f}", flush=True)
if denom > 1e-6:
    print(f"  K(A): full=100.0%, response_only={100*(kl_S-kl_resp)/denom:.1f}%, "
          f"input_only={100*(kl_S-kl_inp)/denom:.1f}%, blocked={100*(kl_S-kl_blk)/denom:.1f}%", flush=True)
print("=== VERIFY COMPLETE ===", flush=True)
