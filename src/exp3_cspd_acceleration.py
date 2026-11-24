"""
Suite 1 (exp3): Accelerating & Upgrading Original CSPD (Efficiency, Convergence Speed & Architecture)
Frozen Student: Jackwang111/M2RL-SFT (Qwen3-4B)
RL Teachers:    Jackwang111/M2RL-RL_IF & Jackwang111/M2RL-RL_Math

Systematically benchmarks 4 efficiency & convergence-speed axes on identical held-out traces:
  Axis 1 (Initialization): Random Vocab (CSPD default) vs Low-Norm (0.15x) vs Task-Instruction Tile
  Axis 2 (Loss & Truncation): 1-Hot SeqNLL vs RB-Top64 (768 tok) vs Short-Prefix RB-Top64 (256 tok, ~2.5x faster!)
                              vs Fork-Token Top-30% Disagreement RB-KL vs Skew/Hybrid Fwd+Rev KL
  Axis 3 (Optimizer & Parameterization): Muon (default) vs Norm-Clipped Muon (RMS<=0.055) vs AdamW
                                         vs Low-Rank Factorized SP (r=32) vs Prompt Length Frontier (L=16, 64, 256)
  Axis 4 (Architecture Paradigm Shift): KV-CSPD (Direct Layerwise KV-Cache Prefix, L_kv=8 [0.59M, 32x shorter]
                                        & L_kv=16 [1.18M, 16x shorter]) + Zero-Training KV Concatenation [KV_M; KV_I]!
  Plus End-to-End IFBench Verification for top winners!
"""

import gc
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, "/content/cspd")
sys.path.insert(0, "/content/IFBench")

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluation_lib as ifbench_eval

from benchmarks import batch_generate_with_soft_prompts, strip_thinking
from cspd_core import (
    ChatTemplateTokens,
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
from kv_prefix_core import (
    LayerwiseMuonKV,
    batch_generate_with_kv_prefixes,
    build_chat_ids_for_kv,
    forward_with_kv_prefix,
    init_kv_prefix_from_tokens,
)

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"
OUT_DIR = "/content/cspd/exp3_results"
os.makedirs(OUT_DIR, exist_ok=True)

IF_TASK_INST = (
    "Follow every formatting, length, keyword, structural, and stylistic constraint in the user's "
    "request with extreme precision. Plan carefully and double-check each constraint before answering."
)
MATH_TASK_INST = (
    "Solve the math problem step by step. After every intermediate derivation, pause to verify "
    "your algebra and edge cases before moving on or concluding."
)


# ---------------------------------------------------------------------------
# 1. Data & Precomputed Exact Teacher Distributions (with Student KL for Fork-Token!)
# ---------------------------------------------------------------------------

def collect_fast_rl_traces(tokenizer, tmpl: ChatTemplateTokens) -> Dict[str, Any]:
    cache_p = os.path.join(OUT_DIR, "rl_traces_exp3.json")
    if os.path.exists(cache_p):
        return json.load(open(cache_p))

    rng = random.Random(42)
    ds_if = load_dataset("google/IFEval", split="train")
    if_qs = [r["prompt"] for r in ds_if]
    rng.shuffle(if_qs)
    if_qs = if_qs[:110]

    ds_m = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    m_qs = [r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            for r in ds_m if any(k in str(r.get("level", "")) for k in ["3", "4"])]
    rng.shuffle(m_qs)
    m_qs = m_qs[:60]

    out: Dict[str, Any] = {}

    print(f"=== Collecting 110 IF teacher traces ({IF_TEACHER_ID}) ===", flush=True)
    t_if = AutoModelForCausalLM.from_pretrained(IF_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    t_if.eval()
    torch.manual_seed(42)
    if_rs = batch_generate_with_soft_prompts(
        t_if, tokenizer, tmpl, if_qs, [], batch_size=55, max_new_tokens=1536,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    if_all = [{"question": q, "response": r} for q, r in zip(if_qs, if_rs) if len(tokenizer.encode(r, add_special_tokens=False)) >= 40]
    out["if_val"] = if_all[:20]
    out["if_train"] = if_all[20:110]
    del t_if; gc.collect(); torch.cuda.empty_cache()

    print(f"=== Collecting 60 Math teacher traces ({MATH_TEACHER_ID}) ===", flush=True)
    t_m = AutoModelForCausalLM.from_pretrained(MATH_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    t_m.eval()
    torch.manual_seed(43)
    m_rs = batch_generate_with_soft_prompts(
        t_m, tokenizer, tmpl, m_qs, [], batch_size=60, max_new_tokens=1536,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    m_all = [{"question": q, "response": r} for q, r in zip(m_qs, m_rs) if len(tokenizer.encode(r, add_special_tokens=False)) >= 60]
    out["math_val"] = m_all[:12]
    out["math_train"] = m_all[12:60]
    del t_m; gc.collect(); torch.cuda.empty_cache()

    print(f"Traces ready | IF: tr={len(out['if_train'])}, val={len(out['if_val'])} | Math: tr={len(out['math_train'])}, val={len(out['math_val'])}", flush=True)
    with open(cache_p, "w") as f:
        json.dump(out, f)
    return out


@torch.no_grad()
def precompute_teacher_and_fork_weights(
    teacher_id: str, student, tokenizer, tmpl: ChatTemplateTokens,
    train_traces: List[Dict[str, str]], val_traces: List[Dict[str, str]],
    max_tr_resp: int = 768, max_val_resp: int = 1024, topk: int = 64, tag: str = "",
):
    t0 = time.time()
    t_model = AutoModelForCausalLM.from_pretrained(teacher_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    t_model.eval()
    t_emb = t_model.get_input_embeddings()
    s_emb = student.get_input_embeddings()

    tr_cache = []
    stream: List[int] = []
    for tr in train_traces:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:max_tr_resp]
        if len(r_ids) < 24:
            continue
        t_lp, _ = compute_response_logprobs(t_model, t_emb, tmpl, [], q_ids, r_ids)
        s_lp, _ = compute_response_logprobs(student, s_emb, tmpl, [], q_ids, r_ids)
        # Per-token baseline KL(T || S) -> identifies high-disagreement RL "fork tokens"!
        n = min(t_lp.size(0), s_lp.size(0))
        tok_kl = (t_lp[:n].exp() * (t_lp[:n] - s_lp[:n])).sum(dim=-1).cpu()  # [n]
        topv, topi = t_lp[:n].topk(topk, dim=-1)
        topp = F.softmax(topv, dim=-1)
        tr_cache.append((tr["question"], q_ids, r_ids[:n], topi.cpu(), topp.to(torch.bfloat16).cpu(), tok_kl.to(torch.float16)))
        stream.extend(tmpl.sys_close_user_open + q_ids + tmpl.user_close_asst_open)

    val_cache = []
    for tr in val_traces:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:max_val_resp]
        if len(r_ids) < 24:
            continue
        t_lp, _ = compute_response_logprobs(t_model, t_emb, tmpl, [], q_ids, r_ids)
        val_cache.append((q_ids, r_ids, t_lp.cpu()))

    del t_model; gc.collect(); torch.cuda.empty_cache()
    print(f"  [{tag}] Precomputed {len(tr_cache)} train (with fork-KL) + {len(val_cache)} val in {time.time()-t0:.1f}s", flush=True)
    return tr_cache, val_cache, stream


@torch.no_grad()
def eval_val_kl_sp(student, s_emb, tmpl, val_cache, sps: List[torch.Tensor]) -> float:
    kls = []
    for q_ids, r_ids, t_lp in val_cache:
        lp, _ = compute_response_logprobs(student, s_emb, tmpl, sps, q_ids, r_ids)
        kls.append(mean_kl_divergence(t_lp.cuda(), lp))
    return float(sum(kls) / max(1, len(kls)))


@torch.no_grad()
def eval_val_kl_kv(student, tmpl, val_cache, kv_list: List[torch.Tensor]) -> float:
    dev = next(student.parameters()).device
    kls = []
    for q_ids, r_ids, t_lp in val_cache:
        full_ids, labels = build_chat_ids_for_kv(tmpl, q_ids, r_ids, None, True, None)
        inp = torch.tensor([full_ids], dtype=torch.long, device=dev)
        sm = torch.ones_like(inp)
        lbl = torch.tensor([labels], dtype=torch.long, device=dev)
        out = forward_with_kv_prefix(student, inp, sm, kv_list)
        shift_logits = out.logits[0, :-1, :]
        shift_labels = lbl[0, 1:]
        mask = shift_labels != -100
        lp = F.log_softmax(shift_logits[mask].float(), dim=-1)
        kls.append(mean_kl_divergence(t_lp.cuda(), lp))
    return float(sum(kls) / max(1, len(kls)))


# ---------------------------------------------------------------------------
# 2. Unified Input-SP Trainer Supporting ALL Acceleration Variants
# ---------------------------------------------------------------------------

def train_sp_accel(
    student, tokenizer, tmpl, tr_cache, val_cache, stream, kl_base: float,
    num_vectors: int = 256,
    num_updates: int = 45,
    batch_size: int = 16,
    micro_batch: int = 2,
    init_type: str = "random_vocab",   # "random_vocab" | "low_norm" | "task_inst"
    task_inst_text: str = IF_TASK_INST,
    loss_type: str = "rb_top64",       # "seq_nll" | "rb_top64" | "fork_top30" | "hybrid_fwd_rev"
    max_resp_trunc: int = 768,         # e.g., 256 for Short-Prefix Truncated speedup!
    optimizer_type: str = "muon",      # "muon" | "adamw"
    lr_max: float = 0.025,
    max_rms: Optional[float] = None,   # Norm-Clipped Muon!
    low_rank: Optional[int] = None,    # Low-Rank Factorized SP: P = A @ B
    prefix_robust: bool = True,
    seed: int = 42,
    tag: str = "",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).detach().to(dtype)
    d_model = s_emb.weight.size(1)

    # --- Initialize P ---
    if init_type == "random_vocab":
        init_mat = init_soft_prompt_from_vocab(s_emb.weight, num_vectors, seed).detach()
    elif init_type == "low_norm":
        init_mat = init_soft_prompt_from_vocab(s_emb.weight, num_vectors, seed).detach() * 0.15
    elif init_type == "task_inst":
        ids = tokenizer.encode(task_inst_text, add_special_tokens=False)
        reps = (num_vectors // max(1, len(ids))) + 1
        ids = (ids * reps)[:num_vectors]
        init_mat = s_emb.weight[torch.tensor(ids, device=device)].detach().float().clone()
    else:
        raise ValueError(init_type)

    if low_rank is not None:
        # Factorize P = init_mat_frozen + A @ B (with B=0 so starts cleanly at init_mat!)
        base_frozen = init_mat.clone().detach()
        A = nn.Parameter(torch.randn((num_vectors, low_rank), device=device) * (0.02 / math.sqrt(low_rank)))
        B = nn.Parameter(torch.zeros((low_rank, d_model), device=device))
        params = [A, B]
        get_sp = lambda: base_frozen + (A @ B)
        opt = torch.optim.AdamW(params, lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
        n_params = A.numel() + B.numel()
    else:
        sp = nn.Parameter(init_mat.clone())
        params = [sp]
        get_sp = lambda: sp
        if optimizer_type == "muon":
            opt = MuonSoftPrompt(params, lr=lr_max, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)
        else:
            opt = torch.optim.AdamW(params, lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
        n_params = sp.numel()

    eval_steps = {0, 5, 10, 15, 20, 30, num_updates}
    curve = []
    train_elapsed = 0.0  # Pure training wall-clock time (excluding validation eval pauses!)

    def log_eval(st: int):
        kl = eval_val_kl_sp(student, s_emb, tmpl, val_cache, [get_sp().detach()])
        g = round(100.0 * (1.0 - kl / kl_base), 2)
        curve.append({"step": st, "train_sec": round(train_elapsed, 1), "val_kl": round(kl, 5), "G_pct": g})
        if st in {0, 10, 20, 30, num_updates}:
            print(f"  [{tag:26s}] step {st:2d}/{num_updates} | t_train={train_elapsed:5.1f}s | val_KL={kl:.5f} | G={g:+6.2f}%", flush=True)

    log_eval(0)

    rng = random.Random(seed)
    num_mb = max(1, batch_size // micro_batch)
    lr_min = lr_max * 0.1
    warmup = 3

    for step in range(1, num_updates + 1):
        ts = time.time()
        if step <= warmup:
            lr = lr_max * (0.3 + 0.7 * (step / warmup))
        else:
            prog = (step - warmup) / max(1, num_updates - warmup)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        student.config.use_cache = False
        student.train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt.zero_grad()

        batch = [tr_cache[rng.randrange(len(tr_cache))] for _ in range(batch_size)]
        batch.sort(key=lambda x: len(x[1]) + min(len(x[2]), max_resp_trunc))

        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            cur_sp = get_sp()
            el, ll, tis, tps, fkls = [], [], [], [], []
            for qt, qi, ri, ti, tp, fkl in sub:
                ri_c = ri[:max_resp_trunc]
                pf = sample_random_prefix(stream, rng, 128, 256) if (prefix_robust and should_add_prefix(qt)) else None
                e, l, _ = build_soft_prompt_inputs(s_emb, tmpl, [cur_sp], qi, ri_c, None, pf, device, dtype)
                el.append(e); ll.append(l)
                tis.append(ti[: len(ri_c)]); tps.append(tp[: len(ri_c)]); fkls.append(fkl[: len(ri_c)])

            be, bm, bl = pad_and_stack_train_batch(el, ll, pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm)

            if loss_type == "seq_nll":
                loss = length_normalized_nll_loss(out.logits, bl)
            else:
                shift_logits = out.logits[:, :-1, :]
                shift_labels = bl[:, 1:]
                per_s = []
                for i in range(len(sub)):
                    mask = shift_labels[i] != -100
                    s_resp = shift_logits[i][mask].float()
                    ti = tis[i].to(device)
                    tp = tps[i].float().to(device)
                    n = min(s_resp.size(0), ti.size(0))
                    s_lp = F.log_softmax(s_resp[:n], dim=-1)
                    s_top_lp = s_lp.gather(-1, ti[:n])
                    fwd_ce = -(tp[:n] * s_top_lp).sum(dim=-1)  # [n]
                    if loss_type == "fork_top30":
                        fkl = fkls[i][:n].float().to(device)
                        k_keep = max(8, int(0.30 * n))
                        _, keep_idx = torch.topk(fkl, k_keep)
                        per_s.append(fwd_ce[keep_idx].mean())
                    elif loss_type == "hybrid_fwd_rev":
                        # Mode-seeking reverse KL on Top-64 teacher support + forward CE
                        s_top_p = F.softmax(s_top_lp, dim=-1)
                        t_top_lp = tp[:n].clamp_min(1e-7).log()
                        rev_kl = (s_top_p * (s_top_p.clamp_min(1e-7).log() - t_top_lp)).sum(dim=-1)
                        per_s.append((0.7 * fwd_ce + 0.3 * rev_kl).mean())
                    else:
                        per_s.append(fwd_ce.mean())
                loss = torch.stack(per_s).mean()

            (loss / num_mb).backward()
            del be, bm, bl, out

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if max_rms is not None and low_rank is None:
            with torch.no_grad():
                cur_r = sp.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
                sp.mul_(torch.clamp(max_rms / cur_r, max=1.0))

        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True
        train_elapsed += time.time() - ts

        if step in eval_steps:
            log_eval(step)

    final_sp = get_sp().detach().clone()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    meta = {
        "tag": tag, "arch": "input_sp", "num_vectors": num_vectors, "params": int(n_params),
        "init_type": init_type, "loss_type": loss_type, "max_resp_trunc": max_resp_trunc,
        "optimizer": optimizer_type, "max_rms": max_rms, "low_rank": low_rank,
        "final_rms": round(float(final_sp.pow(2).mean().sqrt().item()), 5),
        "train_seconds": round(train_elapsed, 1), "peak_vram_gb": round(peak_gb, 2),
        "final_G_pct": curve[-1]["G_pct"], "best_G_pct": max(c["G_pct"] for c in curve),
        "curve": curve,
    }
    return final_sp, meta


# ---------------------------------------------------------------------------
# 3. KV-CSPD Trainer for RL Experts (Direct Layerwise KV-Cache Prefix!)
# ---------------------------------------------------------------------------

def train_kv_accel(
    student, tokenizer, tmpl, tr_cache, val_cache, kl_base: float,
    num_kv_tokens: int = 16,
    task_inst_text: str = IF_TASK_INST,
    num_updates: int = 45,
    batch_size: int = 16,
    micro_batch: int = 2,
    lr_max: float = 0.025,
    max_resp_trunc: int = 512,
    kv_prefix_robust: bool = True,
    max_delta_rms: Optional[float] = 0.7,
    seed: int = 42,
    tag: str = "",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    torch.cuda.reset_peak_memory_stats()
    device = next(student.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    inst_ids = tokenizer.encode(task_inst_text, add_special_tokens=False)
    kv = init_kv_prefix_from_tokens(student, inst_ids, num_kv_tokens=num_kv_tokens)
    init_kv = kv.detach().clone()

    # Build 3 natural KV distractors for Prefix-Robust KV composition!
    d_pool = []
    if kv_prefix_robust:
        for d_txt in [
            "Read the user request carefully and provide a clear, accurate, well-structured response.",
            "Think through each step logically, check calculations carefully, and present the final answer.",
            "Analyze the problem requirements systematically and verify every detail before responding.",
        ]:
            d_ids = tokenizer.encode(d_txt, add_special_tokens=False)
            d_pool.append(init_kv_prefix_from_tokens(student, d_ids, num_kv_tokens=num_kv_tokens).detach())

    opt = torch.optim.AdamW([kv], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    student.gradient_checkpointing_disable()
    student.eval()
    student.config.use_cache = True

    eval_steps = {0, 2, 5, 10, 15, 20, 30, num_updates}
    curve = []
    train_elapsed = 0.0

    def log_eval(st: int):
        kl = eval_val_kl_kv(student, tmpl, val_cache, [kv.detach()])
        g = round(100.0 * (1.0 - kl / kl_base), 2)
        curve.append({"step": st, "train_sec": round(train_elapsed, 1), "val_kl": round(kl, 5), "G_pct": g})
        if st in {0, 5, 10, 20, 30, num_updates}:
            print(f"  [{tag:26s}] step {st:2d}/{num_updates} | t_train={train_elapsed:5.1f}s | val_KL={kl:.5f} | G={g:+6.2f}%", flush=True)

    log_eval(0)
    rng = random.Random(seed)
    num_mb = max(1, batch_size // micro_batch)
    lr_min = lr_max * 0.1
    warmup = 3

    for step in range(1, num_updates + 1):
        ts = time.time()
        if step <= warmup:
            lr = lr_max * (0.3 + 0.7 * (step / warmup))
        else:
            prog = (step - warmup) / max(1, num_updates - warmup)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        batch = [tr_cache[rng.randrange(len(tr_cache))] for _ in range(batch_size)]
        batch.sort(key=lambda x: len(x[1]) + min(len(x[2]), max_resp_trunc))

        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            f_seqs, l_seqs, tis, tps = [], [], [], []
            for qt, qi, ri, ti, tp, _ in sub:
                ri_c = ri[:max_resp_trunc]
                fids, lbls = build_chat_ids_for_kv(tmpl, qi, ri_c, None, True, None)
                f_seqs.append(fids); l_seqs.append(lbls)
                tis.append(ti[: len(ri_c)]); tps.append(tp[: len(ri_c)])

            B = len(sub)
            ml = max(len(x) for x in f_seqs)
            inp = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
            sm = torch.zeros((B, ml), dtype=torch.long, device=device)
            lb = torch.full((B, ml), -100, dtype=torch.long, device=device)
            for i, (fids, lvals) in enumerate(zip(f_seqs, l_seqs)):
                L = len(fids)
                inp[i, :L] = torch.tensor(fids, dtype=torch.long, device=device)
                sm[i, :L] = 1
                lb[i, :L] = torch.tensor(lvals, dtype=torch.long, device=device)

            pre_d, post_d = None, None
            if kv_prefix_robust and d_pool:
                rm = rng.random()
                if rm < 0.35:
                    pre_d = d_pool[rng.randrange(len(d_pool))]
                elif rm < 0.70:
                    post_d = d_pool[rng.randrange(len(d_pool))]

            out = forward_with_kv_prefix(student, inp, sm, [kv], pre_d, post_d)
            shift_logits = out.logits[:, :-1, :]
            shift_labels = lb[:, 1:]
            per_s = []
            for i in range(B):
                mask = shift_labels[i] != -100
                s_resp = shift_logits[i][mask].float()
                ti = tis[i].to(device)
                tp = tps[i].float().to(device)
                n = min(s_resp.size(0), ti.size(0))
                s_lp = F.log_softmax(s_resp[:n], dim=-1)
                ce = -(tp[:n] * s_lp.gather(-1, ti[:n])).sum(dim=-1).mean()
                per_s.append(ce)
            loss = torch.stack(per_s).mean()
            (loss / num_mb).backward()
            del inp, sm, lb, out

        torch.nn.utils.clip_grad_norm_([kv], 1.0)
        opt.step()
        if max_delta_rms is not None:
            with torch.no_grad():
                delta = kv - init_kv
                d_r = delta.pow(2).mean().sqrt().clamp_min(1e-8)
                if d_r > max_delta_rms:
                    kv.copy_(init_kv + delta * (max_delta_rms / d_r))

        train_elapsed += time.time() - ts
        if step in eval_steps:
            log_eval(step)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    meta = {
        "tag": tag, "arch": "kv_prefix", "num_kv_tokens": num_kv_tokens, "params": int(kv.numel()),
        "max_resp_trunc": max_resp_trunc, "kv_prefix_robust": kv_prefix_robust,
        "delta_rms": round(float((kv.detach() - init_kv).pow(2).mean().sqrt().item()), 4),
        "train_seconds": round(train_elapsed, 1), "peak_vram_gb": round(peak_gb, 2),
        "final_G_pct": curve[-1]["G_pct"], "best_G_pct": max(c["G_pct"] for c in curve),
        "curve": curve,
    }
    return kv.detach().clone(), meta


# ---------------------------------------------------------------------------
# 4. Official IFBench Evaluator (32 stratified prompts, max_new_tokens=2560)
# ---------------------------------------------------------------------------

def eval_ifbench_quick(model, tokenizer, tmpl, inps, mode: str, adapters: List[torch.Tensor], tag: str = "") -> Dict[str, Any]:
    torch.manual_seed(42)
    t0 = time.time()
    prompts = [x.prompt for x in inps]
    if mode == "kv":
        outs = batch_generate_with_kv_prefixes(
            model, tokenizer, tmpl, prompts, adapters, batch_size=32, max_new_tokens=2560,
            temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
        )
    else:
        outs = batch_generate_with_soft_prompts(
            model, tokenizer, tmpl, prompts, adapters, batch_size=32, max_new_tokens=2560,
            temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
        )
    p2r = {}
    closed = 0
    for inp, r in zip(inps, outs):
        closed += int("</think>" in r)
        p2r[inp.prompt] = strip_thinking(r)
    s_ok = sum(int(ifbench_eval.test_instruction_following_strict(inp, p2r).follow_all_instructions) for inp in inps)
    inst_l = []
    for inp in inps:
        inst_l.extend(ifbench_eval.test_instruction_following_strict(inp, p2r).follow_instruction_list)
    n = len(inps)
    res = {
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * sum(inst_l) / max(1, len(inst_l)), 2),
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFBench32 | {tag:26s}] strict={res['prompt_strict']:.1f}% ({s_ok}/{n}) | inst={res['inst_strict']:.1f}% | closed={closed}/{n} ({res['closed_acc']:.1f}%) in {res['eval_sec']:.1f}s", flush=True)
    return res


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Suite 1 (exp3): CSPD Acceleration, Convergence & KV-CSPD Benchmark ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    traces = collect_fast_rl_traces(tokenizer, tmpl)

    student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    student.eval()
    for p in student.parameters():
        p.requires_grad = False
    s_emb = student.get_input_embeddings()

    if_tr, if_val, if_stream = precompute_teacher_and_fork_weights(IF_TEACHER_ID, student, tokenizer, tmpl, traces["if_train"], traces["if_val"], tag="IF")
    m_tr, m_val, m_stream = precompute_teacher_and_fork_weights(MATH_TEACHER_ID, student, tokenizer, tmpl, traces["math_train"], traces["math_val"], tag="Math")

    kl_S_if = eval_val_kl_sp(student, s_emb, tmpl, if_val, [])
    kl_S_math = eval_val_kl_sp(student, s_emb, tmpl, m_val, [])
    print(f"Baselines | KL(T_IF||S)={kl_S_if:.5f} | KL(T_Math||S)={kl_S_math:.5f}", flush=True)

    results: Dict[str, Any] = {
        "baselines": {"kl_S_if": round(kl_S_if, 5), "kl_S_math": round(kl_S_math, 5)},
        "axis1_init": {},
        "axis2_loss_trunc": {},
        "axis3_opt_len": {},
        "axis4_kv_cspd": {},
        "dual_composition_G": {},
        "ifbench32_verify": {},
    }

    def save():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # === AXIS 1: Initialization Speedup (IF, L=256, 45 updates) ===
    print("\n=== Axis 1: Initialization Speedup (Random Vocab vs Low-Norm vs Task-Instruction) ===", flush=True)
    for itype in ["random_vocab", "low_norm", "task_inst"]:
        _, meta = train_sp_accel(
            student, tokenizer, tmpl, if_tr, if_val, if_stream, kl_S_if,
            num_vectors=256, num_updates=45, init_type=itype, loss_type="rb_top64",
            optimizer_type="muon", lr_max=0.025, tag=f"Init_{itype}"
        )
        results["axis1_init"][itype] = meta
        save()

    # === AXIS 2: Loss Function & Short-Prefix Truncation Speedup ===
    print("\n=== Axis 2: Loss Formulation & Short-Prefix Truncation Efficiency ===", flush=True)
    for name, ltype, trunc in [
        ("seq_nll_768", "seq_nll", 768),
        ("rb_top64_trunc256", "rb_top64", 256),
        ("fork_top30_768", "fork_top30", 768),
        ("hybrid_fwd_rev_768", "hybrid_fwd_rev", 768),
    ]:
        _, meta = train_sp_accel(
            student, tokenizer, tmpl, if_tr, if_val, if_stream, kl_S_if,
            num_vectors=256, num_updates=45, init_type="low_norm", loss_type=ltype,
            max_resp_trunc=trunc, optimizer_type="muon", lr_max=0.025, tag=f"Loss_{name}"
        )
        results["axis2_loss_trunc"][name] = meta
        save()

    # === AXIS 3: Optimizer (Norm-Clipped Muon vs AdamW), Low-Rank & Length Pareto ===
    print("\n=== Axis 3: Norm-Clipped Muon, Low-Rank SP & Length Pareto Frontier ===", flush=True)
    sp_if_ncm, meta_ncm = train_sp_accel(
        student, tokenizer, tmpl, if_tr, if_val, if_stream, kl_S_if,
        num_vectors=256, num_updates=45, init_type="low_norm", loss_type="rb_top64",
        optimizer_type="muon", lr_max=0.025, max_rms=0.055, tag="Opt_Muon_NormClip055"
    )
    results["axis3_opt_len"]["muon_normclip_055"] = meta_ncm

    _, meta_aw = train_sp_accel(
        student, tokenizer, tmpl, if_tr, if_val, if_stream, kl_S_if,
        num_vectors=256, num_updates=45, init_type="low_norm", loss_type="rb_top64",
        optimizer_type="adamw", lr_max=0.008, tag="Opt_AdamW_LowNorm"
    )
    results["axis3_opt_len"]["adamw_lownorm"] = meta_aw

    _, meta_lr32 = train_sp_accel(
        student, tokenizer, tmpl, if_tr, if_val, if_stream, kl_S_if,
        num_vectors=256, num_updates=45, init_type="low_norm", loss_type="rb_top64",
        low_rank=32, lr_max=0.012, tag="Param_LowRank_r32"
    )
    results["axis3_opt_len"]["low_rank_r32"] = meta_lr32

    for L in [16, 64]:
        _, meta_L = train_sp_accel(
            student, tokenizer, tmpl, if_tr, if_val, if_stream, kl_S_if,
            num_vectors=L, num_updates=45, init_type="low_norm", loss_type="rb_top64",
            optimizer_type="muon", lr_max=0.025, tag=f"Len_L{L}"
        )
        results["axis3_opt_len"][f"len_L{L}"] = meta_L
    save()

    # === AXIS 4: Architectural Paradigm Shift — KV-CSPD (L_kv=8 & L_kv=16) ===
    print("\n=== Axis 4: Direct Layerwise KV-CSPD (L_kv=8 [32x shorter] & L_kv=16 [16x shorter]) ===", flush=True)
    kv_if_8, meta_kv8 = train_kv_accel(
        student, tokenizer, tmpl, if_tr, if_val, kl_S_if,
        num_kv_tokens=8, task_inst_text=IF_TASK_INST, num_updates=45, lr_max=0.025, max_resp_trunc=512, tag="KV_CSPD_IF_L8"
    )
    kv_if_16, meta_kv16 = train_kv_accel(
        student, tokenizer, tmpl, if_tr, if_val, kl_S_if,
        num_kv_tokens=16, task_inst_text=IF_TASK_INST, num_updates=45, lr_max=0.025, max_resp_trunc=512, tag="KV_CSPD_IF_L16"
    )
    results["axis4_kv_cspd"]["kv_if_L8"] = meta_kv8
    results["axis4_kv_cspd"]["kv_if_L16"] = meta_kv16
    save()

    # === Dual-Domain Zero-Training Composition Check: Norm-Clipped Muon [M;I] vs KV-CSPD [KV_M; KV_I]! ===
    print("\n=== Dual-Domain Zero-Training Composition Check (Norm-Clipped Muon vs KV-CSPD) ===", flush=True)
    sp_m_ncm, meta_m_ncm = train_sp_accel(
        student, tokenizer, tmpl, m_tr, m_val, m_stream, kl_S_math,
        num_vectors=256, num_updates=45, init_type="low_norm", loss_type="rb_top64",
        optimizer_type="muon", lr_max=0.020, max_rms=0.055, seed=43, tag="Math_Muon_NormClip055"
    )
    kv_m_16, meta_kvm16 = train_kv_accel(
        student, tokenizer, tmpl, m_tr, m_val, kl_S_math,
        num_kv_tokens=16, task_inst_text=MATH_TASK_INST, num_updates=45, lr_max=0.020, max_resp_trunc=512, seed=43, tag="KV_CSPD_Math_L16"
    )
    results["axis3_opt_len"]["math_muon_normclip_055"] = meta_m_ncm
    results["axis4_kv_cspd"]["kv_math_L16"] = meta_kvm16

    def g_pct(kl_a, kl_s):
        return round(100.0 * (1.0 - kl_a / max(kl_s, 1e-8)), 2)

    for cname, mode, ads in [
        ("NormClipMuon_Single_IF", "sp", [sp_if_ncm]),
        ("NormClipMuon_Single_Math", "sp", [sp_m_ncm]),
        ("NormClipMuon_Concat_M_I", "sp", [sp_m_ncm, sp_if_ncm]),
        ("NormClipMuon_Concat_I_M", "sp", [sp_if_ncm, sp_m_ncm]),
        ("KV16_Single_IF", "kv", [kv_if_16]),
        ("KV16_Single_Math", "kv", [kv_m_16]),
        ("KV16_Concat_M_I", "kv", [kv_m_16, kv_if_16]),
        ("KV16_Concat_I_M", "kv", [kv_if_16, kv_m_16]),
    ]:
        if mode == "sp":
            ki = eval_val_kl_sp(student, s_emb, tmpl, if_val, ads)
            km = eval_val_kl_sp(student, s_emb, tmpl, m_val, ads)
        else:
            ki = eval_val_kl_kv(student, tmpl, if_val, ads)
            km = eval_val_kl_kv(student, tmpl, m_val, ads)
        gi, gm = g_pct(ki, kl_S_if), g_pct(km, kl_S_math)
        results["dual_composition_G"][cname] = {"G_if_pct": gi, "G_math_pct": gm, "G_avg_pct": round((gi + gm) / 2, 2)}
        print(f"  [Comp | {cname:26s}] G_IF={gi:+6.2f}% | G_Math={gm:+6.2f}% | Avg={(gi+gm)/2:+6.2f}%", flush=True)
    save()

    # === Official IFBench End-to-End Verification (32 prompts, max_new_tokens=2560) ===
    print("\n=== End-to-End Official IFBench Verification (32 prompts) ===", flush=True)
    all_inps = ifbench_eval.read_prompt_list("/content/IFBench/data/IFBench_test.jsonl")
    rng_ifb = random.Random(42)
    idxs = list(range(len(all_inps)))
    rng_ifb.shuffle(idxs)
    inps32 = [all_inps[i] for i in sorted(idxs[:32])]

    results["ifbench32_verify"]["student_no_sp"] = eval_ifbench_quick(student, tokenizer, tmpl, inps32, "sp", [], "Student (No SP)")
    results["ifbench32_verify"]["normclip_muon_single_IF"] = eval_ifbench_quick(student, tokenizer, tmpl, inps32, "sp", [sp_if_ncm], "NormClipMuon Single P_if")
    results["ifbench32_verify"]["normclip_muon_concat_M_I"] = eval_ifbench_quick(student, tokenizer, tmpl, inps32, "sp", [sp_m_ncm, sp_if_ncm], "NormClipMuon [M;I]")
    results["ifbench32_verify"]["kv16_single_IF"] = eval_ifbench_quick(student, tokenizer, tmpl, inps32, "kv", [kv_if_16], "KV-CSPD L16 Single P_if")
    results["ifbench32_verify"]["kv16_concat_M_I"] = eval_ifbench_quick(student, tokenizer, tmpl, inps32, "kv", [kv_m_16, kv_if_16], "KV-CSPD L16 [M;I] (32tok!)")

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== Suite 1 (exp3) COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
