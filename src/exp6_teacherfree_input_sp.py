"""
Paper Reference: Section 6.1 (Single-Slot Teacher-Free Baselines: RFT, iRFT, DPO, GRPO, KV2SP).
Suite 4 (exp6): Teacher-Free Input-Space Soft Prompt Acquisition & Arbitrary-Context Assembly
===========================================================================================
ZERO Post-RL teacher models are ever loaded or used during training!
Only the frozen Pre-RL Student (Jackwang111/M2RL-SFT) + deterministic verifiers are used.

Tests 5 Teacher-Free Input-Space SP acquisition methods (L=256, Task-Inst Init, Arbitrary-Context PR):
  1. TF-RFT         : Single-round hardness-weighted Best-of-6 Rejection Sampling (BoN-RFT)
  2. TF-IterRFT     : 2-round On-Policy Expert Iteration (STaR bootstrap with P^(1) rollouts)
  3. TF-Contrastive : Verifier-Guided DPO + NLL Anchor on Student self-sampled (win, lose) pairs
  4. TF-GRPO        : True Online On-Policy GRPO Policy Gradient directly on Layer-0 Input-SP P
  5. TF-KV2SP       : Teacher-Free 2-stage bridge: fast 16-tok TF-KV scaffold (0 weight updates)
                      -> dense Top-64 logit compilation into Arbitrary-Context Input-SP!

Also trains TF-Math (Teacher-Free Input-SP on MATH) and evaluates:
  - IFBench (40 stratified prompts, max_new_tokens=2560) across all 5 methods + Dual [Math; IF]
  - Held-Out MATH-500 (40 disjoint Level 3-5 prompts) for Student, TF-Math, Dual [Math; IF]
  - CRITICAL Arbitrary-Context Assembly Stress Test (C0 Clean vs C1 Dynamic 120-tok System Persona
    vs C2 180-tok Pre-Context / Multi-Turn History placed BEFORE Input-SP vs fixed KV prefix!)
  - Post-hoc mechanistic check: does Teacher-Free Input-SP spontaneously move closer (G > 0) to
    the UNSEEN Post-RL expert M2RL-RL_IF and shift the same Check/But/?/Thus tokens?
"""

import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content/cspd")
sys.path.insert(0, "/content/IFBench")

from cspd_core import (
    ChatTemplateTokens,
    build_soft_prompt_inputs,
    get_qwen_template_tokens,
    length_normalized_nll_loss,
    pad_and_stack_train_batch,
    sample_random_prefix,
)
from benchmarks import (
    SUPPORTED_IFEVAL_IDS,
    batch_generate_with_soft_prompts,
    strip_thinking,
    verify_ifeval_response,
    verify_math_response,
)
from kv_prefix_core import (
    batch_generate_with_kv_prefixes,
    build_chat_ids_for_kv,
    forward_with_kv_prefix,
    init_kv_prefix_from_tokens,
)


def pad_kv_train_batch(
    ids_list: List[List[int]], labels_list: List[List[int]], pad_id: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(ids_list)
    T = max(len(x) for x in ids_list)
    bi = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    bm = torch.zeros((B, T), dtype=torch.long, device=device)
    bl = torch.full((B, T), -100, dtype=torch.long, device=device)
    for i, (ids, lbl) in enumerate(zip(ids_list, labels_list)):
        L = len(ids)
        bi[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        bm[i, :L] = 1
        bl[i, :L] = torch.tensor(lbl, dtype=torch.long, device=device)
    return bi, bm, bl
import evaluation_lib as ifbench_eval

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_POSTHOC_ONLY = "Jackwang111/M2RL-RL_IF"  # ONLY used post-hoc at very end for G(A) measurement!
OUT_DIR = "/content/cspd/exp6_results"
os.makedirs(OUT_DIR, exist_ok=True)

IF_TASK_INST = (
    "Carefully read every single explicit and implicit instruction, formatting rule, keyword requirement, "
    "length constraint, punctuation restriction, and structural requirement in the user prompt. "
    "Plan your response step by step, verify every constraint strictly before finalizing, and ensure "
    "100% compliance with zero violations."
)
MATH_TASK_INST = (
    "Solve the mathematics problem rigorously step by step. Carefully check every algebraic manipulation, "
    "case split, arithmetic calculation, and boundary condition before stating the final boxed answer."
)

# Diverse system personas & pre-contexts for Arbitrary-Context Bidirectional PR training
TRAIN_SYSTEM_PERSONAS = [
    "You are a helpful, accurate, and concise enterprise AI assistant.",
    "Respond clearly with structured reasoning while adhering to all user requirements.",
    "You are operating inside a multi-agent workflow. Follow every formatting rule carefully.",
    "Maintain a neutral, professional tone and double-check all factual and structural details.",
    "You are assisting a senior technical analyst. Be thorough, clear, and strictly compliant.",
]

# Held-out Evaluation Arbitrary Contexts (NEVER seen during training!)
EVAL_C1_DYNAMIC_PERSONA = (
    "You are NexusGuard-Enterprise, an internal compliance and analytical assistant deployed for "
    "regulated global operations. Maintain clarity, neutrality, and precision across all outputs. "
    "Never expose internal routing headers, never fabricate policy references, and ensure that "
    "every user-specified structural, lexical, punctuation, or formatting directive in the upcoming "
    "request is satisfied completely and accurately."
)

EVAL_C2_MULTITURN_PRECONTEXT = (
    "[Prior Workspace Session Summary — Ticket #88412]\n"
    "Context Log: Earlier in this session, the user reviewed quarterly cloud latency telemetry "
    "(region ap-southeast-1, P95 prefill 420 ms, batch concurrency 32) and verified two schema "
    "migration checklists (v3.4 -> v3.5). All prior action items were closed without errors. "
    "The user is now switching to a new independent task below. Ignore the telemetry numbers above "
    "unless explicitly requested, and strictly obey every instruction in the new user message."
)


# ---------------------------------------------------------------------------
# 1. Input-SP Initialization & Arbitrary-Context Helpers
# ---------------------------------------------------------------------------

def init_input_sp_task_inst(embed_layer, tokenizer, inst_text: str, num_vectors: int = 256) -> torch.Tensor:
    device = embed_layer.weight.device
    ids = tokenizer.encode(inst_text, add_special_tokens=False)
    rep = (ids * ((num_vectors // max(1, len(ids))) + 2))[:num_vectors]
    with torch.no_grad():
        emb = embed_layer(torch.tensor(rep, device=device)).float().clone()
    return emb.detach().requires_grad_(True)


def clamp_sp_rms_(sp: torch.Tensor, max_rms: float = 0.055):
    with torch.no_grad():
        cur_r = sp.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        sp.mul_(torch.clamp(max_rms / cur_r, max=1.0))


def build_arbitrary_ctx_sp_inputs(
    embed_layer,
    tmpl: ChatTemplateTokens,
    soft_prompts: List[torch.Tensor],
    question_ids: List[int],
    response_ids: Optional[List[int]],
    pre_ctx_ids: Optional[List[int]],
    post_sys_ids: Optional[List[int]],
    device: torch.device,
    model_dtype: torch.dtype,
):
    """
    Wrapper around build_soft_prompt_inputs that places arbitrary pre_ctx_ids BEFORE
    soft_prompts and arbitrary post_sys_ids AFTER soft_prompts inside the system turn:
      <|im_start|>system\n [pre_ctx] [P_1; ...; P_k] [post_sys] <|im_end|>\n ...
    """
    return build_soft_prompt_inputs(
        embed_layer=embed_layer,
        tmpl=tmpl,
        soft_prompts=soft_prompts,
        question_ids=question_ids,
        response_ids=response_ids,
        system_text_ids=post_sys_ids,
        prefix_F_ids=pre_ctx_ids,
        device=device,
        model_dtype=model_dtype,
    )


@torch.no_grad()
def batch_generate_sp_arbitrary_ctx(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    questions: List[str],
    soft_prompts: List[torch.Tensor],
    pre_ctx_text: Optional[str] = None,
    post_sys_text: Optional[str] = None,
    batch_size: int = 40,
    max_new_tokens: int = 2560,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    enable_thinking: bool = True,
) -> List[str]:
    """
    Batched generation supporting arbitrary text placed BEFORE and/or AFTER input-space soft prompts.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = embed_layer(torch.tensor([pad_id], device=device)).to(dtype)

    pre_ids = tokenizer.encode(pre_ctx_text + "\n\n", add_special_tokens=False) if pre_ctx_text else None
    post_ids = tokenizer.encode(post_sys_text, add_special_tokens=False) if post_sys_text else None

    all_outs: List[str] = []
    for start in range(0, len(questions), batch_size):
        bq = questions[start : start + batch_size]
        seq_embeds = []
        for q in bq:
            q_ids = tokenizer.encode(q, add_special_tokens=False)
            emb, _, _ = build_arbitrary_ctx_sp_inputs(
                embed_layer, tmpl, soft_prompts, q_ids, None, pre_ids, post_ids, device, dtype
            )
            seq_embeds.append(emb.squeeze(0))

        max_len = max(e.size(0) for e in seq_embeds)
        bs = len(seq_embeds)
        d = seq_embeds[0].size(1)
        padded = pad_vec.expand(bs, max_len, d).clone()
        attn_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for i, e in enumerate(seq_embeds):
            L = e.size(0)
            padded[i, max_len - L :, :] = e
            attn_mask[i, max_len - L :] = 1

        gen_kwargs = dict(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            gen_kwargs.update(do_sample=False)

        out_ids = model.generate(**gen_kwargs)
        for row in out_ids:
            all_outs.append(tokenizer.decode(row, skip_special_tokens=True))
    return all_outs


def seq_mean_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return length-normalized mean log-prob per sequence [B] over labels != -100."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    B = shift_logits.size(0)
    out = []
    for i in range(B):
        m = shift_labels[i] != -100
        if m.sum() == 0:
            out.append(shift_logits[i, 0, 0] * 0.0)
            continue
        lp = F.log_softmax(shift_logits[i][m], dim=-1)
        tok_lp = lp.gather(-1, shift_labels[i][m].unsqueeze(-1)).squeeze(-1)
        out.append(tok_lp.mean())
    return torch.stack(out)


@torch.no_grad()
def compute_ref_mean_logprobs(
    student, s_emb, tmpl, items: List[Tuple[List[int], List[int]]], pad_vec, max_resp_trunc: int = 768
) -> List[float]:
    """Compute bare-student (P=None) length-normalized mean log-prob for each (q_ids, r_ids)."""
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    student.eval()
    student.config.use_cache = False
    res = []
    for start in range(0, len(items), 8):
        sub = items[start : start + 8]
        el, ll = [], []
        for qi, ri in sub:
            e, l, _ = build_soft_prompt_inputs(s_emb, tmpl, [], qi, ri[:max_resp_trunc], None, None, device, dtype)
            el.append(e); ll.append(l)
        be, bm, bl = pad_and_stack_train_batch(el, ll, pad_vec)
        out = student(inputs_embeds=be, attention_mask=bm)
        lps = seq_mean_logprob(out.logits, bl)
        res.extend([float(x.item()) for x in lps])
        del be, bm, bl, out
    student.config.use_cache = True
    return res


# ---------------------------------------------------------------------------
# 2. Data Loading & Teacher-Free Rollout Harvesting from Frozen Student ONLY
# ---------------------------------------------------------------------------

def load_verifiable_datasets():
    # IF training pool: google/IFEval (strictly disjoint from IFBench test!)
    ifeval_ds = load_dataset("google/IFEval", split="train")
    if_pool = []
    for row in ifeval_ds:
        iids = row["instruction_id_list"]
        if iids and all(iid in SUPPORTED_IFEVAL_IDS for iid in iids):
            if_pool.append({
                "prompt": row["prompt"],
                "iids": iids,
                "kwargs": row["kwargs"],
            })
    rng = random.Random(42)
    rng.shuffle(if_pool)
    if_train_prompts = if_pool[:64]

    # Math train & held-out test: HuggingFaceH4/MATH-500 (strictly disjoint split!)
    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_all = []
    for row in math_ds:
        math_all.append({
            "prompt": row["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
            "answer": str(row["answer"]),
            "level": int(row.get("level", 3)),
        })
    rng_m = random.Random(42)
    rng_m.shuffle(math_all)
    # Held-out test: 32 Level 3-5 problems (1 single batch of 32!)
    hard_math = [x for x in math_all if x["level"] >= 3]
    math_test = hard_math[:32]
    test_set_prompts = {x["prompt"] for x in math_test}
    math_train_prompts = [x for x in math_all if x["prompt"] not in test_set_prompts][:48]

    # Official IFBench test: EXACT SAME 32 stratified prompts as Suite 1 (exp3) for 100% apples-to-apples comparison!
    all_ifb = ifbench_eval.read_prompt_list("/content/IFBench/data/IFBench_test.jsonl")
    rng_ifb = random.Random(42)
    idxs = list(range(len(all_ifb)))
    rng_ifb.shuffle(idxs)
    ifbench32 = [all_ifb[i] for i in sorted(idxs[:32])]

    print(f"Datasets ready | IFEval train={len(if_train_prompts)} | MATH train={len(math_train_prompts)} | MATH test={len(math_test)} | IFBench32 test={len(ifbench32)}", flush=True)
    return if_train_prompts, math_train_prompts, math_test, ifbench32


def harvest_if_rollouts(
    student, tokenizer, tmpl, prompts_meta: List[Dict[str, Any]],
    soft_prompts: List[torch.Tensor], n_samples: int = 6, max_new_tokens: int = 1536,
    temperature: float = 0.8, seed: int = 101, tag: str = "R1",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate n_samples rollouts per prompt using ONLY frozen student (+ optional soft_prompts).
    Returns:
      wins  : verified strict-passing closed-think trajectories with hardness weight
      pairs : same-prompt (win, lose) contrastive pairs
      stats : pass@1, pass@N, closed rate, etc.
    """
    t0 = time.time()
    torch.manual_seed(seed)
    flat_q = []
    flat_meta = []
    for idx, item in enumerate(prompts_meta):
        for k in range(n_samples):
            flat_q.append(item["prompt"])
            flat_meta.append(idx)

    outs = batch_generate_with_soft_prompts(
        student, tokenizer, tmpl, flat_q, soft_prompts,
        batch_size=32, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=0.95, top_k=20, enable_thinking=True,
    )

    per_prompt = [[] for _ in range(len(prompts_meta))]
    total_strict = 0
    total_inst = 0.0
    total_closed = 0
    for idx, resp in zip(flat_meta, outs):
        meta = prompts_meta[idx]
        closed = "</think>" in resp
        strict, inst_r = verify_ifeval_response(resp, meta["iids"], meta["kwargs"])
        if not closed:
            strict = False
        total_strict += int(strict)
        total_inst += inst_r
        total_closed += int(closed)
        per_prompt[idx].append({
            "resp": resp, "strict": strict, "inst_r": inst_r, "closed": closed,
        })

    wins, pairs = [], []
    prompts_with_win = 0
    for idx, rollouts in enumerate(per_prompt):
        prompt = prompts_meta[idx]["prompt"]
        q_ids = tokenizer.encode(prompt, add_special_tokens=False)
        good = [r for r in rollouts if r["strict"] and r["closed"]]
        bad_closed = [r for r in rollouts if (not r["strict"]) and r["closed"]]
        bad_any = [r for r in rollouts if not r["strict"]]
        if good:
            prompts_with_win += 1
            # Hardness weight: prompts solved 1/6 times get much higher weight than 6/6
            w_hard = 1.0 / (len(good) + 0.5)
            for g in good:
                r_ids = tokenizer.encode(g["resp"], add_special_tokens=False)
                if len(r_ids) >= 16:
                    wins.append({
                        "prompt_idx": idx, "prompt": prompt, "q_ids": q_ids,
                        "r_ids": r_ids, "weight": w_hard, "pass_cnt": len(good),
                    })
            bad_pool = bad_closed if bad_closed else bad_any
            if bad_pool:
                # Up to 2 contrastive pairs per prompt (prefer closed-think failures!)
                bad_pool_sorted = sorted(bad_pool, key=lambda x: x["inst_r"])
                for pair_i in range(min(2, len(good), len(bad_pool_sorted))):
                    g = good[pair_i]
                    b = bad_pool_sorted[pair_i]
                    g_ids = tokenizer.encode(g["resp"], add_special_tokens=False)
                    b_ids = tokenizer.encode(b["resp"], add_special_tokens=False)
                    if len(g_ids) >= 16 and len(b_ids) >= 16:
                        pairs.append({
                            "prompt_idx": idx, "prompt": prompt, "q_ids": q_ids,
                            "win_ids": g_ids, "lose_ids": b_ids,
                            "reward_gap": round(1.0 - b["inst_r"], 3),
                        })

    n_tot = len(flat_q)
    stats = {
        "tag": tag,
        "num_prompts": len(prompts_meta),
        "n_samples": n_samples,
        "pass_at_1_pct": round(100.0 * total_strict / max(1, n_tot), 2),
        "inst_rate_pct": round(100.0 * total_inst / max(1, n_tot), 2),
        "pass_at_N_pct": round(100.0 * prompts_with_win / max(1, len(prompts_meta)), 2),
        "closed_think_pct": round(100.0 * total_closed / max(1, n_tot), 2),
        "num_wins": len(wins),
        "num_pairs": len(pairs),
        "harvest_sec": round(time.time() - t0, 1),
    }
    print(f"  [Harvest IF {tag}] pass@1={stats['pass_at_1_pct']:.1f}% | pass@{n_samples}={stats['pass_at_N_pct']:.1f}% | closed={stats['closed_think_pct']:.1f}% | wins={len(wins)} | pairs={len(pairs)} in {stats['harvest_sec']}s", flush=True)
    return wins, pairs, stats


def harvest_math_rollouts(
    student, tokenizer, tmpl, prompts_meta: List[Dict[str, Any]],
    n_samples: int = 6, max_new_tokens: int = 1536, temperature: float = 0.8, seed: int = 202,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    t0 = time.time()
    torch.manual_seed(seed)
    flat_q, flat_idx = [], []
    for idx, item in enumerate(prompts_meta):
        for _ in range(n_samples):
            flat_q.append(item["prompt"])
            flat_idx.append(idx)

    outs = batch_generate_with_soft_prompts(
        student, tokenizer, tmpl, flat_q, [],
        batch_size=32, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=0.95, top_k=20, enable_thinking=True,
    )
    per_p = [[] for _ in range(len(prompts_meta))]
    tot_ok = 0
    for idx, resp in zip(flat_idx, outs):
        closed = "</think>" in resp
        ok = closed and verify_math_response(resp, prompts_meta[idx]["answer"])
        tot_ok += int(ok)
        per_p[idx].append({"resp": resp, "ok": ok, "closed": closed})

    wins, pairs = [], []
    p_solved = 0
    for idx, rollouts in enumerate(per_p):
        prompt = prompts_meta[idx]["prompt"]
        q_ids = tokenizer.encode(prompt, add_special_tokens=False)
        good = [r for r in rollouts if r["ok"]]
        bad = [r for r in rollouts if (not r["ok"]) and r["closed"]] or [r for r in rollouts if not r["ok"]]
        if good:
            p_solved += 1
            w_hard = 1.0 / (len(good) + 0.5)
            for g in good:
                r_ids = tokenizer.encode(g["resp"], add_special_tokens=False)
                if len(r_ids) >= 16:
                    wins.append({"prompt_idx": idx, "prompt": prompt, "q_ids": q_ids, "r_ids": r_ids, "weight": w_hard})
            if bad:
                for pi in range(min(2, len(good), len(bad))):
                    g_ids = tokenizer.encode(good[pi]["resp"], add_special_tokens=False)
                    b_ids = tokenizer.encode(bad[pi]["resp"], add_special_tokens=False)
                    if len(g_ids) >= 16 and len(b_ids) >= 16:
                        pairs.append({"prompt_idx": idx, "prompt": prompt, "q_ids": q_ids, "win_ids": g_ids, "lose_ids": b_ids})

    stats = {
        "pass_at_1_pct": round(100.0 * tot_ok / max(1, len(flat_q)), 2),
        "pass_at_N_pct": round(100.0 * p_solved / max(1, len(prompts_meta)), 2),
        "num_wins": len(wins),
        "num_pairs": len(pairs),
        "harvest_sec": round(time.time() - t0, 1),
    }
    print(f"  [Harvest Math] pass@1={stats['pass_at_1_pct']:.1f}% | pass@{n_samples}={stats['pass_at_N_pct']:.1f}% | wins={len(wins)} | pairs={len(pairs)} in {stats['harvest_sec']}s", flush=True)
    return wins, pairs, stats


# ---------------------------------------------------------------------------
# 3. Five Teacher-Free Input-Space Soft Prompt Trainers
# ---------------------------------------------------------------------------

def make_ctx_stream(tokenizer, prompts: List[str]) -> List[int]:
    stream = []
    for p in prompts:
        stream.extend(tokenizer.encode(p, add_special_tokens=False))
    for s in TRAIN_SYSTEM_PERSONAS:
        stream.extend(tokenizer.encode(s, add_special_tokens=False))
    return stream


def sample_arb_ctx_pair(stream: List[int], persona_ids_list: List[List[int]], rng: random.Random):
    """Arbitrary-Context Bidirectional PR: 50% pre-context before SP, 35% post-system text after SP."""
    pre = sample_random_prefix(stream, rng, 64, 192) if rng.random() < 0.5 else None
    post = rng.choice(persona_ids_list) if rng.random() < 0.35 else None
    return pre, post


def train_tf_rft_sp(
    student, tokenizer, tmpl, wins: List[Dict[str, Any]], stream: List[int],
    task_inst: str = IF_TASK_INST, num_vectors: int = 256, num_updates: int = 45,
    batch_size: int = 16, micro_batch: int = 2, lr_max: float = 0.005, max_rms: float = 0.055,
    max_resp_trunc: int = 768, init_sp: Optional[torch.Tensor] = None, seed: int = 42, tag: str = "TF_RFT",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Method 1 & 2: Hardness-Weighted Rejection-Sampling Fine-Tuning on Input-Space SP."""
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()
    persona_ids = [tokenizer.encode(s, add_special_tokens=False) for s in TRAIN_SYSTEM_PERSONAS]

    sp = init_sp.detach().clone().float().requires_grad_(True) if init_sp is not None else init_input_sp_task_inst(s_emb, tokenizer, task_inst, num_vectors)
    opt = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)
    weights = [w["weight"] for w in wins]
    num_mb = batch_size // micro_batch

    nll_log = {}
    for step in range(1, num_updates + 1):
        prog = (step - 1) / max(1, num_updates - 1)
        lr = lr_max * 0.1 + 0.5 * (lr_max * 0.9) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        student.config.use_cache = False
        student.train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt.zero_grad()

        batch = rng.choices(wins, weights=weights, k=batch_size)
        batch.sort(key=lambda x: len(x["q_ids"]) + min(len(x["r_ids"]), max_resp_trunc))
        step_loss = 0.0
        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            el, ll = [], []
            for item in sub:
                pre, post = sample_arb_ctx_pair(stream, persona_ids, rng)
                e, l, _ = build_arbitrary_ctx_sp_inputs(
                    s_emb, tmpl, [sp], item["q_ids"], item["r_ids"][:max_resp_trunc], pre, post, device, dtype
                )
                el.append(e); ll.append(l)
            be, bm, bl = pad_and_stack_train_batch(el, ll, pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm)
            loss = length_normalized_nll_loss(out.logits, bl)
            (loss / num_mb).backward()
            step_loss += float(loss.item()) / num_mb
            del be, bm, bl, out

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)

        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True
        if step in {1, 15, 30, num_updates}:
            nll_log[str(step)] = round(step_loss, 4)
            print(f"  [{tag:18s}] step {step:2d}/{num_updates} | nll={step_loss:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "method": "rft_nll", "num_vectors": num_vectors, "num_wins": len(wins),
        "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_seconds": round(time.time() - t0, 1), "nll_log": nll_log,
    }
    return sp.detach().clone(), meta


def train_tf_contrastive_sp(
    student, tokenizer, tmpl, pairs: List[Dict[str, Any]], stream: List[int],
    task_inst: str = IF_TASK_INST, num_vectors: int = 256, num_updates: int = 45,
    batch_size: int = 8, lr_max: float = 0.004, beta_dpo: float = 2.5, alpha_nll: float = 0.45,
    max_rms: float = 0.055, max_resp_trunc: int = 768, seed: int = 42, tag: str = "TF_Contrastive",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Method 3: Verifier-Guided DPO + NLL Anchor on Student self-sampled (win, lose) pairs!
    Zero Post-RL teacher, zero extra reference model (precomputes bare student ref logprobs once).
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()
    persona_ids = [tokenizer.encode(s, add_special_tokens=False) for s in TRAIN_SYSTEM_PERSONAS]

    # Precompute bare-student reference length-normalized logprobs once
    ref_win_lps = compute_ref_mean_logprobs(student, s_emb, tmpl, [(p["q_ids"], p["win_ids"]) for p in pairs], pad_vec, max_resp_trunc)
    ref_lose_lps = compute_ref_mean_logprobs(student, s_emb, tmpl, [(p["q_ids"], p["lose_ids"]) for p in pairs], pad_vec, max_resp_trunc)
    for p, rw, rl in zip(pairs, ref_win_lps, ref_lose_lps):
        p["ref_w"] = rw
        p["ref_l"] = rl

    sp = init_input_sp_task_inst(s_emb, tokenizer, task_inst, num_vectors)
    opt = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)
    log_hist = {}

    for step in range(1, num_updates + 1):
        prog = (step - 1) / max(1, num_updates - 1)
        lr = lr_max * 0.1 + 0.5 * (lr_max * 0.9) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        student.config.use_cache = False
        student.train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt.zero_grad()

        batch = [pairs[rng.randrange(len(pairs))] for _ in range(batch_size)]
        tot_dpo, tot_nll, tot_acc = 0.0, 0.0, 0.0
        for item in batch:
            pre, post = sample_arb_ctx_pair(stream, persona_ids, rng)
            ew, lw, _ = build_arbitrary_ctx_sp_inputs(s_emb, tmpl, [sp], item["q_ids"], item["win_ids"][:max_resp_trunc], pre, post, device, dtype)
            el, ll, _ = build_arbitrary_ctx_sp_inputs(s_emb, tmpl, [sp], item["q_ids"], item["lose_ids"][:max_resp_trunc], pre, post, device, dtype)
            be, bm, bl = pad_and_stack_train_batch([ew, el], [lw, ll], pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm)
            lps = seq_mean_logprob(out.logits, bl)  # [2]: win, lose
            d_win = lps[0] - item["ref_w"]
            d_lose = lps[1] - item["ref_l"]
            margin = beta_dpo * (d_win - d_lose)
            loss_dpo = -F.logsigmoid(margin)
            loss_nll = -lps[0]
            loss = loss_dpo + alpha_nll * loss_nll
            (loss / batch_size).backward()
            tot_dpo += float(loss_dpo.item()) / batch_size
            tot_nll += float(loss_nll.item()) / batch_size
            tot_acc += float((d_win > d_lose).item()) / batch_size
            del be, bm, bl, out

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)

        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True
        if step in {1, 15, 30, num_updates}:
            log_hist[str(step)] = {"dpo": round(tot_dpo, 4), "nll": round(tot_nll, 4), "pref_acc": round(100.0 * tot_acc, 1)}
            print(f"  [{tag:18s}] step {step:2d}/{num_updates} | dpo={tot_dpo:.4f} | nll={tot_nll:.4f} | pref_acc={100*tot_acc:.1f}% | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "method": "contrastive_dpo_nll", "num_vectors": num_vectors, "num_pairs": len(pairs),
        "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_seconds": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


def train_tf_grpo_sp(
    student, tokenizer, tmpl, prompts_meta: List[Dict[str, Any]], stream: List[int],
    task_inst: str = IF_TASK_INST, num_vectors: int = 256, num_steps: int = 20,
    prompts_per_step: int = 4, group_size: int = 6, max_new_tokens: int = 1024,
    lr_max: float = 0.0018, beta_kl: float = 0.04, max_rms: float = 0.055,
    seed: int = 42, tag: str = "TF_GRPO",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Method 4: True Online On-Policy GRPO Policy Gradient directly on Layer-0 Input-Space Soft Prompt!
    Zero Post-RL teacher! Uses dense verifiable reward (0.5*strict + 0.5*inst_rate - 0.2*unclosed)
    and exact Schulman reverse-KL penalty to bare frozen student (P=None).
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()
    persona_ids = [tokenizer.encode(s, add_special_tokens=False) for s in TRAIN_SYSTEM_PERSONAS]

    sp = init_input_sp_task_inst(s_emb, tokenizer, task_inst, num_vectors)
    opt = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)
    online_curve = []

    for step in range(1, num_steps + 1):
        prog = (step - 1) / max(1, num_steps - 1)
        lr = lr_max * 0.2 + 0.5 * (lr_max * 0.8) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        # 1. Sample online rollouts from current policy p_{theta_0}(.|I(sp, x))
        step_prompts = [prompts_meta[rng.randrange(len(prompts_meta))] for _ in range(prompts_per_step)]
        flat_q = []
        for pm in step_prompts:
            flat_q.extend([pm["prompt"]] * group_size)

        student.eval()
        student.config.use_cache = True
        torch.manual_seed(seed * 100 + step)
        outs = batch_generate_with_soft_prompts(
            student, tokenizer, tmpl, flat_q, [sp.detach()],
            batch_size=prompts_per_step * group_size, max_new_tokens=max_new_tokens,
            temperature=0.85, top_p=0.95, top_k=20, enable_thinking=True,
        )

        # 2. Compute exact group-relative advantages
        samples_to_opt = []
        tot_r, tot_strict, tot_closed = 0.0, 0, 0
        for p_i, pm in enumerate(step_prompts):
            grp = outs[p_i * group_size : (p_i + 1) * group_size]
            rewards = []
            for resp in grp:
                closed = "</think>" in resp
                strict, inst_r = verify_ifeval_response(resp, pm["iids"], pm["kwargs"])
                if not closed:
                    strict = False
                rew = (0.5 * float(strict) + 0.5 * inst_r) if closed else (0.2 * inst_r - 0.2)
                rewards.append(rew)
                tot_r += rew; tot_strict += int(strict); tot_closed += int(closed)
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if std_r < 1e-4:
                continue
            q_ids = tokenizer.encode(pm["prompt"], add_special_tokens=False)
            for resp, rew in zip(grp, rewards):
                adv = (rew - mean_r) / (std_r + 1e-4)
                r_ids = tokenizer.encode(resp, add_special_tokens=False)[:768]
                if len(r_ids) >= 12 and abs(adv) > 0.05:
                    samples_to_opt.append((q_ids, r_ids, adv))

        mean_step_r = tot_r / max(1, len(flat_q))
        strict_pct = 100.0 * tot_strict / max(1, len(flat_q))
        closed_pct = 100.0 * tot_closed / max(1, len(flat_q))

        # 3. Precompute bare-student reference token log-probs (P=None) & apply GRPO step
        mean_kl = 0.0
        if samples_to_opt:
            student.eval()
            student.config.use_cache = False
            ref_tok_lps = []
            with torch.no_grad():
                for qi, ri, _ in samples_to_opt:
                    e, l, _ = build_soft_prompt_inputs(s_emb, tmpl, [], qi, ri, None, None, device, dtype)
                    out = student(inputs_embeds=e)
                    sl = out.logits[0, :-1, :].float()
                    lb = l[0, 1:]
                    m = lb != -100
                    lp = F.log_softmax(sl[m], dim=-1).gather(-1, lb[m].unsqueeze(-1)).squeeze(-1)
                    ref_tok_lps.append(lp.detach())
                    del e, l, out

            student.train()
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            opt.zero_grad()
            n_opt = len(samples_to_opt)
            for (qi, ri, adv), ref_lp in zip(samples_to_opt, ref_tok_lps):
                pre, post = sample_arb_ctx_pair(stream, persona_ids, rng)
                e, l, _ = build_arbitrary_ctx_sp_inputs(s_emb, tmpl, [sp], qi, ri, pre, post, device, dtype)
                out = student(inputs_embeds=e)
                sl = out.logits[0, :-1, :].float()
                lb = l[0, 1:]
                m = lb != -100
                cur_lp = F.log_softmax(sl[m], dim=-1).gather(-1, lb[m].unsqueeze(-1)).squeeze(-1)
                n_tok = min(cur_lp.size(0), ref_lp.size(0))
                cur_lp = cur_lp[:n_tok]
                r_lp = ref_lp[:n_tok]
                # Schulman non-negative KL estimator: exp(log p_ref - log p_cur) - (log p_ref - log p_cur) - 1
                log_ratio = (r_lp - cur_lp).clamp(-10.0, 10.0)
                kl_est = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
                pg_loss = -(adv * cur_lp.mean()) + beta_kl * kl_est
                (pg_loss / n_opt).backward()
                mean_kl += float(kl_est.item()) / n_opt
                del e, l, out

            torch.nn.utils.clip_grad_norm_([sp], 1.0)
            opt.step()
            clamp_sp_rms_(sp, max_rms)
            student.gradient_checkpointing_disable()
            student.eval()
            student.config.use_cache = True

        rec = {
            "step": step, "mean_reward": round(mean_step_r, 4), "strict_pct": round(strict_pct, 1),
            "closed_pct": round(closed_pct, 1), "kl_to_ref": round(mean_kl, 4),
            "n_opt": len(samples_to_opt), "rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 4),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        online_curve.append(rec)
        if step in {1, 5, 10, 15, num_steps}:
            print(f"  [{tag:18s}] step {step:2d}/{num_steps} | rew={rec['mean_reward']:.3f} | strict={rec['strict_pct']:.1f}% | kl={rec['kl_to_ref']:.4f} | n_opt={rec['n_opt']} | rms={rec['rms']:.4f} | t={rec['elapsed_sec']}s", flush=True)

    meta = {
        "tag": tag, "method": "online_grpo", "num_vectors": num_vectors, "num_steps": num_steps,
        "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_seconds": round(time.time() - t0, 1), "online_curve": online_curve,
    }
    return sp.detach().clone(), meta


def train_tf_kv2sp(
    student, tokenizer, tmpl, wins: List[Dict[str, Any]], pairs: List[Dict[str, Any]], stream: List[int],
    task_inst: str = IF_TASK_INST, num_kv_tokens: int = 16, num_vectors: int = 256,
    kv_updates: int = 60, sp_updates: int = 45, max_resp_trunc: int = 640, seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Method 5: Teacher-Free Two-Stage Bridge (TF-KV2SP)!
      Stage 5a: Train tiny 16-tok TF-KV scaffold via Verifier DPO+NLL on Student self-sampled pairs
                (ZERO Post-RL model, ZERO weight updates to 4B backbone!)
      Stage 5b: Precompute dense Top-64 logits from (Frozen Student + TF-KV) on verified wins
      Stage 5c: Compile dense Top-64 logits into a pure Layer-0 Arbitrary-Context Input-SP!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()
    persona_ids = [tokenizer.encode(s, add_special_tokens=False) for s in TRAIN_SYSTEM_PERSONAS]

    # --- Stage 5a: Train Teacher-Free KV_L16 scaffold with Verifier DPO+NLL ---
    inst_ids = tokenizer.encode(task_inst, add_special_tokens=False)
    kv = init_kv_prefix_from_tokens(student, inst_ids, num_kv_tokens=num_kv_tokens)
    init_kv = kv.detach().clone()

    # Ensure ref_w / ref_l exist on pairs
    if "ref_w" not in pairs[0]:
        rw = compute_ref_mean_logprobs(student, s_emb, tmpl, [(p["q_ids"], p["win_ids"]) for p in pairs], pad_vec, max_resp_trunc)
        rl = compute_ref_mean_logprobs(student, s_emb, tmpl, [(p["q_ids"], p["lose_ids"]) for p in pairs], pad_vec, max_resp_trunc)
        for p, a, b in zip(pairs, rw, rl):
            p["ref_w"], p["ref_l"] = a, b

    student.gradient_checkpointing_disable()
    student.eval()
    student.config.use_cache = True
    opt_kv = torch.optim.AdamW([kv], lr=0.02, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)
    kv_log = {}

    for step in range(1, kv_updates + 1):
        prog = (step - 1) / max(1, kv_updates - 1)
        lr = 0.002 + 0.5 * 0.018 * (1.0 + math.cos(math.pi * prog))
        for pg in opt_kv.param_groups:
            pg["lr"] = lr
        opt_kv.zero_grad()

        batch = [pairs[rng.randrange(len(pairs))] for _ in range(8)]
        tot_dpo, tot_nll, tot_acc = 0.0, 0.0, 0.0
        for item in batch:
            iw, lw = build_chat_ids_for_kv(tmpl, item["q_ids"], item["win_ids"][:max_resp_trunc], None, True, None)
            il, ll = build_chat_ids_for_kv(tmpl, item["q_ids"], item["lose_ids"][:max_resp_trunc], None, True, None)
            bi, bm, bl = pad_kv_train_batch([iw, il], [lw, ll], pad_id, device)
            out_kv = forward_with_kv_prefix(student, bi, bm, [kv])
            lps = seq_mean_logprob(out_kv.logits, bl)
            d_w = lps[0] - item["ref_w"]
            d_l = lps[1] - item["ref_l"]
            loss_dpo = -F.logsigmoid(2.5 * (d_w - d_l))
            loss_nll = -lps[0]
            ((loss_dpo + 0.45 * loss_nll) / 8.0).backward()
            tot_dpo += float(loss_dpo.item()) / 8.0
            tot_nll += float(loss_nll.item()) / 8.0
            tot_acc += float((d_w > d_l).item()) / 8.0
            del bi, bm, bl, out_kv

        torch.nn.utils.clip_grad_norm_([kv], 1.0)
        opt_kv.step()
        with torch.no_grad():
            delta = kv - init_kv
            dr = delta.pow(2).mean().sqrt().clamp_min(1e-8)
            if dr > 0.55:
                kv.copy_(init_kv + delta * (0.55 / dr))
        if step in {1, 20, 40, kv_updates}:
            kv_log[str(step)] = {"dpo": round(tot_dpo, 4), "nll": round(tot_nll, 4), "pref_acc": round(100.0 * tot_acc, 1)}
            print(f"  [TF_KV2SP 5a-KV    ] step {step:2d}/{kv_updates} | dpo={tot_dpo:.4f} | nll={tot_nll:.4f} | pref_acc={100*tot_acc:.1f}% | d_rms={(kv.detach()-init_kv).pow(2).mean().sqrt().item():.3f} | t={time.time()-t0:.1f}s", flush=True)

    kv_sec = round(time.time() - t0, 1)
    kv_frozen = kv.detach().clone()

    # --- Stage 5b: Precompute Dense Top-64 Logits from (Frozen Student + TF-KV) ---
    t_pre = time.time()
    # Select up to 110 diverse verified wins (deduplicated/hardness prioritized)
    wins_sorted = sorted(wins, key=lambda x: (-x["weight"], x["prompt_idx"]))[:110]
    dense_cache = []
    with torch.no_grad():
        for item in wins_sorted:
            ri = item["r_ids"][:max_resp_trunc]
            ids, lbl = build_chat_ids_for_kv(tmpl, item["q_ids"], ri, None, True, None)
            bi = torch.tensor([ids], dtype=torch.long, device=device)
            bm = torch.ones_like(bi)
            out_kv = forward_with_kv_prefix(student, bi, bm, [kv_frozen])
            sl = out_kv.logits[0, :-1, :].float()
            lb = torch.tensor(lbl[1:], dtype=torch.long, device=device)
            m = lb != -100
            probs = F.softmax(sl[m], dim=-1)
            tp, ti = torch.topk(probs, 64, dim=-1)
            tp = tp / tp.sum(dim=-1, keepdim=True)
            dense_cache.append((item["q_ids"], ri, ti.cpu(), tp.to(torch.float16).cpu(), item["weight"]))
            del bi, bm, out_kv
    pre_sec = round(time.time() - t_pre, 1)
    print(f"  [TF_KV2SP 5b-Cache ] Precomputed {len(dense_cache)} dense Top-64 KV-scaffold targets in {pre_sec}s", flush=True)

    # --- Stage 5c: Compile Dense Top-64 Logits into Arbitrary-Context Layer-0 Input-SP ---
    t_sp = time.time()
    sp = init_input_sp_task_inst(s_emb, tokenizer, task_inst, num_vectors)
    opt_sp = torch.optim.AdamW([sp], lr=0.006, betas=(0.9, 0.999), weight_decay=0.0)
    sp_log = {}
    weights = [x[4] for x in dense_cache]
    num_mb = 8  # batch_size=16, micro_batch=2

    for step in range(1, sp_updates + 1):
        prog = (step - 1) / max(1, sp_updates - 1)
        lr = 0.0006 + 0.5 * 0.0054 * (1.0 + math.cos(math.pi * prog))
        for pg in opt_sp.param_groups:
            pg["lr"] = lr

        student.config.use_cache = False
        student.train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt_sp.zero_grad()

        batch = rng.choices(dense_cache, weights=weights, k=16)
        batch.sort(key=lambda x: len(x[0]) + len(x[1]))
        step_ce = 0.0
        for mb_i in range(0, 16, 2):
            sub = batch[mb_i : mb_i + 2]
            el, ll = [], []
            for qi, ri, _, _, _ in sub:
                pre, post = sample_arb_ctx_pair(stream, persona_ids, rng)
                e, l, _ = build_arbitrary_ctx_sp_inputs(s_emb, tmpl, [sp], qi, ri, pre, post, device, dtype)
                el.append(e); ll.append(l)
            be, bm, bl = pad_and_stack_train_batch(el, ll, pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm)
            sl = out.logits[:, :-1, :]
            lb = bl[:, 1:]
            per_s = []
            for i, (_, _, ti_c, tp_c, _) in enumerate(sub):
                m = lb[i] != -100
                s_resp = sl[i][m].float()
                ti = ti_c.to(device)
                tp = tp_c.float().to(device)
                n = min(s_resp.size(0), ti.size(0))
                s_lp = F.log_softmax(s_resp[:n], dim=-1)
                s_top_lp = s_lp.gather(-1, ti[:n])
                fwd_ce = -(tp[:n] * s_top_lp).sum(dim=-1)
                # Hybrid 0.75 fwd + 0.25 rev KL for crisp distillation
                s_top_p = F.softmax(s_top_lp, dim=-1)
                rev_kl = (s_top_p * (s_top_p.clamp_min(1e-7).log() - tp[:n].clamp_min(1e-7).log())).sum(dim=-1)
                per_s.append((0.75 * fwd_ce + 0.25 * rev_kl).mean())
            loss = torch.stack(per_s).mean()
            (loss / num_mb).backward()
            step_ce += float(loss.item()) / num_mb
            del be, bm, bl, out

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt_sp.step()
        clamp_sp_rms_(sp, 0.055)
        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True
        if step in {1, 15, 30, sp_updates}:
            sp_log[str(step)] = round(step_ce, 4)
            print(f"  [TF_KV2SP 5c-SP    ] step {step:2d}/{sp_updates} | dist_loss={step_ce:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t_sp:.1f}s", flush=True)

    meta = {
        "tag": "TF_KV2SP", "method": "kv_scaffold_to_input_sp",
        "kv_tokens": num_kv_tokens, "sp_vectors": num_vectors,
        "kv_delta_rms": round(float((kv_frozen - init_kv).pow(2).mean().sqrt().item()), 4),
        "sp_final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "kv_train_sec": kv_sec, "precompute_sec": pre_sec, "sp_compile_sec": round(time.time() - t_sp, 1),
        "total_train_sec": round(time.time() - t0, 1),
        "kv_log": kv_log, "sp_log": sp_log,
    }
    return kv_frozen, sp.detach().clone(), meta


# ---------------------------------------------------------------------------
# 4. Evaluators: Official IFBench40, Arbitrary-Context C0/C1/C2, MATH-500, Post-Hoc G(A)
# ---------------------------------------------------------------------------

def eval_ifbench40(
    model, tokenizer, tmpl, inps40, mode: str, adapters: List[torch.Tensor],
    pre_ctx: Optional[str] = None, post_sys: Optional[str] = None,
    max_new_tokens: int = 2048, tag: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(42)
    prompts = [x.prompt for x in inps40]
    if mode == "kv":
        # Fixed KV prefix cannot condition on text before it; any dynamic context must sit in system_texts AFTER KV!
        sys_txt = "\n\n".join([x for x in [pre_ctx, post_sys] if x]) or ""
        outs = batch_generate_with_kv_prefixes(
            model, tokenizer, tmpl, prompts, adapters, system_texts=[sys_txt] * len(prompts),
            batch_size=32, max_new_tokens=max_new_tokens,
            temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
        )
    else:
        outs = batch_generate_sp_arbitrary_ctx(
            model, tokenizer, tmpl, prompts, adapters,
            pre_ctx_text=pre_ctx, post_sys_text=post_sys,
            batch_size=32, max_new_tokens=max_new_tokens,
            temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
        )

    p2r = {}
    closed = 0
    total_chars = 0
    for inp, r in zip(inps40, outs):
        closed += int("</think>" in r)
        total_chars += len(r)
        p2r[inp.prompt] = strip_thinking(r)

    s_ok = sum(int(ifbench_eval.test_instruction_following_strict(inp, p2r).follow_all_instructions) for inp in inps40)
    inst_l = []
    for inp in inps40:
        inst_l.extend(ifbench_eval.test_instruction_following_strict(inp, p2r).follow_instruction_list)
    n = len(inps40)
    res = {
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * sum(inst_l) / max(1, len(inst_l)), 2),
        "n_strict": s_ok,
        "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "avg_chars": round(total_chars / n, 1),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFBench40 | {tag:30s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | inst={res['inst_strict']:5.2f}% | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def eval_math40(model, tokenizer, tmpl, math_test: List[Dict[str, Any]], soft_prompts: List[torch.Tensor], tag: str = "") -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(42)
    prompts = [x["prompt"] for x in math_test]
    outs = batch_generate_with_soft_prompts(
        model, tokenizer, tmpl, prompts, soft_prompts,
        batch_size=32, max_new_tokens=2048, temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    ok = 0
    closed = 0
    for item, r in zip(math_test, outs):
        c = "</think>" in r
        closed += int(c)
        if c and verify_math_response(r, item["answer"]):
            ok += 1
    n = len(math_test)
    res = {
        "acc": round(100.0 * ok / n, 2),
        "n_ok": ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [MATH40    | {tag:30s}] acc={res['acc']:5.2f}% ({ok:2d}/{n}) | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


@torch.no_grad()
def posthoc_alignment_check(student, tokenizer, tmpl, sp_dict: Dict[str, torch.Tensor], ifbench_prompts: List[str]) -> Dict[str, Any]:
    """
    ONLY executed after ALL teacher-free training & evaluation are 100% finished!
    Checks whether Teacher-Free Input-SPs spontaneously closed the KL gap G(A) toward
    the UNSEEN Post-RL expert M2RL-RL_IF and shifted verification token probabilities.
    """
    print("\n=== Post-Hoc Alignment Check vs UNSEEN M2RL-RL_IF (20 traces) ===", flush=True)
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()

    # Load unseen teacher temporarily on GPU (move student to CPU briefly to stay <20GB VRAM)
    student.cpu()
    torch.cuda.empty_cache()
    teacher = AutoModelForCausalLM.from_pretrained(IF_TEACHER_POSTHOC_ONLY, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    teacher.eval()
    torch.manual_seed(999)
    t_prompts = ifbench_prompts[:20]
    t_resps = batch_generate_with_soft_prompts(
        teacher, tokenizer, tmpl, t_prompts, [],
        batch_size=20, max_new_tokens=1024, temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    t_emb = teacher.get_input_embeddings()
    cached = []
    for p, r in zip(t_prompts, t_resps):
        qi = tokenizer.encode(p, add_special_tokens=False)
        ri = tokenizer.encode(r, add_special_tokens=False)[:768]
        if len(ri) < 16:
            continue
        e, l, _ = build_soft_prompt_inputs(t_emb, tmpl, [], qi, ri, None, None, device, dtype)
        out = teacher(inputs_embeds=e)
        sl = out.logits[0, :-1, :].float()
        lb = l[0, 1:]
        m = lb != -100
        probs = F.softmax(sl[m], dim=-1)
        tp, ti = torch.topk(probs, 64, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
        cached.append((qi, ri, ti.cpu(), tp.to(torch.float16).cpu(), ent.cpu()))
        del e, l, out
    del teacher
    torch.cuda.empty_cache()
    student.to(device)
    student.eval()
    s_emb = student.get_input_embeddings()

    def eval_kl_and_tokens(ads: List[torch.Tensor]):
        kls = []
        tok_ids = {k: tokenizer.encode(k, add_special_tokens=False)[-1] for k in ["Check", " But", " Let", "?", "Thus"]}
        t_prob_sum = {k: 0.0 for k in tok_ids}
        s_prob_sum = {k: 0.0 for k in tok_ids}
        n_pos = 0
        for qi, ri, ti_c, tp_c, ent_c in cached:
            e, l, _ = build_soft_prompt_inputs(s_emb, tmpl, ads, qi, ri, None, None, device, dtype)
            out = student(inputs_embeds=e)
            sl = out.logits[0, :-1, :].float()
            lb = l[0, 1:]
            m = lb != -100
            s_resp = sl[m]
            ti = ti_c.to(device)
            tp = tp_c.float().to(device)
            ent = ent_c.to(device)
            n = min(s_resp.size(0), ti.size(0))
            s_lp = F.log_softmax(s_resp[:n], dim=-1)
            ce = -(tp[:n] * s_lp.gather(-1, ti[:n])).sum(dim=-1)
            kls.append(float((ce - ent[:n]).clamp_min(0.0).mean().item()))
            s_p = s_lp.exp()
            # Track discourse token probability sum
            for k, tid in tok_ids.items():
                s_prob_sum[k] += float(s_p[:, tid].sum().item())
                # Teacher approx prob from top-64
                hit = (ti[:n] == tid).float()
                t_prob_sum[k] += float((hit * tp[:n]).sum().item())
            n_pos += n
            del e, l, out
        mean_kl = sum(kls) / max(1, len(kls))
        rel = {k: round(s_prob_sum[k] / max(1e-8, t_prob_sum[k]), 3) for k in tok_ids}
        return mean_kl, rel

    kl_s, rel_s = eval_kl_and_tokens([])
    out_res = {"kl_student_vs_unseen_rl": round(kl_s, 5), "student_tok_rel": rel_s, "models": {}}
    print(f"  [PostHoc | Student (No SP)     ] KL={kl_s:.5f} | tok_rel={rel_s}", flush=True)
    for name, sp in sp_dict.items():
        kl_m, rel_m = eval_kl_and_tokens([sp])
        g = round(100.0 * (1.0 - kl_m / max(kl_s, 1e-8)), 2)
        out_res["models"][name] = {"kl": round(kl_m, 5), "G_vs_unseen_RL_pct": g, "tok_rel": rel_m}
        print(f"  [PostHoc | {name:20s}] KL={kl_m:.5f} | G_vs_unseen_RL={g:+6.2f}% | tok_rel={rel_m}", flush=True)
    return out_res


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Suite 4 (exp6): Teacher-Free Input-Space Soft Prompt Acquisition & Arbitrary-Context Assembly ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    if_train, math_train, math_test, ifbench40 = load_verifiable_datasets()

    student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    results: Dict[str, Any] = {
        "backbone": STUDENT_ID,
        "teacher_free": True,
        "harvest": {},
        "train_meta": {},
        "ifbench40_C0_clean": {},
        "ifbench40_arbitrary_ctx": {},
        "math40_eval": {},
        "posthoc_unseen_RL_alignment": {},
    }

    def save():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # --- STEP 1: Teacher-Free Rollout Harvesting from Frozen Student ONLY ---
    print("\n=== Step 1: Teacher-Free Rollout Harvesting (Frozen M2RL-SFT ONLY, 0 Post-RL Teachers) ===", flush=True)
    if_wins_r1, if_pairs_r1, st_if_r1 = harvest_if_rollouts(student, tokenizer, tmpl, if_train, [], n_samples=4, max_new_tokens=960, seed=101, tag="IF_R1")
    results["harvest"]["IF_R1"] = st_if_r1
    math_wins, math_pairs, st_math = harvest_math_rollouts(student, tokenizer, tmpl, math_train, n_samples=4, max_new_tokens=960, seed=202)
    results["harvest"]["Math_R1"] = st_math
    save()

    if_stream = make_ctx_stream(tokenizer, [x["prompt"] for x in if_train])
    math_stream = make_ctx_stream(tokenizer, [x["prompt"] for x in math_train])

    # --- STEP 2: Train ALL 5 Teacher-Free Input-Space SP Methods on IF + TF-Math ---
    print("\n=== Step 2: Training 5 Teacher-Free Input-Space SP Methods (L=256, Arbitrary-Context PR) ===", flush=True)

    # Method 1: TF-RFT (1-Round BoN)
    sp_rft, meta_rft = train_tf_rft_sp(student, tokenizer, tmpl, if_wins_r1, if_stream, num_updates=40, tag="TF_RFT_R1")
    results["train_meta"]["TF_RFT"] = meta_rft
    torch.save(sp_rft.cpu(), os.path.join(OUT_DIR, "sp_tf_rft.pt"))
    save()

    # Method 2: TF-IterRFT (2-Round On-Policy Expert Iteration)
    print("\n  --- Round 2 On-Policy Rollout Harvesting with P_RFT_R1 ---", flush=True)
    win_cnt_by_idx = {}
    for w in if_wins_r1:
        win_cnt_by_idx[w["prompt_idx"]] = w["pass_cnt"]
    hard_indices = sorted(range(len(if_train)), key=lambda i: win_cnt_by_idx.get(i, 0))[:32]
    hard_if_prompts = [if_train[i] for i in hard_indices]
    if_wins_r2, if_pairs_r2, st_if_r2 = harvest_if_rollouts(student, tokenizer, tmpl, hard_if_prompts, [sp_rft], n_samples=4, max_new_tokens=960, seed=303, tag="IF_R2_OnPolicy")
    r1_unsolved = {i for i in hard_indices if win_cnt_by_idx.get(i, 0) == 0}
    r2_solved_orig = {hard_indices[w["prompt_idx"]] for w in if_wins_r2}
    newly_solved = len(r1_unsolved & r2_solved_orig)
    st_if_r2["r1_unsolved_in_subset"] = len(r1_unsolved)
    st_if_r2["newly_solved_in_r2"] = newly_solved
    print(f"  [IterRFT Bootstrap] Newly solved in Round 2 (0/4 bare -> solved): {newly_solved}/{len(r1_unsolved)}", flush=True)
    results["harvest"]["IF_R2_OnPolicy"] = st_if_r2

    merged_wins = list(if_wins_r1) + [{**w, "weight": w["weight"] * 1.5} for w in if_wins_r2]
    sp_iter_rft, meta_iter = train_tf_rft_sp(student, tokenizer, tmpl, merged_wins, if_stream, num_updates=40, init_sp=sp_rft, lr_max=0.0035, tag="TF_IterRFT_R2")
    results["train_meta"]["TF_IterRFT"] = meta_iter
    torch.save(sp_iter_rft.cpu(), os.path.join(OUT_DIR, "sp_tf_iter_rft.pt"))
    save()

    # Method 3: TF-Contrastive (Verifier-Guided DPO + NLL Anchor)
    merged_pairs = list(if_pairs_r1) + list(if_pairs_r2)
    sp_dpo, meta_dpo = train_tf_contrastive_sp(student, tokenizer, tmpl, merged_pairs, if_stream, num_updates=40, tag="TF_Contrastive")
    results["train_meta"]["TF_Contrastive"] = meta_dpo
    torch.save(sp_dpo.cpu(), os.path.join(OUT_DIR, "sp_tf_contrastive.pt"))
    save()

    # Method 4: TF-GRPO (True Online On-Policy GRPO directly on Layer-0 Input-SP!)
    sp_grpo, meta_grpo = train_tf_grpo_sp(student, tokenizer, tmpl, hard_if_prompts, if_stream, num_steps=12, max_new_tokens=768, tag="TF_GRPO")
    results["train_meta"]["TF_GRPO"] = meta_grpo
    torch.save(sp_grpo.cpu(), os.path.join(OUT_DIR, "sp_tf_grpo.pt"))
    save()

    # Method 5: TF-KV2SP (Teacher-Free KV Scaffold -> Dense Top-64 Compilation into Arbitrary-Context Input-SP!)
    kv_tf_if, sp_kv2sp, meta_kv2sp = train_tf_kv2sp(student, tokenizer, tmpl, merged_wins, merged_pairs, if_stream)
    results["train_meta"]["TF_KV2SP"] = meta_kv2sp
    torch.save(kv_tf_if.cpu(), os.path.join(OUT_DIR, "kv_tf_if_L16.pt"))
    torch.save(sp_kv2sp.cpu(), os.path.join(OUT_DIR, "sp_tf_kv2sp.pt"))
    save()

    # Teacher-Free Math Input-SP (for Zero-Training Dual Composition [P_math_TF; P_if_TF]!)
    sp_math_tf, meta_math_tf = train_tf_contrastive_sp(
        student, tokenizer, tmpl, math_pairs, math_stream, task_inst=MATH_TASK_INST, tag="TF_Math_DPO"
    )
    results["train_meta"]["TF_Math"] = meta_math_tf
    torch.save(sp_math_tf.cpu(), os.path.join(OUT_DIR, "sp_tf_math.pt"))
    save()

    # --- STEP 3: Official IFBench40 Benchmark Across ALL 5 Teacher-Free Methods ---
    print("\n=== Step 3: IFBench (40 prompts, C0 Clean System) ===", flush=True)
    results["ifbench40_C0_clean"]["student_no_sp"] = eval_ifbench40(student, tokenizer, tmpl, ifbench40, "sp", [], tag="C0: Student (No SP)")
    results["ifbench40_C0_clean"]["TF_RFT"] = eval_ifbench40(student, tokenizer, tmpl, ifbench40, "sp", [sp_rft], tag="C0: TF-RFT (1-Rd BoN)")
    results["ifbench40_C0_clean"]["TF_IterRFT"] = eval_ifbench40(student, tokenizer, tmpl, ifbench40, "sp", [sp_iter_rft], tag="C0: TF-IterRFT (2-Rd)")
    results["ifbench40_C0_clean"]["TF_Contrastive"] = eval_ifbench40(student, tokenizer, tmpl, ifbench40, "sp", [sp_dpo], tag="C0: TF-Contrastive (DPO+NLL)")
    results["ifbench40_C0_clean"]["TF_GRPO"] = eval_ifbench40(student, tokenizer, tmpl, ifbench40, "sp", [sp_grpo], tag="C0: TF-GRPO (Online RL)")
    results["ifbench40_C0_clean"]["TF_KV_L16_scaffold"] = eval_ifbench40(student, tokenizer, tmpl, ifbench40, "kv", [kv_tf_if], tag="C0: TF-KV L16 Scaffold")
    results["ifbench40_C0_clean"]["TF_KV2SP"] = eval_ifbench40(student, tokenizer, tmpl, ifbench40, "sp", [sp_kv2sp], tag="C0: TF-KV2SP (Input-SP)")
    save()

    # Select best Teacher-Free Input-SP for Dual Composition & Arbitrary-Context Stress Tests!
    cand_map = {
        "TF_RFT": sp_rft, "TF_IterRFT": sp_iter_rft, "TF_Contrastive": sp_dpo,
        "TF_GRPO": sp_grpo, "TF_KV2SP": sp_kv2sp,
    }
    best_name = max(
        cand_map.keys(),
        key=lambda k: (results["ifbench40_C0_clean"][k]["prompt_strict"], results["ifbench40_C0_clean"][k]["closed_acc"]),
    )
    sp_if_best = cand_map[best_name]
    results["best_tf_input_sp_name"] = best_name
    print(f"\n>>> Best Teacher-Free Input-SP on IFBench40 C0: {best_name} <<<", flush=True)

    # Zero-Training Dual Composition [P_math_TF; P_if_best] & reverse [P_if_best; P_math_TF]
    results["ifbench40_C0_clean"]["TF_Dual_Math_IF"] = eval_ifbench40(student, tokenizer, tmpl, ifbench40, "sp", [sp_math_tf, sp_if_best], tag=f"C0: TF-Dual [Math; {best_name}]")
    results["ifbench40_C0_clean"]["TF_Dual_IF_Math"] = eval_ifbench40(student, tokenizer, tmpl, ifbench40, "sp", [sp_if_best, sp_math_tf], tag=f"C0: TF-Dual [{best_name}; Math]")
    save()

    # --- STEP 4: CRITICAL Arbitrary-Context Assembly Stress Test (Input-SP vs Fixed KV!) ---
    print("\n=== Step 4: Arbitrary-Context Assembly Stress Test (C1 Dynamic Persona & C2 Pre-Context History) ===", flush=True)
    # C1: Dynamic 120-tok Enterprise System Persona
    results["ifbench40_arbitrary_ctx"]["C1_student_no_sp"] = eval_ifbench40(
        student, tokenizer, tmpl, ifbench40, "sp", [], post_sys=EVAL_C1_DYNAMIC_PERSONA, tag="C1-Persona: Student"
    )
    results["ifbench40_arbitrary_ctx"]["C1_TF_KV_L16"] = eval_ifbench40(
        student, tokenizer, tmpl, ifbench40, "kv", [kv_tf_if], post_sys=EVAL_C1_DYNAMIC_PERSONA, tag="C1-Persona: Fixed TF-KV L16"
    )
    results["ifbench40_arbitrary_ctx"]["C1_best_InputSP_after_persona"] = eval_ifbench40(
        student, tokenizer, tmpl, ifbench40, "sp", [sp_if_best], pre_ctx=EVAL_C1_DYNAMIC_PERSONA, tag=f"C1-Persona: [Persona; {best_name}]"
    )
    if best_name != "TF_KV2SP":
        results["ifbench40_arbitrary_ctx"]["C1_KV2SP_after_persona"] = eval_ifbench40(
            student, tokenizer, tmpl, ifbench40, "sp", [sp_kv2sp], pre_ctx=EVAL_C1_DYNAMIC_PERSONA, tag="C1-Persona: [Persona; KV2SP]"
        )
    save()

    # C2: 180-tok Multi-Turn Pre-Context placed BEFORE Input-SP (vs fixed KV prefix!)
    results["ifbench40_arbitrary_ctx"]["C2_student_no_sp"] = eval_ifbench40(
        student, tokenizer, tmpl, ifbench40, "sp", [], pre_ctx=EVAL_C2_MULTITURN_PRECONTEXT, tag="C2-History: Student"
    )
    results["ifbench40_arbitrary_ctx"]["C2_TF_KV_L16"] = eval_ifbench40(
        student, tokenizer, tmpl, ifbench40, "kv", [kv_tf_if], pre_ctx=EVAL_C2_MULTITURN_PRECONTEXT, tag="C2-History: Fixed TF-KV L16"
    )
    results["ifbench40_arbitrary_ctx"]["C2_best_InputSP_after_history"] = eval_ifbench40(
        student, tokenizer, tmpl, ifbench40, "sp", [sp_if_best], pre_ctx=EVAL_C2_MULTITURN_PRECONTEXT, tag=f"C2-History: [Hist; {best_name}]"
    )
    results["ifbench40_arbitrary_ctx"]["C2_Dual_after_history"] = eval_ifbench40(
        student, tokenizer, tmpl, ifbench40, "sp", [sp_math_tf, sp_if_best], pre_ctx=EVAL_C2_MULTITURN_PRECONTEXT, tag="C2-History: [Hist; Math; IF]"
    )
    save()

    # --- STEP 5: Held-Out MATH-500 (40 Level 3-5 Problems) ---
    print("\n=== Step 5: Held-Out MATH-500 (40 Level 3-5 Problems) ===", flush=True)
    results["math40_eval"]["student_no_sp"] = eval_math40(student, tokenizer, tmpl, math_test, [], tag="MATH40: Student (No SP)")
    results["math40_eval"]["TF_Math"] = eval_math40(student, tokenizer, tmpl, math_test, [sp_math_tf], tag="MATH40: TF-Math (DPO+NLL)")
    results["math40_eval"]["TF_Dual_Math_IF"] = eval_math40(student, tokenizer, tmpl, math_test, [sp_math_tf, sp_if_best], tag="MATH40: TF-Dual [Math; IF]")
    save()

    # --- STEP 6: Post-Hoc Distributional Alignment Check vs UNSEEN M2RL-RL_IF ---
    results["posthoc_unseen_RL_alignment"] = posthoc_alignment_check(
        student, tokenizer, tmpl, cand_map, [x.prompt for x in ifbench40]
    )
    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== Suite 4 (exp6) COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
