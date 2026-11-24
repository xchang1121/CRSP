"""
Suite 7 (exp9): Scaled Multi-Seed Algorithm B (CR-MS-GRPO+) & Zero-Interference
                Multi-Expert Arbitrary-Slot Composition (IF + Math + Domain-Doc)
================================================================================
100% ZERO Post-RL Teacher Models are EVER loaded or used!
Only frozen M2RL-SFT (Qwen3-4B) + deterministic verifiers + enterprise doc HeliosPay v4.2.

Two core research tracks requested as direct follow-up to Suite 6 (exp8):

  TRACK 1 (Direction 2 — Scaled Multi-Seed Validation of Algorithm B: CR-MS-GRPO+):
    - Tests Stage-1 TF3 (KL-Anchored Verified Self-Bootstrap) Warm-Start -> Stage-2 TF2
      (Direct Multi-Slot Online GRPO) vs Cold-Start TF2 vs Stage-1 Only TF3 at compact L=64.
    - 3 independent random seeds (42, 108, 202) with paired prompt/slot trajectories.
    - Mid-point checkpoints at step 5 vs final step 10 to measure 2x convergence speedup.
    - Scaled held-out benchmarks:
        * IFEval96 (96 strictly disjoint google/IFEval prompts x 3 seeds = 288 eval pairs/method!)
        * IFBench64 (64 stratified prompts, 2x larger than IFBench32!)

  TRACK 2 (Direction 1 — Zero-Interference Multi-Expert Arbitrary-Slot Composition):
    - Constructs 3 compact (L=64) Zero-Post-RL specialists across disjoint domains:
        * E1_IF_CSCR   : Instruction-Following (Alg B + Co-Presence Distractor Aug)
        * E2_Math_CSCR : Mathematical Reasoning on MATH-500 (Alg B + Co-Presence + L_null)
        * E3_Doc_CSCR  : Vertical Domain Doc HeliosPay v4.2 (Zero-Leakage CR Self-Distill + Co-Presence + L_null)
      PLUS matched Naive Controls (without Co-Presence & without Null-Task Identity L_null).
    - Evaluates Single, 2-Expert Split, 2-Expert Permutation Invariance ([IF;Math] vs [Math;IF]),
      AND 3-EXPERT SIMULTANEOUS SPLIT-SLOT COMPOSITION:
          [Persona + Doc@P1] + [2-Turn History] + [Math@P2 + Query + IF@P4]  (192 vecs total!)
      across ALL THREE tasks (IFEval96, IFBench64, MATH48, HeliosPay Doc36) + exact
      Cross-Expert KL Interference Matrix!
"""

import json
import math
import os
import random
import statistics
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
    get_qwen_template_tokens,
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
from domain_docs_data import DOC_A_TEXT, build_domain_datasets, grade_domain_response
from exp7_context_robust_sp import (
    EVAL_HISTORY_TURNS,
    EVAL_PERSONA,
    EVAL_RAG_BLOCK,
    IF_TASK_INST,
    TRAIN_DIALOG_TURNS,
    TRAIN_PERSONAS,
    TRAIN_RAG_CHUNKS,
    build_multislot_sp_inputs,
    clamp_sp_rms_,
    cross_context_invariance_kl,
    hybrid_top64_distill_loss,
    init_anchored_sp,
)
import evaluation_lib as ifbench_eval

STUDENT_ID = "Jackwang111/M2RL-SFT"
OUT_DIR = "/content/cspd/exp9_results"
os.makedirs(OUT_DIR, exist_ok=True)

MATH_TASK_INST = (
    "Solve the mathematics problem rigorously step by step. Carefully check every algebraic manipulation, "
    "case split, arithmetic calculation, and boundary condition before placing the exact final answer in \\boxed{}."
)
DOC_TASK_INST = (
    "You have internalized the HeliosPay v4.2 Internal Cross-Border Settlement & Compliance Specification. "
    "When asked about HeliosPay v4.2 policies, constants, corridors, error codes, or wire risk assessments, "
    "apply the exact rules and constants accurately; for unrelated tasks, remain completely neutral."
)

# Synthetic foreign expert texts for zero-shot Co-Presence Distractor Augmentation (STRICTLY unrelated!)
FOREIGN_EXPERT_PROMPTS = [
    "Optimize PostgreSQL execution plans by checking composite B-tree index selectivity and avoiding sequential scans.",
    "Validate JSON Schema draft-2020-12 documents ensuring required keys, enums, and numeric ranges conform strictly.",
    "Translate formal business correspondence into fluent idiomatic French while preserving paragraph formatting.",
    "Review commercial software licensing clauses for indemnity caps, liability exclusions, and renewal notice periods.",
    "Assign accurate ICD-10-CM diagnostic codes from clinical discharge summaries following primary-diagnosis rules.",
    "Write deterministic pytest unit tests with parametrized edge cases, fixtures, and strict assertion messages.",
    "Compute multi-echelon safety stock levels using lead-time demand variance and target cycle service levels.",
    "Edit scientific manuscript abstracts to tighten active voice, highlight quantitative findings, and respect word limits.",
]


# ---------------------------------------------------------------------------
# 1. Datasets & Synthetic Foreign Co-Presence Distractor Pool
# ---------------------------------------------------------------------------

def load_datasets_exp9():
    # 1) google/IFEval: 96 train + 96 strictly disjoint held-out test!
    ds_if = load_dataset("google/IFEval", split="train")
    if_all = []
    for row in ds_if:
        iids = list(row["instruction_id_list"])
        if iids and all(i in SUPPORTED_IFEVAL_IDS for i in iids):
            if_all.append({"prompt": row["prompt"], "iids": iids, "kwargs": list(row["kwargs"])})
    random.Random(2026).shuffle(if_all)
    if_train96 = if_all[:96]
    ifeval_test96 = if_all[96:192]

    # 2) IFBench: 64 stratified prompts (2x larger than IFBench32!)
    ifb_all = ifbench_eval.read_prompt_list("/content/IFBench/data/IFBench_test.jsonl")
    step = len(ifb_all) / 64.0
    ifbench64 = [ifb_all[int(i * step)] for i in range(64)]

    # 3) HuggingFaceH4/MATH-500: 64 train + 48 strictly disjoint Level 3-5 test!
    ds_m = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_all = []
    for row in ds_m:
        math_all.append({
            "prompt": row["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
            "answer": str(row["answer"]),
            "level": int(row.get("level", 3)),
        })
    random.Random(42).shuffle(math_all)
    hard_math = [x for x in math_all if x["level"] >= 3]
    math_test48 = hard_math[:48]
    m_test_set = {x["prompt"] for x in math_test48}
    math_train64 = [x for x in math_all if x["prompt"] not in m_test_set][:64]

    # 4) Zero-Leakage Vertical Domain Doc A (HeliosPay v4.2)
    dom_ds = build_domain_datasets(seed=42)["A_heliospay"]
    doc_train = dom_ds["train"]       # 80 rules + 48 upsampled facts
    doc_test_rules = dom_ds["test_rules"]  # 24 unseen rule scenarios
    doc_test_facts = dom_ds["test_facts"]  # 12 paraphrased facts

    print(
        f"Loaded datasets: if_train96={len(if_train96)}, ifeval_test96={len(ifeval_test96)}, "
        f"ifbench64={len(ifbench64)}, math_train64={len(math_train64)}, math_test48={len(math_test48)}, "
        f"doc_train={len(doc_train)}, doc_test={len(doc_test_rules)}+{len(doc_test_facts)}",
        flush=True,
    )
    return if_train96, ifeval_test96, ifbench64, math_train64, math_test48, doc_train, doc_test_rules, doc_test_facts


@torch.no_grad()
def build_foreign_distractor_pool(s_emb, tokenizer, num_vectors: int = 64, seed: int = 77) -> List[torch.Tensor]:
    """
    Build 8 synthetic foreign soft-prompt distractors (L=64, rms=0.032) from strictly unrelated
    domains so experts learn zero-shot cross-expert non-interference without ever seeing each other!
    """
    g = torch.Generator(device=s_emb.weight.device).manual_seed(seed)
    pool = []
    for txt in FOREIGN_EXPERT_PROMPTS:
        sp = init_anchored_sp(s_emb, tokenizer, txt, num_vectors=num_vectors, anchored=True).detach()
        noise = torch.randn(sp.shape, generator=g, device=sp.device, dtype=sp.dtype) * 0.014
        sp = sp + noise
        cur_r = sp.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        sp = sp * (0.032 / cur_r)
        pool.append(sp.detach())
    return pool


def sample_slot_view_cscr(
    rng: random.Random, train_pools: Dict[str, Any], sp: torch.Tensor,
    foreign_pool: Optional[List[torch.Tensor]] = None, co_prob: float = 0.60,
) -> Tuple[Dict[str, Any], str]:
    """
    Sample a multi-slot runtime context view covering P0_RandPre, P1_SysEnd, P2_MultiTurn,
    P3_PostRAG, and P4_PostQuery. When foreign_pool is given, with probability co_prob also
    inject 1 or 2 synthetic foreign L=64 SPs into UNUSED slots (Co-Presence Augmentation!).
    Returns (slot_kwargs, target_sp_key).
    """
    slot = rng.choices(["P1_sys", "P2_turn", "P3_rag", "P4_post", "P0_rand"], weights=[0.18, 0.27, 0.20, 0.25, 0.10])[0]
    kw: Dict[str, Any] = {}
    if slot == "P0_rand":
        pre = sample_random_prefix(train_pools["stream"], rng, 32, 160)
        kw = {"sys_text_ids": pre, "sp_sys_end": [sp]}
        sp_key = "sp_sys_end"
    elif slot == "P1_sys":
        kw = {"sys_text_ids": rng.choice(train_pools["personas"]), "sp_sys_end": [sp]}
        sp_key = "sp_sys_end"
    elif slot == "P2_turn":
        k_t = rng.choice([1, 2])
        turns = [rng.choice(train_pools["turns"]) for _ in range(k_t)]
        kw = {"sys_text_ids": rng.choice(train_pools["personas"]), "history_turns_ids": turns, "sp_pre_query": [sp]}
        sp_key = "sp_pre_query"
    elif slot == "P3_rag":
        kw = {"sys_text_ids": rng.choice(train_pools["personas"]), "rag_ids": rng.choice(train_pools["rags"]), "sp_pre_query": [sp]}
        sp_key = "sp_pre_query"
    else:  # P4_post
        k_t = rng.choice([1, 2])
        turns = [rng.choice(train_pools["turns"]) for _ in range(k_t)]
        kw = {"sys_text_ids": rng.choice(train_pools["personas"]), "history_turns_ids": turns, "sp_post_query": [sp]}
        sp_key = "sp_post_query"

    # Co-Presence Distractor Augmentation: inject 1-2 synthetic foreign experts into OTHER slots (or adjacent!)
    if foreign_pool and rng.random() < co_prob:
        other_slots = [k for k in ["sp_sys_end", "sp_pre_query", "sp_post_query"] if k != sp_key]
        if rng.random() < 0.78:
            # Split-slot co-presence (1 or 2 other slots occupied!)
            n_for = rng.choice([1, 2])
            for os_k in rng.sample(other_slots, k=n_for):
                kw[os_k] = [rng.choice(foreign_pool)]
        else:
            # Adjacent co-presence in the SAME slot (before or after target sp -> permutation invariance!)
            f_sp = rng.choice(foreign_pool)
            kw[sp_key] = [f_sp, sp] if rng.random() < 0.5 else [sp, f_sp]
    return kw, sp_key


# ---------------------------------------------------------------------------
# 2. Universal Multi-Slot Batched Generator & Task Evaluators
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_generate_slots(
    model, tokenizer, tmpl: ChatTemplateTokens, questions: List[str],
    sys_text_ids: Optional[List[int]] = None,
    history_turns_ids: Optional[List[Tuple[List[int], List[int]]]] = None,
    rag_ids: Optional[List[int]] = None,
    sp_sys_start: Optional[List[torch.Tensor]] = None,
    sp_sys_end: Optional[List[torch.Tensor]] = None,
    sp_pre_query: Optional[List[torch.Tensor]] = None,
    sp_post_query: Optional[List[torch.Tensor]] = None,
    batch_size: int = 48, max_new_tokens: int = 1024,
    temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20, seed: int = 42,
) -> List[str]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    s_emb = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype)

    model.eval()
    model.config.use_cache = True
    torch.manual_seed(seed)
    all_outs: List[str] = []
    for start in range(0, len(questions), batch_size):
        bq = questions[start : start + batch_size]
        seq_e = []
        for q in bq:
            qi = tokenizer.encode(q, add_special_tokens=False)
            e, _, _ = build_multislot_sp_inputs(
                s_emb, tmpl, qi, None,
                sys_text_ids=sys_text_ids, history_turns_ids=history_turns_ids, rag_ids=rag_ids,
                sp_sys_start=sp_sys_start, sp_sys_end=sp_sys_end,
                sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
                device=device, model_dtype=dtype,
            )
            seq_e.append(e.squeeze(0))
        ml = max(x.size(0) for x in seq_e)
        bs = len(seq_e)
        d = seq_e[0].size(1)
        padded = pad_vec.expand(bs, ml, d).clone()
        am = torch.zeros((bs, ml), dtype=torch.long, device=device)
        for i, x in enumerate(seq_e):
            L = x.size(0)
            padded[i, ml - L :, :] = x
            am[i, ml - L :] = 1
        out_ids = model.generate(
            inputs_embeds=padded, attention_mask=am, max_new_tokens=max_new_tokens,
            pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
        )
        for row in out_ids:
            all_outs.append(tokenizer.decode(row, skip_special_tokens=True))
        del padded, am, out_ids
    return all_outs


def eval_ifeval96_slots(
    model, tokenizer, tmpl, eval_ctx, items96: List[Dict[str, Any]],
    sp_sys_end=None, sp_pre_query=None, sp_post_query=None, tag: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    outs = batch_generate_slots(
        model, tokenizer, tmpl, [x["prompt"] for x in items96],
        sys_text_ids=eval_ctx["persona_ids"], history_turns_ids=eval_ctx["history_ids"],
        sp_sys_end=sp_sys_end, sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
        batch_size=48, max_new_tokens=1024, seed=42,
    )
    s_ok, closed, inst_sum = 0, 0, 0.0
    for it, resp in zip(items96, outs):
        c = "</think>" in resp
        closed += int(c)
        st, ir = verify_ifeval_response(resp, it["iids"], it["kwargs"])
        if not c:
            st = False
        s_ok += int(st)
        inst_sum += ir
    n = len(items96)
    res = {
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * inst_sum / n, 2),
        "n_strict": s_ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFEval96  | {tag:36s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | inst={res['inst_strict']:5.2f}% | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def eval_ifbench64_slots(
    model, tokenizer, tmpl, eval_ctx, inps64,
    sp_sys_end=None, sp_pre_query=None, sp_post_query=None, tag: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    outs = batch_generate_slots(
        model, tokenizer, tmpl, [x.prompt for x in inps64],
        sys_text_ids=eval_ctx["persona_ids"], history_turns_ids=eval_ctx["history_ids"],
        sp_sys_end=sp_sys_end, sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
        batch_size=32, max_new_tokens=2048, seed=42,
    )
    p2r = {}
    closed = 0
    for inp, r in zip(inps64, outs):
        closed += int("</think>" in r)
        p2r[inp.prompt] = strip_thinking(r)
    s_ok = sum(int(ifbench_eval.test_instruction_following_strict(inp, p2r).follow_all_instructions) for inp in inps64)
    inst_l = []
    for inp in inps64:
        inst_l.extend(ifbench_eval.test_instruction_following_strict(inp, p2r).follow_instruction_list)
    n = len(inps64)
    res = {
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * sum(inst_l) / max(1, len(inst_l)), 2),
        "n_strict": s_ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFBench64 | {tag:36s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | inst={res['inst_strict']:5.2f}% | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def eval_math48_slots(
    model, tokenizer, tmpl, eval_ctx, items48: List[Dict[str, Any]],
    sp_sys_end=None, sp_pre_query=None, sp_post_query=None, tag: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    outs = batch_generate_slots(
        model, tokenizer, tmpl, [x["prompt"] for x in items48],
        sys_text_ids=eval_ctx["persona_ids"], history_turns_ids=eval_ctx["history_ids"],
        sp_sys_end=sp_sys_end, sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
        batch_size=48, max_new_tokens=1536, seed=42,
    )
    ok, closed = 0, 0
    for it, resp in zip(items48, outs):
        c = "</think>" in resp
        closed += int(c)
        if c and verify_math_response(resp, it["answer"]):
            ok += 1
    n = len(items48)
    res = {
        "acc": round(100.0 * ok / n, 2),
        "n_ok": ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [MATH48    | {tag:36s}] acc={res['acc']:5.2f}% ({ok:2d}/{n}) | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def eval_doc36_slots(
    model, tokenizer, tmpl, eval_ctx, rules24: List[Dict[str, Any]], facts12: List[Dict[str, Any]],
    sp_sys_end=None, sp_pre_query=None, sp_post_query=None, extra_sys_text: Optional[str] = None, tag: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    items = list(rules24) + list(facts12)
    sys_ids = list(eval_ctx["persona_ids"])
    if extra_sys_text:
        sys_ids = sys_ids + tokenizer.encode("\n\n" + extra_sys_text, add_special_tokens=False)
    outs = batch_generate_slots(
        model, tokenizer, tmpl, [x["question"] for x in items],
        sys_text_ids=sys_ids, history_turns_ids=eval_ctx["history_ids"],
        sp_sys_end=sp_sys_end, sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
        batch_size=36, max_new_tokens=1024, seed=42,
    )
    r_ok = sum(int(grade_domain_response(r, it)) for r, it in zip(outs[:len(rules24)], rules24))
    f_ok = sum(int(grade_domain_response(r, it)) for r, it in zip(outs[len(rules24):], facts12))
    tot_ok = r_ok + f_ok
    res = {
        "rule_acc": round(100.0 * r_ok / len(rules24), 2),
        "fact_acc": round(100.0 * f_ok / len(facts12), 2),
        "overall_acc": round(100.0 * tot_ok / len(items), 2),
        "n_rule": r_ok, "n_fact": f_ok, "n_total": len(items),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [Doc36     | {tag:36s}] overall={res['overall_acc']:5.2f}% ({tot_ok:2d}/36) | rules={res['rule_acc']:5.1f}% ({r_ok:2d}/24) | facts={res['fact_acc']:5.1f}% ({f_ok:2d}/12) | t={res['eval_sec']}s", flush=True)
    return res


# ---------------------------------------------------------------------------
# 3. Shared Bare-Student Harvest & Anchor Caching (100% Zero Post-RL!)
# ---------------------------------------------------------------------------

@torch.no_grad()
def harvest_verified_wins_with_anchors(
    student, tokenizer, tmpl: ChatTemplateTokens, prompts_meta: List[Dict[str, Any]],
    task: str = "if", n_samples: int = 3, max_new_tokens: int = 768,
    max_resp_trunc: int = 512, seed: int = 202, tag: str = "Harvest",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()

    flat_q = []
    for pm in prompts_meta:
        flat_q.extend([pm["prompt"]] * n_samples)

    student.eval()
    student.config.use_cache = True
    torch.manual_seed(seed)
    outs = batch_generate_with_soft_prompts(
        student, tokenizer, tmpl, flat_q, [], batch_size=48, max_new_tokens=max_new_tokens,
        temperature=0.85, top_p=0.95, top_k=20, enable_thinking=True,
    )

    wins = []
    n_pass_any, n_pass_all_rollouts = 0, 0
    for i, pm in enumerate(prompts_meta):
        grp = outs[n_samples * i : n_samples * (i + 1)]
        passed = []
        for resp in grp:
            if "</think>" not in resp:
                continue
            if task == "if":
                ok, _ = verify_ifeval_response(resp, pm["iids"], pm["kwargs"])
            else:
                ok = verify_math_response(resp, pm["answer"])
            if ok:
                passed.append(resp)
                n_pass_all_rollouts += 1
        if passed:
            n_pass_any += 1
            w_val = 1.0 / max(1, len(passed))
            for resp in passed[:2]:
                qi = tokenizer.encode(pm["prompt"], add_special_tokens=False)
                ri = tokenizer.encode(resp, add_special_tokens=False)[:max_resp_trunc]
                if len(ri) >= 16:
                    wins.append({"q_ids": qi, "r_ids": ri, "weight": w_val})

    # Precompute bare-student clean Top-32 KL anchors
    for item in wins:
        e, l, _ = build_multislot_sp_inputs(s_emb, tmpl, item["q_ids"], item["r_ids"], device=device, model_dtype=dtype)
        out = student(inputs_embeds=e, use_cache=False)
        sl = out.logits[0, :-1, :].float()
        m = l[0, 1:] != -100
        p_full = F.softmax(sl[m], dim=-1)
        tp, ti = torch.topk(p_full, 32, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-8).log()).sum(dim=-1)
        item["anc_ti"] = ti.cpu()
        item["anc_tp"] = tp.to(torch.float16).cpu()
        item["anc_ent"] = ent.cpu()
        del e, l, out

    stats = {
        "task": task, "n_prompts": len(prompts_meta), "n_samples": n_samples,
        "pass_at_1_pct": round(100.0 * n_pass_all_rollouts / max(1, len(flat_q)), 2),
        "pass_at_N_pct": round(100.0 * n_pass_any / max(1, len(prompts_meta)), 2),
        "num_wins": len(wins), "sec": round(time.time() - t0, 1),
    }
    print(f"  [{tag:22s}] {stats}", flush=True)
    return wins, stats


# ---------------------------------------------------------------------------
# 4. Algorithm B Components: Stage-1 TF3 (KL-Anchored SelfBoot) & Stage-2 TF2 (Multi-Slot GRPO)
# ---------------------------------------------------------------------------

def train_stage1_tf3_selfboot(
    student, tokenizer, tmpl: ChatTemplateTokens, wins: List[Dict[str, Any]],
    train_pools: Dict[str, Any], task_inst: str, num_vectors: int = 64,
    num_updates: int = 24, batch_size: int = 10, lr_max: float = 0.0055,
    lambda_anc: float = 0.55, lambda_inv: float = 0.35, max_rms: float = 0.045,
    foreign_pool: Optional[List[torch.Tensor]] = None, co_prob: float = 0.55,
    null_pool: Optional[List[Dict[str, Any]]] = None, lambda_null: float = 0.0,
    seed: int = 42, tag: str = "Stage1_TF3",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    sp = init_anchored_sp(s_emb, tokenizer, task_inst, num_vectors=num_vectors, anchored=True)
    opt = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)
    weights = [w["weight"] for w in wins]
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

        batch = rng.choices(wins, weights=weights, k=batch_size)
        tot_nll, tot_anc, tot_inv, tot_null = 0.0, 0.0, 0.0, 0.0

        for item in batch:
            qi, ri = item["q_ids"], item["r_ids"]
            e0, l0, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, sp_sys_start=[sp], device=device, model_dtype=dtype)
            slot_kw, _ = sample_slot_view_cscr(rng, train_pools, sp, foreign_pool=foreign_pool, co_prob=co_prob)
            ec, lc, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **slot_kw)
            be, bm, bl = pad_and_stack_train_batch([e0, ec], [l0, lc], pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm)
            sl = out.logits[:, :-1, :]
            lb = bl[:, 1:]
            m0, mc = lb[0] != -100, lb[1] != -100
            z0, zc = sl[0][m0].float(), sl[1][mc].float()

            nll_c = F.cross_entropy(zc, lb[1][mc], reduction="mean")
            nll_0 = F.cross_entropy(z0, lb[0][m0], reduction="mean")
            ati, atp, aent = item["anc_ti"].to(device), item["anc_tp"].float().to(device), item["anc_ent"].to(device)
            na = min(zc.size(0), ati.size(0))
            ce_anc = -(atp[:na] * F.log_softmax(zc[:na], dim=-1).gather(-1, ati[:na])).sum(dim=-1)
            kl_anc = (ce_anc - aent[:na]).clamp_min(0.0).mean()
            l_inv = cross_context_invariance_kl(z0, zc, topk=32)

            loss = (nll_c + 0.30 * nll_0 + lambda_anc * kl_anc + lambda_inv * l_inv) / batch_size
            loss.backward()
            tot_nll += float(nll_c.item()) / batch_size
            tot_anc += float(kl_anc.item()) / batch_size
            tot_inv += float(l_inv.item()) / batch_size
            del be, bm, bl, out, z0, zc

        # Optional Null-Task Identity Preservation Regularizer L_null on off-domain queries!
        if null_pool and lambda_null > 0.0:
            n_null = 2
            for _ in range(n_null):
                ni = null_pool[rng.randrange(len(null_pool))]
                slot_kw, _ = sample_slot_view_cscr(rng, train_pools, sp, foreign_pool=foreign_pool, co_prob=co_prob)
                en, ln, _ = build_multislot_sp_inputs(s_emb, tmpl, ni["q_ids"], ni["r_ids"], device=device, model_dtype=dtype, **slot_kw)
                out_n = student(inputs_embeds=en)
                sl = out_n.logits[0, :-1, :].float()
                m = ln[0, 1:] != -100
                zn = sl[m]
                ati, atp, aent = ni["anc_ti"].to(device), ni["anc_tp"].float().to(device), ni["anc_ent"].to(device)
                na = min(zn.size(0), ati.size(0))
                ce_n = -(atp[:na] * F.log_softmax(zn[:na], dim=-1).gather(-1, ati[:na])).sum(dim=-1)
                kl_n = (ce_n - aent[:na]).clamp_min(0.0).mean()
                (lambda_null * kl_n / n_null).backward()
                tot_null += float(kl_n.item()) / n_null
                del en, ln, out_n

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)
        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True

        if step in {1, 12, num_updates}:
            log_hist[str(step)] = {"nll": round(tot_nll, 4), "anc": round(tot_anc, 4), "inv": round(tot_inv, 4), "null": round(tot_null, 4)}
            print(f"  [{tag:24s}] step {step:2d}/{num_updates} | nll={tot_nll:.4f} | anc={tot_anc:.4f} | inv={tot_inv:.4f} | null={tot_null:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "num_vectors": num_vectors, "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


def train_stage2_tf2_multislot_grpo(
    student, tokenizer, tmpl: ChatTemplateTokens, prompts_meta: List[Dict[str, Any]],
    train_pools: Dict[str, Any], task: str = "if", task_inst: str = IF_TASK_INST,
    init_sp: Optional[torch.Tensor] = None, num_vectors: int = 64, num_steps: int = 10,
    mid_ckpt_step: int = 5, prompts_per_step: int = 4, group_size: int = 6,
    max_new_tokens: int = 768, lr_max: float = 0.0022, beta_kl: float = 0.05,
    lambda_inv: float = 0.25, max_rms: float = 0.045,
    foreign_pool: Optional[List[torch.Tensor]] = None, co_prob: float = 0.55,
    null_pool: Optional[List[Dict[str, Any]]] = None, lambda_null: float = 0.0,
    seed: int = 42, tag: str = "Stage2_TF2",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    Direct Multi-Slot Online GRPO on compact L=64 Input-SP with:
      - Paired deterministic prompt/slot sampling per (seed, step) for apples-to-apples convergence comparison
      - Optional Stage-1 warm-start init_sp (Full Algorithm B)
      - Optional Co-Presence Distractor Augmentation & Null-Task Identity L_null (CS-CR)
      - Mid-point checkpoint capture at mid_ckpt_step (proving 2x convergence speed!)
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    if init_sp is not None:
        sp = init_sp.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    else:
        sp = init_anchored_sp(s_emb, tokenizer, task_inst, num_vectors=num_vectors, anchored=True)

    opt = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    online_curve = []
    mid_sp: Optional[torch.Tensor] = None

    for step in range(1, num_steps + 1):
        # Paired RNG per (seed, step) so Cold-Start & Warm-Start see EXACT same prompts & context slots!
        rng_step = random.Random(seed * 1000 + step)
        prog = (step - 1) / max(1, num_steps - 1)
        lr = lr_max * 0.2 + 0.5 * (lr_max * 0.8) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        step_prompts = [prompts_meta[rng_step.randrange(len(prompts_meta))] for _ in range(prompts_per_step)]
        step_views = [
            sample_slot_view_cscr(rng_step, train_pools, sp.detach(), foreign_pool=foreign_pool, co_prob=co_prob)
            for _ in step_prompts
        ]

        # Batched multi-slot on-policy generation (24 rollouts)
        student.eval()
        student.config.use_cache = True
        seq_e = []
        for pm, (skw, _) in zip(step_prompts, step_views):
            qi = tokenizer.encode(pm["prompt"], add_special_tokens=False)
            e, _, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, None, device=device, model_dtype=dtype, **skw)
            for _ in range(group_size):
                seq_e.append(e.squeeze(0))
        ml = max(x.size(0) for x in seq_e)
        bs = len(seq_e)
        padded = pad_vec.expand(bs, ml, seq_e[0].size(1)).clone()
        am = torch.zeros((bs, ml), dtype=torch.long, device=device)
        for i, x in enumerate(seq_e):
            L = x.size(0)
            padded[i, ml - L :, :] = x
            am[i, ml - L :] = 1

        torch.manual_seed(seed * 100 + step)
        with torch.no_grad():
            out_ids = student.generate(
                inputs_embeds=padded, attention_mask=am, max_new_tokens=max_new_tokens,
                pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True, do_sample=True, temperature=0.85, top_p=0.95, top_k=20,
            )
        outs = [tokenizer.decode(row, skip_special_tokens=True) for row in out_ids]
        del padded, am, out_ids

        samples_to_opt = []
        tot_r, tot_strict, tot_closed = 0.0, 0, 0
        for p_i, (pm, (skw, sp_key)) in enumerate(zip(step_prompts, step_views)):
            grp = outs[p_i * group_size : (p_i + 1) * group_size]
            rewards = []
            for resp in grp:
                closed = "</think>" in resp
                if task == "if":
                    strict, inst_r = verify_ifeval_response(resp, pm["iids"], pm["kwargs"])
                    if not closed:
                        strict = False
                    rew = (0.5 * float(strict) + 0.5 * inst_r) if closed else (0.2 * inst_r - 0.2)
                else:
                    strict = closed and verify_math_response(resp, pm["answer"])
                    rew = (1.0 if strict else 0.0) if closed else -0.25
                rewards.append(rew)
                tot_r += rew; tot_strict += int(strict); tot_closed += int(closed)
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if std_r < 1e-4:
                continue
            qi = tokenizer.encode(pm["prompt"], add_special_tokens=False)
            # Re-bind live sp tensor inside skw (preserving any foreign distractors in other slots or same slot!)
            target_idx_in_list = None
            slot_list_ref = skw[sp_key]
            for idx_item, t_item in enumerate(slot_list_ref):
                if t_item.data_ptr() == sp.detach().data_ptr() or torch.equal(t_item, sp.detach()):
                    target_idx_in_list = idx_item
                    break
            for resp, rew in zip(grp, rewards):
                adv = (rew - mean_r) / (std_r + 1e-4)
                ri = tokenizer.encode(resp, add_special_tokens=False)[:640]
                if len(ri) >= 12 and abs(adv) > 0.05:
                    samples_to_opt.append((qi, ri, adv, skw, sp_key, target_idx_in_list))

        mean_kl = 0.0
        if samples_to_opt:
            # Same-context bare-student reference (remove ONLY target sp, keep same context/history/persona!)
            ref_lps = []
            with torch.no_grad():
                for qi, ri, _, skw, sp_key, tidx in samples_to_opt:
                    kw_ref = {k: v for k, v in skw.items() if not k.startswith("sp_")}
                    e_r, l_r, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **kw_ref)
                    out_r = student(inputs_embeds=e_r, use_cache=False)
                    sl = out_r.logits[0, :-1, :].float()
                    lb = l_r[0, 1:]
                    m = lb != -100
                    ref_lps.append(F.log_softmax(sl[m], dim=-1).gather(-1, lb[m].unsqueeze(-1)).squeeze(-1).detach())
                    del e_r, l_r, out_r

            student.config.use_cache = False
            student.train()
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            opt.zero_grad()
            n_opt = len(samples_to_opt)
            for (qi, ri, adv, skw, sp_key, tidx), r_lp in zip(samples_to_opt, ref_lps):
                kw_cur = dict(skw)
                new_list = list(skw[sp_key])
                new_list[tidx if tidx is not None else 0] = sp
                kw_cur[sp_key] = new_list

                ec, lc, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **kw_cur)
                e0, l0, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, sp_sys_start=[sp], device=device, model_dtype=dtype)
                be, bm, bl = pad_and_stack_train_batch([e0, ec], [l0, lc], pad_vec)
                out = student(inputs_embeds=be, attention_mask=bm)
                sl = out.logits[:, :-1, :]
                lb = bl[:, 1:]
                m0, mc = lb[0] != -100, lb[1] != -100
                z0, zc = sl[0][m0].float(), sl[1][mc].float()
                cur_lp = F.log_softmax(zc, dim=-1).gather(-1, lb[1][mc].unsqueeze(-1)).squeeze(-1)
                n_tok = min(cur_lp.size(0), r_lp.size(0))
                log_ratio = (r_lp[:n_tok] - cur_lp[:n_tok]).clamp(-10.0, 10.0)
                kl_est = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
                l_inv = cross_context_invariance_kl(z0, zc, topk=32)
                pg_loss = -(adv * cur_lp[:n_tok].mean()) + beta_kl * kl_est + lambda_inv * l_inv
                (pg_loss / n_opt).backward()
                mean_kl += float(kl_est.item()) / n_opt
                del be, bm, bl, out, z0, zc

            # Optional Null-Task Identity preservation on off-domain items
            if null_pool and lambda_null > 0.0:
                for _ in range(2):
                    ni = null_pool[rng_step.randrange(len(null_pool))]
                    skw_n, _ = sample_slot_view_cscr(rng_step, train_pools, sp, foreign_pool=foreign_pool, co_prob=co_prob)
                    en, ln, _ = build_multislot_sp_inputs(s_emb, tmpl, ni["q_ids"], ni["r_ids"], device=device, model_dtype=dtype, **skw_n)
                    out_n = student(inputs_embeds=en)
                    zn = out_n.logits[0, :-1, :].float()[ln[0, 1:] != -100]
                    ati, atp, aent = ni["anc_ti"].to(device), ni["anc_tp"].float().to(device), ni["anc_ent"].to(device)
                    na = min(zn.size(0), ati.size(0))
                    ce_n = -(atp[:na] * F.log_softmax(zn[:na], dim=-1).gather(-1, ati[:na])).sum(dim=-1)
                    kl_n = (ce_n - aent[:na]).clamp_min(0.0).mean()
                    (lambda_null * kl_n / 2.0).backward()
                    del en, ln, out_n

            torch.nn.utils.clip_grad_norm_([sp], 1.0)
            opt.step()
            clamp_sp_rms_(sp, max_rms)
            student.gradient_checkpointing_disable()
            student.eval()
            student.config.use_cache = True

        rec = {
            "step": step,
            "mean_reward": round(tot_r / max(1, bs), 4),
            "strict_pct": round(100.0 * tot_strict / max(1, bs), 1),
            "closed_pct": round(100.0 * tot_closed / max(1, bs), 1),
            "kl_to_ref": round(mean_kl, 4),
            "rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 4),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        online_curve.append(rec)
        if step == mid_ckpt_step:
            mid_sp = sp.detach().clone()
        if step in {1, mid_ckpt_step, num_steps}:
            print(f"  [{tag:24s}] step {step:2d}/{num_steps} | rew={rec['mean_reward']:.3f} | strict={rec['strict_pct']:5.1f}% | kl={rec['kl_to_ref']:.4f} | rms={rec['rms']:.4f} | t={rec['elapsed_sec']}s", flush=True)

    meta = {
        "tag": tag, "num_vectors": num_vectors, "warm_started": init_sp is not None,
        "co_presence": foreign_pool is not None, "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "online_curve": online_curve,
    }
    return sp.detach().clone(), mid_sp, meta


# ---------------------------------------------------------------------------
# 5. Track 2: Zero-Post-RL Vertical Domain Document CR-SP Compiler (HeliosPay v4.2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_doc_self_distill_cache(
    student, tokenizer, tmpl: ChatTemplateTokens, doc_text: str, doc_train: List[Dict[str, Any]], topk: int = 64
) -> List[Dict[str, Any]]:
    """
    Zero-Post-RL In-Context Self-Distillation Cache:
      Uses the EXACT SAME frozen student conditioned on DOC_A_TEXT (HeliosPay v4.2) to extract
      Top-64 logits on gold rule/fact solutions!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    doc_ids = tokenizer.encode(
        "Use the following internal specification strictly to answer every question accurately:\n\n" + doc_text,
        add_special_tokens=False,
    )
    nt_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)

    cache = []
    student.eval()
    for item in doc_train:
        qi = tokenizer.encode(item["question"], add_special_tokens=False)
        ri = nt_ids + tokenizer.encode(item["gold_answer"], add_special_tokens=False)
        e, l, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, sys_text_ids=doc_ids, device=device, model_dtype=dtype)
        out = student(inputs_embeds=e, use_cache=False)
        sl = out.logits[0, :-1, :].float()
        m = l[0, 1:] != -100
        probs = F.softmax(sl[m], dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        cache.append({"q_ids": qi, "r_ids": ri, "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "kind": item["kind"]})
        del e, l, out
    print(f"  [DocSelfCache HeliosPay ] Cached {len(cache)} Top-64 targets from In-Context Student in {time.time()-t0:.1f}s", flush=True)
    return cache


def train_doc_cr_sp(
    student, tokenizer, tmpl: ChatTemplateTokens, doc_cache: List[Dict[str, Any]],
    train_pools: Dict[str, Any], num_vectors: int = 64, num_updates: int = 45,
    batch_size: int = 10, lr_max: float = 0.008, lambda_nll: float = 0.45,
    lambda_inv: float = 0.35, max_rms: float = 0.045,
    foreign_pool: Optional[List[torch.Tensor]] = None, co_prob: float = 0.65,
    null_pool: Optional[List[Dict[str, Any]]] = None, lambda_null: float = 0.65,
    seed: int = 42, tag: str = "E3_Doc_CSCR",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    sp = init_anchored_sp(s_emb, tokenizer, DOC_TASK_INST, num_vectors=num_vectors, anchored=True)
    opt = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)
    log_hist = {}

    for step in range(1, num_updates + 1):
        prog = (step - 1) / max(1, num_updates - 1)
        lr = lr_max * 0.08 + 0.5 * (lr_max * 0.92) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        student.config.use_cache = False
        student.train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt.zero_grad()

        batch = [doc_cache[rng.randrange(len(doc_cache))] for _ in range(batch_size)]
        tot_dist, tot_nll, tot_inv, tot_null = 0.0, 0.0, 0.0, 0.0

        for item in batch:
            qi, ri = item["q_ids"], item["r_ids"]
            ti, tp = item["ti"].to(device), item["tp"].float().to(device)
            e0, l0, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, sp_sys_start=[sp], device=device, model_dtype=dtype)
            slot_kw, _ = sample_slot_view_cscr(rng, train_pools, sp, foreign_pool=foreign_pool, co_prob=co_prob)
            ec, lc, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **slot_kw)
            be, bm, bl = pad_and_stack_train_batch([e0, ec], [l0, lc], pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm)
            sl = out.logits[:, :-1, :]
            lb = bl[:, 1:]
            m0, mc = lb[0] != -100, lb[1] != -100
            z0, zc = sl[0][m0].float(), sl[1][mc].float()

            l_dist = hybrid_top64_distill_loss(zc, ti, tp) + 0.35 * hybrid_top64_distill_loss(z0, ti, tp)
            l_nll = F.cross_entropy(zc, lb[1][mc], reduction="mean")
            l_inv = cross_context_invariance_kl(z0, zc, topk=32)
            loss = (l_dist + lambda_nll * l_nll + lambda_inv * l_inv) / batch_size
            loss.backward()
            tot_dist += float(l_dist.item()) / batch_size
            tot_nll += float(l_nll.item()) / batch_size
            tot_inv += float(l_inv.item()) / batch_size
            del be, bm, bl, out, z0, zc

        # Null-Task Identity Regularizer L_null: keeps Doc_CR SILENT on non-HeliosPay queries!
        if null_pool and lambda_null > 0.0:
            n_null = 3
            for _ in range(n_null):
                ni = null_pool[rng.randrange(len(null_pool))]
                slot_kw, _ = sample_slot_view_cscr(rng, train_pools, sp, foreign_pool=foreign_pool, co_prob=co_prob)
                en, ln, _ = build_multislot_sp_inputs(s_emb, tmpl, ni["q_ids"], ni["r_ids"], device=device, model_dtype=dtype, **slot_kw)
                out_n = student(inputs_embeds=en)
                zn = out_n.logits[0, :-1, :].float()[ln[0, 1:] != -100]
                ati, atp, aent = ni["anc_ti"].to(device), ni["anc_tp"].float().to(device), ni["anc_ent"].to(device)
                na = min(zn.size(0), ati.size(0))
                ce_n = -(atp[:na] * F.log_softmax(zn[:na], dim=-1).gather(-1, ati[:na])).sum(dim=-1)
                kl_n = (ce_n - aent[:na]).clamp_min(0.0).mean()
                (lambda_null * kl_n / n_null).backward()
                tot_null += float(kl_n.item()) / n_null
                del en, ln, out_n

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)
        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True

        if step in {1, 20, num_updates}:
            log_hist[str(step)] = {"dist": round(tot_dist, 4), "nll": round(tot_nll, 4), "inv": round(tot_inv, 4), "null": round(tot_null, 4)}
            print(f"  [{tag:24s}] step {step:2d}/{num_updates} | dist={tot_dist:.4f} | nll={tot_nll:.4f} | inv={tot_inv:.4f} | null={tot_null:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "num_vectors": num_vectors, "co_presence": foreign_pool is not None,
        "lambda_null": lambda_null, "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


# ---------------------------------------------------------------------------
# 6. Exact Cross-Expert Interference Matrix (Pairwise & 3-Way KL Drift)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_interference_matrix(
    student, tokenizer, tmpl, eval_ctx, probes: Dict[str, List[Tuple[List[int], List[int]]]],
    experts: Dict[str, torch.Tensor], tag: str = "",
) -> Dict[str, Any]:
    """
    For each domain d in {IF, Math, Doc}, compare single-expert distribution vs:
      - Bare student (measures expert activation signal strength on its OWN domain vs OFF domains!)
      - 3-expert Triple-Split [Doc@P1, Math@P2, IF@P4] (measures cross-expert interference KL!)
    """
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    student.eval()

    def get_top32(q_ids, r_ids, sp_sys_end=None, sp_pre_query=None, sp_post_query=None):
        e, l, _ = build_multislot_sp_inputs(
            s_emb, tmpl, q_ids, r_ids,
            sys_text_ids=eval_ctx["persona_ids"], history_turns_ids=eval_ctx["history_ids"],
            sp_sys_end=sp_sys_end, sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
            device=device, model_dtype=dtype,
        )
        out = student(inputs_embeds=e, use_cache=False)
        z = out.logits[0, :-1, :].float()[l[0, 1:] != -100]
        p = F.softmax(z, dim=-1)
        tp, ti = torch.topk(p, 32, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        return z, ti, tp

    def kl_from_ref(z_cur, ti_ref, tp_ref):
        n = min(z_cur.size(0), ti_ref.size(0))
        lp = F.log_softmax(z_cur[:n], dim=-1).gather(-1, ti_ref[:n])
        ent = -(tp_ref[:n] * tp_ref[:n].clamp_min(1e-8).log()).sum(dim=-1)
        ce = -(tp_ref[:n] * lp).sum(dim=-1)
        return float((ce - ent).clamp_min(0.0).mean().item())

    sp_if, sp_math, sp_doc = experts["if"], experts["math"], experts["doc"]
    out_m: Dict[str, Any] = {}
    for dom, pairs in probes.items():
        # Reference slot for each expert: Doc@P1, Math@P2, IF@P4
        slot_single = {
            "doc": dict(sp_sys_end=[sp_doc]),
            "math": dict(sp_pre_query=[sp_math]),
            "if": dict(sp_post_query=[sp_if]),
        }
        kl_bare_vs_exp = {"doc": [], "math": [], "if": []}
        kl_own_vs_triple = []
        for qi, ri in pairs:
            _, ti_b, tp_b = get_top32(qi, ri)
            for ek, kw in slot_single.items():
                z_e, ti_e, tp_e = get_top32(qi, ri, **kw)
                kl_bare_vs_exp[ek].append(kl_from_ref(z_e, ti_b, tp_b))
                if ek == dom:
                    ti_own, tp_own = ti_e, tp_e
            z_tri, _, _ = get_top32(qi, ri, sp_sys_end=[sp_doc], sp_pre_query=[sp_math], sp_post_query=[sp_if])
            kl_own_vs_triple.append(kl_from_ref(z_tri, ti_own, tp_own))

        out_m[dom] = {
            "kl_from_bare_doc": round(sum(kl_bare_vs_exp["doc"]) / len(pairs), 5),
            "kl_from_bare_math": round(sum(kl_bare_vs_exp["math"]) / len(pairs), 5),
            "kl_from_bare_if": round(sum(kl_bare_vs_exp["if"]) / len(pairs), 5),
            "triple_interference_kl": round(sum(kl_own_vs_triple) / len(pairs), 5),
        }
        print(f"  [Interf {tag:8s} | {dom:4s}] bare->(doc={out_m[dom]['kl_from_bare_doc']:.4f}, math={out_m[dom]['kl_from_bare_math']:.4f}, if={out_m[dom]['kl_from_bare_if']:.4f}) | own->Triple KL={out_m[dom]['triple_interference_kl']:.4f}", flush=True)
    return out_m


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Suite 7 (exp9): Multi-Seed Alg B (CR-MS-GRPO+) & 3-Expert Arbitrary-Slot Composition ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    if_train96, ifeval_test96, ifbench64, math_train64, math_test48, doc_train, doc_test_rules, doc_test_facts = load_datasets_exp9()

    train_stream: List[int] = []
    for x in if_train96:
        train_stream.extend(tokenizer.encode(x["prompt"], add_special_tokens=False))
    train_pools = {
        "stream": train_stream,
        "personas": [tokenizer.encode(s, add_special_tokens=False) for s in TRAIN_PERSONAS],
        "turns": [(tokenizer.encode(u, add_special_tokens=False), tokenizer.encode(a, add_special_tokens=False)) for u, a in TRAIN_DIALOG_TURNS],
        "rags": [tokenizer.encode(c + "\n\n", add_special_tokens=False) for c in TRAIN_RAG_CHUNKS],
    }
    eval_ctx = {
        "persona_ids": tokenizer.encode(EVAL_PERSONA, add_special_tokens=False),
        "history_ids": [(tokenizer.encode(u, add_special_tokens=False), tokenizer.encode(a, add_special_tokens=False)) for u, a in EVAL_HISTORY_TURNS],
        "rag_ids": tokenizer.encode(EVAL_RAG_BLOCK + "\n\n", add_special_tokens=False),
    }

    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad = False
    s_emb = student.get_input_embeddings()

    foreign_pool = build_foreign_distractor_pool(s_emb, tokenizer, num_vectors=64, seed=77)

    results: Dict[str, Any] = {
        "backbone": STUDENT_ID,
        "teacher_free": True,
        "seeds": [42, 108, 202],
        "harvest": {},
        "track1_multiseed_train": {},
        "track1_ifeval96_multiseed": {},
        "track1_ifeval96_summary": {},
        "track1_ifbench64": {},
        "track2_experts_train": {},
        "track2_interference_kl": {},
        "track2_composition_evals": {
            "ifeval96": {},
            "ifbench64": {},
            "math48": {},
            "doc36": {},
        },
    }

    def save():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # =========================================================================
    # PHASE 1: TRACK 1 — 3-SEED ALGORITHM B (TF3 Warm-Start -> TF2 GRPO) & SCALED EVAL
    # =========================================================================
    print("\n=== PHASE 1 (Track 1): 3-Seed Algorithm B Convergence & Scaled IFEval96 / IFBench64 ===", flush=True)
    if_wins, st_if = harvest_verified_wins_with_anchors(
        student, tokenizer, tmpl, if_train96, task="if", n_samples=3, max_new_tokens=768, seed=202, tag="IF96_BareHarvest"
    )
    results["harvest"]["if_train96"] = st_if
    save()

    # Bare Student baseline on IFEval96 (Multi-Turn context: P2 and P4 without SP are identical!)
    results["track1_ifeval96_multiseed"]["Bare_Student"] = eval_ifeval96_slots(
        student, tokenizer, tmpl, eval_ctx, ifeval_test96, tag="Bare Student (MultiTurn)"
    )
    save()

    seeds = [42, 108, 202]
    seed_models: Dict[int, Dict[str, torch.Tensor]] = {}

    for sd in seeds:
        print(f"\n  --- Track 1 Seed {sd}: Training Stage-1 TF3, Cold-Start TF2, and Full Alg B (TF3->TF2) [L=64] ---", flush=True)
        # 1) Stage-1 Only: TF3 KL-Anchored Self-Bootstrap (with Co-Presence Aug)
        sp_s1, meta_s1 = train_stage1_tf3_selfboot(
            student, tokenizer, tmpl, if_wins, train_pools, IF_TASK_INST,
            num_vectors=64, num_updates=24, lr_max=0.0055,
            foreign_pool=foreign_pool, co_prob=0.55, seed=sd, tag=f"S1_TF3_seed{sd}",
        )
        # 2) Cold-Start TF2 (Direct Multi-Slot GRPO, NO warm-start, NO Co-Presence -> also serves as Naive IF control!)
        sp_cold, sp_cold_mid, meta_cold = train_stage2_tf2_multislot_grpo(
            student, tokenizer, tmpl, if_train96, train_pools, task="if", task_inst=IF_TASK_INST,
            init_sp=None, num_vectors=64, num_steps=10, mid_ckpt_step=5, lr_max=0.0025,
            foreign_pool=None, co_prob=0.0, seed=sd, tag=f"Cold_TF2_seed{sd}",
        )
        # 3) Full Algorithm B Hybrid: Stage-1 TF3 Warm-Start -> Stage-2 TF2 Multi-Slot GRPO WITH Co-Presence!
        sp_algb, sp_algb_mid, meta_algb = train_stage2_tf2_multislot_grpo(
            student, tokenizer, tmpl, if_train96, train_pools, task="if", task_inst=IF_TASK_INST,
            init_sp=sp_s1, num_vectors=64, num_steps=10, mid_ckpt_step=5, lr_max=0.0020,
            foreign_pool=foreign_pool, co_prob=0.55, seed=sd, tag=f"Full_AlgB_seed{sd}",
        )

        seed_models[sd] = {
            "s1": sp_s1, "cold": sp_cold, "cold_mid": sp_cold_mid,
            "algb": sp_algb, "algb_mid": sp_algb_mid,
        }
        results["track1_multiseed_train"][f"seed_{sd}"] = {"stage1_tf3": meta_s1, "cold_tf2": meta_cold, "full_algb": meta_algb}
        torch.save(sp_algb.cpu(), os.path.join(OUT_DIR, f"sp_if_algb_seed{sd}.pt"))
        torch.save(sp_cold.cpu(), os.path.join(OUT_DIR, f"sp_if_cold_seed{sd}.pt"))
        save()

        # Immediate IFEval96 evaluation for this seed (P2 & P4 + mid-step 5 convergence check!)
        r_sd: Dict[str, Any] = {}
        r_sd["P2_Stage1_TF3"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[sp_s1], tag=f"s{sd} P2 Stage1_TF3")
        r_sd["P4_Stage1_TF3"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_post_query=[sp_s1], tag=f"s{sd} P4 Stage1_TF3")
        r_sd["P4_Cold_TF2_step5"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_post_query=[sp_cold_mid], tag=f"s{sd} P4 Cold_TF2@step5")
        r_sd["P2_Cold_TF2_step10"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[sp_cold], tag=f"s{sd} P2 Cold_TF2@step10")
        r_sd["P4_Cold_TF2_step10"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_post_query=[sp_cold], tag=f"s{sd} P4 Cold_TF2@step10")
        r_sd["P4_Full_AlgB_step5"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_post_query=[sp_algb_mid], tag=f"s{sd} P4 Full_AlgB@step5")
        r_sd["P2_Full_AlgB_step10"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[sp_algb], tag=f"s{sd} P2 Full_AlgB@step10")
        r_sd["P4_Full_AlgB_step10"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_post_query=[sp_algb], tag=f"s{sd} P4 Full_AlgB@step10")
        results["track1_ifeval96_multiseed"][f"seed_{sd}"] = r_sd
        save()

    # Aggregate Mean +/- Std across the 3 seeds on IFEval96!
    print("\n  --- Track 1 Multi-Seed Summary on Held-Out IFEval96 (Mean +/- Std across 3 Seeds) ---", flush=True)
    keys_agg = [
        "P2_Stage1_TF3", "P4_Stage1_TF3",
        "P4_Cold_TF2_step5", "P2_Cold_TF2_step10", "P4_Cold_TF2_step10",
        "P4_Full_AlgB_step5", "P2_Full_AlgB_step10", "P4_Full_AlgB_step10",
    ]
    for k in keys_agg:
        vals = [results["track1_ifeval96_multiseed"][f"seed_{sd}"][k]["prompt_strict"] for sd in seeds]
        insts = [results["track1_ifeval96_multiseed"][f"seed_{sd}"][k]["inst_strict"] for sd in seeds]
        cls = [results["track1_ifeval96_multiseed"][f"seed_{sd}"][k]["closed_acc"] for sd in seeds]
        results["track1_ifeval96_summary"][k] = {
            "strict_mean": round(statistics.mean(vals), 2),
            "strict_std": round(statistics.stdev(vals), 2),
            "strict_per_seed": vals,
            "inst_mean": round(statistics.mean(insts), 2),
            "closed_acc_mean": round(statistics.mean(cls), 2),
        }
        s = results["track1_ifeval96_summary"][k]
        print(f"    {k:22s}: strict = {s['strict_mean']:5.2f} +/- {s['strict_std']:4.2f}%  {vals} | inst = {s['inst_mean']:5.2f}% | closed_acc = {s['closed_acc_mean']:5.2f}%", flush=True)
    save()

    # Scaled Official IFBench64 (2x IFBench32!) on Seed-42 Champion Models
    print("\n  --- Scaled IFBench64 (64 Stratified Prompts, 2x IFBench32!) ---", flush=True)
    e1_if_cscr = seed_models[42]["algb"]
    e1_if_naive = seed_models[42]["cold"]
    e1_if_s1 = seed_models[42]["s1"]

    results["track1_ifbench64"]["Bare_Student"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, tag="Bare Student (MultiTurn)")
    results["track1_ifbench64"]["P4_Stage1_TF3"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_post_query=[e1_if_s1], tag="P4 Stage1_TF3 (L64)")
    results["track1_ifbench64"]["P4_Cold_TF2"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_post_query=[e1_if_naive], tag="P4 Cold_TF2 (L64)")
    results["track1_ifbench64"]["P2_Full_AlgB"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_pre_query=[e1_if_cscr], tag="P2 Full_AlgB CS-CR (L64)")
    results["track1_ifbench64"]["P4_Full_AlgB"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_post_query=[e1_if_cscr], tag="P4 Full_AlgB CS-CR (L64)")
    save()

    # =========================================================================
    # PHASE 2: TRACK 2 — BUILD ZERO-TEACHER MATH & DOMAIN-DOC SPECIALISTS (L=64)
    # =========================================================================
    print("\n=== PHASE 2 (Track 2): Zero-Post-RL Math (MATH-500) & Domain-Doc (HeliosPay v4.2) Specialists ===", flush=True)
    math_wins, st_math = harvest_verified_wins_with_anchors(
        student, tokenizer, tmpl, math_train64, task="math", n_samples=3, max_new_tokens=960, seed=303, tag="MATH64_BareHarvest"
    )
    results["harvest"]["math_train64"] = st_math
    save()

    # Off-domain null pools (free from bare-student anchor caches!)
    null_for_math = if_wins[:24]
    null_for_doc = if_wins[:16] + math_wins[:16]

    # E2_Math_Naive (NO Co-Presence, NO L_null)
    e2_math_naive, meta_m_naive = train_stage1_tf3_selfboot(
        student, tokenizer, tmpl, math_wins, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=24, lr_max=0.0055, lambda_anc=0.35,
        foreign_pool=None, co_prob=0.0, null_pool=None, lambda_null=0.0, seed=42, tag="E2_Math_Naive",
    )
    # E2_Math_CSCR (Full Alg B: Stage-1 TF3 -> Stage-2 TF2 Online GRPO + Co-Presence + L_null!)
    e2_math_s1, meta_m_s1 = train_stage1_tf3_selfboot(
        student, tokenizer, tmpl, math_wins, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=24, lr_max=0.0055, lambda_anc=0.55,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="E2_Math_S1_CSCR",
    )
    e2_math_cscr, _, meta_m_s2 = train_stage2_tf2_multislot_grpo(
        student, tokenizer, tmpl, math_train64, train_pools, task="math", task_inst=MATH_TASK_INST,
        init_sp=e2_math_s1, num_vectors=64, num_steps=8, mid_ckpt_step=4, max_new_tokens=960,
        lr_max=0.0018, beta_kl=0.05, lambda_inv=0.25,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.30, seed=42, tag="E2_Math_CSCR",
    )
    results["track2_experts_train"]["E2_Math_Naive"] = meta_m_naive
    results["track2_experts_train"]["E2_Math_S1_CSCR"] = meta_m_s1
    results["track2_experts_train"]["E2_Math_CSCR"] = meta_m_s2
    torch.save(e2_math_cscr.cpu(), os.path.join(OUT_DIR, "sp_math_cscr.pt"))
    save()

    # E3_Doc (HeliosPay v4.2, Zero Pretraining Leakage, Zero Post-RL!)
    doc_cache = precompute_doc_self_distill_cache(student, tokenizer, tmpl, DOC_A_TEXT, doc_train, topk=64)
    # Control: E3_Doc_NaiveCR (NO Co-Presence, NO L_null)
    e3_doc_naive, meta_d_naive = train_doc_cr_sp(
        student, tokenizer, tmpl, doc_cache, train_pools, num_vectors=64, num_updates=75,
        lr_max=0.010, max_rms=0.055,
        foreign_pool=None, co_prob=0.0, null_pool=None, lambda_null=0.0, seed=42, tag="E3_Doc_NaiveCR",
    )
    # Proposed: E3_Doc_CSCR (Co-Presence Distractors + Null-Task Identity Preservation L_null!)
    e3_doc_cscr, meta_d_cscr = train_doc_cr_sp(
        student, tokenizer, tmpl, doc_cache, train_pools, num_vectors=64, num_updates=75,
        lr_max=0.010, max_rms=0.055,
        foreign_pool=foreign_pool, co_prob=0.65, null_pool=null_for_doc, lambda_null=0.65, seed=42, tag="E3_Doc_CSCR",
    )
    results["track2_experts_train"]["E3_Doc_NaiveCR"] = meta_d_naive
    results["track2_experts_train"]["E3_Doc_CSCR"] = meta_d_cscr
    torch.save(e3_doc_cscr.cpu(), os.path.join(OUT_DIR, "sp_doc_cscr.pt"))
    save()

    # =========================================================================
    # PHASE 3: ZERO-INTERFERENCE MULTI-EXPERT ARBITRARY-SLOT COMPOSITION SUITE
    # =========================================================================
    print("\n=== PHASE 3 (Track 2): Multi-Expert Arbitrary-Slot Composition & Zero-Interference Suite ===", flush=True)

    # 3.1 Exact Cross-Expert KL Interference Matrix (Naive vs CS-CR!)
    probes = {
        "if": [(x["q_ids"], x["r_ids"][:256]) for x in if_wins[:10]],
        "math": [(x["q_ids"], x["r_ids"][:256]) for x in math_wins[:10]],
        "doc": [(x["q_ids"], x["r_ids"][:256]) for x in doc_cache[:10]],
    }
    results["track2_interference_kl"]["Naive"] = compute_interference_matrix(
        student, tokenizer, tmpl, eval_ctx, probes,
        {"if": e1_if_naive, "math": e2_math_naive, "doc": e3_doc_naive}, tag="Naive",
    )
    results["track2_interference_kl"]["CSCR"] = compute_interference_matrix(
        student, tokenizer, tmpl, eval_ctx, probes,
        {"if": e1_if_cscr, "math": e2_math_cscr, "doc": e3_doc_cscr}, tag="CS-CR",
    )
    save()

    # 3.2 Domain-Doc HeliosPay36 Evaluations (Bare, In-Context Upper Bound, Single P1/P2/P4, Dual, Triple!)
    print("\n  --- [Domain-Doc HeliosPay v4.2] Arbitrary-Slot & Multi-Expert Composition (36 items) ---", flush=True)
    d_ev = results["track2_composition_evals"]["doc36"]
    d_ev["Bare_Student"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, tag="Bare Student (0-Leakage)")
    d_ev["InContext_Doc_UpperBound"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, extra_sys_text=DOC_A_TEXT, tag="In-Context Raw Doc (Upper)")
    d_ev["Single_Doc_Naive_P1"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_naive], tag="Single Doc_Naive @ P1")
    d_ev["Single_Doc_CSCR_P1"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_cscr], tag="Single Doc_CSCR @ P1")
    d_ev["Single_Doc_CSCR_P2"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_pre_query=[e3_doc_cscr], tag="Single Doc_CSCR @ P2")
    d_ev["Single_Doc_CSCR_P4"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_post_query=[e3_doc_cscr], tag="Single Doc_CSCR @ P4")
    d_ev["Dual_Naive_DocP1_IFP4"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_naive], sp_post_query=[e1_if_naive], tag="Dual Naive [Doc@P1 + IF@P4]")
    d_ev["Dual_CSCR_DocP1_IFP4"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_cscr], sp_post_query=[e1_if_cscr], tag="Dual CS-CR [Doc@P1 + IF@P4]")
    d_ev["Triple_Naive_Split"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_naive], sp_pre_query=[e2_math_naive], sp_post_query=[e1_if_naive], tag="Triple Naive [P1+P2+P4]")
    d_ev["Triple_CSCR_Split"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_cscr], sp_pre_query=[e2_math_cscr], sp_post_query=[e1_if_cscr], tag="Triple CS-CR [P1+P2+P4]")
    save()

    # 3.3 MATH48 Evaluations (Bare, Single P2/P4, Dual Split, Permutation Invariance, Triple!)
    print("\n  --- [MATH48 Held-Out Level 3-5] Arbitrary-Slot & Multi-Expert Composition ---", flush=True)
    m_ev = results["track2_composition_evals"]["math48"]
    m_ev["Bare_Student"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, tag="Bare Student (MultiTurn)")
    m_ev["Single_Math_Naive_P2"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_pre_query=[e2_math_naive], tag="Single Math_Naive @ P2")
    m_ev["Single_Math_CSCR_P2"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_pre_query=[e2_math_cscr], tag="Single Math_CSCR @ P2")
    m_ev["Single_Math_CSCR_P4"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_post_query=[e2_math_cscr], tag="Single Math_CSCR @ P4")
    m_ev["Dual_CSCR_MathP2_IFP4"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_pre_query=[e2_math_cscr], sp_post_query=[e1_if_cscr], tag="Dual CS-CR [Math@P2 + IF@P4]")
    m_ev["Dual_CSCR_Adj_IF_Math_P2"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_pre_query=[e1_if_cscr, e2_math_cscr], tag="Dual Adj [IF; Math] @ P2")
    m_ev["Dual_CSCR_Adj_Math_IF_P2"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_pre_query=[e2_math_cscr, e1_if_cscr], tag="Dual Adj [Math; IF] @ P2")
    m_ev["Triple_Naive_Split"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_sys_end=[e3_doc_naive], sp_pre_query=[e2_math_naive], sp_post_query=[e1_if_naive], tag="Triple Naive [P1+P2+P4]")
    m_ev["Triple_CSCR_Split"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_sys_end=[e3_doc_cscr], sp_pre_query=[e2_math_cscr], sp_post_query=[e1_if_cscr], tag="Triple CS-CR [P1+P2+P4]")
    save()

    # 3.4 IFEval96 Composition Evaluations (Dual, Permutation Invariance, Triple Naive vs Triple CS-CR!)
    print("\n  --- [IFEval96] Multi-Expert Composition & Permutation Invariance ---", flush=True)
    i_ev = results["track2_composition_evals"]["ifeval96"]
    i_ev["Bare_Student"] = results["track1_ifeval96_multiseed"]["Bare_Student"]
    i_ev["Single_IF_Naive_P4"] = results["track1_ifeval96_multiseed"]["seed_42"]["P4_Cold_TF2_step10"]
    i_ev["Single_IF_CSCR_P4"] = results["track1_ifeval96_multiseed"]["seed_42"]["P4_Full_AlgB_step10"]
    i_ev["Dual_Naive_DocP1_IFP4"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_sys_end=[e3_doc_naive], sp_post_query=[e1_if_naive], tag="Dual Naive [Doc@P1 + IF@P4]")
    i_ev["Dual_CSCR_DocP1_IFP4"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_sys_end=[e3_doc_cscr], sp_post_query=[e1_if_cscr], tag="Dual CS-CR [Doc@P1 + IF@P4]")
    i_ev["Dual_CSCR_MathP2_IFP4"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[e2_math_cscr], sp_post_query=[e1_if_cscr], tag="Dual CS-CR [Math@P2 + IF@P4]")
    i_ev["Dual_CSCR_Adj_IF_Math_P2"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[e1_if_cscr, e2_math_cscr], tag="Dual Adj [IF; Math] @ P2")
    i_ev["Dual_CSCR_Adj_Math_IF_P2"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[e2_math_cscr, e1_if_cscr], tag="Dual Adj [Math; IF] @ P2")
    i_ev["Triple_Naive_Split"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_sys_end=[e3_doc_naive], sp_pre_query=[e2_math_naive], sp_post_query=[e1_if_naive], tag="Triple Naive [P1+P2+P4]")
    i_ev["Triple_CSCR_Split"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_sys_end=[e3_doc_cscr], sp_pre_query=[e2_math_cscr], sp_post_query=[e1_if_cscr], tag="Triple CS-CR [P1+P2+P4]")
    save()

    # 3.5 Official IFBench64 Grand Finale: 3-Expert Triple-Split Naive vs CS-CR!
    print("\n  --- [Official IFBench64] Grand Finale: 3-Expert Simultaneous Split-Slot Composition ---", flush=True)
    b_ev = results["track2_composition_evals"]["ifbench64"]
    b_ev["Bare_Student"] = results["track1_ifbench64"]["Bare_Student"]
    b_ev["Single_IF_CSCR_P4"] = results["track1_ifbench64"]["P4_Full_AlgB"]
    b_ev["Triple_Naive_Split"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_sys_end=[e3_doc_naive], sp_pre_query=[e2_math_naive], sp_post_query=[e1_if_naive], tag="Triple Naive [P1+P2+P4]")
    b_ev["Triple_CSCR_Split"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_sys_end=[e3_doc_cscr], sp_pre_query=[e2_math_cscr], sp_post_query=[e1_if_cscr], tag="Triple CS-CR [P1+P2+P4]")
    save()

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== Suite 7 (exp9) COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
