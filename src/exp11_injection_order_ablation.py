"""
Suite 9 (exp11): Comprehensive Injection-Order & Slot-Permutation Study (Short & Long Context)
==============================================================================================
Systematically completes all injection-order dimensions across CSPD, KV-PD, and CRSP:
  1) Part A — Dual-Expert Split-Slot Swap ([Math@P2 + IF@P4] vs. [IF@P2 + Math@P4]) &
              Same-Slot Adjacent Swap ([Math; IF] vs. [IF; Math] at P2 & P4) across L_ctx in {0, 1024, 4096, 8192}
              for both Direct CRSP (L=64) and Algorithm 1 KV-Inv-CRSP (L=64).
  2) Part B — Three-Expert All-6-Permutations Study ({Doc, Math, IF}, all 3! = 6 orders) at L_ctx in {0, 4096}:
              (i)   PR-CSPD fixed-P0 3-expert concatenation (6 permutations)
              (ii)  KV-PD fixed-P0 3-expert concatenation (6 permutations)
              (iii) Alg. 1 CRSP 3-expert split-slot assignment across (P1, P2, P4) (6 permutations)
              Reports per-permutation gains, mean +- std, min-max spread, and optimal slot ordering.
  3) Part C — End-to-End Generative Order-Swap Verification at L_ctx in {0, 4096}:
              - MATH32    (N=32, 1536 tok): Alg. 1 Split [Math@P2 + IF@P4] vs. [IF@P2 + Math@P4] (L=0 & 4096)
              - IFBench32 (N=32, 2048 tok): Alg. 1 Split [Math@P2 + IF@P4] vs. [IF@P2 + Math@P4] (L=0 & 4096)
                                            + PR-CSPD [Math; IF] vs. [IF; Math] @ P0 (L=4096)
                                            + KV-PD   [Math; IF] vs. [IF; Math] @ P0 (L=4096)
              - IFEval48  (N=48, 1536 tok): Alg. 1 Split [Math@P2 + IF@P4] vs. [IF@P2 + Math@P4] (L=0 & 4096)
"""

import itertools
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content/cspd")
sys.path.insert(0, "/content/IFBench")

from cspd_core import get_qwen_template_tokens
from domain_docs_data import DOC_A_TEXT
from exp7_context_robust_sp import (
    IF_TASK_INST,
    TRAIN_DIALOG_TURNS,
    TRAIN_PERSONAS,
    TRAIN_RAG_CHUNKS,
    build_multislot_sp_inputs,
)
from exp9_multiseed_cr_grpo_and_composition import (
    DOC_TASK_INST,
    MATH_TASK_INST,
    build_foreign_distractor_pool,
    load_datasets_exp9,
)
from exp9b_postrl_crsp_and_composition import (
    cache_kv_pure_logit_targets,
    cache_postrl_teacher_top64,
    precompute_doc_cache_eos,
    train_doc_cscr_muon_l128,
    train_kv_distilled_l16_custom,
    train_postrl_cscr_sp,
)
from exp10_long_context_attenuation_and_composition import (
    LENGTHS_COMP,
    build_long_context_bundles,
    build_null_anchor_pool,
    eval_config_on_val_cache,
    eval_pair_interference_kl,
    train_input_cspd_sp,
    train_kv_doc_helios,
)
from exp10c_fast_4096_e2e import (
    run_ifbench_fast,
    run_ifeval_fast,
    run_math_fast,
)

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"

OUT_DIR = "/content/cspd/exp11_results"
os.makedirs(OUT_DIR, exist_ok=True)
RESULT_PATH = os.path.join(OUT_DIR, "exp11_results.json")


def save_results(res: Dict[str, Any]):
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def mean_std(vals: List[float]) -> Dict[str, float]:
    m = sum(vals) / max(1, len(vals))
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / max(1, len(vals)))
    return {
        "mean": round(m, 2),
        "std": round(sd, 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "spread": round(max(vals) - min(vals), 2),
    }


@torch.no_grad()
def eval_3slot_sp_on_val_cache(
    student, tmpl, val_cache, ctx_bundle,
    sp_p1: List[torch.Tensor], sp_p2: List[torch.Tensor], sp_p4: List[torch.Tensor],
) -> Dict[str, float]:
    """Evaluate 3-slot [A@P1 + B@P2 + C@P4] where P1 sits mid-system (after sys_text, before history)."""
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    student.eval()

    kls = []
    for item in val_cache:
        qi, ri = item["q_ids"], item["r_ids"]
        ti = item["ti"].to(device)
        tp = item["tp"].float().to(device)
        ent = item["ent"].float().to(device)

        e, l, _ = build_multislot_sp_inputs(
            s_emb, tmpl, qi, ri,
            sys_text_ids=ctx_bundle["sys_text_ids"],
            history_turns_ids=ctx_bundle["history_turns_ids"],
            rag_ids=ctx_bundle["rag_ids"],
            sp_sys_end=sp_p1,
            sp_pre_query=sp_p2,
            sp_post_query=sp_p4,
            device=device, model_dtype=dtype,
        )
        out = student(inputs_embeds=e, use_cache=False)
        m = l[0, 1:] != -100
        z = out.logits[0, :-1, :][m].float()
        del e, l, out

        n = min(z.size(0), ti.size(0))
        logp = torch.nn.functional.log_softmax(z[:n], dim=-1)
        ce = -(tp[:n] * logp.gather(-1, ti[:n])).sum(dim=-1)
        kls.append((ce - ent[:n]).clamp_min(0.0).mean().item())
        del z, logp

    return {"kl": round(sum(kls) / max(1, len(kls)), 5)}


def main():
    t0 = time.time()
    print("=" * 94, flush=True)
    print("Suite 9 (exp11): Comprehensive Injection-Order & Slot-Permutation Study (0 -> 8192)", flush=True)
    print("=" * 94, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    if_train96, ifeval_test96, ifbench64, math_train64, math_test48, doc_train, doc_rules24, doc_facts12 = load_datasets_exp9()

    if_train_prompts = [x["prompt"] for x in if_train96[:64]]
    if_val_prompts = [x["prompt"] for x in ifeval_test96[:24]]
    math_train_prompts = [x["prompt"] for x in math_train64[:48]]
    math_val_prompts = [x["prompt"] for x in math_test48[:24]]
    ifeval48 = ifeval_test96[:48]
    ifbench32 = ifbench64[:32]
    math32 = math_test48[:32]

    ctx_bundles = build_long_context_bundles(tokenizer, tmpl, LENGTHS_COMP)

    results: Dict[str, Any] = {
        "meta": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "lengths": LENGTHS_COMP,
            "actual_ctx_tokens": {str(k): v["actual_ctx_tokens"] for k, v in ctx_bundles.items()},
        },
        "partA_split_and_adj_order": {"if": {}, "math": {}, "interference": {}},
        "partB_3expert_all6_perms": {"L0": {}, "L4096": {}},
        "partC_e2e_order_swap": {"math32": {}, "ifbench32": {}, "ifeval48": {}},
    }

    # ------------------------------------------------------------------
    # PHASE 1: Teacher Top-64 Caches
    # ------------------------------------------------------------------
    print("\n--- PHASE 1: Caching Post-RL Teacher Top-64 Targets ---", flush=True)
    if_all = cache_postrl_teacher_top64(
        IF_TEACHER_ID, tokenizer, tmpl, if_train_prompts + if_val_prompts,
        max_resp_gen=768, max_resp_trunc=512, topk=64, seed=777, tag="IF_Teacher",
    )
    if_train_cache, if_val_cache = if_all[:len(if_train_prompts)], if_all[len(if_train_prompts):]
    if len(if_val_cache) < 16:
        if_train_cache, if_val_cache = if_all[:-24], if_all[-24:]

    math_all = cache_postrl_teacher_top64(
        MATH_TEACHER_ID, tokenizer, tmpl, math_train_prompts + math_val_prompts,
        max_resp_gen=768, max_resp_trunc=512, topk=64, seed=888, tag="Math_Teacher",
    )
    math_train_cache, math_val_cache = math_all[:len(math_train_prompts)], math_all[len(math_train_prompts):]
    if len(math_val_cache) < 16:
        math_train_cache, math_val_cache = math_all[:-24], math_all[-24:]

    print(f"\n--- Loading Frozen Backbone: {STUDENT_ID} ---", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    for p in student.parameters():
        p.requires_grad_(False)
    student.eval()
    s_emb = student.get_input_embeddings()

    # ------------------------------------------------------------------
    # PHASE 2: Train Complete 3-Domain Expert Library (IF, Math, Doc)
    # ------------------------------------------------------------------
    print("\n--- PHASE 2: Training 3-Domain Experts (PR-CSPD, KV-PD, Direct CRSP, Alg. 1) ---", flush=True)
    train_stream_text = " ".join(TRAIN_PERSONAS + [u + " " + a for u, a in TRAIN_DIALOG_TURNS] + TRAIN_RAG_CHUNKS)
    train_pools = {
        "personas": [tokenizer.encode(p, add_special_tokens=False) for p in TRAIN_PERSONAS],
        "turns": [(tokenizer.encode(u, add_special_tokens=False), tokenizer.encode(a, add_special_tokens=False)) for u, a in TRAIN_DIALOG_TURNS],
        "rags": [tokenizer.encode(r, add_special_tokens=False) for r in TRAIN_RAG_CHUNKS],
        "stream": tokenizer.encode(train_stream_text, add_special_tokens=False),
    }
    foreign_pool = build_foreign_distractor_pool(s_emb, tokenizer, num_vectors=64, seed=77)
    null_for_if = build_null_anchor_pool(student, tmpl, math_train_cache, max_items=24, topk=32)
    null_for_math = build_null_anchor_pool(student, tmpl, if_train_cache, max_items=24, topk=32)
    doc_eos_cache = precompute_doc_cache_eos(student, tokenizer, tmpl, DOC_A_TEXT, doc_train, topk=64)
    doc_val_cache = precompute_doc_cache_eos(
        student, tokenizer, tmpl, DOC_A_TEXT, list(doc_rules24) + list(doc_facts12), topk=64
    )
    for it in doc_val_cache:
        p = it["tp"].float()
        it["ent"] = -(p * p.clamp_min(1e-9).log()).sum(dim=-1)

    # PR-CSPD (IF, Math, Doc)
    sp_pr_if, _ = train_input_cspd_sp(
        student, tokenizer, tmpl, if_train_cache, train_pools["stream"], IF_TASK_INST,
        num_vectors=256, prefix_robust=True, num_updates=45, batch_size=10, lr_max=0.005, max_rms=0.048, seed=42, tag="PR_CSPD_IF_L256",
    )
    sp_pr_math, _ = train_input_cspd_sp(
        student, tokenizer, tmpl, math_train_cache, train_pools["stream"], MATH_TASK_INST,
        num_vectors=256, prefix_robust=True, num_updates=45, batch_size=10, lr_max=0.005, max_rms=0.048, seed=42, tag="PR_CSPD_Math_L256",
    )
    sp_pr_doc, _ = train_input_cspd_sp(
        student, tokenizer, tmpl, doc_eos_cache, train_pools["stream"], DOC_TASK_INST,
        num_vectors=128, prefix_robust=True, num_updates=60, batch_size=10, lr_max=0.008, max_rms=0.10, seed=42, tag="PR_CSPD_Doc_L128",
    )

    # KV-PD (IF, Math, Doc — using verified HeliosPay v4.2 EOS trainer)
    kv_if, _ = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, if_train_cache, inst_text=IF_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.022, seed=42, tag="KVPD_IF_L16",
    )
    kv_math, _ = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, math_train_cache, inst_text=MATH_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.020, seed=42, tag="KVPD_Math_L16",
    )
    kv_doc, _ = train_kv_doc_helios(
        student, tokenizer, tmpl, doc_eos_cache,
        num_kv_tokens=16, num_updates=80, batch_size=8, lr_max=0.03, seed=42, tag="KVPD_Doc_Helios",
    )

    # Direct CRSP (L=64 for IF & Math; Muon L=128 Declarative CRSP for Doc)
    sp_dir_if, _ = train_postrl_cscr_sp(
        student, tokenizer, tmpl, if_train_cache, train_pools, IF_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Direct_CRSP_IF_L64",
    )
    sp_dir_math, _ = train_postrl_cscr_sp(
        student, tokenizer, tmpl, math_train_cache, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="Direct_CRSP_Math_L64",
    )
    sp_cscr_doc, _ = train_doc_cscr_muon_l128(
        student, tokenizer, tmpl, doc_eos_cache, train_pools,
        num_vectors=128, num_updates=80, batch_size=10, lr_max=0.045, lambda_nll=0.85,
        lambda_inv=0.25, max_rms=0.14, foreign_pool=foreign_pool, co_prob=0.65,
        null_pool=null_for_if, lambda_null=0.75, seed=42, tag="CSCR_Doc_MuonL128",
    )

    # Algorithm 1 KV-Inv-CRSP (L=64 for IF & Math)
    kv_if_self = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_if, if_train_prompts, max_resp_gen=768, max_resp_trunc=512, topk=64, seed=606, tag="KV_IF_Self",
    )
    sp_alg1_if, _ = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_if_self, train_pools, IF_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Alg1_KVInv_IF_L64",
    )
    kv_math_self = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_math, math_train_prompts, max_resp_gen=768, max_resp_trunc=512, topk=64, seed=607, tag="KV_Math_Self",
    )
    sp_alg1_math, _ = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_math_self, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="Alg1_KVInv_Math_L64",
    )
    torch.save(
        {
            "sp_pr_if": sp_pr_if.cpu(), "sp_pr_math": sp_pr_math.cpu(), "sp_pr_doc": sp_pr_doc.cpu(),
            "kv_if": kv_if.cpu(), "kv_math": kv_math.cpu(), "kv_doc": kv_doc.cpu(),
            "sp_dir_if": sp_dir_if.cpu(), "sp_dir_math": sp_dir_math.cpu(), "sp_cscr_doc": sp_cscr_doc.cpu(),
            "sp_alg1_if": sp_alg1_if.cpu(), "sp_alg1_math": sp_alg1_math.cpu(),
        },
        os.path.join(OUT_DIR, "exp11_modules.pt"),
    )

    # ------------------------------------------------------------------
    # PART A: Dual-Expert Split-Slot Swap & Adjacent P2/P4 Order Sweep (0..8192)
    # ------------------------------------------------------------------
    print("\n--- PART A: Split-Slot Swap ([M@P2+I@P4] vs [I@P2+M@P4]) & Adj P2/P4 Swap (0..8192) ---", flush=True)
    base_if_kl, base_m_kl = {}, {}
    solo_dir_if_p4, solo_dir_if_p2, solo_alg1_if_p4, solo_alg1_if_p2 = {}, {}, {}, {}
    for L in LENGTHS_COMP:
        bi, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="base")
        bm, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode="base")
        base_if_kl[L], base_m_kl[L] = bi["kl"], bm["kl"]
        _, solo_dir_if_p4[L]  = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="sp", sp_post_query=[sp_dir_if], save_ref_top32=True)
        _, solo_dir_if_p2[L]  = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="sp", sp_pre_query=[sp_dir_if], save_ref_top32=True)
        _, solo_alg1_if_p4[L] = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="sp", sp_post_query=[sp_alg1_if], save_ref_top32=True)
        _, solo_alg1_if_p2[L] = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="sp", sp_pre_query=[sp_alg1_if], save_ref_top32=True)

    dual_specs = [
        ("Direct_Split_M@P2_I@P4", {"sp_pre_query": [sp_dir_math],  "sp_post_query": [sp_dir_if]},   solo_dir_if_p4),
        ("Direct_Split_I@P2_M@P4", {"sp_pre_query": [sp_dir_if],    "sp_post_query": [sp_dir_math]}, solo_dir_if_p2),
        ("Direct_Adj_M_I@P2",      {"sp_pre_query": [sp_dir_math, sp_dir_if]},                        solo_dir_if_p2),
        ("Direct_Adj_I_M@P2",      {"sp_pre_query": [sp_dir_if, sp_dir_math]},                        solo_dir_if_p2),
        ("Direct_Adj_M_I@P4",      {"sp_post_query": [sp_dir_math, sp_dir_if]},                       solo_dir_if_p4),
        ("Direct_Adj_I_M@P4",      {"sp_post_query": [sp_dir_if, sp_dir_math]},                       solo_dir_if_p4),
        ("Alg1_Split_M@P2_I@P4",   {"sp_pre_query": [sp_alg1_math], "sp_post_query": [sp_alg1_if]},  solo_alg1_if_p4),
        ("Alg1_Split_I@P2_M@P4",   {"sp_pre_query": [sp_alg1_if],   "sp_post_query": [sp_alg1_math]}, solo_alg1_if_p2),
        ("Alg1_Adj_M_I@P2",        {"sp_pre_query": [sp_alg1_math, sp_alg1_if]},                      solo_alg1_if_p2),
        ("Alg1_Adj_I_M@P2",        {"sp_pre_query": [sp_alg1_if, sp_alg1_math]},                      solo_alg1_if_p2),
        ("Alg1_Adj_M_I@P4",        {"sp_post_query": [sp_alg1_math, sp_alg1_if]},                     solo_alg1_if_p4),
        ("Alg1_Adj_I_M@P4",        {"sp_post_query": [sp_alg1_if, sp_alg1_math]},                     solo_alg1_if_p4),
    ]
    for name, kw, ref_map in dual_specs:
        results["partA_split_and_adj_order"]["if"][name] = {}
        results["partA_split_and_adj_order"]["math"][name] = {}
        results["partA_split_and_adj_order"]["interference"][name] = {}
        for L in LENGTHS_COMP:
            ri, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="sp", **kw)
            rm, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode="sp", **kw)
            intf = eval_pair_interference_kl(student, tmpl, if_val_cache, ctx_bundles[L], ref_map[L], mode="sp", **kw)
            results["partA_split_and_adj_order"]["if"][name][str(L)] = {**ri, "g_vs_ctx": round(100.0 * (1.0 - ri["kl"] / base_if_kl[L]), 2)}
            results["partA_split_and_adj_order"]["math"][name][str(L)] = {**rm, "g_vs_ctx": round(100.0 * (1.0 - rm["kl"] / base_m_kl[L]), 2)}
            results["partA_split_and_adj_order"]["interference"][name][str(L)] = intf
        si = " | ".join(f"{L}:{results['partA_split_and_adj_order']['if'][name][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_COMP)
        sm = " | ".join(f"{L}:{results['partA_split_and_adj_order']['math'][name][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_COMP)
        sk = " | ".join(f"{L}:{results['partA_split_and_adj_order']['interference'][name][str(L)]:.4f}" for L in LENGTHS_COMP)
        print(f"  [{name:24s}] IF=[{si}] | Math=[{sm}] | intf=[{sk}]", flush=True)
    save_results(results)

    # ------------------------------------------------------------------
    # PART B: Three-Expert All-6-Permutations ({Doc, Math, IF}) at L=0 & L=4096
    #         Evaluated on ALL THREE domains (IF, Math, HeliosPay Doc36)
    # ------------------------------------------------------------------
    print("\n--- PART B: Three-Expert All-6-Permutations ({Doc, Math, IF}) at L=0 & L=4096 ---", flush=True)
    perms = list(itertools.permutations(["D", "M", "I"]))
    pr_map = {"D": sp_pr_doc, "M": sp_pr_math, "I": sp_pr_if}
    kv_map = {"D": kv_doc,    "M": kv_math,    "I": kv_if}
    a1_map = {"D": sp_cscr_doc, "M": sp_alg1_math, "I": sp_alg1_if}

    base_d_kl = {}
    for L, lkey in [(0, "L0"), (4096, "L4096")]:
        bd, _ = eval_config_on_val_cache(student, tmpl, doc_val_cache, ctx_bundles[L], mode="base")
        base_d_kl[L] = bd["kl"]
        block: Dict[str, Any] = {"base_kl": {"if": base_if_kl[L], "math": base_m_kl[L], "doc": base_d_kl[L]}}

        # Solo single-expert reference gains at this L_ctx
        solo_ref = {}
        for tag_s, m_s, kw_s in [
            ("PR_Doc@P0",   "sp", {"sp_sys_start": [sp_pr_doc]}),
            ("PR_Math@P0",  "sp", {"sp_sys_start": [sp_pr_math]}),
            ("PR_IF@P0",    "sp", {"sp_sys_start": [sp_pr_if]}),
            ("KV_Doc@P0",   "kv", {"kv_list": [kv_doc]}),
            ("KV_Math@P0",  "kv", {"kv_list": [kv_math]}),
            ("KV_IF@P0",    "kv", {"kv_list": [kv_if]}),
            ("Alg1_Doc@P2", "sp", {"sp_pre_query": [sp_cscr_doc]}),
            ("Alg1_Math@P2","sp", {"sp_pre_query": [sp_alg1_math]}),
            ("Alg1_IF@P4",  "sp", {"sp_post_query": [sp_alg1_if]}),
        ]:
            vcache_s = doc_val_cache if "Doc" in tag_s else (math_val_cache if "Math" in tag_s else if_val_cache)
            bkl_s = base_d_kl[L] if "Doc" in tag_s else (base_m_kl[L] if "Math" in tag_s else base_if_kl[L])
            rs, _ = eval_config_on_val_cache(student, tmpl, vcache_s, ctx_bundles[L], mode=m_s, **kw_s)
            solo_ref[tag_s] = round(100.0 * (1.0 - rs["kl"] / bkl_s), 2)
        block["solo_ref_g"] = solo_ref
        print(f"  [3-Exp {lkey:5s} Solo Ref] " + " | ".join(f"{k}={v:+.1f}%" for k, v in solo_ref.items()), flush=True)

        fams = [
            ("PR_CSPD_P0_3Exp",        "pr"),
            ("KVPD_P0_3Exp",           "kv"),
            ("Alg1_Adj_P2_3Exp",       "a1_adj"),
            ("Alg1_Split_P1P2P4_3Exp", "a1_split"),
        ]
        for fam, mode in fams:
            per_perm = {}
            if_vals, m_vals, d_vals = [], [], []
            for p in perms:
                pkey = "".join(p)
                if mode == "pr":
                    ri, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="sp", sp_sys_start=[pr_map[x] for x in p])
                    rm, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode="sp", sp_sys_start=[pr_map[x] for x in p])
                    rd, _ = eval_config_on_val_cache(student, tmpl, doc_val_cache, ctx_bundles[L], mode="sp", sp_sys_start=[pr_map[x] for x in p])
                elif mode == "kv":
                    ri, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="kv", kv_list=[kv_map[x] for x in p])
                    rm, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode="kv", kv_list=[kv_map[x] for x in p])
                    rd, _ = eval_config_on_val_cache(student, tmpl, doc_val_cache, ctx_bundles[L], mode="kv", kv_list=[kv_map[x] for x in p])
                elif mode == "a1_adj":
                    ri, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="sp", sp_pre_query=[a1_map[x] for x in p])
                    rm, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode="sp", sp_pre_query=[a1_map[x] for x in p])
                    rd, _ = eval_config_on_val_cache(student, tmpl, doc_val_cache, ctx_bundles[L], mode="sp", sp_pre_query=[a1_map[x] for x in p])
                else:
                    ri = eval_3slot_sp_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], [a1_map[p[0]]], [a1_map[p[1]]], [a1_map[p[2]]])
                    rm = eval_3slot_sp_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], [a1_map[p[0]]], [a1_map[p[1]]], [a1_map[p[2]]])
                    rd = eval_3slot_sp_on_val_cache(student, tmpl, doc_val_cache, ctx_bundles[L], [a1_map[p[0]]], [a1_map[p[1]]], [a1_map[p[2]]])
                gi = round(100.0 * (1.0 - ri["kl"] / base_if_kl[L]), 2)
                gm = round(100.0 * (1.0 - rm["kl"] / base_m_kl[L]), 2)
                gd = round(100.0 * (1.0 - rd["kl"] / base_d_kl[L]), 2)
                if_vals.append(gi)
                m_vals.append(gm)
                d_vals.append(gd)
                per_perm[pkey] = {
                    "if_g": gi, "math_g": gm, "doc_g": gd,
                    "avg2_g": round(0.5 * (gi + gm), 2),
                    "avg3_g": round((gi + gm + gd) / 3.0, 2),
                    "if_kl": ri["kl"], "math_kl": rm["kl"], "doc_kl": rd["kl"],
                }
            st_if   = mean_std(if_vals)
            st_m    = mean_std(m_vals)
            st_d    = mean_std(d_vals)
            st_avg2 = mean_std([0.5 * (a + b) for a, b in zip(if_vals, m_vals)])
            st_avg3 = mean_std([(a + b + c) / 3.0 for a, b, c in zip(if_vals, m_vals, d_vals)])
            block[fam] = {
                "perms": per_perm, "if_summary": st_if, "math_summary": st_m,
                "doc_summary": st_d, "avg2_summary": st_avg2, "avg3_summary": st_avg3,
            }
            print(
                f"  [3-Exp {lkey:5s} | {fam:22s}] IF={st_if['mean']:+5.1f}+-{st_if['std']:4.1f}% (sp {st_if['spread']:4.1f}) | "
                f"Math={st_m['mean']:+5.1f}+-{st_m['std']:4.1f}% (sp {st_m['spread']:4.1f}) | "
                f"Doc={st_d['mean']:+5.1f}+-{st_d['std']:4.1f}% (sp {st_d['spread']:4.1f}) | Avg3={st_avg3['mean']:+5.1f}%",
                flush=True,
            )
            for pkey, pv in per_perm.items():
                print(f"      perm {pkey}: IF={pv['if_g']:+6.2f}% | Math={pv['math_g']:+6.2f}% | Doc={pv['doc_g']:+6.2f}% | Avg3={pv['avg3_g']:+6.2f}%", flush=True)
        results["partB_3expert_all6_perms"][lkey] = block
        save_results(results)

    # ------------------------------------------------------------------
    # PART C: End-to-End Generative Order-Swap Verification (L=0 & L=4096)
    # ------------------------------------------------------------------
    print("\n--- PART C1: MATH32 Split-Slot Swap (L=0 & L=4096) ---", flush=True)
    for L in [0, 4096]:
        results["partC_e2e_order_swap"]["math32"][str(L)] = {
            "Alg1_Split_M@P2_I@P4": run_math_fast(
                student, tokenizer, tmpl, ctx_bundles[L], math32, mode="sp",
                sp_pre_query=[sp_alg1_math], sp_post_query=[sp_alg1_if], tag="Alg1_Split_M@P2_I@P4",
            ),
            "Alg1_Split_I@P2_M@P4": run_math_fast(
                student, tokenizer, tmpl, ctx_bundles[L], math32, mode="sp",
                sp_pre_query=[sp_alg1_if], sp_post_query=[sp_alg1_math], tag="Alg1_Split_I@P2_M@P4",
            ),
        }
        save_results(results)

    print("\n--- PART C2: IFBench32 Order-Swap Across Split CRSP, PR-CSPD & KV-PD (L=0 & L=4096) ---", flush=True)
    for L in [0, 4096]:
        blk = {
            "Alg1_Split_M@P2_I@P4": run_ifbench_fast(
                student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="sp",
                sp_pre_query=[sp_alg1_math], sp_post_query=[sp_alg1_if], tag="Alg1_Split_M@P2_I@P4",
            ),
            "Alg1_Split_I@P2_M@P4": run_ifbench_fast(
                student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="sp",
                sp_pre_query=[sp_alg1_if], sp_post_query=[sp_alg1_math], tag="Alg1_Split_I@P2_M@P4",
            ),
        }
        if L == 4096:
            blk["PR_CSPD_P0_Math_IF"] = run_ifbench_fast(
                student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="sp",
                sp_sys_start=[sp_pr_math, sp_pr_if], tag="PR_CSPD_P0_Math_IF",
            )
            blk["PR_CSPD_P0_IF_Math"] = run_ifbench_fast(
                student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="sp",
                sp_sys_start=[sp_pr_if, sp_pr_math], tag="PR_CSPD_P0_IF_Math",
            )
            blk["KVPD_P0_Math_IF"] = run_ifbench_fast(
                student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="kv",
                kv_list=[kv_math, kv_if], tag="KVPD_P0_Math_IF",
            )
            blk["KVPD_P0_IF_Math"] = run_ifbench_fast(
                student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="kv",
                kv_list=[kv_if, kv_math], tag="KVPD_P0_IF_Math",
            )
        results["partC_e2e_order_swap"]["ifbench32"][str(L)] = blk
        save_results(results)

    print("\n--- PART C3: IFEval48 Split-Slot Swap (L=0 & L=4096) ---", flush=True)
    for L in [0, 4096]:
        results["partC_e2e_order_swap"]["ifeval48"][str(L)] = {
            "Alg1_Split_M@P2_I@P4": run_ifeval_fast(
                student, tokenizer, tmpl, ctx_bundles[L], ifeval48, mode="sp",
                sp_pre_query=[sp_alg1_math], sp_post_query=[sp_alg1_if], tag="Alg1_Split_M@P2_I@P4",
            ),
            "Alg1_Split_I@P2_M@P4": run_ifeval_fast(
                student, tokenizer, tmpl, ctx_bundles[L], ifeval48, mode="sp",
                sp_pre_query=[sp_alg1_if], sp_post_query=[sp_alg1_math], tag="Alg1_Split_I@P2_M@P4",
            ),
        }
        save_results(results)

    results["meta"]["total_wall_sec"] = round(time.time() - t0, 1)
    save_results(results)
    print(f"\n=== ALL DONE! exp11_results.json saved in {results['meta']['total_wall_sec']}s ===", flush=True)


if __name__ == "__main__":
    main()
