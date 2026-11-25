"""
Paper Reference: Section 7.2–7.3, Tables 14–15, Fig. 13(c) (Long-Context E2E & Private-Doc KV-PD 0->4096).
Suite 8B (exp10b): Long-Context End-to-End Generation (IFEval48, IFBench32, MATH32) &
                   Zero-Leakage Private Document KV-PD Long-Context Attenuation (HeliosPay v4.2)
=================================================================================================
Companion to exp10 (which completed Part A 0..8192 single-expert KL/drift sweep, Part B
layerwise attention-mass diagnostic, and Part C 0..8192 multi-expert composition & interference).
Runs on A100-80GB in ~25 min:
  1) Part E (Fixed using exact exp4/exp5 train_kv_prefix_doc recipe: L_kv=16 & L_kv=32):
       Zero-leakage HeliosPay v4.2 (24 rules + 12 facts) under L_ctx in {0, 1024, 2048, 4096}
       comparing KV-PD alone, KV-PD + Micro-RAG, and In-Context Full Manual @ Start!
  2) Part C_KV_Verify:
       Re-verifies KV-PD (lr=0.020, smooth cosine) single & dual concat [Math; IF] / [IF; Math]
       across L_ctx in {0, 1024, 4096, 8192} to confirm zero late-step instability!
  3) Part D (Full End-to-End Autoregressive Generation with strict programmatic verification):
       - IFEval48  (N=48, 1536 tok) at L_ctx in {0, 2048, 4096}
       - IFBench32 (N=32, 2048 tok) at L_ctx in {0, 4096}
       - MATH32    (N=32, 1536 tok) at L_ctx in {0, 4096}
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content/cspd")
sys.path.insert(0, "/content/IFBench")

from cspd_core import get_qwen_template_tokens
from domain_docs_data import DOC_A_TEXT, grade_domain_response
from exp4_doc_cl_breakthroughs import train_kv_prefix_doc
from exp7_context_robust_sp import (
    IF_TASK_INST,
    TRAIN_DIALOG_TURNS,
    TRAIN_PERSONAS,
    TRAIN_RAG_CHUNKS,
)
from exp9_multiseed_cr_grpo_and_composition import (
    MATH_TASK_INST,
    build_foreign_distractor_pool,
    load_datasets_exp9,
)
from exp9b_postrl_crsp_and_composition import (
    cache_kv_pure_logit_targets,
    cache_postrl_teacher_top64,
    train_kv_distilled_l16_custom,
    train_postrl_cscr_sp,
)
from exp10_long_context_attenuation_and_composition import (
    LENGTHS_COMP,
    LENGTHS_SWEEP,
    batch_generate_kv_longctx,
    build_long_context_bundles,
    build_null_anchor_pool,
    eval_config_on_val_cache,
    eval_doc36_kv_longctx,
    eval_ifbench_longctx,
    eval_ifeval_longctx,
    eval_math_longctx,
    train_input_cspd_sp,
)

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"

OUT_DIR = "/content/cspd/exp10b_results"
os.makedirs(OUT_DIR, exist_ok=True)
RESULT_PATH = os.path.join(OUT_DIR, "exp10b_results.json")


def save_results(res: Dict[str, Any]):
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def eval_doc36_incontext_start(
    model, tokenizer, tmpl, ctx_bundle, rules24, facts12, doc_text: str, tag: str = "InContext_Manual_Start"
) -> Dict[str, Any]:
    """Upper-bound reference: Full HeliosPay manual placed in system turn at Start before L_ctx history."""
    t0 = time.time()
    L = ctx_bundle["target_len"]
    doc_ids = tokenizer.encode(
        "Use the following internal specification strictly to answer every question accurately:\n\n" + doc_text,
        add_special_tokens=False,
    )
    sys_ids = doc_ids if ctx_bundle["sys_text_ids"] is None else (doc_ids + tokenizer.encode("\n\n", add_special_tokens=False) + list(ctx_bundle["sys_text_ids"]))
    mod_bundle = dict(ctx_bundle)
    mod_bundle["sys_text_ids"] = sys_ids
    items = list(rules24) + list(facts12)
    bs = 18 if L <= 1024 else 9
    outs = batch_generate_kv_longctx(
        model, tokenizer, tmpl, [x["question"] for x in items], [], mod_bundle,
        batch_size=bs, max_new_tokens=320, temperature=0.0, enable_thinking=False,
    )
    r_ok = sum(int(grade_domain_response(r, it)) for r, it in zip(outs[:len(rules24)], rules24))
    f_ok = sum(int(grade_domain_response(r, it)) for r, it in zip(outs[len(rules24):], facts12))
    tot = r_ok + f_ok
    res = {
        "rule_acc": round(100.0 * r_ok / len(rules24), 2),
        "fact_acc": round(100.0 * f_ok / len(facts12), 2),
        "overall_acc": round(100.0 * tot / len(items), 2),
        "n_rule": r_ok, "n_fact": f_ok, "n_total": len(items),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [Doc36    L={L:4d} | {tag:30s}] overall={res['overall_acc']:5.2f}% ({tot:2d}/36) | rules={res['rule_acc']:5.1f}% ({r_ok:2d}/24) | facts={res['fact_acc']:5.1f}% ({f_ok:2d}/12) | t={res['eval_sec']}s", flush=True)
    return res


def main():
    t0 = time.time()
    print("=" * 92, flush=True)
    print("Suite 8B (exp10b): Long-Context E2E Generation & Fixed Doc KV-PD Attenuation", flush=True)
    print("=" * 92, flush=True)

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

    ctx_bundles = build_long_context_bundles(tokenizer, tmpl, LENGTHS_SWEEP)

    results: Dict[str, Any] = {
        "meta": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "actual_ctx_tokens": {str(k): v["actual_ctx_tokens"] for k, v in ctx_bundles.items()},
        },
        "training": {},
        "partE_doc_kvpd_longctx": {},
        "partC_kvpd_clean_verify": {"if": {}, "math": {}},
        "partD_e2e_ifeval48": {},
        "partD_e2e_ifbench32": {},
        "partD_e2e_math32": {},
    }

    # ------------------------------------------------------------------
    # PHASE 1: Teacher Top-64 Caches (IF & Math)
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
    # PART E FIRST: Fixed Zero-Leakage HeliosPay v4.2 KV-PD (L_kv=16 & L_kv=32)
    # ------------------------------------------------------------------
    print("\n--- PART E: Fixed Zero-Leakage HeliosPay v4.2 KV-PD Under Long Context (0 -> 4096) ---", flush=True)
    kv_doc16, m = train_kv_prefix_doc(
        student, tokenizer, tmpl, doc_train, DOC_A_TEXT,
        num_kv_tokens=16, num_updates=90, batch_size=16, micro_batch=4,
        optimizer_type="adamw", lr_max=0.025, enable_thinking=False, seed=42, tag="KVPD_Doc_L16",
    )
    results["training"]["KVPD_Doc_L16"] = m

    kv_doc32, m = train_kv_prefix_doc(
        student, tokenizer, tmpl, doc_train, DOC_A_TEXT,
        num_kv_tokens=32, num_updates=90, batch_size=16, micro_batch=4,
        optimizer_type="adamw", lr_max=0.025, enable_thinking=False, seed=42, tag="KVPD_Doc_L32",
    )
    results["training"]["KVPD_Doc_L32"] = m

    for L in [0, 1024, 2048, 4096]:
        slot_res = {
            "KVPD_L16_Only": eval_doc36_kv_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], doc_rules24, doc_facts12, [kv_doc16],
                use_micro_rag=False, tag="KVPD_L16_Only",
            ),
            "KVPD_L16_MicroRAG": eval_doc36_kv_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], doc_rules24, doc_facts12, [kv_doc16],
                use_micro_rag=True, tag="KVPD_L16_MicroRAG",
            ),
            "KVPD_L32_Only": eval_doc36_kv_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], doc_rules24, doc_facts12, [kv_doc32],
                use_micro_rag=False, tag="KVPD_L32_Only",
            ),
            "KVPD_L32_MicroRAG": eval_doc36_kv_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], doc_rules24, doc_facts12, [kv_doc32],
                use_micro_rag=True, tag="KVPD_L32_MicroRAG",
            ),
        }
        if L in (0, 4096):
            slot_res["Base_NoDoc"] = eval_doc36_kv_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], doc_rules24, doc_facts12, [],
                use_micro_rag=False, tag="Base_NoDoc",
            )
            slot_res["InContext_Manual_Start"] = eval_doc36_incontext_start(
                student, tokenizer, tmpl, ctx_bundles[L], doc_rules24, doc_facts12, DOC_A_TEXT,
            )
        results["partE_doc_kvpd_longctx"][str(L)] = slot_res
        save_results(results)

    # ------------------------------------------------------------------
    # PHASE 2: Train Procedural Experts Needed for Part D & KV-PD Verify
    # ------------------------------------------------------------------
    print("\n--- PHASE 2: Training Procedural Experts (CSPD, KV-PD, Direct CRSP, Alg. 1) ---", flush=True)
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

    sp_pr_if, m = train_input_cspd_sp(
        student, tokenizer, tmpl, if_train_cache, train_pools["stream"], IF_TASK_INST,
        num_vectors=256, prefix_robust=True, num_updates=45, batch_size=10, lr_max=0.005, max_rms=0.048, seed=42, tag="PR_CSPD_IF_L256",
    )
    results["training"]["PR_CSPD_IF_L256"] = m

    sp_pr_math, m = train_input_cspd_sp(
        student, tokenizer, tmpl, math_train_cache, train_pools["stream"], MATH_TASK_INST,
        num_vectors=256, prefix_robust=True, num_updates=45, batch_size=10, lr_max=0.005, max_rms=0.048, seed=42, tag="PR_CSPD_Math_L256",
    )
    results["training"]["PR_CSPD_Math_L256"] = m

    kv_if, m = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, if_train_cache, inst_text=IF_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.022, seed=42, tag="KVPD_IF_L16",
    )
    results["training"]["KVPD_IF_L16"] = m

    kv_math, m = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, math_train_cache, inst_text=MATH_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.020, seed=42, tag="KVPD_Math_L16",
    )
    results["training"]["KVPD_Math_L16"] = m

    sp_crsp_if_256, m = train_postrl_cscr_sp(
        student, tokenizer, tmpl, if_train_cache, train_pools, IF_TASK_INST,
        num_vectors=256, num_updates=45, batch_size=10, lr_max=0.0055, lambda_inv=0.35, max_rms=0.038,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Direct_CRSP_IF_L256",
    )
    results["training"]["Direct_CRSP_IF_L256"] = m

    sp_crsp_math_64, m = train_postrl_cscr_sp(
        student, tokenizer, tmpl, math_train_cache, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="Direct_CRSP_Math_L64",
    )
    results["training"]["Direct_CRSP_Math_L64"] = m

    kv_if_self = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_if, if_train_prompts, max_resp_gen=768, max_resp_trunc=512, topk=64, seed=606, tag="KV_IF_Self",
    )
    sp_alg1_if_64, m = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_if_self, train_pools, IF_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Alg1_KVInv_IF_L64",
    )
    results["training"]["Alg1_KVInv_IF_L64"] = m

    kv_math_self = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_math, math_train_prompts, max_resp_gen=768, max_resp_trunc=512, topk=64, seed=607, tag="KV_Math_Self",
    )
    sp_alg1_math_64, m = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_math_self, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="Alg1_KVInv_Math_L64",
    )
    results["training"]["Alg1_KVInv_Math_L64"] = m
    save_results(results)

    # Quick clean verification of KV-PD single & dual concat across 0/1024/4096/8192
    print("\n--- PART C_KV_VERIFY: Smooth-LR KV-PD Single & Dual Concat Across 0..8192 ---", flush=True)
    base_if_kls, base_m_kls = {}, {}
    for L in LENGTHS_COMP:
        rb_i, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="base")
        rb_m, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode="base")
        base_if_kls[L], base_m_kls[L] = rb_i["kl"], rb_m["kl"]
    for cname, kvl in [
        ("KVPD_IF_Solo", [kv_if]),
        ("KVPD_Math_Solo", [kv_math]),
        ("KVPD_Dual_Math_IF", [kv_math, kv_if]),
        ("KVPD_Dual_IF_Math", [kv_if, kv_math]),
    ]:
        results["partC_kvpd_clean_verify"]["if"][cname] = {}
        results["partC_kvpd_clean_verify"]["math"][cname] = {}
        for L in LENGTHS_COMP:
            ri, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="kv", kv_list=kvl)
            rm, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode="kv", kv_list=kvl)
            results["partC_kvpd_clean_verify"]["if"][cname][str(L)] = {**ri, "g_vs_ctx": round(100.0 * (1.0 - ri["kl"] / base_if_kls[L]), 2)}
            results["partC_kvpd_clean_verify"]["math"][cname][str(L)] = {**rm, "g_vs_ctx": round(100.0 * (1.0 - rm["kl"] / base_m_kls[L]), 2)}
        si = " | ".join(f"{L}:{results['partC_kvpd_clean_verify']['if'][cname][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_COMP)
        sm = " | ".join(f"{L}:{results['partC_kvpd_clean_verify']['math'][cname][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_COMP)
        print(f"  [KV-Verify {cname:18s}] IF: {si} || Math: {sm}", flush=True)
    save_results(results)

    # ------------------------------------------------------------------
    # PART D: Complete End-to-End Generative Long-Context Benchmarks
    # ------------------------------------------------------------------
    print("\n--- PART D1: IFEval48 (N=48, 1536 tok) at L_ctx in {0, 2048, 4096} ---", flush=True)
    ifeval_cfgs = [
        ("Base_Frozen",               "sp", {}),
        ("PR_CSPD_L256_at_Start",        "sp", {"sp_sys_start": [sp_pr_if]}),
        ("KVPD_L16_at_Start",            "kv", {"kv_list": [kv_if]}),
        ("Direct_CRSP_L256_at_PreQuery",    "sp", {"sp_pre_query": [sp_crsp_if_256]}),
        ("Direct_CRSP_L256_at_PostQuery",    "sp", {"sp_post_query": [sp_crsp_if_256]}),
        ("Alg1_KVInv_L64_at_PostQuery",      "sp", {"sp_post_query": [sp_alg1_if_64]}),
        ("PR_CSPD_Dual_Start",           "sp", {"sp_sys_start": [sp_pr_math, sp_pr_if]}),
        ("KVPD_Dual_Start",              "kv", {"kv_list": [kv_math, kv_if]}),
        ("Alg1_CRSP_Split_Math_PreQuery_IF_PostQuery", "sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]}),
    ]
    for L in [0, 2048, 4096]:
        results["partD_e2e_ifeval48"][str(L)] = {}
        for name, mode, kw in ifeval_cfgs:
            results["partD_e2e_ifeval48"][str(L)][name] = eval_ifeval_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], ifeval48, mode=mode, tag=name, **kw
            )
        save_results(results)

    print("\n--- PART D2: Official IFBench32 (N=32, 2048 tok) at L_ctx in {0, 4096} ---", flush=True)
    ifbench_cfgs = [
        ("Base_Frozen",               "sp", {}),
        ("PR_CSPD_L256_at_Start",        "sp", {"sp_sys_start": [sp_pr_if]}),
        ("KVPD_L16_at_Start",            "kv", {"kv_list": [kv_if]}),
        ("Direct_CRSP_L256_at_PreQuery",    "sp", {"sp_pre_query": [sp_crsp_if_256]}),
        ("Alg1_KVInv_L64_at_PostQuery",      "sp", {"sp_post_query": [sp_alg1_if_64]}),
        ("PR_CSPD_Dual_Start",           "sp", {"sp_sys_start": [sp_pr_math, sp_pr_if]}),
        ("KVPD_Dual_Start",              "kv", {"kv_list": [kv_math, kv_if]}),
        ("Alg1_CRSP_Split_Math_PreQuery_IF_PostQuery", "sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]}),
    ]
    for L in [0, 4096]:
        results["partD_e2e_ifbench32"][str(L)] = {}
        for name, mode, kw in ifbench_cfgs:
            results["partD_e2e_ifbench32"][str(L)][name] = eval_ifbench_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode=mode, tag=name, **kw
            )
        save_results(results)

    print("\n--- PART D3: MATH32 (N=32, 1536 tok) at L_ctx in {0, 4096} ---", flush=True)
    math_cfgs = [
        ("Base_Frozen",               "sp", {}),
        ("PR_CSPD_Math_Start",           "sp", {"sp_sys_start": [sp_pr_math]}),
        ("KVPD_Math_Start",              "kv", {"kv_list": [kv_math]}),
        ("Direct_CRSP_Math_PreQuery",       "sp", {"sp_pre_query": [sp_crsp_math_64]}),
        ("Alg1_Math_PreQuery",              "sp", {"sp_pre_query": [sp_alg1_math_64]}),
        ("PR_CSPD_Dual_Start",           "sp", {"sp_sys_start": [sp_pr_math, sp_pr_if]}),
        ("KVPD_Dual_Start",              "kv", {"kv_list": [kv_math, kv_if]}),
        ("Alg1_CRSP_Split_Math_PreQuery_IF_PostQuery", "sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]}),
    ]
    for L in [0, 4096]:
        results["partD_e2e_math32"][str(L)] = {}
        for name, mode, kw in math_cfgs:
            results["partD_e2e_math32"][str(L)][name] = eval_math_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], math32, mode=mode, tag=name, **kw
            )
        save_results(results)

    results["meta"]["total_wall_sec"] = round(time.time() - t0, 1)
    save_results(results)
    print(f"\n=== ALL DONE! exp10b_results.json saved in {results['meta']['total_wall_sec']}s ===", flush=True)


if __name__ == "__main__":
    main()
