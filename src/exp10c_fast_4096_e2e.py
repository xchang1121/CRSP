"""
Suite 8C (exp10c): Fast Batched L=4096 End-to-End Long-Context Generation (MATH32, IFEval48, IFBench32)
=======================================================================================================
Complements exp10 (Part A 0..8192 sweep, Part B attention mass, Part C dual composition & interference)
and exp10b (Part E HeliosPay Doc36 KV-PD 0..4096, Part C smooth-LR KV-PD verify, Part D IFEval48 L=0/2048).

Optimized for high throughput (large VRAM-aware batch sizes + saving trained modules to disk):
  1) Smooth-LR KV-PD 6-point sweep {0, 512, 1024, 2048, 4096, 8192} for both IF and Math (~15s).
  2) End-to-end MATH32    (N=32, 1536 tok) at L_ctx in {0, 4096} across all 8 single & dual configs.
  3) End-to-end IFEval48  (N=48, 1536 tok) at L_ctx = 4096       across all 8 single & dual configs.
  4) End-to-end IFBench32 (N=32, 2048 tok) at L_ctx in {0, 4096} across all 8 single & dual configs.
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

import evaluation_lib as ifbench_eval
from benchmarks import (
    strip_thinking,
    verify_ifeval_response,
    verify_math_response,
)
from cspd_core import get_qwen_template_tokens
from exp7_context_robust_sp import (
    IF_TASK_INST,
    TRAIN_DIALOG_TURNS,
    TRAIN_PERSONAS,
    TRAIN_RAG_CHUNKS,
)
from exp9_multiseed_cr_grpo_and_composition import (
    MATH_TASK_INST,
    batch_generate_slots,
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
    LENGTHS_SWEEP,
    batch_generate_kv_longctx,
    build_long_context_bundles,
    build_null_anchor_pool,
    eval_config_on_val_cache,
    train_input_cspd_sp,
)

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"

OUT_DIR = "/content/cspd/exp10c_results"
os.makedirs(OUT_DIR, exist_ok=True)
RESULT_PATH = os.path.join(OUT_DIR, "exp10c_results.json")


def save_results(res: Dict[str, Any]):
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def _fast_bs(L: int, n_items: int, max_new: int = 1536) -> int:
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 40.0
    if vram >= 85.0:
        return min(n_items, 48 if L == 0 else 32)
    elif vram >= 75.0:
        return min(n_items, 48 if L == 0 else 24)
    elif vram >= 38.0:
        if L == 0:
            return min(n_items, 32)
        return min(n_items, 16 if max_new <= 1536 else 12)
    else:
        return min(n_items, 12 if L == 0 else 6)


def run_math_fast(model, tokenizer, tmpl, ctx_bundle, items, mode: str,
                  kv_list=None, sp_sys_start=None, sp_pre_query=None, sp_post_query=None, tag: str = "") -> Dict[str, Any]:
    t0 = time.time()
    L = ctx_bundle["target_len"]
    bs = _fast_bs(L, len(items), 1536)
    if mode == "kv":
        outs = batch_generate_kv_longctx(
            model, tokenizer, tmpl, [x["prompt"] for x in items], kv_list, ctx_bundle,
            batch_size=bs, max_new_tokens=1536, temperature=0.6, seed=42, enable_thinking=True,
        )
    else:
        outs = batch_generate_slots(
            model, tokenizer, tmpl, [x["prompt"] for x in items],
            sys_text_ids=ctx_bundle["sys_text_ids"],
            history_turns_ids=ctx_bundle["history_turns_ids"],
            rag_ids=ctx_bundle["rag_ids"],
            sp_sys_start=sp_sys_start, sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
            batch_size=bs, max_new_tokens=1536, temperature=0.6, seed=42,
        )
    ok, closed = 0, 0
    for it, resp in zip(items, outs):
        c = "</think>" in resp
        closed += int(c)
        if c and verify_math_response(resp, it["answer"]):
            ok += 1
    n = len(items)
    res = {
        "acc": round(100.0 * ok / n, 2), "n_ok": ok, "n_total": n,
        "closed_think": closed, "closed_acc": round(100.0 * ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [MATH32   L={L:4d} bs={bs:2d} | {tag:28s}] acc={res['acc']:5.2f}% ({ok:2d}/{n}) | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def run_ifeval_fast(model, tokenizer, tmpl, ctx_bundle, items, mode: str,
                    kv_list=None, sp_sys_start=None, sp_pre_query=None, sp_post_query=None, tag: str = "") -> Dict[str, Any]:
    t0 = time.time()
    L = ctx_bundle["target_len"]
    bs = _fast_bs(L, len(items), 1536)
    if mode == "kv":
        outs = batch_generate_kv_longctx(
            model, tokenizer, tmpl, [x["prompt"] for x in items], kv_list, ctx_bundle,
            batch_size=bs, max_new_tokens=1536, temperature=0.6, seed=42, enable_thinking=True,
        )
    else:
        outs = batch_generate_slots(
            model, tokenizer, tmpl, [x["prompt"] for x in items],
            sys_text_ids=ctx_bundle["sys_text_ids"],
            history_turns_ids=ctx_bundle["history_turns_ids"],
            rag_ids=ctx_bundle["rag_ids"],
            sp_sys_start=sp_sys_start, sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
            batch_size=bs, max_new_tokens=1536, temperature=0.6, seed=42,
        )
    s_ok, closed, inst_sum = 0, 0, 0.0
    for it, resp in zip(items, outs):
        c = "</think>" in resp
        closed += int(c)
        st, ir = verify_ifeval_response(resp, it["iids"], it["kwargs"])
        if not c:
            st = False
        s_ok += int(st)
        inst_sum += ir
    n = len(items)
    res = {
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * inst_sum / n, 2),
        "n_strict": s_ok, "n_total": n,
        "closed_think": closed, "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFEval48 L={L:4d} bs={bs:2d} | {tag:28s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def run_ifbench_fast(model, tokenizer, tmpl, ctx_bundle, inps, mode: str,
                     kv_list=None, sp_sys_start=None, sp_pre_query=None, sp_post_query=None, tag: str = "") -> Dict[str, Any]:
    t0 = time.time()
    L = ctx_bundle["target_len"]
    bs = _fast_bs(L, len(inps), 2048)
    if mode == "kv":
        outs = batch_generate_kv_longctx(
            model, tokenizer, tmpl, [x.prompt for x in inps], kv_list, ctx_bundle,
            batch_size=bs, max_new_tokens=2048, temperature=0.6, seed=42, enable_thinking=True,
        )
    else:
        outs = batch_generate_slots(
            model, tokenizer, tmpl, [x.prompt for x in inps],
            sys_text_ids=ctx_bundle["sys_text_ids"],
            history_turns_ids=ctx_bundle["history_turns_ids"],
            rag_ids=ctx_bundle["rag_ids"],
            sp_sys_start=sp_sys_start, sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
            batch_size=bs, max_new_tokens=2048, temperature=0.6, seed=42,
        )
    p2r, closed = {}, 0
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
        "n_strict": s_ok, "n_total": n,
        "closed_think": closed, "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFB32    L={L:4d} bs={bs:2d} | {tag:28s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def main():
    t0 = time.time()
    print("=" * 92, flush=True)
    print("Suite 8C (exp10c): Fast Batched L=4096 End-to-End Generation (MATH32, IFEval48, IFBench32)", flush=True)
    print("=" * 92, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    if_train96, ifeval_test96, ifbench64, math_train64, math_test48, _, _, _ = load_datasets_exp9()

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
        "kvpd_smooth_6pt": {"if": {}, "math": {}},
        "e2e_math32": {},
        "e2e_ifeval48_4096": {},
        "e2e_ifbench32": {},
    }

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

    print("\n--- PHASE 2: Training Procedural Experts ---", flush=True)
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

    sp_pr_if, _ = train_input_cspd_sp(
        student, tokenizer, tmpl, if_train_cache, train_pools["stream"], IF_TASK_INST,
        num_vectors=256, prefix_robust=True, num_updates=45, batch_size=10, lr_max=0.005, max_rms=0.048, seed=42, tag="PR_CSPD_IF_L256",
    )
    sp_pr_math, _ = train_input_cspd_sp(
        student, tokenizer, tmpl, math_train_cache, train_pools["stream"], MATH_TASK_INST,
        num_vectors=256, prefix_robust=True, num_updates=45, batch_size=10, lr_max=0.005, max_rms=0.048, seed=42, tag="PR_CSPD_Math_L256",
    )
    kv_if, _ = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, if_train_cache, inst_text=IF_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.022, seed=42, tag="KVPD_IF_L16",
    )
    kv_math, _ = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, math_train_cache, inst_text=MATH_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.020, seed=42, tag="KVPD_Math_L16",
    )
    sp_crsp_if_256, _ = train_postrl_cscr_sp(
        student, tokenizer, tmpl, if_train_cache, train_pools, IF_TASK_INST,
        num_vectors=256, num_updates=45, batch_size=10, lr_max=0.0055, lambda_inv=0.35, max_rms=0.038,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Direct_CRSP_IF_L256",
    )
    sp_crsp_math_64, _ = train_postrl_cscr_sp(
        student, tokenizer, tmpl, math_train_cache, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="Direct_CRSP_Math_L64",
    )
    kv_if_self = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_if, if_train_prompts, max_resp_gen=768, max_resp_trunc=512, topk=64, seed=606, tag="KV_IF_Self",
    )
    sp_alg1_if_64, _ = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_if_self, train_pools, IF_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Alg1_KVInv_IF_L64",
    )
    kv_math_self = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_math, math_train_prompts, max_resp_gen=768, max_resp_trunc=512, topk=64, seed=607, tag="KV_Math_Self",
    )
    sp_alg1_math_64, _ = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_math_self, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="Alg1_KVInv_Math_L64",
    )

    # Smooth-LR KV-PD full 6-point single-expert sweep {0,512,1024,2048,4096,8192}
    print("\n--- PHASE 2B: Smooth-LR KV-PD 6-Point Sweep (0..8192) ---", flush=True)
    for dom, vcache, kvl in [("if", if_val_cache, [kv_if]), ("math", math_val_cache, [kv_math])]:
        ref0 = None
        for L in LENGTHS_SWEEP:
            rb, _ = eval_config_on_val_cache(student, tmpl, vcache, ctx_bundles[L], mode="base")
            rk, ref_out = eval_config_on_val_cache(
                student, tmpl, vcache, ctx_bundles[L], mode="kv", kv_list=kvl,
                ref_p0_logits_list=None if L == 0 else ref0, save_ref_top32=(L == 0),
            )
            if L == 0:
                ref0 = ref_out
            results["kvpd_smooth_6pt"][dom][str(L)] = {
                **rk, "base_kl": rb["kl"], "g_vs_ctx": round(100.0 * (1.0 - rk["kl"] / rb["kl"]), 2)
            }
        print(f"  [KVPD_Smooth {dom:4s}] " + " | ".join(
            f"{L}:{results['kvpd_smooth_6pt'][dom][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_SWEEP
        ), flush=True)
    save_results(results)

    # 1. MATH32 at L=0 & L=4096 (most dramatic PR_CSPD collapse domain!)
    print("\n--- PHASE 3: MATH32 (N=32, 1536 tok) at L_ctx in {0, 4096} ---", flush=True)
    math_cfgs = [
        ("Base_Frozen",               "sp", {}),
        ("PR_CSPD_Math_P0",           "sp", {"sp_sys_start": [sp_pr_math]}),
        ("KVPD_Math_P0",              "kv", {"kv_list": [kv_math]}),
        ("Direct_CRSP_Math_P2",       "sp", {"sp_pre_query": [sp_crsp_math_64]}),
        ("Alg1_Math_P2",              "sp", {"sp_pre_query": [sp_alg1_math_64]}),
        ("PR_CSPD_Dual_P0",           "sp", {"sp_sys_start": [sp_pr_math, sp_pr_if]}),
        ("KVPD_Dual_P0",              "kv", {"kv_list": [kv_math, kv_if]}),
        ("Alg1_CRSP_Split_M@P2_I@P4", "sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]}),
    ]
    for L in [4096, 0]:
        results["e2e_math32"][str(L)] = {}
        for name, mode, kw in math_cfgs:
            results["e2e_math32"][str(L)][name] = run_math_fast(
                student, tokenizer, tmpl, ctx_bundles[L], math32, mode=mode, tag=name, **kw
            )
            save_results(results)

    # 2. IFEval48 at L=4096 (completes exp10b's L=0 and L=2048!)
    print("\n--- PHASE 4: IFEval48 (N=48, 1536 tok) at L_ctx = 4096 ---", flush=True)
    ifeval_cfgs = [
        ("Base_Frozen",               "sp", {}),
        ("PR_CSPD_L256_at_P0",        "sp", {"sp_sys_start": [sp_pr_if]}),
        ("KVPD_L16_at_P0",            "kv", {"kv_list": [kv_if]}),
        ("Direct_CRSP_L256_at_P2",    "sp", {"sp_pre_query": [sp_crsp_if_256]}),
        ("Alg1_KVInv_L64_at_P4",      "sp", {"sp_post_query": [sp_alg1_if_64]}),
        ("PR_CSPD_Dual_P0",           "sp", {"sp_sys_start": [sp_pr_math, sp_pr_if]}),
        ("KVPD_Dual_P0",              "kv", {"kv_list": [kv_math, kv_if]}),
        ("Alg1_CRSP_Split_M@P2_I@P4", "sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]}),
    ]
    results["e2e_ifeval48_4096"] = {}
    for name, mode, kw in ifeval_cfgs:
        results["e2e_ifeval48_4096"][name] = run_ifeval_fast(
            student, tokenizer, tmpl, ctx_bundles[4096], ifeval48, mode=mode, tag=name, **kw
        )
        save_results(results)

    # 3. Official IFBench32 at L=4096 & L=0
    print("\n--- PHASE 5: Official IFBench32 (N=32, 2048 tok) at L_ctx in {4096, 0} ---", flush=True)
    for L in [4096, 0]:
        results["e2e_ifbench32"][str(L)] = {}
        for name, mode, kw in ifeval_cfgs:
            results["e2e_ifbench32"][str(L)][name] = run_ifbench_fast(
                student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode=mode, tag=name, **kw
            )
            save_results(results)

    results["meta"]["total_wall_sec"] = round(time.time() - t0, 1)
    save_results(results)
    print(f"\n=== ALL DONE! exp10c_results.json saved in {results['meta']['total_wall_sec']}s ===", flush=True)


if __name__ == "__main__":
    main()
