"""
Suite 10 (exp12): Reference Step-200 Soft-Prompt Audit, Complete-EOS Trajectory Upgrade,
                  36-Layer Causal Pathway Ablation & Virtual-Token/Fast-Head Verification
=========================================================================================
Evaluates 6 scientific & engineering extensions:
  Part A — Reference step-200 CSPD checkpoints (M, C, S, I; loaded via REF_PROMPT_MANIFEST):
           (A1) Six-slot (P0-P5) positional drift & collapse audit on held-out IF & Math.
           (A2) Long-context (0 -> 8192) single-expert attenuation & 9-layer attention mass.
           (A3) Long-context (0 -> 8192) dual ([M;I]/[I;M]) & 4-expert (MCSI/ISCM) composition & interference.
  Part B — Complete EOS-Terminated & Closed-Think Trajectory Upgrade (vs. [:512] truncated):
           Strict verify_complete_eos_trace gating (closed </think> + <|im_end|> EOS) with
           response-only lm_head projection for KV-PD-EOS, Direct-CRSP-EOS, and Alg1-CRSP-EOS.
  Part C — 36-Layer Cumulative Causal Attention Pathway Ablation (open [0, l) vs. open [l, 36)):
           Direct head-to-head comparison of Reference CSPD I @ P0, KV-PD-EOS @ P0, and
           Alg. 1 CRSP-EOS @ P4 at both L_ctx = 0 and L_ctx = 4096.
  Part D — Engineering Benchmarks:
           (D1) Response-only lm_head projection peak VRAM & wall-time vs. full logits (512..4096).
           (D2) Byte-exact virtual-token derived checkpoint equivalence (P0 & P4) + tied-embedding
                high-norm virtual row output-logit hazard verification (-100 vs. -1e9).
  Part E — End-to-End L=0 & L=4096 Generation Verification (MATH32, IFBench32, IFEval48):
           Reference M / I / [M;I] / MCSI vs. Complete-EOS KV-PD & Alg. 1 Split [M@P2+I@P4].
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content/cspd")
sys.path.insert(0, "/content/IFBench")

from benchmarks import batch_generate_with_soft_prompts, verify_math_response
from cspd_core import (
    QWEN_END_TOKENS,
    compute_response_logits_fast,
    final_answer_strict,
    get_qwen_template_tokens,
    pad_and_stack_train_batch,
    verify_complete_eos_trace,
    verify_qwen_system_scaffold,
)
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
from exp9_multiseed_cr_grpo_and_composition import (
    MATH_TASK_INST,
    build_foreign_distractor_pool,
    load_datasets_exp9,
    sample_slot_view_cscr,
)
from exp9b_postrl_crsp_and_composition import (
    cache_postrl_teacher_top64,
    train_kv_distilled_l16_custom,
)
from exp10_long_context_attenuation_and_composition import (
    LENGTHS_COMP,
    LENGTHS_SWEEP,
    build_long_context_bundles,
    build_longctx_ids_for_kv,
    build_null_anchor_pool,
    eval_config_on_val_cache,
    eval_pair_interference_kl,
    measure_response_attention_mass,
)
from exp10c_fast_4096_e2e import (
    run_ifbench_fast,
    run_ifeval_fast,
    run_math_fast,
)
from kv_prefix_core import forward_with_kv_prefix, make_mutable_kv_cache
from virtual_tokens_export import (
    build_derived_virtual_checkpoint,
    build_hard_output_mask_ids,
    render_multislot_virtual_ids,
)

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"

REF_MANIFEST_ENV = "REF_PROMPT_MANIFEST"
DEFAULT_MANIFEST_PATH = "/content/cspd/ref_prompts_manifest.json"

OUT_DIR = "/content/cspd/exp12_results"
os.makedirs(OUT_DIR, exist_ok=True)
RESULT_PATH = os.path.join(OUT_DIR, "exp12_results.json")


def save_results(res: Dict[str, Any]):
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. Load & Audit Reference Step-200 Soft Prompts (via runtime manifest JSON)
# ---------------------------------------------------------------------------

def fetch_official_prompts(device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    manifest_path = os.environ.get(REF_MANIFEST_ENV, DEFAULT_MANIFEST_PATH)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Reference prompt manifest not found at {manifest_path}. "
            f"Provide a JSON mapping expert keys (M, C, S, I) to {{'path' or 'url', 'sha256'}} "
            f"via {REF_MANIFEST_ENV}."
        )
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    prompts: Dict[str, torch.Tensor] = {}
    stats: Dict[str, Any] = {}
    for k, info in manifest.items():
        cache_p = os.path.join(OUT_DIR, f"ref_{k}.pt")
        if info.get("path") and os.path.exists(info["path"]):
            raw = open(info["path"], "rb").read()
        elif os.path.exists(cache_p):
            raw = open(cache_p, "rb").read()
        else:
            raw = urlopen(info["url"], timeout=120).read()
            open(cache_p, "wb").write(raw)
        got = hashlib.sha256(raw).hexdigest()
        if info.get("sha256") and got != info["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for reference {k}.pt: {got} != {info['sha256']}")
        st = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        p = st["prompt"].float() if isinstance(st, dict) and "prompt" in st else st.float()
        row_rms = p.pow(2).mean(dim=-1).sqrt()
        sv = torch.linalg.svdvals(p)
        prob = (sv / sv.sum()).clamp_min(1e-12)
        erank = float(torch.exp(-(prob * prob.log()).sum()).item())
        stats[k] = {
            "shape": list(p.shape),
            "sha256_verified": bool(info.get("sha256")),
            "global_rms": round(float(p.pow(2).mean().sqrt().item()), 5),
            "row_rms_min": round(float(row_rms.min().item()), 4),
            "row_rms_median": round(float(row_rms.median().item()), 4),
            "row_rms_max": round(float(row_rms.max().item()), 4),
            "effective_rank": round(erank, 2),
        }
        prompts[k] = p.to(device)
        print(
            f"  [Ref {k}.pt] SHA256 OK | shape={tuple(p.shape)} | rms={stats[k]['global_rms']:.4f} "
            f"(row med {stats[k]['row_rms_median']:.4f}, max {stats[k]['row_rms_max']:.4f}) | erank={erank:.1f}",
            flush=True,
        )
    return prompts, stats


# ---------------------------------------------------------------------------
# 2. Part A1 Helper: Six-Slot (P0-P5) Positional Audit
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_official_6slots(student, tokenizer, tmpl, val_cache, sp: torch.Tensor, tag: str) -> Dict[str, Any]:
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    student.eval()

    eval_persona_ids = tokenizer.encode(EVAL_PERSONA, add_special_tokens=False)
    eval_hist_ids = [
        (tokenizer.encode(u, add_special_tokens=False), tokenizer.encode(a, add_special_tokens=False))
        for u, a in EVAL_HISTORY_TURNS
    ]
    eval_rag_ids = tokenizer.encode(EVAL_RAG_BLOCK, add_special_tokens=False)

    slot_specs = [
        ("P0_Clean",       dict(sys_text_ids=None,             history_turns_ids=None,          rag_ids=None),         dict(sp_sys_start=[sp])),
        ("P0_FullCtx",     dict(sys_text_ids=eval_persona_ids, history_turns_ids=eval_hist_ids, rag_ids=eval_rag_ids), dict(sp_sys_start=[sp])),
        ("P1_SysEnd",      dict(sys_text_ids=eval_persona_ids, history_turns_ids=None,          rag_ids=None),         dict(sp_sys_end=[sp])),
        ("P2_MultiTurn",   dict(sys_text_ids=eval_persona_ids, history_turns_ids=eval_hist_ids, rag_ids=None),         dict(sp_pre_query=[sp])),
        ("P3_PostRAG",     dict(sys_text_ids=eval_persona_ids, history_turns_ids=None,          rag_ids=eval_rag_ids), dict(sp_pre_query=[sp])),
        ("P4_PostQuery",   dict(sys_text_ids=eval_persona_ids, history_turns_ids=eval_hist_ids, rag_ids=None),         dict(sp_post_query=[sp])),
        ("P5_AllCombined", dict(sys_text_ids=eval_persona_ids, history_turns_ids=eval_hist_ids, rag_ids=eval_rag_ids), dict(sp_post_query=[sp])),
    ]

    ref_p0: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    base_p0_kl = 0.0
    out_res: Dict[str, Any] = {}

    for sname, ctx_kw, sp_kw in slot_specs:
        base_kls, sp_kls, drifts = [], [], []
        for idx, item in enumerate(val_cache):
            qi, ri = item["q_ids"], item["r_ids"]
            ti, tp, ent = item["ti"].to(device), item["tp"].float().to(device), item["ent"].float().to(device)

            # Base at this slot context
            eb, lb, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **ctx_kw)
            zb = compute_response_logits_fast(student, eb, lb)[0]
            nb = min(zb.size(0), ti.size(0))
            lpb = F.log_softmax(zb[:nb], dim=-1)
            base_kls.append((-(tp[:nb] * lpb.gather(-1, ti[:nb])).sum(dim=-1) - ent[:nb]).clamp_min(0.0).mean().item())
            del eb, lb, zb, lpb

            # With Official SP at this slot
            es, ls, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **ctx_kw, **sp_kw)
            zs = compute_response_logits_fast(student, es, ls)[0]
            ns = min(zs.size(0), ti.size(0))
            lps = F.log_softmax(zs[:ns], dim=-1)
            sp_kls.append((-(tp[:ns] * lps.gather(-1, ti[:ns])).sum(dim=-1) - ent[:ns]).clamp_min(0.0).mean().item())

            if sname == "P0_Clean":
                probs = lps.exp()
                rp, ri_top = torch.topk(probs, 32, dim=-1)
                rp = rp / rp.sum(dim=-1, keepdim=True)
                rent = -(rp * rp.clamp_min(1e-9).log()).sum(dim=-1)
                ref_p0.append((ri_top.cpu(), rp.to(torch.float16).cpu(), rent.cpu()))
            else:
                r_ti, r_tp, r_ent = ref_p0[idx]
                r_ti, r_tp, r_ent = r_ti.to(device), r_tp.float().to(device), r_ent.float().to(device)
                nr = min(lps.size(0), r_ti.size(0))
                d_ce = -(r_tp[:nr] * lps[:nr].gather(-1, r_ti[:nr])).sum(dim=-1)
                drifts.append((d_ce - r_ent[:nr]).clamp_min(0.0).mean().item())
            del es, ls, zs, lps

        bkl = sum(base_kls) / max(1, len(base_kls))
        skl = sum(sp_kls) / max(1, len(sp_kls))
        if sname == "P0_Clean":
            base_p0_kl = bkl
        g_vs_slot = round(100.0 * (1.0 - skl / bkl), 2)
        g_vs_p0 = round(100.0 * (1.0 - skl / base_p0_kl), 2)
        row = {"base_kl": round(bkl, 5), "sp_kl": round(skl, 5), "g_vs_slot": g_vs_slot, "g_vs_p0": g_vs_p0}
        if drifts:
            row["drift_kl_vs_p0"] = round(sum(drifts) / len(drifts), 5)
        out_res[sname] = row
        d_str = f" | drift={row['drift_kl_vs_p0']:.4f}" if "drift_kl_vs_p0" in row else ""
        print(f"  [6-Slot {tag:12s} | {sname:14s}] G_slot={g_vs_slot:+6.2f}% | G_p0={g_vs_p0:+6.2f}%{d_str}", flush=True)
    return out_res


# ---------------------------------------------------------------------------
# 3. Part B Helpers: Complete EOS-Terminated Teacher Caching & Fast CRSP Trainer
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_complete_eos_teacher_top64(
    model_id: str,
    tokenizer,
    tmpl,
    items: List[Dict[str, Any]],
    domain: str,  # "if" or "math"
    max_resp_gen: int = 1536,
    max_keep: int = 64,
    topk: int = 64,
    seed: int = 777,
    tag: str = "EOS_Teacher",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate complete closed-think EOS-terminated teacher trajectories:
      - Strictly requires single well-formed </think> + non-empty final answer + trailing <|im_end|> (151645).
      - Uses response-only lm_head projection (compute_response_logits_fast) to cache exact Top-64 targets
        across the ENTIRE complete trajectory including </think> and <|im_end|>!
    """
    t0 = time.time()
    print(f"  [EOS-Cache {tag:14s}] Loading {model_id} for {len(items)} prompts (max_gen={max_resp_gen})...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    teacher.eval()
    device = next(teacher.parameters()).device
    dtype = next(teacher.parameters()).dtype
    t_emb = teacher.get_input_embeddings()

    prompts = [x["prompt"] for x in items]
    torch.manual_seed(seed)
    resps = batch_generate_with_soft_prompts(
        teacher, tokenizer, tmpl, prompts, [], batch_size=32, max_new_tokens=max_resp_gen,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )

    eos_id = int(QWEN_END_TOKENS[0])  # 151645 (<|im_end|>)
    verified_correct, clean_closed = [], []
    for item, r in zip(items, resps):
        r_ids = tokenizer.encode(r, add_special_tokens=False) + [eos_id]
        if not verify_complete_eos_trace(r_ids, r, require_think_close=True):
            continue
        if len(r_ids) < 24 or len(r_ids) > max_resp_gen + 8:
            continue
        is_ok = verify_math_response(r, item["answer"]) if domain == "math" else True
        if is_ok:
            verified_correct.append((item["prompt"], r_ids))
        else:
            clean_closed.append((item["prompt"], r_ids))

    chosen = (verified_correct + clean_closed)[:max_keep]
    cache = []
    for p, ri in chosen:
        qi = tokenizer.encode(p, add_special_tokens=False)
        e, l, _ = build_multislot_sp_inputs(t_emb, tmpl, qi, ri, device=device, model_dtype=dtype)
        z = compute_response_logits_fast(teacher, e, l)[0]
        probs = F.softmax(z, dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
        cache.append({"q_ids": qi, "r_ids": ri, "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "ent": ent.cpu()})
        del e, l, z

    mean_len = sum(len(x["r_ids"]) for x in cache) / max(1, len(cache))
    stats = {
        "input_prompts": len(items),
        "verified_closed_eos": len(verified_correct),
        "total_closed_eos": len(verified_correct) + len(clean_closed),
        "kept": len(cache),
        "mean_resp_tokens": round(mean_len, 1),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"  [EOS-Cache {tag:14s}] kept={stats['kept']} (verified={stats['verified_closed_eos']}, "
        f"closed_eos={stats['total_closed_eos']}/{len(items)}, mean_tok={stats['mean_resp_tokens']}) in {stats['elapsed_sec']}s",
        flush=True,
    )
    return cache, stats


@torch.no_grad()
def cache_kv_complete_eos_self_targets(
    student, tokenizer, tmpl, kv_prefix: torch.Tensor, prompts: List[str],
    max_resp_gen: int = 1536, max_keep: int = 56, topk: int = 64, seed: int = 606, tag: str = "KV_Self_EOS",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Algorithm 1 self-generation upgraded with strict complete closed-think + EOS gating!"""
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = int(QWEN_END_TOKENS[0])
    student.eval()
    student.config.use_cache = True
    torch.manual_seed(seed)

    all_outs: List[str] = []
    for start in range(0, len(prompts), 32):
        bp = prompts[start : start + 32]
        seqs = [
            list(tmpl.sys_open) + list(tmpl.sys_close_user_open) + tokenizer.encode(q, add_special_tokens=False) + list(tmpl.user_close_asst_open)
            for q in bp
        ]
        B = len(seqs)
        ml = max(len(s) for s in seqs)
        inp = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
        sm = torch.zeros((B, ml), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            inp[i, ml - len(s) :] = torch.tensor(s, dtype=torch.long, device=device)
            sm[i, ml - len(s) :] = 1
        cache = make_mutable_kv_cache([kv_prefix], B, device, dtype)
        L_kv = cache.get_seq_length()
        am = torch.cat([torch.ones((B, L_kv), dtype=torch.long, device=device), sm], dim=1)
        out_ids = student.generate(
            input_ids=inp, attention_mask=am, past_key_values=cache,
            max_new_tokens=max_resp_gen, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
        )
        for row in out_ids:
            all_outs.append(tokenizer.decode(row[ml:], skip_special_tokens=True))
        del inp, sm, am, out_ids
        torch.cuda.empty_cache()

    kept_pairs = []
    for p, r in zip(prompts, all_outs):
        ri = tokenizer.encode(r, add_special_tokens=False) + [eos_id]
        if verify_complete_eos_trace(ri, r, require_think_close=True) and 24 <= len(ri) <= max_resp_gen + 8:
            kept_pairs.append((p, ri))
    kept_pairs = kept_pairs[:max_keep]

    cache_list = []
    for p, ri in kept_pairs:
        qi = tokenizer.encode(p, add_special_tokens=False)
        ids = list(tmpl.sys_open) + list(tmpl.sys_close_user_open) + qi + list(tmpl.user_close_asst_open) + ri
        lbl = [-100] * (len(ids) - len(ri)) + ri
        bi = torch.tensor([ids], dtype=torch.long, device=device)
        bm = torch.ones_like(bi)
        out = forward_with_kv_prefix(student, bi, bm, [kv_prefix])
        m = torch.tensor(lbl[1:], dtype=torch.long, device=device) != -100
        z = out.logits[0, :-1, :][m].float()
        probs = F.softmax(z, dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
        cache_list.append({"q_ids": qi, "r_ids": ri, "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "ent": ent.cpu()})
        del bi, bm, out, z

    stats = {
        "input_prompts": len(prompts),
        "closed_eos_kept": len(cache_list),
        "mean_resp_tokens": round(sum(len(x["r_ids"]) for x in cache_list) / max(1, len(cache_list)), 1),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"  [Alg1-EOS {tag:15s}] kept={stats['closed_eos_kept']}/{len(prompts)} (mean_tok={stats['mean_resp_tokens']}) in {stats['elapsed_sec']}s", flush=True)
    return cache_list, stats


def train_cscr_sp_fast_eos(
    student, tokenizer, tmpl, train_cache: List[Dict[str, Any]], train_pools: Dict[str, Any],
    task_inst: str, num_vectors: int = 64, num_updates: int = 45, batch_size: int = 10,
    lr_max: float = 0.007, lambda_inv: float = 0.35, max_rms: float = 0.047,
    foreign_pool=None, co_prob: float = 0.60, null_pool=None, lambda_null: float = 0.20,
    seed: int = 42, tag: str = "CRSP_EOS",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Upgraded CRSP trainer using response-only lm_head projection (compute_response_logits_fast)
    so complete 1536-token EOS-terminated trajectories train fast with minimal logit VRAM!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    sp = init_anchored_sp(s_emb, tokenizer, task_inst, num_vectors=num_vectors, anchored=True)
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

        batch = [train_cache[rng.randrange(len(train_cache))] for _ in range(batch_size)]
        tot_dist, tot_inv, tot_null = 0.0, 0.0, 0.0
        for item in batch:
            qi, ri = item["q_ids"], item["r_ids"]
            ti, tp = item["ti"].to(device), item["tp"].float().to(device)
            e0, l0, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, sp_sys_start=[sp], device=device, model_dtype=dtype)
            slot_kw, _ = sample_slot_view_cscr(rng, train_pools, sp, foreign_pool=foreign_pool, co_prob=co_prob)
            ec, lc, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **slot_kw)
            be, bm, bl = pad_and_stack_train_batch([e0, ec], [l0, lc], pad_vec)
            # Response-only lm_head projection (Item 4 upgrade!)
            z0, zc = compute_response_logits_fast(student, be, bl, attention_mask=bm)
            l_dist = hybrid_top64_distill_loss(zc, ti, tp)
            l_inv = cross_context_invariance_kl(z0, zc, topk=32)
            loss = (l_dist + lambda_inv * l_inv) / batch_size
            loss.backward()
            tot_dist += float(l_dist.item()) / batch_size
            tot_inv += float(l_inv.item()) / batch_size
            del be, bm, bl, z0, zc

        if null_pool and lambda_null > 0.0:
            n_null = 2
            for _ in range(n_null):
                ni = null_pool[rng.randrange(len(null_pool))]
                slot_kw, _ = sample_slot_view_cscr(rng, train_pools, sp, foreign_pool=foreign_pool, co_prob=co_prob)
                en, ln, _ = build_multislot_sp_inputs(s_emb, tmpl, ni["q_ids"], ni["r_ids"], device=device, model_dtype=dtype, **slot_kw)
                zn = compute_response_logits_fast(student, en, ln)[0]
                ati, atp, aent = ni["anc_ti"].to(device), ni["anc_tp"].float().to(device), ni["anc_ent"].to(device)
                na = min(zn.size(0), ati.size(0))
                ce_n = -(atp[:na] * F.log_softmax(zn[:na], dim=-1).gather(-1, ati[:na])).sum(dim=-1)
                kl_n = (ce_n - aent[:na]).clamp_min(0.0).mean()
                (lambda_null * kl_n / n_null).backward()
                tot_null += float(kl_n.item()) / n_null
                del en, ln, zn

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)
        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True

        if step in {1, 20, num_updates}:
            log_hist[str(step)] = {"dist": round(tot_dist, 4), "inv": round(tot_inv, 4), "null": round(tot_null, 4)}
            print(f"  [{tag:24s}] step {step:2d}/{num_updates} | dist={tot_dist:.4f} | inv={tot_inv:.4f} | null={tot_null:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "num_vectors": num_vectors, "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


# ---------------------------------------------------------------------------
# 4. Part C: 36-Layer Cumulative Causal Attention Pathway Ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_layerwise_causal_pathway(
    student,
    tmpl,
    val_items: List[Dict[str, Any]],
    ctx_bundle: Dict[str, Any],
    mode: str,  # "sp" or "kv"
    kv_list: Optional[List[torch.Tensor]] = None,
    sp_sys_start: Optional[List[torch.Tensor]] = None,
    sp_post_query: Optional[List[torch.Tensor]] = None,
    cutpoints: List[int] = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36),
    tag: str = "",
) -> Dict[str, Any]:
    """
    36-layer cumulative direct-path (R -> module) causal ablation:
      - open_first_l: response tokens may attend to module ONLY in layers [0, l)
      - open_last_l:  response tokens may attend to module ONLY in layers [l, 36)
    Uses exact causal additive mask inside hooked F.scaled_dot_product_attention (zero OOM).
    """
    t0 = time.time()
    n_layers = int(student.config.num_hidden_layers)
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    student.eval()

    # Pre-compute base KL (no module) and full unhooked KL (all 36 layers open)
    rb, _ = eval_config_on_val_cache(student, tmpl, val_items, ctx_bundle, mode="base")
    rf, _ = eval_config_on_val_cache(
        student, tmpl, val_items, ctx_bundle, mode=mode,
        kv_list=kv_list, sp_sys_start=sp_sys_start, sp_post_query=sp_post_query,
    )
    base_kl, full_kl = rb["kl"], rf["kl"]

    orig_sdpa = F.scaled_dot_product_attention

    def run_with_blocked_layers(blocked_layers: Set[int]) -> float:
        kls = []
        for item in val_items:
            qi, ri = item["q_ids"], item["r_ids"]
            ti, tp, ent = item["ti"].to(device), item["tp"].float().to(device), item["ent"].float().to(device)

            if mode == "kv":
                ids, lbl, spans = build_longctx_ids_for_kv(
                    tmpl, qi, ri,
                    sys_text_ids=ctx_bundle["sys_text_ids"],
                    history_turns_ids=ctx_bundle["history_turns_ids"],
                    rag_ids=ctx_bundle["rag_ids"],
                    enable_thinking=True,
                )
                L_kv = sum(k.size(2) for k in kv_list)
                mod_s, mod_e = 0, L_kv
                resp_q_start = spans["response"][0] - 1  # in Q coordinates (0..len(ids)-1)
            else:
                e, l, spans = build_multislot_sp_inputs(
                    s_emb, tmpl, qi, ri,
                    sys_text_ids=ctx_bundle["sys_text_ids"],
                    history_turns_ids=ctx_bundle["history_turns_ids"],
                    rag_ids=ctx_bundle["rag_ids"],
                    sp_sys_start=sp_sys_start,
                    sp_post_query=sp_post_query,
                    device=device, model_dtype=dtype,
                )
                sp_key = "sp_sys_start" if sp_sys_start else "sp_post_query"
                mod_s, mod_e = spans[sp_key]
                resp_s = int((l[0] != -100).nonzero()[0].item())
                resp_q_start = resp_s - 1

            state = {"call": 0}

            def hooked_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
                li = state["call"] % n_layers
                state["call"] += 1
                if li not in blocked_layers:
                    return orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
                ql, kl = q.size(-2), k.size(-2)
                neg_inf = torch.finfo(q.dtype).min
                if attn_mask is not None and attn_mask.dtype != torch.bool:
                    m = attn_mask.clone()
                else:
                    # Build exact causal mask accounting for past_key_values offset (kl - ql)
                    m = torch.triu(torch.full((1, 1, ql, kl), neg_inf, dtype=q.dtype, device=q.device), diagonal=1 + (kl - ql))
                m[:, :, resp_q_start:, mod_s:mod_e] = neg_inf
                return orig_sdpa(q, k, v, attn_mask=m, dropout_p=dropout_p, is_causal=False, scale=scale, enable_gqa=enable_gqa)

            F.scaled_dot_product_attention = hooked_sdpa
            try:
                if mode == "kv":
                    bi = torch.tensor([ids], dtype=torch.long, device=device)
                    bm = torch.ones_like(bi)
                    out = forward_with_kv_prefix(student, bi, bm, kv_list)
                    rmask = torch.tensor(lbl[1:], dtype=torch.long, device=device) != -100
                    z = out.logits[0, :-1, :][rmask].float()
                    del bi, bm, out
                else:
                    z = compute_response_logits_fast(student, e, l)[0]
                    del e, l
            finally:
                F.scaled_dot_product_attention = orig_sdpa

            n = min(z.size(0), ti.size(0))
            logp = F.log_softmax(z[:n], dim=-1)
            ce = -(tp[:n] * logp.gather(-1, ti[:n])).sum(dim=-1)
            kls.append((ce - ent[:n]).clamp_min(0.0).mean().item())
            del z, logp
        return round(sum(kls) / max(1, len(kls)), 5)

    # Sanity check: empty blocked set matches unhooked full_kl
    sanity_kl = run_with_blocked_layers(set())
    all_blk_kl = run_with_blocked_layers(set(range(n_layers)))

    def rec_pct(kl_val: float) -> float:
        den = base_kl - full_kl
        if abs(den) < 1e-6:
            return 0.0
        return round(100.0 * (base_kl - kl_val) / den, 2)

    open_first: Dict[str, Any] = {}
    open_last: Dict[str, Any] = {}
    for l in cutpoints:
        # open [0, l) -> block [l, n_layers)
        kl_f = all_blk_kl if l == 0 else (sanity_kl if l == n_layers else run_with_blocked_layers(set(range(l, n_layers))))
        # open [l, n_layers) -> block [0, l)
        kl_l = sanity_kl if l == 0 else (all_blk_kl if l == n_layers else run_with_blocked_layers(set(range(0, l))))
        open_first[str(l)] = {"kl": kl_f, "recovery_vs_base_pct": rec_pct(kl_f), "g_vs_ctx": round(100.0 * (1.0 - kl_f / base_kl), 2)}
        open_last[str(l)] = {"kl": kl_l, "recovery_vs_base_pct": rec_pct(kl_l), "g_vs_ctx": round(100.0 * (1.0 - kl_l / base_kl), 2)}

    res = {
        "base_kl": base_kl, "full_kl": full_kl, "sanity_open_all_kl": sanity_kl, "all_blocked_kl": all_blk_kl,
        "sanity_diff": round(abs(full_kl - sanity_kl), 6),
        "open_first_0_to_l": open_first,
        "open_last_l_to_36": open_last,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    sf = " | ".join(f"0..{l}:{open_first[str(l)]['recovery_vs_base_pct']:+5.1f}%" for l in [0, 8, 12, 16, 24, 36])
    sl = " | ".join(f"{l}..36:{open_last[str(l)]['recovery_vs_base_pct']:+5.1f}%" for l in [0, 8, 12, 16, 24, 36])
    print(f"  [Pathway {tag:22s}] sanity_diff={res['sanity_diff']:.6f} | first=[{sf}] | last=[{sl}] ({res['elapsed_sec']}s)", flush=True)
    return res


# ---------------------------------------------------------------------------
# 5. Part D: Engineering Verification (Fast lm_head VRAM + Virtual Tokens)
# ---------------------------------------------------------------------------

def benchmark_response_only_lm_head(student, tmpl, ctx_bundles, sp: torch.Tensor) -> Dict[str, Any]:
    """Compare full-sequence student(inputs_embeds).logits vs. compute_response_logits_fast."""
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    out: Dict[str, Any] = {}
    qi = [100] * 64
    ri = [200] * 256

    for L in [512, 2048, 4096]:
        cb = ctx_bundles[L]
        sp_var = sp.detach().clone().requires_grad_(True)
        e, l, _ = build_multislot_sp_inputs(
            s_emb, tmpl, qi, ri,
            sys_text_ids=cb["sys_text_ids"], history_turns_ids=cb["history_turns_ids"], rag_ids=cb["rag_ids"],
            sp_post_query=[sp_var], device=device, model_dtype=dtype,
        )
        T_total = int(e.size(1))
        student.config.use_cache = False
        student.train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Full logits baseline
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        res_full = student(inputs_embeds=e)
        z_full = res_full.logits[0, :-1, :][l[0, 1:] != -100].float()
        loss_full = F.cross_entropy(z_full, l[0, 1:][l[0, 1:] != -100])
        loss_full.backward()
        torch.cuda.synchronize()
        full_ms = round((time.time() - t0) * 1000.0, 1)
        full_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        z_full_ref = z_full.detach().clone()
        del res_full, z_full, loss_full
        sp_var.grad = None

        # Response-only lm_head projection
        e2, l2, _ = build_multislot_sp_inputs(
            s_emb, tmpl, qi, ri,
            sys_text_ids=cb["sys_text_ids"], history_turns_ids=cb["history_turns_ids"], rag_ids=cb["rag_ids"],
            sp_post_query=[sp_var], device=device, model_dtype=dtype,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.time()
        z_fast = compute_response_logits_fast(student, e2, l2)[0]
        loss_fast = F.cross_entropy(z_fast, l2[0, 1:][l2[0, 1:] != -100])
        loss_fast.backward()
        torch.cuda.synchronize()
        fast_ms = round((time.time() - t1) * 1000.0, 1)
        fast_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        max_diff = float((z_fast.detach() - z_full_ref).abs().max().item())

        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True
        del e, l, e2, l2, z_fast, loss_fast, z_full_ref
        torch.cuda.empty_cache()

        out[str(L)] = {
            "total_seq_len": T_total,
            "full_peak_mb": full_peak_mb,
            "fast_peak_mb": fast_peak_mb,
            "vram_saved_mb": round(full_peak_mb - fast_peak_mb, 1),
            "full_ms": full_ms,
            "fast_ms": fast_ms,
            "speedup": round(full_ms / max(0.1, fast_ms), 2),
            "max_abs_logit_diff": round(max_diff, 6),
        }
        print(
            f"  [FastHead L={L:4d} T={T_total:4d}] VRAM {full_peak_mb:.0f}->{fast_peak_mb:.0f} MB "
            f"(saved {full_peak_mb-fast_peak_mb:.0f} MB) | {full_ms:.0f}->{fast_ms:.0f} ms ({out[str(L)]['speedup']:.2f}x) | diff={max_diff:.6f}",
            flush=True,
        )
    return out


@torch.no_grad()
def verify_virtual_token_export_and_tied_hazard(
    student, tokenizer, tmpl, official_i: torch.Tensor, crsp_if: torch.Tensor
) -> Dict[str, Any]:
    """
    1. Build derived safetensors checkpoint appending Official I.pt (rms=0.37) + Alg1 CRSP (rms=0.042).
    2. Verify exact logit equivalence between inputs_embeds and virtual token IDs at P0 and P4.
    3. Measure tied-embedding output logit hazard (max logit on appended high-norm virtual rows vs.
       top real vocabulary token), empirically verifying why -100 fails and -1e9 is mandatory!
    """
    from huggingface_hub import hf_hub_download

    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    src_dir = Path(hf_hub_download(STUDENT_ID, "config.json", local_files_only=True)).parent

    with tempfile.TemporaryDirectory(prefix="cspd_vtok_") as tmp:
        dst = Path(tmp) / "derived"
        meta = build_derived_virtual_checkpoint(
            src_dir, dst, [("official_I", official_i), ("alg1_crsp_IF", crsp_if)]
        )
        derived = AutoModelForCausalLM.from_pretrained(str(dst), dtype=dtype, device_map="cuda", attn_implementation="sdpa")
        derived.eval()

        q_ids = tokenizer.encode("Write a 2-sentence summary in uppercase.", add_special_tokens=False)
        user_open = tmpl.sys_close_user_open[len(tmpl.im_end) :]
        asst_open = tmpl.user_close_asst_open[len(tmpl.im_end) :]
        s_emb = student.get_input_embeddings()

        # Check 1: P0 (Official I.pt)
        e_p0, _, _ = build_multislot_sp_inputs(s_emb, tmpl, q_ids, None, sp_sys_start=[official_i], device=device, model_dtype=dtype)
        z_emb_p0 = student(inputs_embeds=e_p0, use_cache=False).logits[0, -1, :].float()
        vids_p0 = render_multislot_virtual_ids(
            tmpl.sys_open, tmpl.im_end, user_open, asst_open, q_ids,
            vids_p0_sys_start=meta["expert_token_ids"]["official_I"],
        )
        z_vt_p0 = derived(input_ids=torch.tensor([vids_p0], device=device), use_cache=False).logits[0, -1, :].float()
        orig_v = meta["original_vocab_size"]
        diff_p0 = float((z_emb_p0 - z_vt_p0[:orig_v]).abs().max().item())

        # Tied-embedding logit hazard check on Official I.pt virtual rows!
        real_max_logit = float(z_vt_p0[: len(tokenizer)].max().item())
        off_i_ids = meta["expert_token_ids"]["official_I"]
        crsp_ids = meta["expert_token_ids"]["alg1_crsp_IF"]
        off_i_max_logit = float(z_vt_p0[off_i_ids[0] : off_i_ids[-1] + 1].max().item())
        crsp_max_logit_at_p0 = float(z_vt_p0[crsp_ids[0] : crsp_ids[-1] + 1].max().item())

        # Check 2: P4 (Alg1 CRSP after persona + history)
        p_ids = tokenizer.encode(EVAL_PERSONA, add_special_tokens=False)
        e_p4, _, _ = build_multislot_sp_inputs(
            s_emb, tmpl, q_ids, None, sys_text_ids=p_ids, sp_post_query=[crsp_if], device=device, model_dtype=dtype
        )
        z_emb_p4 = student(inputs_embeds=e_p4, use_cache=False).logits[0, -1, :].float()
        vids_p4 = render_multislot_virtual_ids(
            tmpl.sys_open, tmpl.im_end, user_open, asst_open, q_ids,
            sys_text_ids=p_ids, vids_p4_post_query=crsp_ids,
        )
        z_vt_p4 = derived(input_ids=torch.tensor([vids_p4], device=device), use_cache=False).logits[0, -1, :].float()
        diff_p4 = float((z_emb_p4 - z_vt_p4[:orig_v]).abs().max().item())
        crsp_max_logit_at_p4 = float(z_vt_p4[crsp_ids[0] : crsp_ids[-1] + 1].max().item())
        real_max_p4 = float(z_vt_p4[: len(tokenizer)].max().item())
        off_i_max_p4 = float(z_vt_p4[off_i_ids[0] : off_i_ids[-1] + 1].max().item())

        masked_ids = build_hard_output_mask_ids(len(tokenizer), meta["extended_vocab_size"])
        del derived
        gc.collect()
        torch.cuda.empty_cache()

    res = {
        "original_vocab_size": orig_v,
        "extended_vocab_size": meta["extended_vocab_size"],
        "hard_masked_token_count": len(masked_ids),
        "p0_official_I_max_logit_diff": round(diff_p0, 6),
        "p4_alg1_crsp_max_logit_diff": round(diff_p4, 6),
        "tied_hazard_at_p0": {
            "top_real_vocab_logit": round(real_max_logit, 2),
            "max_official_I_virtual_logit": round(off_i_max_logit, 2),
            "max_crsp_virtual_logit": round(crsp_max_logit_at_p0, 2),
            "exceeds_minus100_guard": bool(off_i_max_logit - 100.0 > real_max_logit),
        },
        "tied_hazard_at_p4": {
            "top_real_vocab_logit": round(real_max_p4, 2),
            "max_official_I_virtual_logit": round(off_i_max_p4, 2),
            "max_crsp_virtual_logit": round(crsp_max_logit_at_p4, 2),
        },
    }
    print(
        f"  [VirtualToken Export] P0 diff={diff_p0:.6f} | P4 diff={diff_p4:.6f} | "
        f"Tied logits @P0: real_max={real_max_logit:.1f}, Official_I_max={off_i_max_logit:.1f} "
        f"(beats -100 guard={res['tied_hazard_at_p0']['exceeds_minus100_guard']}), CRSP_max={crsp_max_logit_at_p0:.1f}",
        flush=True,
    )
    return res


def run_parts_d_and_e(
    student, tokenizer, tmpl, ctx_bundles, off_p, kv_if_eos, kv_math_eos, sp_alg1_if_eos, sp_alg1_math_eos,
    math32, ifbench32, ifeval48, results, t0,
):
    print("\n--- PART D: Engineering Verification (Response-Only lm_head & Virtual-Token Export) ---", flush=True)
    if not results["partD_engineering_verify"].get("response_only_lm_head"):
        results["partD_engineering_verify"]["response_only_lm_head"] = benchmark_response_only_lm_head(student, tmpl, ctx_bundles, sp_alg1_if_eos)
        save_results(results)
    results["partD_engineering_verify"]["virtual_token_export"] = verify_virtual_token_export_and_tied_hazard(student, tokenizer, tmpl, off_p["I"], sp_alg1_if_eos)
    save_results(results)

    print("\n--- PART E1: MATH32 End-to-End (L=0 & L=4096) ---", flush=True)
    for L in [0, 4096]:
        results["partE_e2e_benchmarks"]["math32"][str(L)] = {
            "Official_M@P0": run_math_fast(student, tokenizer, tmpl, ctx_bundles[L], math32, mode="sp", sp_sys_start=[off_p["M"]], tag="Official_M@P0"),
            "Official_Dual_MI@P0": run_math_fast(student, tokenizer, tmpl, ctx_bundles[L], math32, mode="sp", sp_sys_start=[off_p["M"], off_p["I"]], tag="Official_Dual_MI@P0"),
            "Official_MCSI@P0": run_math_fast(student, tokenizer, tmpl, ctx_bundles[L], math32, mode="sp", sp_sys_start=[off_p["M"], off_p["C"], off_p["S"], off_p["I"]], tag="Official_MCSI@P0"),
            "KVPD_Math_EOS@P0": run_math_fast(student, tokenizer, tmpl, ctx_bundles[L], math32, mode="kv", kv_list=[kv_math_eos], tag="KVPD_Math_EOS@P0"),
            "Alg1_Split_EOS_M@P2_I@P4": run_math_fast(student, tokenizer, tmpl, ctx_bundles[L], math32, mode="sp", sp_pre_query=[sp_alg1_math_eos], sp_post_query=[sp_alg1_if_eos], tag="Alg1_Split_EOS_M@P2_I@P4"),
        }
        save_results(results)

    print("\n--- PART E2: IFBench32 End-to-End (L=0 & L=4096) ---", flush=True)
    for L in [0, 4096]:
        results["partE_e2e_benchmarks"]["ifbench32"][str(L)] = {
            "Official_I@P0": run_ifbench_fast(student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="sp", sp_sys_start=[off_p["I"]], tag="Official_I@P0"),
            "Official_Dual_MI@P0": run_ifbench_fast(student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="sp", sp_sys_start=[off_p["M"], off_p["I"]], tag="Official_Dual_MI@P0"),
            "Official_MCSI@P0": run_ifbench_fast(student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="sp", sp_sys_start=[off_p["M"], off_p["C"], off_p["S"], off_p["I"]], tag="Official_MCSI@P0"),
            "KVPD_IF_EOS@P0": run_ifbench_fast(student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="kv", kv_list=[kv_if_eos], tag="KVPD_IF_EOS@P0"),
            "Alg1_Split_EOS_M@P2_I@P4": run_ifbench_fast(student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode="sp", sp_pre_query=[sp_alg1_math_eos], sp_post_query=[sp_alg1_if_eos], tag="Alg1_Split_EOS_M@P2_I@P4"),
        }
        save_results(results)

    print("\n--- PART E3: IFEval48 End-to-End (L=0 & L=4096) ---", flush=True)
    for L in [0, 4096]:
        results["partE_e2e_benchmarks"]["ifeval48"][str(L)] = {
            "Official_I@P0": run_ifeval_fast(student, tokenizer, tmpl, ctx_bundles[L], ifeval48, mode="sp", sp_sys_start=[off_p["I"]], tag="Official_I@P0"),
            "Official_Dual_MI@P0": run_ifeval_fast(student, tokenizer, tmpl, ctx_bundles[L], ifeval48, mode="sp", sp_sys_start=[off_p["M"], off_p["I"]], tag="Official_Dual_MI@P0"),
            "Official_MCSI@P0": run_ifeval_fast(student, tokenizer, tmpl, ctx_bundles[L], ifeval48, mode="sp", sp_sys_start=[off_p["M"], off_p["C"], off_p["S"], off_p["I"]], tag="Official_MCSI@P0"),
            "KVPD_IF_EOS@P0": run_ifeval_fast(student, tokenizer, tmpl, ctx_bundles[L], ifeval48, mode="kv", kv_list=[kv_if_eos], tag="KVPD_IF_EOS@P0"),
            "Alg1_Split_EOS_M@P2_I@P4": run_ifeval_fast(student, tokenizer, tmpl, ctx_bundles[L], ifeval48, mode="sp", sp_pre_query=[sp_alg1_math_eos], sp_post_query=[sp_alg1_if_eos], tag="Alg1_Split_EOS_M@P2_I@P4"),
        }
        save_results(results)

    results["meta"]["total_wall_sec"] = round(time.time() - t0, 1)
    save_results(results)
    print(f"\n=== ALL DONE! exp12_results.json saved in {results['meta']['total_wall_sec']}s ===", flush=True)


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 94, flush=True)
    print("Suite 10 (exp12): Reference Step-200 CSPD Audit, Complete-EOS Upgrade & 36-Layer Pathway", flush=True)
    print("=" * 94, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    verify_qwen_system_scaffold(tokenizer)
    print("  [Scaffold Audit] verify_qwen_system_scaffold PASSED!", flush=True)
    tmpl = get_qwen_template_tokens(tokenizer)

    if_train96, ifeval_test96, ifbench64, math_train64, math_test48, _, _, _ = load_datasets_exp9()
    # Expand Math train pool to 96 Level 3-4 problems so strict EOS+verified filter gets >=48 clean traces
    ds_math_tr = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    m_test_set = {x["prompt"] for x in math_test48}
    math_pool96 = list(math_train64)
    seen_m = {x["prompt"] for x in math_pool96} | m_test_set
    for row in ds_math_tr:
        lvl = str(row.get("level", ""))
        if any(k in lvl for k in ["3", "4"]):
            q = row["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            if q not in seen_m:
                math_pool96.append({"prompt": q, "answer": row["solution"]})
                seen_m.add(q)
                if len(math_pool96) >= 96:
                    break

    if_val_prompts = [x["prompt"] for x in ifeval_test96[:24]]
    math_val_prompts = [x["prompt"] for x in math_test48[:24]]
    ifeval48 = ifeval_test96[:48]
    ifbench32 = ifbench64[:32]
    math32 = math_test48[:32]

    ctx_bundles = build_long_context_bundles(tokenizer, tmpl, LENGTHS_SWEEP)

    results: Dict[str, Any] = {
        "meta": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "scaffold_verified": True,
            "actual_ctx_tokens": {str(k): v["actual_ctx_tokens"] for k, v in ctx_bundles.items()},
        },
        "official_prompt_stats": {},
        "partA1_official_6slots": {},
        "partA2_official_longctx_single": {"if": {}, "math": {}, "attn_mass_if": {}},
        "partA3_official_longctx_comp": {"if": {}, "math": {}, "interference": {}},
        "partB_eos_training": {},
        "partC_36layer_causal_pathway": {},
        "partD_engineering_verify": {},
        "partE_e2e_benchmarks": {"math32": {}, "ifbench32": {}, "ifeval48": {}},
    }

    if "--resume-de" in sys.argv and os.path.exists(RESULT_PATH) and os.path.exists(os.path.join(OUT_DIR, "exp12_eos_modules.pt")):
        print("\n--- RESUME MODE (--resume-de): Loading existing exp12_results.json & exp12_eos_modules.pt ---", flush=True)
        results = json.load(open(RESULT_PATH, "r", encoding="utf-8"))
        student = AutoModelForCausalLM.from_pretrained(
            STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
        )
        for p in student.parameters():
            p.requires_grad_(False)
        student.eval()
        device = next(student.parameters()).device
        off_p, _ = fetch_official_prompts(device)
        mods = torch.load(os.path.join(OUT_DIR, "exp12_eos_modules.pt"), map_location=device, weights_only=True)
        run_parts_d_and_e(
            student, tokenizer, tmpl, ctx_bundles, off_p,
            mods["kv_if_eos"], mods["kv_math_eos"], mods["sp_alg1_if_eos"], mods["sp_alg1_math_eos"],
            math32, ifbench32, ifeval48, results, t0,
        )
        return

    # ------------------------------------------------------------------
    # PHASE 1: Cache Val Targets (consistent with exp10/11) + Complete-EOS Train Targets
    # ------------------------------------------------------------------
    print("\n--- PHASE 1: Caching Val Targets & Complete EOS-Terminated Train Targets ---", flush=True)
    if_val_cache = cache_postrl_teacher_top64(
        IF_TEACHER_ID, tokenizer, tmpl, if_val_prompts,
        max_resp_gen=768, max_resp_trunc=512, topk=64, seed=777, tag="IF_Val24",
    )
    math_val_cache = cache_postrl_teacher_top64(
        MATH_TEACHER_ID, tokenizer, tmpl, math_val_prompts,
        max_resp_gen=768, max_resp_trunc=512, topk=64, seed=888, tag="Math_Val24",
    )
    if_eos_train, if_eos_stats = cache_complete_eos_teacher_top64(
        IF_TEACHER_ID, tokenizer, tmpl, if_train96, domain="if",
        max_resp_gen=1536, max_keep=64, topk=64, seed=1777, tag="IF_EOS_Train",
    )
    math_eos_train, math_eos_stats = cache_complete_eos_teacher_top64(
        MATH_TEACHER_ID, tokenizer, tmpl, math_pool96, domain="math",
        max_resp_gen=1536, max_keep=64, topk=64, seed=1888, tag="Math_EOS_Train",
    )
    results["partB_eos_training"]["teacher_eos_stats"] = {"if": if_eos_stats, "math": math_eos_stats}

    print(f"\n--- Loading Frozen Backbone: {STUDENT_ID} ---", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    for p in student.parameters():
        p.requires_grad_(False)
    student.eval()
    device = next(student.parameters()).device
    s_emb = student.get_input_embeddings()

    # Load reference step-200 checkpoints (SHA-256 verified)
    print("\n--- Loading Reference Step-200 Prompts (M, C, S, I) ---", flush=True)
    off_p, off_stats = fetch_official_prompts(device)
    results["official_prompt_stats"] = off_stats
    save_results(results)

    # ------------------------------------------------------------------
    # PART A: Reference Step-200 CSPD Audit (P0-P5 & 0 -> 8192)
    # ------------------------------------------------------------------
    print("\n--- PART A1: Official I.pt & M.pt Across Six Deployment Slots (P0-P5) ---", flush=True)
    results["partA1_official_6slots"]["official_I_on_IF"] = eval_official_6slots(student, tokenizer, tmpl, if_val_cache, off_p["I"], "Official_I")
    results["partA1_official_6slots"]["official_M_on_Math"] = eval_official_6slots(student, tokenizer, tmpl, math_val_cache, off_p["M"], "Official_M")
    save_results(results)

    print("\n--- PART A2: Official I.pt & M.pt Long-Context Sweep (0 -> 8192) & 9-Layer Attn Mass ---", flush=True)
    base_if_kl, base_m_kl = {}, {}
    for L in LENGTHS_SWEEP:
        bi, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="base")
        bm, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode="base")
        base_if_kl[L], base_m_kl[L] = bi["kl"], bm["kl"]

    for dom, vcache, bmap, sp_t, name in [
        ("if",   if_val_cache,   base_if_kl, off_p["I"], "Official_I_at_P0"),
        ("math", math_val_cache, base_m_kl,  off_p["M"], "Official_M_at_P0"),
    ]:
        ref0 = None
        results["partA2_official_longctx_single"][dom][name] = {}
        for L in LENGTHS_SWEEP:
            r, saved = eval_config_on_val_cache(
                student, tmpl, vcache, ctx_bundles[L], mode="sp", sp_sys_start=[sp_t],
                ref_p0_logits_list=ref0, save_ref_top32=(L == 0),
            )
            if L == 0:
                ref0 = saved
            results["partA2_official_longctx_single"][dom][name][str(L)] = {
                **r, "g_vs_ctx": round(100.0 * (1.0 - r["kl"] / bmap[L]), 2)
            }
        line = " | ".join(f"{L}:{results['partA2_official_longctx_single'][dom][name][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_SWEEP)
        d8k = results["partA2_official_longctx_single"][dom][name]["8192"]["drift_kl"]
        print(f"  [{name:22s}] {dom.upper()}=[{line}] | drift8k={d8k:.4f}", flush=True)

    # Exact 9-layer attention mass for Official I.pt @ P0 across 0..8192
    for L in LENGTHS_SWEEP:
        am = measure_response_attention_mass(student, tmpl, if_val_cache[:8], ctx_bundles[L], mode="sp", sp_sys_start=[off_p["I"]])
        results["partA2_official_longctx_single"]["attn_mass_if"][str(L)] = am
    am_line = " | ".join(f"{L}:{results['partA2_official_longctx_single']['attn_mass_if'][str(L)]['mean_mod_pct']:.2f}%" for L in LENGTHS_SWEEP)
    print(f"  [Official_I_at_P0 Attn ] mean_mod_pct=[{am_line}]", flush=True)
    save_results(results)

    print("\n--- PART A3: Official Dual ([M;I]/[I;M]) & 4-Expert (MCSI/ISCM) Composition (0 -> 8192) ---", flush=True)
    solo_off_i_refs = {}
    for L in LENGTHS_COMP:
        _, solo_off_i_refs[L] = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="sp", sp_sys_start=[off_p["I"]], save_ref_top32=True)

    off_comp_specs = [
        ("Official_Dual_MI@P0", [off_p["M"], off_p["I"]]),
        ("Official_Dual_IM@P0", [off_p["I"], off_p["M"]]),
        ("Official_MCSI@P0",    [off_p["M"], off_p["C"], off_p["S"], off_p["I"]]),
        ("Official_ISCM@P0",    [off_p["I"], off_p["S"], off_p["C"], off_p["M"]]),
    ]
    for name, sps in off_comp_specs:
        results["partA3_official_longctx_comp"]["if"][name] = {}
        results["partA3_official_longctx_comp"]["math"][name] = {}
        results["partA3_official_longctx_comp"]["interference"][name] = {}
        for L in LENGTHS_COMP:
            ri, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode="sp", sp_sys_start=sps)
            rm, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode="sp", sp_sys_start=sps)
            intf = eval_pair_interference_kl(student, tmpl, if_val_cache, ctx_bundles[L], solo_off_i_refs[L], mode="sp", sp_sys_start=sps)
            results["partA3_official_longctx_comp"]["if"][name][str(L)] = {**ri, "g_vs_ctx": round(100.0 * (1.0 - ri["kl"] / base_if_kl[L]), 2)}
            results["partA3_official_longctx_comp"]["math"][name][str(L)] = {**rm, "g_vs_ctx": round(100.0 * (1.0 - rm["kl"] / base_m_kl[L]), 2)}
            results["partA3_official_longctx_comp"]["interference"][name][str(L)] = intf
        si = " | ".join(f"{L}:{results['partA3_official_longctx_comp']['if'][name][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_COMP)
        sm = " | ".join(f"{L}:{results['partA3_official_longctx_comp']['math'][name][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_COMP)
        sk = " | ".join(f"{L}:{results['partA3_official_longctx_comp']['interference'][name][str(L)]:.4f}" for L in LENGTHS_COMP)
        print(f"  [{name:22s}] IF=[{si}] | Math=[{sm}] | intf=[{sk}]", flush=True)
    save_results(results)

    # ------------------------------------------------------------------
    # PART B: Complete EOS-Terminated Training (KV-PD-EOS, Direct-EOS, Alg1-EOS)
    # ------------------------------------------------------------------
    print("\n--- PART B: Complete-EOS Training (KV-PD-EOS & Alg. 1 CRSP-EOS with Fast lm_head) ---", flush=True)
    train_stream_text = " ".join(TRAIN_PERSONAS + [u + " " + a for u, a in TRAIN_DIALOG_TURNS] + TRAIN_RAG_CHUNKS)
    train_pools = {
        "personas": [tokenizer.encode(p, add_special_tokens=False) for p in TRAIN_PERSONAS],
        "turns": [(tokenizer.encode(u, add_special_tokens=False), tokenizer.encode(a, add_special_tokens=False)) for u, a in TRAIN_DIALOG_TURNS],
        "rags": [tokenizer.encode(r, add_special_tokens=False) for r in TRAIN_RAG_CHUNKS],
        "stream": tokenizer.encode(train_stream_text, add_special_tokens=False),
    }
    foreign_pool = build_foreign_distractor_pool(s_emb, tokenizer, num_vectors=64, seed=77)
    null_for_if = build_null_anchor_pool(student, tmpl, math_eos_train, max_items=24, topk=32)
    null_for_math = build_null_anchor_pool(student, tmpl, if_eos_train, max_items=24, topk=32)

    kv_if_eos, m_kv_if = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, if_eos_train, inst_text=IF_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.022, seed=42, tag="KVPD_IF_EOS_L16",
    )
    kv_math_eos, m_kv_math = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, math_eos_train, inst_text=MATH_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.020, seed=42, tag="KVPD_Math_EOS_L16",
    )
    kv_if_self_eos, s_kv_if = cache_kv_complete_eos_self_targets(
        student, tokenizer, tmpl, kv_if_eos, [x["prompt"] for x in if_train96[:80]],
        max_resp_gen=1536, max_keep=56, topk=64, seed=1606, tag="KV_IF_Self_EOS",
    )
    sp_alg1_if_eos, m_a1_if = train_cscr_sp_fast_eos(
        student, tokenizer, tmpl, kv_if_self_eos, train_pools, IF_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Alg1_IF_EOS_L64",
    )
    kv_math_self_eos, s_kv_math = cache_kv_complete_eos_self_targets(
        student, tokenizer, tmpl, kv_math_eos, [x["prompt"] for x in math_pool96[:80]],
        max_resp_gen=1536, max_keep=56, topk=64, seed=1607, tag="KV_Math_Self_EOS",
    )
    sp_alg1_math_eos, m_a1_math = train_cscr_sp_fast_eos(
        student, tokenizer, tmpl, kv_math_self_eos, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="Alg1_Math_EOS_L64",
    )
    results["partB_eos_training"].update({
        "kv_if_meta": m_kv_if, "kv_math_meta": m_kv_math,
        "kv_self_if_stats": s_kv_if, "kv_self_math_stats": s_kv_math,
        "alg1_if_meta": m_a1_if, "alg1_math_meta": m_a1_math,
    })
    torch.save(
        {"kv_if_eos": kv_if_eos.cpu(), "kv_math_eos": kv_math_eos.cpu(), "sp_alg1_if_eos": sp_alg1_if_eos.cpu(), "sp_alg1_math_eos": sp_alg1_math_eos.cpu()},
        os.path.join(OUT_DIR, "exp12_eos_modules.pt"),
    )
    save_results(results)

    # ------------------------------------------------------------------
    # PART C: 36-Layer Cumulative Causal Pathway Ablation (L=0 & L=4096)
    # ------------------------------------------------------------------
    print("\n--- PART C: 36-Layer Cumulative Causal Attention Pathway Ablation (CSPD vs KV-PD vs CRSP) ---", flush=True)
    cutpoints = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
    val_sub = if_val_cache[:12]
    for L, lkey in [(0, "L0"), (4096, "L4096")]:
        results["partC_36layer_causal_pathway"][lkey] = {
            "Official_CSPD_I@P0": eval_layerwise_causal_pathway(
                student, tmpl, val_sub, ctx_bundles[L], mode="sp", sp_sys_start=[off_p["I"]],
                cutpoints=cutpoints, tag=f"Off_I@P0_{lkey}",
            ),
            "KVPD_IF_EOS@P0": eval_layerwise_causal_pathway(
                student, tmpl, val_sub, ctx_bundles[L], mode="kv", kv_list=[kv_if_eos],
                cutpoints=cutpoints, tag=f"KVPD@P0_{lkey}",
            ),
            "Alg1_CRSP_EOS@P4": eval_layerwise_causal_pathway(
                student, tmpl, val_sub, ctx_bundles[L], mode="sp", sp_post_query=[sp_alg1_if_eos],
                cutpoints=cutpoints, tag=f"Alg1@P4_{lkey}",
            ),
        }
        save_results(results)

    run_parts_d_and_e(
        student, tokenizer, tmpl, ctx_bundles, off_p,
        kv_if_eos, kv_math_eos, sp_alg1_if_eos, sp_alg1_math_eos,
        math32, ifbench32, ifeval48, results, t0,
    )


if __name__ == "__main__":
    main()
