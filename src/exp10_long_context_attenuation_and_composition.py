"""
Paper Reference: Section 7.1–7.2, Tables 10 & 12, Fig. 12–13 (0->8192 Attenuation & 9-Layer Attention Mass).
Suite 8 (exp10): Long-Context Capability Injection Attenuation & Multi-Expert Composition
=======================================================================================
Systematically investigates how capability injection and multi-expert composition behave
as realistic preceding context length scales from L_ctx = 0 up to L_ctx = 8192 tokens
(non-repeating multi-turn dialogue + WikiText reference documents) on frozen M2RL-SFT
(36-layer Qwen3-4B, 32k native context window):

  Part A — Single-Expert Long-Context Attenuation Sweep (L_ctx in {0, 512, 1024, 2048, 4096, 8192}):
    Compares start-fixed vs. sliding-slot modules on held-out IF (N=24) & Math (N=24) trajectories:
      * Pos-0 CSPD (L=256, fixed at index 0)
      * Prefix-Robust CSPD (PR-SP, L=256, fixed at index 0)
      * Layerwise KV-PD (L_kv=16, fixed at index 0 via past_key_values)
      * Direct CRSP (L=256) pinned at Start (control) vs. sliding at PreQuery (pre-query) & PostQuery (post-query)
      * Direct CRSP (L=64) sliding at PreQuery & PostQuery
      * Algorithm 1 KV-Inv-CRSP (L=64, no teacher) sliding at PreQuery & PostQuery
    Measures G_vs_ctx(L_ctx), absolute KL, retention ratio G(L_ctx)/G(0), and Drift-KL(L_ctx, 0).

  Part B — Mechanistic Layerwise Attention-Mass Diagnostic (L_ctx in {0..8192}):
    Hooks F.scaled_dot_product_attention (exact post-QK-norm, post-RoPE, post-GQA weights)
    across 9 representative layers (shallow / mid / deep) to measure exact response-to-module
    attention mass alpha_mod(L_ctx) vs. context dilution alpha_ctx(L_ctx), separating RoPE
    distance decay (fixed Start vs. sliding pre_query/post_query) from softmax denominator dilution!

  Part C — Multi-Expert Composition Under Long Context (L_ctx in {0, 1024, 4096, 8192}):
    Compares dual-expert (Math + IF) composition & order gap |Delta_order| + cross-expert
    Top-32 KL interference across:
      * PR-CSPD dual concat at Start ([Math; IF] vs [IF; Math])
      * Start-Fixed KV-PD dual concat at Start ([Z_Math; Z_IF] vs [Z_IF; Z_Math])
      * Direct CRSP (L=64) sliding split-slot [Math@PreQuery + IF@PostQuery] & adjacent [Math; IF]@PreQuery / [IF; Math]@PreQuery
      * Alg. 1 KV-Inv-CRSP (L=64) sliding split-slot [Math@PreQuery + IF@PostQuery] & adjacent [Math; IF]@PreQuery / [IF; Math]@PreQuery

  Part D — End-to-End Autoregressive Generative Evaluation Under Long Context:
    Real generation & strict programmatic verification at L_ctx in {0, 2048, 4096} on:
      * Held-out IFEval (N=48, 1536 tok)
      * Official IFBench (N=32, 2048 tok, L_ctx in {0, 4096})
      * Held-out MATH (N=32, 1536 tok, L_ctx in {0, 4096})

  Part E — Zero-Leakage Private Document KV-PD (HeliosPay v4.2, N=36) Under Long Context:
    Tests whether Stage II start-fixed KV-PD (L_kv=16, with and without Micro-RAG) suffers
    attenuation when L_ctx in {0, 1024, 2048, 4096} tokens of unrelated conversation sit
    between the Start KV cache and the HeliosPay query.
"""

import gc
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
    get_qwen_template_tokens,
    pad_and_stack_train_batch,
    sample_random_prefix,
)
from benchmarks import (
    strip_thinking,
    verify_ifeval_response,
    verify_math_response,
)
from domain_docs_data import (
    DOC_A_TEXT,
    grade_domain_response,
)
from kv_prefix_core import (
    build_chat_ids_for_kv,
    forward_with_kv_prefix,
    init_kv_prefix_from_tokens,
    make_mutable_kv_cache,
    make_readonly_kv_cache,
)
from exp7_context_robust_sp import (
    EVAL_HISTORY_TURNS,
    EVAL_PERSONA,
    IF_TASK_INST,
    TRAIN_DIALOG_TURNS,
    TRAIN_PERSONAS,
    TRAIN_RAG_CHUNKS,
    build_multislot_sp_inputs,
    clamp_sp_rms_,
    hybrid_top64_distill_loss,
    init_anchored_sp,
)
from exp9_multiseed_cr_grpo_and_composition import (
    DOC_TASK_INST,
    MATH_TASK_INST,
    batch_generate_slots,
    build_foreign_distractor_pool,
    load_datasets_exp9,
)
from exp9b_postrl_crsp_and_composition import (
    cache_kv_pure_logit_targets,
    cache_postrl_teacher_top64,
    precompute_doc_cache_eos,
    train_kv_distilled_l16_custom,
    train_postrl_cscr_sp,
)
import evaluation_lib as ifbench_eval

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"

OUT_DIR = "/content/cspd/exp10_results"
os.makedirs(OUT_DIR, exist_ok=True)
RESULT_PATH = os.path.join(OUT_DIR, "exp10_results.json")

LENGTHS_SWEEP = [0, 512, 1024, 2048, 4096, 8192]
LENGTHS_COMP = [0, 1024, 4096, 8192]
ATTN_LAYERS = [0, 5, 11, 17, 23, 29, 32, 34, 35]

HELIOS_MICRO_FACTS = [
    ("version", "System Version: HeliosPay v4.2 (Production Core)."),
    ("cutoff", "Daily Settlement Cutoff: 16:40 SGT."),
    ("reserve", "Mandatory Reserve Buffer: 14.5% of rolling 30-day corridor volume."),
    ("quorum", "Cold Vault Quorum: 4-of-7 hardware security keys."),
    ("retention", "Regulatory Audit Retention Period: 9.5 years."),
    ("officer", "Lead Compliance Officer: Dr. Maren Lindholm."),
    ("sla", "Standard Escalation SLA: 22 minutes."),
    ("limit", "Maximum Single Instant Wire Limit: 420,000 USD."),
    ("4172", "Error Code HP-4172: Velocity Circuit Breaker (Risk Score R >= 75)."),
    ("5089", "Error Code HP-5089: Corridor Liquidity Halt (below 14.5% reserve buffer)."),
    ("3314", "Error Code HP-3314: Beneficiary Sanctions Mismatch (OFAC/EU watchlist hit)."),
    ("6205", "Error Code HP-6205: FX Margin Breach (LATAM-Q9 wires > 420,000 USD)."),
    ("2901", "Error Code HP-2901: Dual-Key Signature Expiry (L2 token > 90 minutes)."),
    ("corridor", "Corridors: APAC-B7 (Base Risk 12, Base Fee 38 bps); EMEA-K3 (18, 52 bps); LATAM-Q9 (31, 74 bps)."),
]


def retrieve_micro_rag_snippet(question: str) -> str:
    ql = question.lower()
    hits = []
    kw_map = {
        "version": ["version", "release", "core"],
        "cutoff": ["cutoff", "cut-off", "sgt", "deadline", "daily settlement"],
        "reserve": ["reserve", "buffer", "liquidity"],
        "quorum": ["quorum", "vault", "hardware", "key"],
        "retention": ["retention", "audit", "years", "keep"],
        "officer": ["officer", "compliance", "lead", "who", "notify"],
        "sla": ["sla", "escalation", "minutes", "review time"],
        "limit": ["limit", "maximum", "single instant", "cap"],
        "4172": ["4172", "velocity", "circuit"],
        "5089": ["5089", "liquidity halt"],
        "3314": ["3314", "sanction", "ofac", "watchlist"],
        "6205": ["6205", "fx margin", "margin breach"],
        "2901": ["2901", "dual-key", "expiry", "signature"],
        "corridor": ["apac-b7", "emea-k3", "latam-q9", "base fee", "base risk", "corridor"],
    }
    for key, text in HELIOS_MICRO_FACTS:
        if any(w in ql for w in kw_map[key]):
            hits.append(text)
    if not hits:
        return ""
    return "[Retrieved Policy Facts]\n" + "\n".join(hits[:3]) + "\n\n"


def save_results(results: Dict[str, Any]):
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. Realistic Non-Repeating Long-Context Builder (0 .. 8192 tokens)
# ---------------------------------------------------------------------------

def load_nonrepeating_prose_stream(tokenizer, min_tokens: int = 12000) -> List[int]:
    """Load clean, non-repeating English encyclopedia passages from Salesforce/wikitext validation."""
    paragraphs: List[str] = []
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        for row in ds:
            t = row["text"].strip()
            if len(t) >= 120 and not t.startswith("="):
                paragraphs.append(t)
            if sum(len(p) for p in paragraphs) > min_tokens * 6:
                break
        print(f"  [LongCtx] Loaded {len(paragraphs)} non-repeating Salesforce/wikitext passages.", flush=True)
    except Exception as e:
        print(f"  [LongCtx] WikiText load warning ({e}); falling back to synthetic corpus.", flush=True)
        for i in range(120):
            paragraphs.append(
                f"Operational Archive Note #{1000+i}: Regional cluster node-{i%17} recorded median "
                f"RPC latency of {11.3 + (i%9)*1.4:.1f} ms across {20000 + i*317} requests during "
                f"batch checkpoint {i}. Storage replication remained healthy with zero dropped frames "
                f"and checksum validation passed across all {4 + i%6} availability zones."
            )
    full_text = "\n\n".join(paragraphs)
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    while len(ids) < min_tokens:
        ids = ids + ids
    return ids[:min_tokens]


def build_long_context_bundles(
    tokenizer, tmpl: ChatTemplateTokens, lengths: List[int]
) -> Dict[int, Dict[str, Any]]:
    """
    Build realistic multi-turn + retrieved reference document context bundles of EXACT
    preceding token length L_ctx in {0, 512, 1024, 2048, 4096, 8192}.
    All bundles share the exact same structure for both KV-PD (at Start) and Input-SP / CRSP!
    """
    prose_stream = load_nonrepeating_prose_stream(tokenizer, min_tokens=14000)
    persona_ids = tokenizer.encode(EVAL_PERSONA, add_special_tokens=False)
    base_turns = [
        (tokenizer.encode(u, add_special_tokens=False), tokenizer.encode(a, add_special_tokens=False))
        for u, a in EVAL_HISTORY_TURNS
    ]
    header_ids = tokenizer.encode("[Background Workspace Reference Archive — Unrelated to New Request]\n", add_special_tokens=False)
    footer_ids = tokenizer.encode("\n[End of Background Archive. Independently follow every instruction in the user prompt below.]\n\n", add_special_tokens=False)

    user_open = tmpl.sys_close_user_open[len(tmpl.im_end):]
    asst_open = tmpl.user_close_asst_open[len(tmpl.im_end):]

    # Overhead tokens contributed by persona + 2 history turns + chat template wrappers
    turn_overhead = len(persona_ids) + sum(
        len(user_open) + len(u) + len(tmpl.im_end) + len(asst_open) + len(a) + len(tmpl.im_end)
        for u, a in base_turns
    )

    bundles: Dict[int, Dict[str, Any]] = {}
    for L in lengths:
        if L == 0:
            bundles[0] = {
                "target_len": 0,
                "actual_ctx_tokens": 0,
                "sys_text_ids": None,
                "history_turns_ids": None,
                "rag_ids": None,
            }
        else:
            rem = max(0, L - turn_overhead - len(header_ids) - len(footer_ids))
            rag_ids = header_ids + prose_stream[:rem] + footer_ids
            actual = turn_overhead + len(rag_ids)
            bundles[L] = {
                "target_len": L,
                "actual_ctx_tokens": actual,
                "sys_text_ids": persona_ids,
                "history_turns_ids": base_turns,
                "rag_ids": rag_ids,
            }
        print(f"  [LongCtx Bundle] target={L:4d} -> actual_ctx_tokens={bundles[L]['actual_ctx_tokens']:4d}", flush=True)
    return bundles


def build_longctx_ids_for_kv(
    tmpl: ChatTemplateTokens,
    question_ids: List[int],
    response_ids: Optional[List[int]] = None,
    sys_text_ids: Optional[List[int]] = None,
    history_turns_ids: Optional[List[Tuple[List[int], List[int]]]] = None,
    rag_ids: Optional[List[int]] = None,
    enable_thinking: bool = True,
    no_think_ids: Optional[List[int]] = None,
) -> Tuple[List[int], Optional[List[int]], Dict[str, Tuple[int, int]]]:
    """
    Exact token-level twin of build_multislot_sp_inputs (with 0 soft prompts), returning
    (ids, labels, spans) so KV-PD at Start and Base(L_ctx) see the EXACT same token sequence!
    """
    ids: List[int] = []
    spans: Dict[str, Tuple[int, int]] = {}

    def append_seg(seg: Optional[List[int]], name: Optional[str] = None):
        if not seg:
            return
        s0 = len(ids)
        ids.extend(seg)
        if name:
            spans[name] = (s0, len(ids))

    user_open = tmpl.sys_close_user_open[len(tmpl.im_end):]
    asst_open = tmpl.user_close_asst_open[len(tmpl.im_end):]

    ctx_start = 0
    append_seg(tmpl.sys_open)
    append_seg(sys_text_ids, "sys_text")
    append_seg(tmpl.im_end)
    if history_turns_ids:
        for u_ids, a_ids in history_turns_ids:
            append_seg(user_open)
            append_seg(u_ids)
            append_seg(tmpl.im_end)
            append_seg(asst_open)
            append_seg(a_ids)
            append_seg(tmpl.im_end)
    append_seg(user_open)
    append_seg(rag_ids, "rag")
    ctx_end = len(ids)
    spans["ctx_all"] = (ctx_start, ctx_end)

    append_seg(question_ids, "question")
    append_seg(tmpl.user_close_asst_open)
    if (not enable_thinking) and no_think_ids:
        append_seg(no_think_ids)
    prompt_len = len(ids)
    if response_ids is None:
        return ids, None, spans
    append_seg(response_ids, "response")
    labels = [-100] * prompt_len + list(response_ids)
    return ids, labels, spans


# ---------------------------------------------------------------------------
# 2. Training Helpers: Pos-0 CSPD, PR-SP CSPD, KV-PD Doc (HeliosPay v4.2)
# ---------------------------------------------------------------------------

def train_input_cspd_sp(
    student, tokenizer, tmpl: ChatTemplateTokens, target_cache: List[Dict[str, Any]],
    stream_ids: List[int], task_inst: str, num_vectors: int = 256,
    prefix_robust: bool = True, num_updates: int = 45, batch_size: int = 10,
    lr_max: float = 0.005, max_rms: float = 0.048, seed: int = 42, tag: str = "CSPD",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Train Input-Space CSPD (either pure Pos-0 SP or Prefix-Robust PR-SP) with AdamW."""
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    sp = init_anchored_sp(s_emb, tokenizer, task_inst, num_vectors=num_vectors, anchored=False)
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

        batch = [target_cache[rng.randrange(len(target_cache))] for _ in range(batch_size)]
        tot_loss = 0.0
        for item in batch:
            qi, ri = item["q_ids"], item["r_ids"]
            ti, tp = item["ti"].to(device), item["tp"].float().to(device)
            if prefix_robust and rng.random() < 0.5:
                pre = sample_random_prefix(stream_ids, rng, 64, 192)
                e, l, _ = build_multislot_sp_inputs(
                    s_emb, tmpl, qi, ri, sys_text_ids=pre, sp_sys_end=[sp], device=device, model_dtype=dtype
                )
            else:
                e, l, _ = build_multislot_sp_inputs(
                    s_emb, tmpl, qi, ri, sp_sys_start=[sp], device=device, model_dtype=dtype
                )
            out = student(inputs_embeds=e)
            sl = out.logits[0, :-1, :].float()
            m = l[0, 1:] != -100
            loss = hybrid_top64_distill_loss(sl[m], ti, tp)
            (loss / batch_size).backward()
            tot_loss += float(loss.item()) / batch_size
            del e, l, out

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)
        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True

        if step in {1, 20, num_updates}:
            log_hist[str(step)] = round(tot_loss, 4)
            print(f"  [{tag:24s}] step {step:2d}/{num_updates} | dist={tot_loss:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "num_vectors": num_vectors, "prefix_robust": prefix_robust,
        "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


def train_kv_doc_helios(
    student, tokenizer, tmpl: ChatTemplateTokens, doc_cache: List[Dict[str, Any]],
    num_kv_tokens: int = 16, num_updates: int = 80, batch_size: int = 8,
    lr_max: float = 0.03, seed: int = 42, tag: str = "KVPD_Doc_Helios",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Train zero-leakage HeliosPay v4.2 KV-PD (L_kv=16, AdamW, 80 steps, matching exp4/exp5)."""
    t0 = time.time()
    device = next(student.parameters()).device
    doc_ids = tokenizer.encode(DOC_A_TEXT, add_special_tokens=False)
    nt_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    kv = init_kv_prefix_from_tokens(student, doc_ids, num_kv_tokens=num_kv_tokens)
    init_kv = kv.detach().clone()
    student.gradient_checkpointing_disable()
    student.eval()
    student.config.use_cache = True
    opt = torch.optim.AdamW([kv], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)
    log_hist = {}

    for step in range(1, num_updates + 1):
        prog = (step - 1) / max(1, num_updates - 1)
        lr = lr_max * 0.1 + 0.5 * (lr_max * 0.9) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad()
        batch = [doc_cache[rng.randrange(len(doc_cache))] for _ in range(batch_size)]
        tot_loss = 0.0
        for item in batch:
            # item["r_ids"] already starts with nt_ids (<think>\n\n</think>\n\n) and ends with im_end
            ids, lbl = build_chat_ids_for_kv(tmpl, item["q_ids"], item["r_ids"], None, True, None)
            bi = torch.tensor([ids], dtype=torch.long, device=device)
            bm = torch.ones_like(bi)
            out = forward_with_kv_prefix(student, bi, bm, [kv])
            sl = out.logits[0, :-1, :].float()
            m = torch.tensor(lbl[1:], dtype=torch.long, device=device) != -100
            nll = F.cross_entropy(sl[m], torch.tensor(item["r_ids"], dtype=torch.long, device=device))
            kd = hybrid_top64_distill_loss(sl[m], item["ti"].to(device), item["tp"].float().to(device))
            loss = 0.5 * nll + 0.5 * kd
            (loss / batch_size).backward()
            tot_loss += float(loss.item()) / batch_size
            del bi, bm, out
        torch.nn.utils.clip_grad_norm_([kv], 1.0)
        opt.step()
        if step in {1, 40, num_updates}:
            log_hist[str(step)] = round(tot_loss, 4)
            print(f"  [{tag:24s}] step {step:2d}/{num_updates} | loss={tot_loss:.4f} | d_rms={(kv.detach()-init_kv).pow(2).mean().sqrt().item():.3f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "num_kv_tokens": num_kv_tokens,
        "delta_rms": round(float((kv.detach() - init_kv).pow(2).mean().sqrt().item()), 4),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return kv.detach().clone(), meta


@torch.no_grad()
def build_null_anchor_pool(
    student, tmpl: ChatTemplateTokens, other_cache: List[Dict[str, Any]], max_items: int = 24, topk: int = 32
) -> List[Dict[str, Any]]:
    """Cache frozen-student Top-32 anchor logits on non-target domain for symmetric L_null."""
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    student.eval()
    pool = []
    for item in other_cache[:max_items]:
        e, l, _ = build_multislot_sp_inputs(s_emb, tmpl, item["q_ids"], item["r_ids"], device=device, model_dtype=dtype)
        out = student(inputs_embeds=e, use_cache=False)
        sl = out.logits[0, :-1, :].float()[l[0, 1:] != -100]
        probs = F.softmax(sl, dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
        pool.append({
            "q_ids": item["q_ids"], "r_ids": item["r_ids"],
            "anc_ti": ti.cpu(), "anc_tp": tp.to(torch.float16).cpu(), "anc_ent": ent.cpu(),
        })
        del e, l, out
    return pool


# ---------------------------------------------------------------------------
# 3. Distributional Long-Context Evaluator (KL vs Teacher & Cross-Length Drift)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_config_on_val_cache(
    student, tmpl: ChatTemplateTokens, val_cache: List[Dict[str, Any]],
    ctx_bundle: Dict[str, Any],
    mode: str,  # "base", "kv", "sp"
    kv_list: Optional[List[torch.Tensor]] = None,
    sp_sys_start: Optional[List[torch.Tensor]] = None,
    sp_pre_query: Optional[List[torch.Tensor]] = None,
    sp_post_query: Optional[List[torch.Tensor]] = None,
    ref_start_logits_list: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    save_ref_top32: bool = False,
) -> Tuple[Dict[str, float], Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    """
    Evaluate KL(Teacher || Model) at a given context bundle L_ctx, optionally computing
    Drift-KL relative to clean L_ctx=0 reference Top-32 distributions!
    """
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    student.eval()

    kls, drifts = [], []
    saved_refs = [] if save_ref_top32 else None

    for idx, item in enumerate(val_cache):
        qi, ri = item["q_ids"], item["r_ids"]
        ti = item["ti"].to(device)
        tp = item["tp"].float().to(device)
        ent = item["ent"].float().to(device)

        if mode == "kv":
            ids, lbl, _ = build_longctx_ids_for_kv(
                tmpl, qi, ri,
                sys_text_ids=ctx_bundle["sys_text_ids"],
                history_turns_ids=ctx_bundle["history_turns_ids"],
                rag_ids=ctx_bundle["rag_ids"],
                enable_thinking=True,
            )
            bi = torch.tensor([ids], dtype=torch.long, device=device)
            bm = torch.ones_like(bi)
            out = forward_with_kv_prefix(student, bi, bm, kv_list)
            m = torch.tensor(lbl[1:], dtype=torch.long, device=device) != -100
            z = out.logits[0, :-1, :][m].float()
            del bi, bm, out
        else:
            e, l, _ = build_multislot_sp_inputs(
                s_emb, tmpl, qi, ri,
                sys_text_ids=ctx_bundle["sys_text_ids"],
                history_turns_ids=ctx_bundle["history_turns_ids"],
                rag_ids=ctx_bundle["rag_ids"],
                sp_sys_start=sp_sys_start,
                sp_pre_query=sp_pre_query,
                sp_post_query=sp_post_query,
                device=device, model_dtype=dtype,
            )
            out = student(inputs_embeds=e, use_cache=False)
            m = l[0, 1:] != -100
            z = out.logits[0, :-1, :][m].float()
            del e, l, out

        n = min(z.size(0), ti.size(0))
        logp = F.log_softmax(z[:n], dim=-1)
        ce = -(tp[:n] * logp.gather(-1, ti[:n])).sum(dim=-1)
        kl = (ce - ent[:n]).clamp_min(0.0).mean().item()
        kls.append(kl)

        if save_ref_top32:
            probs = logp.exp()
            rp, ri_top = torch.topk(probs, 32, dim=-1)
            rp = rp / rp.sum(dim=-1, keepdim=True)
            rent = -(rp * rp.clamp_min(1e-9).log()).sum(dim=-1)
            saved_refs.append((ri_top.cpu(), rp.to(torch.float16).cpu(), rent.cpu()))

        if ref_start_logits_list is not None:
            r_ti, r_tp, r_ent = ref_start_logits_list[idx]
            r_ti = r_ti.to(device)
            r_tp = r_tp.float().to(device)
            r_ent = r_ent.float().to(device)
            nr = min(logp.size(0), r_ti.size(0))
            d_ce = -(r_tp[:nr] * logp[:nr].gather(-1, r_ti[:nr])).sum(dim=-1)
            drifts.append((d_ce - r_ent[:nr]).clamp_min(0.0).mean().item())

        del z, logp

    res = {"kl": round(sum(kls) / max(1, len(kls)), 5)}
    if drifts:
        res["drift_kl"] = round(sum(drifts) / max(1, len(drifts)), 5)
    return res, saved_refs


@torch.no_grad()
def eval_pair_interference_kl(
    student, tmpl: ChatTemplateTokens, val_cache: List[Dict[str, Any]],
    ctx_bundle: Dict[str, Any],
    solo_refs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    mode: str,
    kv_list: Optional[List[torch.Tensor]] = None,
    sp_sys_start: Optional[List[torch.Tensor]] = None,
    sp_pre_query: Optional[List[torch.Tensor]] = None,
    sp_post_query: Optional[List[torch.Tensor]] = None,
) -> float:
    """Measure Top-32 Logit KL(solo_target || composed_pair) at the exact same L_ctx."""
    res, _ = eval_config_on_val_cache(
        student, tmpl, val_cache, ctx_bundle, mode=mode, kv_list=kv_list,
        sp_sys_start=sp_sys_start, sp_pre_query=sp_pre_query, sp_post_query=sp_post_query,
        ref_start_logits_list=solo_refs,
    )
    return res["drift_kl"]


# ---------------------------------------------------------------------------
# 4. Mechanistic Layerwise Attention-Mass Hook (Zero-OOM at L_ctx = 8192!)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_response_attention_mass(
    student, tmpl: ChatTemplateTokens, val_items: List[Dict[str, Any]],
    ctx_bundle: Dict[str, Any],
    mode: str,  # "kv" or "sp"
    kv_list: Optional[List[torch.Tensor]] = None,
    sp_sys_start: Optional[List[torch.Tensor]] = None,
    sp_pre_query: Optional[List[torch.Tensor]] = None,
    sp_post_query: Optional[List[torch.Tensor]] = None,
    resp_window: int = 96,
) -> Dict[str, Any]:
    """
    Intercept F.scaled_dot_product_attention on 9 representative layers (0,5,11,17,23,29,32,34,35)
    to compute EXACT post-QK-norm, post-RoPE, post-GQA attention mass paid by generated response
    tokens to: (1) module tokens (SP or KV-PD), (2) preceding context tokens, (3) user query,
    and (4) self-response tokens. Peak extra VRAM < 80 MB even at L_ctx = 8192!
    """
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    student.eval()

    layer_stats = {li: {"mod": [], "ctx": [], "query": [], "resp": []} for li in ATTN_LAYERS}
    orig_sdpa = F.scaled_dot_product_attention

    for item in val_items:
        qi, ri = item["q_ids"], item["r_ids"][:resp_window]
        if mode == "kv":
            ids, _, spans = build_longctx_ids_for_kv(
                tmpl, qi, ri,
                sys_text_ids=ctx_bundle["sys_text_ids"],
                history_turns_ids=ctx_bundle["history_turns_ids"],
                rag_ids=ctx_bundle["rag_ids"],
                enable_thinking=True,
            )
            L_kv = sum(k.size(2) for k in kv_list)
            # In KV mode, key sequence is [0..L_kv-1] (module) + [L_kv .. L_kv+T-1] (ids)
            mod_span = (0, L_kv)
            ctx_span = (L_kv + spans["ctx_all"][0], L_kv + spans["ctx_all"][1])
            q_span = (L_kv + spans["question"][0], L_kv + spans["question"][1])
            r_span_q = spans["response"]  # indices within query tensor [0..T-1]
            r_span_k = (L_kv + r_span_q[0], L_kv + r_span_q[1])
            q_offset = L_kv
        else:
            e, _, spans = build_multislot_sp_inputs(
                s_emb, tmpl, qi, ri,
                sys_text_ids=ctx_bundle["sys_text_ids"],
                history_turns_ids=ctx_bundle["history_turns_ids"],
                rag_ids=ctx_bundle["rag_ids"],
                sp_sys_start=sp_sys_start,
                sp_pre_query=sp_pre_query,
                sp_post_query=sp_post_query,
                device=device, model_dtype=dtype,
            )
            # Identify primary module span
            mod_span = spans.get("sp_sys_start") or spans.get("sp_pre_query") or spans.get("sp_post_query")
            # Context span = sys_text + history + rag (everything before question excluding mod_span)
            q_span = spans["question"]
            r_span_q = spans["response"]
            r_span_k = r_span_q
            ctx_span = (0, q_span[0])  # we subtract mod_span overlap explicitly below!
            q_offset = 0

        r0, r1 = r_span_q
        call_counter = {"idx": 0}

        def hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            li = call_counter["idx"]
            call_counter["idx"] += 1
            if li in layer_stats:
                q_sub = query[:, :, r0:r1, :].float()  # [1, H_q, R, D_h]
                k_all = key.float()                     # [1, H_kv or H_q, K, D_h]
                if k_all.size(1) != q_sub.size(1):
                    k_all = k_all.repeat_interleave(q_sub.size(1) // k_all.size(1), dim=1)
                sc = scale if scale is not None else (1.0 / math.sqrt(q_sub.size(-1)))
                scores = torch.matmul(q_sub, k_all.transpose(-2, -1)) * sc  # [1, H_q, R, K]
                K_len = k_all.size(-2)
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        scores = scores.masked_fill(~attn_mask[:, :, r0:r1, :K_len], float("-inf"))
                    else:
                        scores = scores + attn_mask[:, :, r0:r1, :K_len].float()
                elif is_causal:
                    q_pos = torch.arange(q_offset + r0, q_offset + r1, device=scores.device).unsqueeze(1)
                    k_pos = torch.arange(0, K_len, device=scores.device).unsqueeze(0)
                    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
                probs = F.softmax(scores, dim=-1)  # [1, H_q, R, K]

                m_s, m_e = mod_span
                q_s, q_e = q_span
                rk_s, rk_e = r_span_k
                mod_m = probs[:, :, :, m_s:m_e].sum(dim=-1).mean().item()
                q_m = probs[:, :, :, q_s:q_e].sum(dim=-1).mean().item()
                r_m = probs[:, :, :, rk_s:rk_e].sum(dim=-1).mean().item()
                # Exact context mass = everything in [0..q_s) EXCEPT module tokens
                if mode == "kv":
                    c_m = probs[:, :, :, ctx_span[0]:ctx_span[1]].sum(dim=-1).mean().item()
                else:
                    pre_q_all = probs[:, :, :, 0:q_s].sum(dim=-1).mean().item()
                    mod_in_pre_q = mod_m if m_e <= q_s else 0.0
                    c_m = max(0.0, pre_q_all - mod_in_pre_q)

                layer_stats[li]["mod"].append(mod_m)
                layer_stats[li]["ctx"].append(c_m)
                layer_stats[li]["query"].append(q_m)
                layer_stats[li]["resp"].append(r_m)
                del q_sub, k_all, scores, probs

            return orig_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            )

        F.scaled_dot_product_attention = hooked_sdpa
        try:
            if mode == "kv":
                bi = torch.tensor([ids], dtype=torch.long, device=device)
                bm = torch.ones_like(bi)
                _ = forward_with_kv_prefix(student, bi, bm, kv_list)
                del bi, bm
            else:
                _ = student(inputs_embeds=e, use_cache=False)
                del e
        finally:
            F.scaled_dot_product_attention = orig_sdpa

    per_layer = {}
    for li in ATTN_LAYERS:
        per_layer[str(li)] = {
            k: round(100.0 * sum(v) / max(1, len(v)), 3) for k, v in layer_stats[li].items()
        }
    shallow = [0, 5, 11]
    mid = [17, 23]
    deep = [29, 32, 34, 35]
    summary = {
        "mean_mod_pct": round(sum(per_layer[str(l)]["mod"] for l in ATTN_LAYERS) / len(ATTN_LAYERS), 3),
        "shallow_mod_pct": round(sum(per_layer[str(l)]["mod"] for l in shallow) / len(shallow), 3),
        "mid_mod_pct": round(sum(per_layer[str(l)]["mod"] for l in mid) / len(mid), 3),
        "deep_mod_pct": round(sum(per_layer[str(l)]["mod"] for l in deep) / len(deep), 3),
        "mean_ctx_pct": round(sum(per_layer[str(l)]["ctx"] for l in ATTN_LAYERS) / len(ATTN_LAYERS), 3),
        "mean_query_pct": round(sum(per_layer[str(l)]["query"] for l in ATTN_LAYERS) / len(ATTN_LAYERS), 3),
        "mean_resp_pct": round(sum(per_layer[str(l)]["resp"] for l in ATTN_LAYERS) / len(ATTN_LAYERS), 3),
        "per_layer": per_layer,
    }
    return summary


# ---------------------------------------------------------------------------
# 5. Long-Context Batched Generation for KV-PD (Exact Twin of batch_generate_slots)
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_generate_kv_longctx(
    model, tokenizer, tmpl: ChatTemplateTokens, questions: List[str],
    kv_list: List[torch.Tensor],
    ctx_bundle: Dict[str, Any],
    batch_size: int = 16, max_new_tokens: int = 1536,
    temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20,
    enable_thinking: bool = True, seed: int = 42,
    micro_rag_prefixes: Optional[List[str]] = None,
) -> List[str]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    nt_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False) if not enable_thinking else None

    model.eval()
    model.config.use_cache = True
    torch.manual_seed(seed)
    all_outs: List[str] = []

    for start in range(0, len(questions), batch_size):
        bq = questions[start : start + batch_size]
        bmr = micro_rag_prefixes[start : start + batch_size] if micro_rag_prefixes else [""] * len(bq)
        seqs = []
        for q, mr in zip(bq, bmr):
            q_ids = tokenizer.encode((mr + q) if mr else q, add_special_tokens=False)
            ids, _, _ = build_longctx_ids_for_kv(
                tmpl, q_ids, None,
                sys_text_ids=ctx_bundle["sys_text_ids"],
                history_turns_ids=ctx_bundle["history_turns_ids"],
                rag_ids=ctx_bundle["rag_ids"],
                enable_thinking=enable_thinking,
                no_think_ids=nt_ids,
            )
            seqs.append(ids)
        B = len(seqs)
        ml = max(len(x) for x in seqs)
        inp = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
        sm = torch.zeros((B, ml), dtype=torch.long, device=device)
        for i, ids in enumerate(seqs):
            L = len(ids)
            inp[i, ml - L :] = torch.tensor(ids, dtype=torch.long, device=device)
            sm[i, ml - L :] = 1

        gen_kw = dict(
            input_ids=inp, max_new_tokens=max_new_tokens,
            pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if kv_list:
            cache = make_mutable_kv_cache(kv_list, B, device, dtype)
            L_kv = cache.get_seq_length()
            gen_kw["past_key_values"] = cache
            gen_kw["attention_mask"] = torch.cat([torch.ones((B, L_kv), dtype=torch.long, device=device), sm], dim=1)
        else:
            gen_kw["attention_mask"] = sm
        if temperature > 0.0:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            gen_kw.update(do_sample=False)

        out_ids = model.generate(**gen_kw)
        for row in out_ids:
            all_outs.append(tokenizer.decode(row[ml:], skip_special_tokens=True))
        del inp, sm, out_ids
        torch.cuda.empty_cache()
    return all_outs


def _vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 40.0


def eval_ifeval_longctx(
    model, tokenizer, tmpl, ctx_bundle, items, mode: str,
    kv_list=None, sp_sys_start=None, sp_pre_query=None, sp_post_query=None,
    tag: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    L = ctx_bundle["target_len"]
    vram = _vram_gb()
    if vram >= 75.0:
        bs = 48 if L == 0 else (24 if L <= 2048 else 16)
    elif vram < 30.0:
        bs = 12 if L == 0 else (6 if L <= 2048 else 4)
    else:
        bs = 24 if L == 0 else (12 if L <= 2048 else 8)
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
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFEval48 L={L:4d} | {tag:30s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def eval_ifbench_longctx(
    model, tokenizer, tmpl, ctx_bundle, inps, mode: str,
    kv_list=None, sp_sys_start=None, sp_pre_query=None, sp_post_query=None,
    tag: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    L = ctx_bundle["target_len"]
    vram = _vram_gb()
    bs = (32 if L == 0 else 16) if vram >= 75.0 else ((8 if L == 0 else 4) if vram < 30.0 else (16 if L == 0 else 8))
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
    pre_query_res, closed = {}, 0
    for inp, r in zip(inps, outs):
        closed += int("</think>" in r)
        pre_query_res[inp.prompt] = strip_thinking(r)
    s_ok = sum(int(ifbench_eval.test_instruction_following_strict(inp, pre_query_res).follow_all_instructions) for inp in inps)
    inst_l = []
    for inp in inps:
        inst_l.extend(ifbench_eval.test_instruction_following_strict(inp, pre_query_res).follow_instruction_list)
    n = len(inps)
    res = {
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * sum(inst_l) / max(1, len(inst_l)), 2),
        "n_strict": s_ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFB32    L={L:4d} | {tag:30s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def eval_math_longctx(
    model, tokenizer, tmpl, ctx_bundle, items, mode: str,
    kv_list=None, sp_sys_start=None, sp_pre_query=None, sp_post_query=None,
    tag: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    L = ctx_bundle["target_len"]
    vram = _vram_gb()
    bs = (32 if L == 0 else 16) if vram >= 75.0 else ((8 if L == 0 else 4) if vram < 30.0 else (16 if L == 0 else 8))
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
        "acc": round(100.0 * ok / n, 2),
        "n_ok": ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [MATH32   L={L:4d} | {tag:30s}] acc={res['acc']:5.2f}% ({ok:2d}/{n}) | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def eval_doc36_kv_longctx(
    model, tokenizer, tmpl, ctx_bundle, rules24, facts12, kv_list,
    use_micro_rag: bool = False, tag: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    L = ctx_bundle["target_len"]
    vram = _vram_gb()
    bs = (36 if L <= 1024 else 18) if vram >= 75.0 else ((9 if L <= 1024 else 4) if vram < 30.0 else (18 if L <= 1024 else 9))
    items = list(rules24) + list(facts12)
    qs = [x["question"] for x in items]
    mrs = [retrieve_micro_rag_snippet(q) if (use_micro_rag and i >= len(rules24)) else "" for i, q in enumerate(qs)]
    outs = batch_generate_kv_longctx(
        model, tokenizer, tmpl, qs, kv_list, ctx_bundle,
        batch_size=bs, max_new_tokens=320, temperature=0.0, enable_thinking=False,
        micro_rag_prefixes=mrs if use_micro_rag else None,
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


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main():
    t_global = time.time()
    print("=" * 92, flush=True)
    print("Suite 8 (exp10): Long-Context Attenuation & Multi-Expert Composition (L_ctx: 0 -> 8192)", flush=True)
    print("=" * 92, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    if_train96, ifeval_test96, ifbench64, math_train64, math_test48, doc_train, doc_rules24, doc_facts12 = load_datasets_exp9()

    # Splits
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
            "student": STUDENT_ID, "if_teacher": IF_TEACHER_ID, "math_teacher": MATH_TEACHER_ID,
            "lengths_sweep": LENGTHS_SWEEP, "lengths_comp": LENGTHS_COMP,
            "actual_ctx_tokens": {str(k): v["actual_ctx_tokens"] for k, v in ctx_bundles.items()},
            "attn_layers": ATTN_LAYERS,
        },
        "training": {},
        "partA_single_sweep_if": {},
        "partA_single_sweep_math": {},
        "partB_attention_mass": {},
        "partC_composition_if": {},
        "partC_composition_math": {},
        "partC_interference": {},
        "partD_e2e_ifeval48": {},
        "partD_e2e_ifbench32": {},
        "partD_e2e_math32": {},
        "partE_doc_kvpd_longctx": {},
    }

    # ------------------------------------------------------------------
    # PHASE 1: Cache Teacher Top-64 Targets (Load -> Cache -> Unload)
    # ------------------------------------------------------------------
    print("\n--- PHASE 1: Caching Post-RL Teacher Top-64 Targets (IF & Math) ---", flush=True)
    if_all_cache = cache_postrl_teacher_top64(
        IF_TEACHER_ID, tokenizer, tmpl, if_train_prompts + if_val_prompts,
        max_resp_gen=768, max_resp_trunc=512, topk=64, seed=777, tag="IF_Teacher",
    )
    n_if_tr = len(if_train_prompts)
    # Partition carefully by matching prompts (all prompts usually succeed >= 16 tok)
    if_train_cache = if_all_cache[:n_if_tr]
    if_val_cache = if_all_cache[n_if_tr:]
    if len(if_val_cache) < 16:
        if_val_cache = if_all_cache[-24:]
        if_train_cache = if_all_cache[:-24]

    math_all_cache = cache_postrl_teacher_top64(
        MATH_TEACHER_ID, tokenizer, tmpl, math_train_prompts + math_val_prompts,
        max_resp_gen=768, max_resp_trunc=512, topk=64, seed=888, tag="Math_Teacher",
    )
    n_m_tr = len(math_train_prompts)
    math_train_cache = math_all_cache[:n_m_tr]
    math_val_cache = math_all_cache[n_m_tr:]
    if len(math_val_cache) < 16:
        math_val_cache = math_all_cache[-24:]
        math_train_cache = math_all_cache[:-24]
    print(f"  [Split] IF train={len(if_train_cache)} val={len(if_val_cache)} | Math train={len(math_train_cache)} val={len(math_val_cache)}", flush=True)

    # Load Frozen Student Backbone
    print(f"\n--- Loading Frozen Backbone: {STUDENT_ID} ---", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    for p in student.parameters():
        p.requires_grad_(False)
    student.eval()
    s_emb = student.get_input_embeddings()

    # Training context pools (strictly disjoint from WikiText-2 / EVAL_PERSONA / EVAL_HISTORY_TURNS)
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

    # ------------------------------------------------------------------
    # PHASE 2: Train All Stage I (CSPD), Stage II (KV-PD), Stage III (CRSP) Modules
    # ------------------------------------------------------------------
    print("\n--- PHASE 2: Training Stage I (CSPD), Stage II (KV-PD), and Stage III (CRSP) Modules ---", flush=True)

    # Stage I: Input-Space CSPD (Pos-0 & Prefix-Robust, L=256, AdamW)
    sp_pos0_if, m = train_input_cspd_sp(
        student, tokenizer, tmpl, if_train_cache, train_pools["stream"], IF_TASK_INST,
        num_vectors=256, prefix_robust=False, num_updates=45, batch_size=10, lr_max=0.005, max_rms=0.048, seed=42, tag="StartOnly_CSPD_IF_L256",
    )
    results["training"]["StartOnly_CSPD_IF_L256"] = m

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

    # Stage II: Layerwise KV-PD (L_kv=16, AdamW)
    kv_if, m = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, if_train_cache, inst_text=IF_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.025, seed=42, tag="KVPD_IF_L16",
    )
    results["training"]["KVPD_IF_L16"] = m

    kv_math, m = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, math_train_cache, inst_text=MATH_TASK_INST,
        num_kv_tokens=16, num_updates=45, batch_size=12, lr_max=0.025, seed=42, tag="KVPD_Math_L16",
    )
    results["training"]["KVPD_Math_L16"] = m

    # Stage III: Direct CS-CR CRSP (L=256 IF, L=64 IF, L=64 Math)
    sp_crsp_if_256, m = train_postrl_cscr_sp(
        student, tokenizer, tmpl, if_train_cache, train_pools, IF_TASK_INST,
        num_vectors=256, num_updates=45, batch_size=10, lr_max=0.0055, lambda_inv=0.35, max_rms=0.038,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Direct_CRSP_IF_L256",
    )
    results["training"]["Direct_CRSP_IF_L256"] = m

    sp_crsp_if_64, m = train_postrl_cscr_sp(
        student, tokenizer, tmpl, if_train_cache, train_pools, IF_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Direct_CRSP_IF_L64",
    )
    results["training"]["Direct_CRSP_IF_L64"] = m

    sp_crsp_math_64, m = train_postrl_cscr_sp(
        student, tokenizer, tmpl, math_train_cache, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="Direct_CRSP_Math_L64",
    )
    results["training"]["Direct_CRSP_Math_L64"] = m

    # Stage III: Algorithm 1 (KV-Inv-CRSP, L=64 IF & Math, pure output distribution from KV-PD!)
    kv_if_self_cache = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_if, if_train_prompts, max_resp_gen=768, max_resp_trunc=512, topk=64, seed=606, tag="KV_IF_Self",
    )
    sp_alg1_if_64, m = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_if_self_cache, train_pools, IF_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_if, lambda_null=0.20, seed=42, tag="Alg1_KVInv_IF_L64",
    )
    results["training"]["Alg1_KVInv_IF_L64"] = m

    kv_math_self_cache = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_math, math_train_prompts, max_resp_gen=768, max_resp_trunc=512, topk=64, seed=607, tag="KV_Math_Self",
    )
    sp_alg1_math_64, m = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_math_self_cache, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, batch_size=10, lr_max=0.007, lambda_inv=0.35, max_rms=0.047,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_for_math, lambda_null=0.35, seed=42, tag="Alg1_KVInv_Math_L64",
    )
    results["training"]["Alg1_KVInv_Math_L64"] = m

    # Stage II Doc KV-PD (HeliosPay v4.2 zero-leakage)
    doc_cache = precompute_doc_cache_eos(student, tokenizer, tmpl, DOC_A_TEXT, doc_train, topk=64)
    kv_doc, m = train_kv_doc_helios(student, tokenizer, tmpl, doc_cache, num_kv_tokens=16, num_updates=80, batch_size=8, lr_max=0.03, seed=42)
    results["training"]["KVPD_Doc_Helios_L16"] = m
    save_results(results)

    # ------------------------------------------------------------------
    # PART A: Single-Expert Long-Context Attenuation Sweep (0 -> 8192)
    # ------------------------------------------------------------------
    print("\n--- PART A: Single-Expert Long-Context Attenuation Sweep (L_ctx: 0 -> 8192) ---", flush=True)

    # 1) Instruction Following (IF held-out N=24)
    if_single_configs = [
        ("Base_Frozen",            "base", {}),
        ("StartOnly_CSPD_L256_at_Start",   "sp",   {"sp_sys_start": [sp_pos0_if]}),
        ("PR_CSPD_L256_at_Start",     "sp",   {"sp_sys_start": [sp_pr_if]}),
        ("KVPD_L16_at_Start",         "kv",   {"kv_list": [kv_if]}),
        ("Direct_CRSP_L256_at_Start", "sp",   {"sp_sys_start": [sp_crsp_if_256]}),
        ("Direct_CRSP_L256_at_PreQuery", "sp",   {"sp_pre_query": [sp_crsp_if_256]}),
        ("Direct_CRSP_L256_at_PostQuery", "sp",   {"sp_post_query": [sp_crsp_if_256]}),
        ("Direct_CRSP_L64_at_PreQuery",  "sp",   {"sp_pre_query": [sp_crsp_if_64]}),
        ("Direct_CRSP_L64_at_PostQuery",  "sp",   {"sp_post_query": [sp_crsp_if_64]}),
        ("Alg1_KVInv_L64_at_PreQuery",   "sp",   {"sp_pre_query": [sp_alg1_if_64]}),
        ("Alg1_KVInv_L64_at_PostQuery",   "sp",   {"sp_post_query": [sp_alg1_if_64]}),
    ]
    if_start_refs: Dict[str, Any] = {}
    base_if_kls: Dict[int, float] = {}
    for L in LENGTHS_SWEEP:
        res_base, ref_b = eval_config_on_val_cache(
            student, tmpl, if_val_cache, ctx_bundles[L], mode="base",
            ref_start_logits_list=if_start_refs.get("Base_Frozen"), save_ref_top32=(L == 0),
        )
        if L == 0:
            if_start_refs["Base_Frozen"] = ref_b
        base_if_kls[L] = res_base["kl"]
        results["partA_single_sweep_if"].setdefault("Base_Frozen", {})[str(L)] = {**res_base, "g_vs_ctx": 0.0}
        print(f"  [IF Sweep L={L:4d}] Base_Frozen            | kl={res_base['kl']:.4f} | drift={res_base.get('drift_kl', 0.0):.4f}", flush=True)

    for name, mode, kw in if_single_configs[1:]:
        results["partA_single_sweep_if"][name] = {}
        g0 = None
        for L in LENGTHS_SWEEP:
            r, ref = eval_config_on_val_cache(
                student, tmpl, if_val_cache, ctx_bundles[L], mode=mode,
                ref_start_logits_list=if_start_refs.get(name), save_ref_top32=(L == 0), **kw,
            )
            if L == 0:
                if_start_refs[name] = ref
            g = round(100.0 * (1.0 - r["kl"] / max(1e-8, base_if_kls[L])), 2)
            if L == 0:
                g0 = g
            ret = round(100.0 * g / g0, 1) if (g0 is not None and abs(g0) > 1e-4) else 0.0
            results["partA_single_sweep_if"][name][str(L)] = {**r, "g_vs_ctx": g, "retention_vs_0_pct": ret}
        row_str = " | ".join(f"{L}:{results['partA_single_sweep_if'][name][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_SWEEP)
        d_last = results["partA_single_sweep_if"][name]["8192"].get("drift_kl", 0.0)
        print(f"  [IF   ] {name:24s} -> {row_str} | drift@8k={d_last:.4f}", flush=True)

    # 2) Mathematics (Math held-out N=24)
    math_single_configs = [
        ("Base_Frozen",            "base", {}),
        ("PR_CSPD_Math_L256_at_Start","sp",   {"sp_sys_start": [sp_pr_math]}),
        ("KVPD_Math_L16_at_Start",    "kv",   {"kv_list": [kv_math]}),
        ("Direct_CRSP_Math_L64_Start","sp",   {"sp_sys_start": [sp_crsp_math_64]}),
        ("Direct_CRSP_Math_L64_PreQuery","sp",   {"sp_pre_query": [sp_crsp_math_64]}),
        ("Direct_CRSP_Math_L64_PostQuery","sp",   {"sp_post_query": [sp_crsp_math_64]}),
        ("Alg1_KVInv_Math_L64_PreQuery", "sp",   {"sp_pre_query": [sp_alg1_math_64]}),
    ]
    math_start_refs: Dict[str, Any] = {}
    base_math_kls: Dict[int, float] = {}
    for L in LENGTHS_SWEEP:
        res_base, ref_b = eval_config_on_val_cache(
            student, tmpl, math_val_cache, ctx_bundles[L], mode="base",
            ref_start_logits_list=math_start_refs.get("Base_Frozen"), save_ref_top32=(L == 0),
        )
        if L == 0:
            math_start_refs["Base_Frozen"] = ref_b
        base_math_kls[L] = res_base["kl"]
        results["partA_single_sweep_math"].setdefault("Base_Frozen", {})[str(L)] = {**res_base, "g_vs_ctx": 0.0}

    for name, mode, kw in math_single_configs[1:]:
        results["partA_single_sweep_math"][name] = {}
        g0 = None
        for L in LENGTHS_SWEEP:
            r, ref = eval_config_on_val_cache(
                student, tmpl, math_val_cache, ctx_bundles[L], mode=mode,
                ref_start_logits_list=math_start_refs.get(name), save_ref_top32=(L == 0), **kw,
            )
            if L == 0:
                math_start_refs[name] = ref
            g = round(100.0 * (1.0 - r["kl"] / max(1e-8, base_math_kls[L])), 2)
            if L == 0:
                g0 = g
            ret = round(100.0 * g / g0, 1) if (g0 is not None and abs(g0) > 1e-4) else 0.0
            results["partA_single_sweep_math"][name][str(L)] = {**r, "g_vs_ctx": g, "retention_vs_0_pct": ret}
        row_str = " | ".join(f"{L}:{results['partA_single_sweep_math'][name][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_SWEEP)
        d_last = results["partA_single_sweep_math"][name]["8192"].get("drift_kl", 0.0)
        print(f"  [Math ] {name:24s} -> {row_str} | drift@8k={d_last:.4f}", flush=True)

    save_results(results)

    # ------------------------------------------------------------------
    # PART B: Mechanistic Layerwise Attention-Mass Diagnostic (0 -> 8192)
    # ------------------------------------------------------------------
    print("\n--- PART B: Mechanistic Response-to-Module Attention Mass (L_ctx: 0 -> 8192) ---", flush=True)
    attn_items = if_val_cache[:6]
    attn_configs = [
        ("PR_CSPD_L256_at_Start",     "sp", {"sp_sys_start": [sp_pr_if]}),
        ("KVPD_L16_at_Start",         "kv", {"kv_list": [kv_if]}),
        ("Direct_CRSP_L256_at_Start", "sp", {"sp_sys_start": [sp_crsp_if_256]}),
        ("Direct_CRSP_L256_at_PreQuery", "sp", {"sp_pre_query": [sp_crsp_if_256]}),
        ("Direct_CRSP_L256_at_PostQuery", "sp", {"sp_post_query": [sp_crsp_if_256]}),
        ("Alg1_KVInv_L64_at_PostQuery",   "sp", {"sp_post_query": [sp_alg1_if_64]}),
    ]
    for name, mode, kw in attn_configs:
        results["partB_attention_mass"][name] = {}
        for L in LENGTHS_SWEEP:
            am = measure_response_attention_mass(
                student, tmpl, attn_items, ctx_bundles[L], mode=mode, resp_window=96, **kw
            )
            results["partB_attention_mass"][name][str(L)] = am
        row_str = " | ".join(f"{L}:{results['partB_attention_mass'][name][str(L)]['mean_mod_pct']:5.2f}%" for L in LENGTHS_SWEEP)
        print(f"  [AttnMod] {name:24s} -> {row_str}", flush=True)

    save_results(results)

    # ------------------------------------------------------------------
    # PART C: Multi-Expert Composition Under Long Context (0, 1024, 4096, 8192)
    # ------------------------------------------------------------------
    print("\n--- PART C: Dual-Expert Composition (Math + IF) & Interference Under Long Context ---", flush=True)
    comp_configs = [
        ("PR_CSPD_Start_Math_IF",          "sp", {"sp_sys_start": [sp_pr_math, sp_pr_if]}),
        ("PR_CSPD_Start_IF_Math",          "sp", {"sp_sys_start": [sp_pr_if, sp_pr_math]}),
        ("KVPD_Start_Math_IF",             "kv", {"kv_list": [kv_math, kv_if]}),
        ("KVPD_Start_IF_Math",             "kv", {"kv_list": [kv_if, kv_math]}),
        ("Direct_CRSP_Split_Math_PreQuery_IF_PostQuery", "sp", {"sp_pre_query": [sp_crsp_math_64], "sp_post_query": [sp_crsp_if_64]}),
        ("Direct_CRSP_Adj_M_I@PreQuery",      "sp", {"sp_pre_query": [sp_crsp_math_64, sp_crsp_if_64]}),
        ("Direct_CRSP_Adj_I_M@PreQuery",      "sp", {"sp_pre_query": [sp_crsp_if_64, sp_crsp_math_64]}),
        ("Alg1_CRSP_Split_Math_PreQuery_IF_PostQuery",   "sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]}),
        ("Alg1_CRSP_Adj_M_I@PreQuery",        "sp", {"sp_pre_query": [sp_alg1_math_64, sp_alg1_if_64]}),
        ("Alg1_CRSP_Adj_I_M@PreQuery",        "sp", {"sp_pre_query": [sp_alg1_if_64, sp_alg1_math_64]}),
    ]

    for name, mode, kw in comp_configs:
        results["partC_composition_if"][name] = {}
        results["partC_composition_math"][name] = {}
        for L in LENGTHS_COMP:
            r_if, _ = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode=mode, **kw)
            g_if = round(100.0 * (1.0 - r_if["kl"] / max(1e-8, base_if_kls[L])), 2)
            results["partC_composition_if"][name][str(L)] = {**r_if, "g_vs_ctx": g_if}

            r_m, _ = eval_config_on_val_cache(student, tmpl, math_val_cache, ctx_bundles[L], mode=mode, **kw)
            g_m = round(100.0 * (1.0 - r_m["kl"] / max(1e-8, base_math_kls[L])), 2)
            results["partC_composition_math"][name][str(L)] = {**r_m, "g_vs_ctx": g_m}
        s_if = " | ".join(f"{L}:{results['partC_composition_if'][name][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_COMP)
        s_m = " | ".join(f"{L}:{results['partC_composition_math'][name][str(L)]['g_vs_ctx']:+5.1f}%" for L in LENGTHS_COMP)
        print(f"  [Comp IF  ] {name:28s} -> {s_if}", flush=True)
        print(f"  [Comp Math] {name:28s} -> {s_m}", flush=True)

    # Cross-Expert Interference KL(Solo_IF || Pair) on IF val trajectories across L_ctx in LENGTHS_COMP
    solo_pair_map = [
        ("PR_CSPD_Start",        ("sp", {"sp_sys_start": [sp_pr_if]}),        ("sp", {"sp_sys_start": [sp_pr_math, sp_pr_if]})),
        ("KVPD_Start",           ("kv", {"kv_list": [kv_if]}),                ("kv", {"kv_list": [kv_math, kv_if]})),
        ("Direct_CRSP_Split", ("sp", {"sp_post_query": [sp_crsp_if_64]}),  ("sp", {"sp_pre_query": [sp_crsp_math_64], "sp_post_query": [sp_crsp_if_64]})),
        ("Alg1_CRSP_Split",   ("sp", {"sp_post_query": [sp_alg1_if_64]}),  ("sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]})),
    ]
    for fam, (sm, skw), (pm, pkw) in solo_pair_map:
        results["partC_interference"][fam] = {}
        for L in LENGTHS_COMP:
            _, s_refs = eval_config_on_val_cache(student, tmpl, if_val_cache, ctx_bundles[L], mode=sm, save_ref_top32=True, **skw)
            ikl = eval_pair_interference_kl(student, tmpl, if_val_cache, ctx_bundles[L], s_refs, mode=pm, **pkw)
            results["partC_interference"][fam][str(L)] = round(ikl, 5)
        s_i = " | ".join(f"{L}:{results['partC_interference'][fam][str(L)]:.4f}" for L in LENGTHS_COMP)
        print(f"  [Interf KL] {fam:20s} -> {s_i}", flush=True)

    save_results(results)

    # ------------------------------------------------------------------
    # PART E: Zero-Leakage Declarative Document KV-PD Under Long Context (0..4096)
    # (Run right before Part D so fast Doc36 results are checkpointed early!)
    # ------------------------------------------------------------------
    print("\n--- PART E: Zero-Leakage HeliosPay v4.2 KV-PD (L_kv=16) Under Long Context ---", flush=True)
    for L in [0, 1024, 2048, 4096]:
        results["partE_doc_kvpd_longctx"][str(L)] = {
            "KVPD_Doc_Only": eval_doc36_kv_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], doc_rules24, doc_facts12, [kv_doc],
                use_micro_rag=False, tag="KVPD_Doc_Only",
            ),
            "KVPD_Doc_MicroRAG": eval_doc36_kv_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], doc_rules24, doc_facts12, [kv_doc],
                use_micro_rag=True, tag="KVPD_Doc_MicroRAG",
            ),
        }
        save_results(results)

    # ------------------------------------------------------------------
    # PART D: End-to-End Generative Long-Context Benchmark (IFEval48, IFBench32, MATH32)
    # ------------------------------------------------------------------
    print("\n--- PART D: End-to-End Generative Long-Context Benchmark ---", flush=True)

    # D1: IFEval48 at L_ctx in {0, 2048, 4096}
    ifeval_configs = [
        ("Base_Frozen",                 "sp", {}),
        ("PR_CSPD_L256_at_Start",          "sp", {"sp_sys_start": [sp_pr_if]}),
        ("KVPD_L16_at_Start",              "kv", {"kv_list": [kv_if]}),
        ("Direct_CRSP_L256_at_PostQuery",      "sp", {"sp_post_query": [sp_crsp_if_256]}),
        ("Alg1_KVInv_L64_at_PostQuery",        "sp", {"sp_post_query": [sp_alg1_if_64]}),
        ("PR_CSPD_Dual_Start",             "sp", {"sp_sys_start": [sp_pr_math, sp_pr_if]}),
        ("KVPD_Dual_Start",                "kv", {"kv_list": [kv_math, kv_if]}),
        ("Alg1_CRSP_Split_Math_PreQuery_IF_PostQuery",   "sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]}),
    ]
    for L in [0, 2048, 4096]:
        results["partD_e2e_ifeval48"][str(L)] = {}
        for name, mode, kw in ifeval_configs:
            results["partD_e2e_ifeval48"][str(L)][name] = eval_ifeval_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], ifeval48, mode=mode, tag=name, **kw
            )
        save_results(results)

    # D2: Official IFBench32 at L_ctx in {0, 4096}
    ifbench_configs = [
        ("Base_Frozen",                 "sp", {}),
        ("PR_CSPD_L256_at_Start",          "sp", {"sp_sys_start": [sp_pr_if]}),
        ("KVPD_L16_at_Start",              "kv", {"kv_list": [kv_if]}),
        ("Direct_CRSP_L256_at_PreQuery",      "sp", {"sp_pre_query": [sp_crsp_if_256]}),
        ("Alg1_KVInv_L64_at_PostQuery",        "sp", {"sp_post_query": [sp_alg1_if_64]}),
        ("KVPD_Dual_Start",                "kv", {"kv_list": [kv_math, kv_if]}),
        ("Alg1_CRSP_Split_Math_PreQuery_IF_PostQuery",   "sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]}),
    ]
    for L in [0, 4096]:
        results["partD_e2e_ifbench32"][str(L)] = {}
        for name, mode, kw in ifbench_configs:
            results["partD_e2e_ifbench32"][str(L)][name] = eval_ifbench_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], ifbench32, mode=mode, tag=name, **kw
            )
        save_results(results)

    # D3: MATH32 at L_ctx in {0, 4096}
    math_e2e_configs = [
        ("Base_Frozen",                 "sp", {}),
        ("PR_CSPD_Math_Start",             "sp", {"sp_sys_start": [sp_pr_math]}),
        ("KVPD_Math_Start",                "kv", {"kv_list": [kv_math]}),
        ("Alg1_Math_PreQuery",                "sp", {"sp_pre_query": [sp_alg1_math_64]}),
        ("PR_CSPD_Dual_Start",             "sp", {"sp_sys_start": [sp_pr_math, sp_pr_if]}),
        ("KVPD_Dual_Start",                "kv", {"kv_list": [kv_math, kv_if]}),
        ("Alg1_CRSP_Split_Math_PreQuery_IF_PostQuery",   "sp", {"sp_pre_query": [sp_alg1_math_64], "sp_post_query": [sp_alg1_if_64]}),
    ]
    for L in [0, 4096]:
        results["partD_e2e_math32"][str(L)] = {}
        for name, mode, kw in math_e2e_configs:
            results["partD_e2e_math32"][str(L)][name] = eval_math_longctx(
                student, tokenizer, tmpl, ctx_bundles[L], math32, mode=mode, tag=name, **kw
            )
        save_results(results)

    results["meta"]["total_wall_sec"] = round(time.time() - t_global, 1)
    save_results(results)
    print(f"\n=== ALL DONE! exp10_results.json saved in {results['meta']['total_wall_sec']}s ===", flush=True)


if __name__ == "__main__":
    main()
