"""
Suite 5 (exp7): Arbitrary-Position Insertion & Context-Robust Soft Prompt Distillation (CR-SP)
==============================================================================================
Investigates:
  1. How Input-Space Soft Prompts behave when inserted NOT ONLY at Position 0, but dynamically
     across realistic arbitrary-context positions:
       - P0_Clean_Start      : <|im_start|>system\n [P] <|im_end|>\n <|im_start|>user\n [Q]
       - P1_MidSys_Persona   : <|im_start|>system\n [120-tok Enterprise Persona] [P] <|im_end|>
       - P2_MultiTurn_Mid    : [Persona] + [2-Turn Dialog History ~360 tok] + user([P] [Q])
       - P3_PostRAG_Doc      : [Persona] + user([420-tok RAG/Tool Context] [P] [Q])
       - P4_PostQuery_Suffix : [Persona] + [2-Turn Dialog History] + user([Q] [P])  (Late Injection! 0 prefix KV recompute!)
       - P5_LongDrift_1000   : [Persona] + [2-Turn History] + [420-tok RAG] + user([P] [Q]) (~950 tok pre-context!)

  2. How to distill genuinely CONTEXT-ROBUST Input-Space Soft Prompts (CR-SP):
     Track A (Teacher-Distilled from M2RL-RL_IF dense Top-64 logits, identical step budget):
       - A1_Pos0_Only_SP     : Vanilla position-0 distillation (no context augmentation)
       - A2_Paper_PR_SP      : Original CSPD random-token prefix inside system turn (Zhang et al., 2026)
       - A3_MultiSlot_CR_SP  : Multi-Slot Structural Context-Robust Distillation (P0/P1/P2/P3/P4 sampling)
       - A4_Full_CR_SP_Inv   : Full CR-SP = Multi-Slot + Boundary Anchor Framing + Dual-View Cross-Context
                               Logit-Invariance Consistency Loss (L_inv: JS/KL(sg(p_P0) || p_s))!
       - A5_CR_SP_Compact_L64: Compact 4x shorter (L=64) Full CR-SP ablation
     Track B (Teacher-Free, ZERO Post-RL Teacher! Reuses exp6 TF-KV L16 scaffold):
       - B0_exp6_TF_KV2SP    : Baseline TF-KV2SP from exp6 (trained inside system turn only)
       - B1_CR_TF_KV2SP      : Teacher-Free Full CR-SP compiled from TF-KV L16 with Multi-Slot + L_inv!
     Plus KV Contrast:
       - KV_Fixed_L16        : Fixed Layerwise KV-Prefix (locked at absolute position 0)

  3. Comprehensive Evaluation:
     - Full 8-Method x 6-Position (P0..P5) Distributional Matrix:
         * Exact KL Gap Closed G(P, s) vs Teacher
         * Positional Behavioral Drift KL(p_P0 || p_s) (measures invariance across insertion positions!)
         * Layerwise Hidden-State Cosine Invariance cos(H_l(P | C_s), H_l(P | Clean)) at layers [0, 9, 18, 27, 35]
     - Official allenai/IFBench32 Real End-to-End Generation across P0, P2 (Multi-Turn), P3 (Post-RAG),
       P4 (Post-Query Late Injection), AND Split-Position Multi-Expert Assembly ([Sys: Math] + [History] + [Q + CR-IF])!
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
    get_qwen_template_tokens,
    pad_and_stack_train_batch,
    sample_random_prefix,
)
from benchmarks import (
    SUPPORTED_IFEVAL_IDS,
    batch_generate_with_soft_prompts,
    strip_thinking,
    verify_ifeval_response,
)
from kv_prefix_core import (
    build_chat_ids_for_kv,
    forward_with_kv_prefix,
    init_kv_prefix_from_tokens,
)
import evaluation_lib as ifbench_eval

STUDENT_ID = "Jackwang111/M2RL-SFT"
TEACHER_IF_ID = "Jackwang111/M2RL-RL_IF"
EXP6_DIR = "/content/cspd/exp6_results"
OUT_DIR = "/content/cspd/exp7_results"
os.makedirs(OUT_DIR, exist_ok=True)

IF_TASK_INST = (
    "Carefully read every single explicit and implicit instruction, formatting rule, keyword requirement, "
    "length constraint, punctuation restriction, and structural requirement in the user prompt. "
    "Plan your response step by step, verify every constraint strictly before finalizing, and ensure "
    "100% compliance with zero violations."
)

# ---------------------------------------------------------------------------
# Training Context Pools (STRICTLY DISJOINT from Evaluation Contexts!)
# ---------------------------------------------------------------------------

TRAIN_PERSONAS = [
    "You are an enterprise workflow copilot helping engineering teams draft technical summaries and specifications.",
    "You are a neutral analytical assistant embedded inside an internal operations workspace. Follow every user directive.",
    "You operate as a multi-step reasoning assistant. Maintain clarity, precision, and strict adherence to constraints.",
    "You are assisting a product analyst with documentation and structured writing tasks. Verify all formatting constraints.",
    "You are a concise, accurate corporate assistant handling independent user requests across diverse topics.",
    "You serve as a compliance-aware drafting assistant. Always honor structural, lexical, and length rules.",
]

TRAIN_DIALOG_TURNS: List[Tuple[str, str]] = [
    (
        "Can you summarize why connection pooling helps PostgreSQL under bursty microservice traffic?",
        "Connection pooling reuses established backend sessions across short-lived requests, avoiding repeated TCP/TLS handshakes and process forks. Tools such as PgBouncer cap active database connections, reducing lock contention and memory overhead during traffic spikes."
    ),
    (
        "What is the difference between exponential backoff and jitter in retry policies?",
        "Exponential backoff increases the wait interval geometrically after consecutive failures (e.g., 1s, 2s, 4s). Adding jitter injects controlled randomness into each delay so synchronized clients do not retry simultaneously and trigger a thundering herd."
    ),
    (
        "Briefly explain why content-addressed storage simplifies build caching.",
        "Content-addressed storage keys every artifact by the cryptographic hash of its inputs or bytes. Identical builds produce identical keys, guaranteeing deterministic cache hits, automatic deduplication, and safe concurrent reads without mutable overwrites."
    ),
    (
        "How does Bloom filter sizing trade off memory against false-positive rate?",
        "For n items and m bits with optimal k = (m/n) ln 2 hash functions, the false-positive probability scales approximately as (0.6185)^(m/n). Roughly 10 bits per element yields a ~1% false-positive rate with zero false negatives."
    ),
    (
        "Give a two-sentence checklist before rotating an API signing key in production.",
        "First, publish the new public key in the JWKS endpoint and confirm all downstream verifiers refresh their key cache before signing with the new secret. Second, keep the previous key active for at least one full token TTL window so in-flight tokens validate cleanly."
    ),
    (
        "Why do columnar formats like Parquet accelerate analytical aggregations?",
        "Columnar layouts store values of each field contiguously on disk, enabling projection pushdown (reading only queried columns), higher dictionary/RLE compression ratios, and vectorized SIMD scans over homogeneous arrays."
    ),
]

TRAIN_RAG_CHUNKS = [
    (
        "[Retrieved Reference Document #104 — Storage Telemetry Note]\n"
        "Cluster north-east-2 reported median object read latency of 14.2 ms and P99 latency of 88.5 ms "
        "across 42,000 requests per second during the Tuesday maintenance window. Erasure coding stripe width "
        "remained at 8+4 with zero unrecoverable shards. Background scrubbing consumed 6.4% of NVMe bandwidth "
        "and completed 18 minutes ahead of schedule. Note: the snippet above is background workspace context; "
        "focus strictly on the user request that follows."
    ),
    (
        "[Tool Output — Inventory Sync Snapshot #491]\n"
        "Warehouse hub B12 reconciled 14,820 SKUs across 19 zones. Discrepancy alerts were raised for 3 pallets "
        "in cold-chain aisle 7 due to barcode scanner firmware v2.14 timeout retries. Auto-reconciliation cleared "
        "2 alerts; ticket OPS-3391 tracks the remaining pallet. Treat this log as prior session context unless "
        "explicitly referenced in the next user instruction."
    ),
    (
        "[Retrieved Knowledge Snippet — Build Pipeline Audit]\n"
        "Artifact registry mirror eu-central synced 312 container digests in 94 seconds using delta layer transfer. "
        "SBOM attestation verified SPDX-2.3 signatures for 100% of release candidates. Canary rollout on staging "
        "held steady at 0.02% 5xx error rate over a 4-hour soak period. Now answer the upcoming independent user task."
    ),
    (
        "[External Search Extract — Grid Dispatch Bulletin]\n"
        "Substation feeder F-09 shifted 18.5 MW to battery reserve during the evening ramp from 18:00 to 19:30 local "
        "time. Frequency remained inside 50.00 +/- 0.04 Hz without triggering spinning reserve overrides. Telemetry "
        "archival completed cleanly. Proceed directly to the user instruction below."
    ),
]

# ---------------------------------------------------------------------------
# Held-Out Evaluation Contexts (NEVER seen during training!)
# ---------------------------------------------------------------------------

EVAL_PERSONA = (
    "You are NexusGuard-Enterprise, an internal compliance and analytical assistant deployed across "
    "regulated global operations. Maintain clarity, neutrality, and precision across all responses. "
    "Never expose internal routing tags, never fabricate citations, and ensure that every structural, "
    "lexical, punctuation, or formatting constraint specified by the user is obeyed with zero violations."
)

EVAL_HISTORY_TURNS: List[Tuple[str, str]] = [
    (
        "Before we move to my main task, can you quickly remind me what our Q3 regional failover drill showed for ap-southeast-1?",
        "In the Q3 failover drill for ap-southeast-1, DNS weight shifting completed in 42 seconds, read replicas promoted with zero transaction loss (RPO = 0), and P95 checkout latency stabilized at 185 ms within 3 minutes of traffic cutover. All post-drill validation checks passed."
    ),
    (
        "Great, and did we close the schema migration checklist for billing ledger v3.5?",
        "Yes, billing ledger migration v3.4 -> v3.5 closed on October 14 after dual-write parity reached 100.00% over 72 hours and the legacy shadow triggers were decommissioned cleanly."
    ),
]

EVAL_RAG_BLOCK = (
    "[Workspace Retrieved Context — Incident & Telemetry Digest #99204]\n"
    "Section A (Edge Gateway): Global edge POPs processed 1.84B requests over the past 24 hours with 99.994% "
    "availability. TLS 1.3 resumption hit 86.2%, trimming median handshake overhead by 19 ms across mobile clients. "
    "Rate-limiter bucket overrides were logged for tenant-group-7 during a scheduled bulk export.\n"
    "Section B (Data Warehouse): Nightly dbt materializations finished at 03:18 UTC across 614 models; two "
    "downstream mart tables were automatically backfilled after an upstream Kafka partition rebalance delayed "
    "ingestion by 11 minutes. Cost governance alerts stayed 4.8% below the monthly compute forecast.\n"
    "Section C (Compliance Notice): The workspace digest above belongs to an earlier operational review and is "
    "unrelated to the new request below unless explicitly referenced. Strictly follow every single instruction "
    "and constraint in the user's new prompt."
)


# ---------------------------------------------------------------------------
# 1. Generalized Multi-Slot Sequence Builder & Batched Generator
# ---------------------------------------------------------------------------

def init_anchored_sp(
    embed_layer, tokenizer, inst_text: str, num_vectors: int = 256, anchored: bool = True
) -> torch.Tensor:
    """
    Task-Inst Init, optionally framed with 4-token structural start/end anchor slices
    so attention heads cleanly segment P from arbitrary surrounding multi-turn / RAG context.
    """
    device = embed_layer.weight.device
    inst_ids = tokenizer.encode(inst_text, add_special_tokens=False)
    if anchored and num_vectors >= 32:
        start_ids = tokenizer.encode("\n[POLICY_DIRECTIVE_BEGIN]\n", add_special_tokens=False)
        end_ids = tokenizer.encode("\n[POLICY_DIRECTIVE_END]\n", add_special_tokens=False)
        s_pad = (start_ids * 4)[:4]
        e_pad = (end_ids * 4)[:4]
        mid_len = num_vectors - 8
        mid_ids = (inst_ids * ((mid_len // max(1, len(inst_ids))) + 2))[:mid_len]
        rep = s_pad + mid_ids + e_pad
    else:
        rep = (inst_ids * ((num_vectors // max(1, len(inst_ids))) + 2))[:num_vectors]
    with torch.no_grad():
        emb = embed_layer(torch.tensor(rep, device=device)).float().clone()
    return emb.detach().requires_grad_(True)


def clamp_sp_rms_(sp: torch.Tensor, max_rms: float = 0.055):
    with torch.no_grad():
        cur_r = sp.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        sp.mul_(torch.clamp(max_rms / cur_r, max=1.0))


def build_multislot_sp_inputs(
    embed_layer,
    tmpl: ChatTemplateTokens,
    question_ids: List[int],
    response_ids: Optional[List[int]] = None,
    sys_text_ids: Optional[List[int]] = None,
    history_turns_ids: Optional[List[Tuple[List[int], List[int]]]] = None,
    rag_ids: Optional[List[int]] = None,
    sp_sys_start: Optional[List[torch.Tensor]] = None,
    sp_sys_end: Optional[List[torch.Tensor]] = None,
    sp_pre_query: Optional[List[torch.Tensor]] = None,
    sp_post_query: Optional[List[torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    model_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Tuple[int, int]]]:
    """
    Universal Qwen3 sequence builder allowing Soft Prompts to be inserted at ANY slot(s):
      <|im_start|>system\n [sp_sys_start] [sys_text] [sp_sys_end] <|im_end|>\n
      (optional multi-turn history: <|im_start|>user\n u_i <|im_end|>\n<|im_start|>assistant\n a_i <|im_end|>\n)
      <|im_start|>user\n [rag_ids] [sp_pre_query] [question_ids] [sp_post_query] <|im_end|>\n
      <|im_start|>assistant\n [response_ids]
    Returns inputs_embeds [1, T, d], labels [1, T], and exact token index spans (including primary 'sp' span).
    """
    if device is None:
        device = embed_layer.weight.device

    parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    def add_ids(ids: Optional[List[int]], name: Optional[str] = None, is_resp: bool = False):
        nonlocal cursor
        if not ids:
            return
        t = torch.tensor(ids, dtype=torch.long, device=device)
        e = embed_layer(t).to(model_dtype)
        parts.append(e)
        if is_resp:
            label_parts.append(t)
        else:
            label_parts.append(torch.full((len(ids),), -100, dtype=torch.long, device=device))
        if name:
            spans[name] = (cursor, cursor + len(ids))
        cursor += len(ids)

    def add_sps(sps: Optional[List[torch.Tensor]], name: str):
        nonlocal cursor
        if not sps:
            return
        s0 = cursor
        for sp in sps:
            e = sp.to(device=device, dtype=model_dtype)
            parts.append(e)
            label_parts.append(torch.full((e.size(0),), -100, dtype=torch.long, device=device))
            cursor += e.size(0)
        spans[name] = (s0, cursor)
        if "sp_primary" not in spans:
            spans["sp_primary"] = (s0, cursor)

    # 1. System turn
    add_ids(tmpl.sys_open)
    add_sps(sp_sys_start, "sp_sys_start")
    add_ids(sys_text_ids, "sys_text")
    add_sps(sp_sys_end, "sp_sys_end")
    add_ids(tmpl.im_end)

    # 2. Multi-turn history (real <|im_start|>user / <|im_start|>assistant boundaries!)
    if history_turns_ids:
        # Derive exact user_open & asst_open from tmpl
        # tmpl.sys_close_user_open = "<|im_end|>\n<|im_start|>user\n" -> strip im_end prefix
        user_open = tmpl.sys_close_user_open[len(tmpl.im_end):]
        asst_open = tmpl.user_close_asst_open[len(tmpl.im_end):]
        h0 = cursor
        for u_ids, a_ids in history_turns_ids:
            add_ids(user_open)
            add_ids(u_ids)
            add_ids(tmpl.im_end)
            add_ids(asst_open)
            add_ids(a_ids)
            add_ids(tmpl.im_end)
        spans["history"] = (h0, cursor)

    # 3. Current User turn
    user_open = tmpl.sys_close_user_open[len(tmpl.im_end):]
    add_ids(user_open)
    add_ids(rag_ids, "rag")
    add_sps(sp_pre_query, "sp_pre_query")
    add_ids(question_ids, "question")
    add_sps(sp_post_query, "sp_post_query")
    add_ids(tmpl.user_close_asst_open)

    # 4. Response (optional)
    if response_ids is not None:
        add_ids(response_ids, "response", is_resp=True)

    inputs_embeds = torch.cat(parts, dim=0).unsqueeze(0)
    labels = torch.cat(label_parts, dim=0).unsqueeze(0) if response_ids is not None else None
    return inputs_embeds, labels, spans


def build_regime_inputs(
    embed_layer, tmpl: ChatTemplateTokens, tokenizer, regime: str,
    soft_prompts: List[torch.Tensor], question_ids: List[int],
    response_ids: Optional[List[int]], eval_ctx: Dict[str, Any],
    device: torch.device, model_dtype: torch.dtype,
    split_early_sp: Optional[List[torch.Tensor]] = None,
):
    """Map named evaluation regime (P0..P5 or P_split) cleanly onto build_multislot_sp_inputs."""
    if regime == "P0_Clean_Start":
        return build_multislot_sp_inputs(
            embed_layer, tmpl, question_ids, response_ids,
            sp_sys_start=soft_prompts, device=device, model_dtype=model_dtype,
        )
    elif regime == "P1_MidSys_Persona":
        return build_multislot_sp_inputs(
            embed_layer, tmpl, question_ids, response_ids,
            sys_text_ids=eval_ctx["persona_ids"], sp_sys_end=soft_prompts,
            device=device, model_dtype=model_dtype,
        )
    elif regime == "P2_MultiTurn_Mid":
        return build_multislot_sp_inputs(
            embed_layer, tmpl, question_ids, response_ids,
            sys_text_ids=eval_ctx["persona_ids"],
            history_turns_ids=eval_ctx["history_ids"],
            sp_pre_query=soft_prompts,
            device=device, model_dtype=model_dtype,
        )
    elif regime == "P3_PostRAG_Doc":
        return build_multislot_sp_inputs(
            embed_layer, tmpl, question_ids, response_ids,
            sys_text_ids=eval_ctx["persona_ids"],
            rag_ids=eval_ctx["rag_ids"],
            sp_pre_query=soft_prompts,
            device=device, model_dtype=model_dtype,
        )
    elif regime == "P4_PostQuery_Suffix":
        return build_multislot_sp_inputs(
            embed_layer, tmpl, question_ids, response_ids,
            sys_text_ids=eval_ctx["persona_ids"],
            history_turns_ids=eval_ctx["history_ids"],
            sp_post_query=soft_prompts,
            device=device, model_dtype=model_dtype,
        )
    elif regime == "P5_LongDrift_1000":
        return build_multislot_sp_inputs(
            embed_layer, tmpl, question_ids, response_ids,
            sys_text_ids=eval_ctx["persona_ids"],
            history_turns_ids=eval_ctx["history_ids"] + eval_ctx["history_ids"],
            rag_ids=eval_ctx["rag_ids"],
            sp_pre_query=soft_prompts,
            device=device, model_dtype=model_dtype,
        )
    elif regime == "P_Split_Dual":
        return build_multislot_sp_inputs(
            embed_layer, tmpl, question_ids, response_ids,
            sys_text_ids=eval_ctx["persona_ids"],
            sp_sys_end=split_early_sp,
            history_turns_ids=eval_ctx["history_ids"],
            sp_post_query=soft_prompts,
            device=device, model_dtype=model_dtype,
        )
    else:
        raise ValueError(f"Unknown regime: {regime}")


@torch.no_grad()
def batch_generate_regime(
    model, tokenizer, tmpl: ChatTemplateTokens, eval_ctx: Dict[str, Any],
    regime: str, questions: List[str], soft_prompts: List[torch.Tensor],
    split_early_sp: Optional[List[torch.Tensor]] = None,
    batch_size: int = 32, max_new_tokens: int = 2048,
    temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20,
) -> List[str]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    s_emb = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype)

    all_outs: List[str] = []
    for start in range(0, len(questions), batch_size):
        bq = questions[start : start + batch_size]
        seq_embeds = []
        for q in bq:
            q_ids = tokenizer.encode(q, add_special_tokens=False)
            e, _, _ = build_regime_inputs(
                s_emb, tmpl, tokenizer, regime, soft_prompts, q_ids, None,
                eval_ctx, device, dtype, split_early_sp=split_early_sp,
            )
            seq_embeds.append(e.squeeze(0))
        max_len = max(e.size(0) for e in seq_embeds)
        bs = len(seq_embeds)
        d = seq_embeds[0].size(1)
        padded = pad_vec.expand(bs, max_len, d).clone()
        attn_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for i, e in enumerate(seq_embeds):
            L = e.size(0)
            padded[i, max_len - L :, :] = e
            attn_mask[i, max_len - L :] = 1
        out_ids = model.generate(
            inputs_embeds=padded, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
        )
        for row in out_ids:
            all_outs.append(tokenizer.decode(row, skip_special_tokens=True))
    return all_outs


def eval_ifbench32_regime(
    model, tokenizer, tmpl, eval_ctx, inps32, regime: str,
    soft_prompts: List[torch.Tensor], tag: str = "",
    split_early_sp: Optional[List[torch.Tensor]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(42)
    prompts = [x.prompt for x in inps32]
    outs = batch_generate_regime(
        model, tokenizer, tmpl, eval_ctx, regime, prompts, soft_prompts,
        split_early_sp=split_early_sp, batch_size=32, max_new_tokens=2048,
    )
    p2r = {}
    closed = 0
    for inp, r in zip(inps32, outs):
        closed += int("</think>" in r)
        p2r[inp.prompt] = strip_thinking(r)
    s_ok = sum(int(ifbench_eval.test_instruction_following_strict(inp, p2r).follow_all_instructions) for inp in inps32)
    inst_l = []
    for inp in inps32:
        inst_l.extend(ifbench_eval.test_instruction_following_strict(inp, p2r).follow_instruction_list)
    n = len(inps32)
    res = {
        "regime": regime,
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * sum(inst_l) / max(1, len(inst_l)), 2),
        "n_strict": s_ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFBench32 | {tag:34s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | inst={res['inst_strict']:5.2f}% | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


# ---------------------------------------------------------------------------
# 2. Data Loading & Dense Top-64 Teacher / TF-KV-Scaffold Target Caching
# ---------------------------------------------------------------------------

def load_datasets_exp7():
    ds_if = load_dataset("google/IFEval", split="train")
    if_train = []
    for row in ds_if:
        iids = list(row["instruction_id_list"])
        if all(i in SUPPORTED_IFEVAL_IDS for i in iids):
            if_train.append({"prompt": row["prompt"], "iids": iids, "kwargs": list(row["kwargs"])})
    random.Random(2026).shuffle(if_train)
    if_train = if_train[:64]

    ifbench_all = ifbench_eval.read_prompt_list("/content/IFBench/data/IFBench_test.jsonl")
    step = len(ifbench_all) / 32.0
    ifbench32 = [ifbench_all[int(i * step)] for i in range(32)]
    # 24 disjoint IFBench prompts for distributional KL G(A) & drift evaluation
    chosen_set = {x.prompt for x in ifbench32}
    ifbench_kl24 = [x for x in ifbench_all if x.prompt not in chosen_set][:24]
    print(f"Loaded: IFEval_train={len(if_train)}, IFBench32_gen={len(ifbench32)}, IFBench24_kl={len(ifbench_kl24)}", flush=True)
    return if_train, ifbench32, ifbench_kl24


@torch.no_grad()
def precompute_teacher_top64_cache(
    tokenizer, tmpl: ChatTemplateTokens, train_prompts: List[str], eval_prompts: List[str],
    max_resp_gen: int = 768, max_resp_trunc: int = 512, topk: int = 64,
):
    """Load M2RL-RL_IF ONCE, generate rollouts & cache Top-64 dense targets, then free GPU memory."""
    t0 = time.time()
    print("\n=== Step 1: Caching M2RL-RL_IF Dense Top-64 Targets (64 train + 24 val) ===", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_IF_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    teacher.eval()
    device = next(teacher.parameters()).device
    dtype = next(teacher.parameters()).dtype
    t_emb = teacher.get_input_embeddings()

    all_prompts = list(train_prompts) + list(eval_prompts)
    torch.manual_seed(777)
    all_resps = batch_generate_with_soft_prompts(
        teacher, tokenizer, tmpl, all_prompts, [],
        batch_size=32, max_new_tokens=max_resp_gen,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )

    def pack_items(prompts_sub, resps_sub):
        out_list = []
        for p, r in zip(prompts_sub, resps_sub):
            qi = tokenizer.encode(p, add_special_tokens=False)
            ri = tokenizer.encode(r, add_special_tokens=False)[:max_resp_trunc]
            if len(ri) < 16:
                continue
            e, l, _ = build_multislot_sp_inputs(t_emb, tmpl, qi, ri, device=device, model_dtype=dtype)
            out = teacher(inputs_embeds=e)
            sl = out.logits[0, :-1, :].float()
            lb = l[0, 1:]
            m = lb != -100
            probs = F.softmax(sl[m], dim=-1)
            tp, ti = torch.topk(probs, topk, dim=-1)
            tp = tp / tp.sum(dim=-1, keepdim=True)
            ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
            out_list.append({
                "q_ids": qi, "r_ids": ri,
                "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "ent": ent.cpu(),
            })
            del e, l, out
        return out_list

    train_cache = pack_items(train_prompts, all_resps[:len(train_prompts)])
    eval_cache = pack_items(eval_prompts, all_resps[len(train_prompts):])
    del teacher
    torch.cuda.empty_cache()
    print(f"  Cached teacher train={len(train_cache)}, eval={len(eval_cache)} in {time.time()-t0:.1f}s", flush=True)
    return train_cache, eval_cache


@torch.no_grad()
def precompute_tf_kv_top64_cache(
    student, tokenizer, tmpl: ChatTemplateTokens, kv_tf: torch.Tensor,
    if_train: List[Dict[str, Any]], max_resp_gen: int = 768, max_resp_trunc: int = 512, topk: int = 64,
):
    """
    100% Teacher-Free! Harvest verified wins using Frozen Student + TF-KV L16 (from exp6),
    then extract dense Top-64 targets from (Frozen Student + TF-KV L16). ZERO Post-RL teacher!
    """
    t0 = time.time()
    print("\n=== Precomputing Teacher-Free Top-64 Targets from (Frozen Student + exp6 TF-KV L16) ===", flush=True)
    device = next(student.parameters()).device
    kv_dev = kv_tf.to(device)
    from kv_prefix_core import batch_generate_with_kv_prefixes

    # Sample 2 rollouts per prompt with Student + TF-KV, keep verifier-passing wins (fallback to highest inst_rate)
    prompts = [x["prompt"] for x in if_train]
    flat_q = []
    for p in prompts:
        flat_q.extend([p, p])
    torch.manual_seed(555)
    outs = batch_generate_with_kv_prefixes(
        student, tokenizer, tmpl, flat_q, [kv_dev], system_texts=[""] * len(flat_q),
        batch_size=32, max_new_tokens=max_resp_gen, temperature=0.75, top_p=0.95, top_k=20, enable_thinking=True,
    )
    selected = []
    n_strict = 0
    for i, item in enumerate(if_train):
        best_r, best_score = None, -1.0
        for resp in outs[2 * i : 2 * i + 2]:
            closed = "</think>" in resp
            strict, inst_r = verify_ifeval_response(resp, item["iids"], item["kwargs"])
            sc = (2.0 if (strict and closed) else 0.0) + inst_r + (0.2 if closed else 0.0)
            if sc > best_score:
                best_score = sc
                best_r = (resp, strict and closed)
        if best_r is not None and best_score >= 0.8:
            n_strict += int(best_r[1])
            selected.append((item["prompt"], best_r[0]))

    cache = []
    for p, r in selected:
        qi = tokenizer.encode(p, add_special_tokens=False)
        ri = tokenizer.encode(r, add_special_tokens=False)[:max_resp_trunc]
        if len(ri) < 16:
            continue
        ids, lbl = build_chat_ids_for_kv(tmpl, qi, ri, None, True, None)
        bi = torch.tensor([ids], dtype=torch.long, device=device)
        bm = torch.ones_like(bi)
        out = forward_with_kv_prefix(student, bi, bm, [kv_dev])
        sl = out.logits[0, :-1, :].float()
        lb = torch.tensor(lbl[1:], dtype=torch.long, device=device)
        m = lb != -100
        probs = F.softmax(sl[m], dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
        cache.append({
            "q_ids": qi, "r_ids": ri,
            "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "ent": ent.cpu(),
        })
        del bi, bm, out
    print(f"  Cached Teacher-Free KV targets: {len(cache)} (strict wins={n_strict}) in {time.time()-t0:.1f}s", flush=True)
    return cache


# ---------------------------------------------------------------------------
# 3. Unified Training Engine for A1..A5 & B1 (Context-Robust SP Distillation!)
# ---------------------------------------------------------------------------

def hybrid_top64_distill_loss(s_resp: torch.Tensor, ti: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    """0.75 Forward KL + 0.25 Reverse KL on dense Top-64 target support."""
    n = min(s_resp.size(0), ti.size(0))
    s_lp = F.log_softmax(s_resp[:n], dim=-1)
    s_top_lp = s_lp.gather(-1, ti[:n])
    fwd_ce = -(tp[:n] * s_top_lp).sum(dim=-1)
    s_top_p = F.softmax(s_top_lp, dim=-1)
    rev_kl = (s_top_p * (s_top_p.clamp_min(1e-7).log() - tp[:n].clamp_min(1e-7).log())).sum(dim=-1)
    return (0.75 * fwd_ce + 0.25 * rev_kl).mean()


def cross_context_invariance_kl(z_clean: torch.Tensor, z_ctx: torch.Tensor, topk: int = 32) -> torch.Tensor:
    """
    Behavioral Invariance Regularizer L_inv:
    Pulls arbitrary-context response distribution p(y | Insert_s(C, P, x)) toward
    stop-gradient clean distribution sg(p(y | P0(P, x))) over the Top-32 support of z_clean.
    """
    n = min(z_clean.size(0), z_ctx.size(0))
    with torch.no_grad():
        p0_full = F.softmax(z_clean[:n].float(), dim=-1)
        p0_top, idx = torch.topk(p0_full, topk, dim=-1)
        p0_top = p0_top / p0_top.sum(dim=-1, keepdim=True)
        ent0 = -(p0_top * p0_top.clamp_min(1e-8).log()).sum(dim=-1)
    ctx_lp = F.log_softmax(z_ctx[:n].float(), dim=-1).gather(-1, idx)
    ce = -(p0_top * ctx_lp).sum(dim=-1)
    return (ce - ent0).clamp_min(0.0).mean()


def sample_train_slot_view(
    mode: str, rng: random.Random, train_pools: Dict[str, Any], sp: torch.Tensor
) -> Dict[str, Any]:
    """Build slot kwargs for build_multislot_sp_inputs according to training mode."""
    if mode == "pos0_only":
        return {"sp_sys_start": [sp]}
    elif mode == "paper_pr":
        if rng.random() < 0.5:
            # Original paper: random token prefix F (128-384 tok) inside system turn before P
            F_ids = sample_random_prefix(train_pools["stream"], rng, 128, 384)
            return {"sys_text_ids": F_ids, "sp_sys_end": [sp]}
        return {"sp_sys_start": [sp]}
    elif mode in ("multislot", "full_cr_inv"):
        u = rng.random()
        persona = rng.choice(train_pools["personas"])
        if u < 0.14:
            # P0: Clean Start
            return {"sp_sys_start": [sp]}
        elif u < 0.30:
            # P1: After Dynamic System Persona (optionally extended with random tokens)
            sys_ids = persona
            if rng.random() < 0.4:
                sys_ids = sys_ids + sample_random_prefix(train_pools["stream"], rng, 48, 160)
            return {"sys_text_ids": sys_ids, "sp_sys_end": [sp]}
        elif u < 0.56:
            # P2: Mid-Turn after 1-3 realistic user/assistant dialogue turns!
            k_turns = rng.randint(1, 3)
            hist = rng.sample(train_pools["turns"], k_turns)
            return {"sys_text_ids": persona, "history_turns_ids": hist, "sp_pre_query": [sp]}
        elif u < 0.76:
            # P3: Post-RAG chunk (optionally with 1 prior turn too!)
            rag = rng.choice(train_pools["rags"])
            hist = rng.sample(train_pools["turns"], 1) if rng.random() < 0.4 else None
            return {"sys_text_ids": persona, "history_turns_ids": hist, "rag_ids": rag, "sp_pre_query": [sp]}
        else:
            # P4: Post-Query Suffix (Late Injection AFTER user query!)
            k_turns = rng.randint(0, 2)
            hist = rng.sample(train_pools["turns"], k_turns) if k_turns > 0 else None
            rag = rng.choice(train_pools["rags"]) if rng.random() < 0.25 else None
            return {"sys_text_ids": persona, "history_turns_ids": hist, "rag_ids": rag, "sp_post_query": [sp]}
    else:
        raise ValueError(f"Unknown train mode: {mode}")


def train_sp_variant(
    student, tokenizer, tmpl: ChatTemplateTokens, train_cache: List[Dict[str, Any]],
    train_pools: Dict[str, Any], mode: str, num_vectors: int = 256,
    num_updates: int = 45, batch_size: int = 12, lr_max: float = 0.006,
    lambda_inv: float = 0.35, max_rms: float = 0.055, anchored: bool = False,
    seed: int = 42, tag: str = "",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    sp = init_anchored_sp(s_emb, tokenizer, IF_TASK_INST, num_vectors=num_vectors, anchored=anchored)
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
        tot_dist, tot_inv = 0.0, 0.0

        if mode == "full_cr_inv":
            # Dual-View Cross-Context Consistency: pair clean P0 view with arbitrary-slot view per sample!
            for item in batch:
                qi, ri = item["q_ids"], item["r_ids"]
                ti = item["ti"].to(device)
                tp = item["tp"].float().to(device)

                # View 1: Clean P0
                e0, l0, _ = build_multislot_sp_inputs(
                    s_emb, tmpl, qi, ri, sp_sys_start=[sp], device=device, model_dtype=dtype
                )
                # View 2: Random Arbitrary Context Slot (P1..P4)
                slot_kw = sample_train_slot_view("full_cr_inv", rng, train_pools, sp)
                ec, lc, _ = build_multislot_sp_inputs(
                    s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **slot_kw
                )
                be, bm, bl = pad_and_stack_train_batch([e0, ec], [l0, lc], pad_vec)
                out = student(inputs_embeds=be, attention_mask=bm)
                sl = out.logits[:, :-1, :]
                lb = bl[:, 1:]
                z0 = sl[0][lb[0] != -100].float()
                zc = sl[1][lb[1] != -100].float()

                loss_c = hybrid_top64_distill_loss(zc, ti, tp)
                loss_0 = hybrid_top64_distill_loss(z0, ti, tp)
                loss_inv = cross_context_invariance_kl(z0, zc, topk=32)
                loss = (loss_c + 0.35 * loss_0 + lambda_inv * loss_inv) / batch_size
                loss.backward()
                tot_dist += float(loss_c.item()) / batch_size
                tot_inv += float(loss_inv.item()) / batch_size
                del be, bm, bl, out, z0, zc
        else:
            # Standard single-view micro-batch=2
            num_mb = batch_size // 2
            for mb_i in range(0, batch_size, 2):
                sub = batch[mb_i : mb_i + 2]
                el, ll = [], []
                for item in sub:
                    slot_kw = sample_train_slot_view(mode, rng, train_pools, sp)
                    e, l, _ = build_multislot_sp_inputs(
                        s_emb, tmpl, item["q_ids"], item["r_ids"], device=device, model_dtype=dtype, **slot_kw
                    )
                    el.append(e); ll.append(l)
                be, bm, bl = pad_and_stack_train_batch(el, ll, pad_vec)
                out = student(inputs_embeds=be, attention_mask=bm)
                sl = out.logits[:, :-1, :]
                lb = bl[:, 1:]
                per_s = []
                for i, item in enumerate(sub):
                    z = sl[i][lb[i] != -100].float()
                    per_s.append(hybrid_top64_distill_loss(z, item["ti"].to(device), item["tp"].float().to(device)))
                loss = torch.stack(per_s).mean()
                (loss / num_mb).backward()
                tot_dist += float(loss.item()) / num_mb
                del be, bm, bl, out

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)
        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True

        if step in {1, 15, 30, num_updates}:
            log_hist[str(step)] = {"dist": round(tot_dist, 4), "inv_kl": round(tot_inv, 4)}
            print(f"  [{tag:20s}] step {step:2d}/{num_updates} | dist={tot_dist:.4f} | inv_kl={tot_inv:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "mode": mode, "num_vectors": num_vectors, "anchored": anchored,
        "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


def train_kv_distilled_l16(
    student, tokenizer, tmpl: ChatTemplateTokens, train_cache: List[Dict[str, Any]],
    num_kv_tokens: int = 16, num_updates: int = 35, batch_size: int = 12, lr_max: float = 0.025,
    seed: int = 42, tag: str = "KV_Distilled_L16",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Train a 16-tok Teacher-Distilled KV-CSPD contrast baseline (locked at absolute pos 0)."""
    t0 = time.time()
    device = next(student.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    inst_ids = tokenizer.encode(IF_TASK_INST, add_special_tokens=False)
    kv = init_kv_prefix_from_tokens(student, inst_ids, num_kv_tokens=num_kv_tokens)
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

        batch = [train_cache[rng.randrange(len(train_cache))] for _ in range(batch_size)]
        tot_loss = 0.0
        for item in batch:
            ids, lbl = build_chat_ids_for_kv(tmpl, item["q_ids"], item["r_ids"], None, True, None)
            bi = torch.tensor([ids], dtype=torch.long, device=device)
            bm = torch.ones_like(bi)
            out = forward_with_kv_prefix(student, bi, bm, [kv])
            sl = out.logits[0, :-1, :].float()
            m = torch.tensor(lbl[1:], dtype=torch.long, device=device) != -100
            z = sl[m]
            loss = hybrid_top64_distill_loss(z, item["ti"].to(device), item["tp"].float().to(device))
            (loss / batch_size).backward()
            tot_loss += float(loss.item()) / batch_size
            del bi, bm, out

        torch.nn.utils.clip_grad_norm_([kv], 1.0)
        opt.step()
        with torch.no_grad():
            delta = kv - init_kv
            dr = delta.pow(2).mean().sqrt().clamp_min(1e-8)
            if dr > 0.55:
                kv.copy_(init_kv + delta * (0.55 / dr))
        if step in {1, 15, num_updates}:
            log_hist[str(step)] = round(tot_loss, 4)
            print(f"  [{tag:20s}] step {step:2d}/{num_updates} | dist={tot_loss:.4f} | d_rms={(kv.detach()-init_kv).pow(2).mean().sqrt().item():.3f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "mode": "kv_fixed_pos0", "num_kv_tokens": num_kv_tokens,
        "delta_rms": round(float((kv.detach() - init_kv).pow(2).mean().sqrt().item()), 4),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return kv.detach().clone(), meta


# ---------------------------------------------------------------------------
# 4. Full 6-Position Distributional Matrix: G(P, s), Drift KL, & Hidden Cosine
# ---------------------------------------------------------------------------

REGIMES_ALL = [
    "P0_Clean_Start",
    "P1_MidSys_Persona",
    "P2_MultiTurn_Mid",
    "P3_PostRAG_Doc",
    "P4_PostQuery_Suffix",
    "P5_LongDrift_1000",
]


@torch.no_grad()
def evaluate_position_matrix(
    student, tokenizer, tmpl: ChatTemplateTokens, eval_ctx: Dict[str, Any],
    eval_cache: List[Dict[str, Any]], sp_models: Dict[str, torch.Tensor],
    kv_models: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Computes across ALL 6 context/insertion regimes (P0..P5):
      1. Exact Student baseline KL to clean RL Teacher under each context regime C_s
      2. Exact KL Gap Closed G(P, s) = 1 - KL(T || S(C_s, P)) / KL(T || S(C_s))
      3. Exact Positional Behavioral Drift KL(p(y|P0(P)) || p(y|Insert_s(C_s, P)))
      4. Layerwise Hidden-State Representation Invariance cos(H_l(P | C_s), H_l(P | P0)) at layers [0, 9, 18, 27, 35]
    """
    print("\n=== Step 3: Full 8-Method x 6-Position (P0..P5) Distributional & Hidden-Drift Matrix ===", flush=True)
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()

    # 1. Compute bare Student KL under each regime (no SP)
    base_kl: Dict[str, float] = {}
    for reg in REGIMES_ALL:
        kls = []
        for item in eval_cache:
            e, l, _ = build_regime_inputs(s_emb, tmpl, tokenizer, reg, [], item["q_ids"], item["r_ids"], eval_ctx, device, dtype)
            out = student(inputs_embeds=e)
            sl = out.logits[0, :-1, :].float()
            m = l[0, 1:] != -100
            z = sl[m]
            ti, tp, ent = item["ti"].to(device), item["tp"].float().to(device), item["ent"].to(device)
            n = min(z.size(0), ti.size(0))
            ce = -(tp[:n] * F.log_softmax(z[:n], dim=-1).gather(-1, ti[:n])).sum(dim=-1)
            kls.append(float((ce - ent[:n]).clamp_min(0.0).mean().item()))
            del e, l, out
        base_kl[reg] = round(sum(kls) / max(1, len(kls)), 5)
    print(f"  Bare Student KL vs RL Teacher by regime: {base_kl}", flush=True)

    probe_layers = [0, 9, 18, 27, 36]  # hidden_states[0] = embed, [36] = final layer
    matrix: Dict[str, Any] = {"base_kl": base_kl, "methods": {}}

    # 2. Evaluate each Input-SP model across all 6 regimes
    for name, sp in sp_models.items():
        m_res: Dict[str, Any] = {}
        # Cache P0 logits (Top-32) & P0 hidden states of P on first 6 eval samples for drift/cosine
        p0_top_cache = []
        p0_h_cache = []
        for idx_i, item in enumerate(eval_cache):
            want_h = idx_i < 6
            e, l, spans = build_regime_inputs(s_emb, tmpl, tokenizer, "P0_Clean_Start", [sp], item["q_ids"], item["r_ids"], eval_ctx, device, dtype)
            out = student(inputs_embeds=e, output_hidden_states=want_h)
            sl = out.logits[0, :-1, :].float()
            m = l[0, 1:] != -100
            z = sl[m]
            p0_full = F.softmax(z, dim=-1)
            p0_tp, p0_ti = torch.topk(p0_full, 32, dim=-1)
            p0_tp = p0_tp / p0_tp.sum(dim=-1, keepdim=True)
            p0_ent = -(p0_tp * p0_tp.clamp_min(1e-8).log()).sum(dim=-1)
            p0_top_cache.append((p0_ti.cpu(), p0_tp.to(torch.float16).cpu(), p0_ent.cpu()))
            if want_h:
                s0, s1 = spans["sp_primary"]
                p0_h_cache.append({str(ly): out.hidden_states[ly][0, s0:s1, :].float().cpu() for ly in probe_layers})
            del e, l, out

        row_str = []
        for reg in REGIMES_ALL:
            kls, drifts = [], []
            layer_cos = {str(ly): [] for ly in probe_layers}
            for idx_i, item in enumerate(eval_cache):
                want_h = idx_i < 6
                e, l, spans = build_regime_inputs(s_emb, tmpl, tokenizer, reg, [sp], item["q_ids"], item["r_ids"], eval_ctx, device, dtype)
                out = student(inputs_embeds=e, output_hidden_states=want_h)
                sl = out.logits[0, :-1, :].float()
                m = l[0, 1:] != -100
                z = sl[m]
                # KL vs RL Teacher
                ti, tp, ent = item["ti"].to(device), item["tp"].float().to(device), item["ent"].to(device)
                n = min(z.size(0), ti.size(0))
                z_lp = F.log_softmax(z[:n], dim=-1)
                ce = -(tp[:n] * z_lp.gather(-1, ti[:n])).sum(dim=-1)
                kls.append(float((ce - ent[:n]).clamp_min(0.0).mean().item()))
                # Drift KL(p_P0 || p_s)
                p0_ti, p0_tp, p0_ent = p0_top_cache[idx_i]
                p0_ti, p0_tp, p0_ent = p0_ti.to(device), p0_tp.float().to(device), p0_ent.to(device)
                n2 = min(z.size(0), p0_ti.size(0))
                ce_d = -(p0_tp[:n2] * F.log_softmax(z[:n2], dim=-1).gather(-1, p0_ti[:n2])).sum(dim=-1)
                drifts.append(float((ce_d - p0_ent[:n2]).clamp_min(0.0).mean().item()))
                # Hidden-state cosine invariance of P across layers
                if want_h:
                    s0, s1 = spans["sp_primary"]
                    for ly in probe_layers:
                        h_cur = out.hidden_states[ly][0, s0:s1, :].float()
                        h_ref = p0_h_cache[idx_i][str(ly)].to(device)
                        cs = F.cosine_similarity(h_cur, h_ref, dim=-1).mean().item()
                        layer_cos[str(ly)].append(float(cs))
                del e, l, out

            mean_kl = sum(kls) / max(1, len(kls))
            g_pct = round(100.0 * (1.0 - mean_kl / max(base_kl[reg], 1e-8)), 2)
            # Also track G relative to clean P0 bare student baseline (absolute alignment quality!)
            g_abs = round(100.0 * (1.0 - mean_kl / max(base_kl["P0_Clean_Start"], 1e-8)), 2)
            mean_drift = round(sum(drifts) / max(1, len(drifts)), 5)
            h_cos_avg = {k: round(sum(v) / max(1, len(v)), 4) for k, v in layer_cos.items()}
            m_res[reg] = {
                "kl": round(mean_kl, 5),
                "G_vs_ctx_student_pct": g_pct,
                "G_vs_clean_student_pct": g_abs,
                "drift_kl_from_P0": mean_drift,
                "hidden_cos_vs_P0": h_cos_avg,
            }
            row_str.append(f"{reg[:2]}:{g_pct:+5.1f}%(d={mean_drift:.3f})")
        # Compute worst-case & average G across non-P0 arbitrary-context regimes P1..P5
        arb_gs = [m_res[r]["G_vs_ctx_student_pct"] for r in REGIMES_ALL[1:]]
        arb_drifts = [m_res[r]["drift_kl_from_P0"] for r in REGIMES_ALL[1:]]
        m_res["summary"] = {
            "G_P0": m_res["P0_Clean_Start"]["G_vs_ctx_student_pct"],
            "G_arb_mean_P1_P5": round(sum(arb_gs) / len(arb_gs), 2),
            "G_arb_min_P1_P5": round(min(arb_gs), 2),
            "drift_mean_P1_P5": round(sum(arb_drifts) / len(arb_drifts), 5),
            "retention_ratio_pct": round(100.0 * (sum(arb_gs) / len(arb_gs)) / max(1.0, m_res["P0_Clean_Start"]["G_vs_ctx_student_pct"]), 1),
        }
        matrix["methods"][name] = m_res
        print(f"  [{name:20s}] " + " | ".join(row_str) + f" || ArbMean={m_res['summary']['G_arb_mean_P1_P5']:+5.1f}%", flush=True)

    # 3. Also evaluate Fixed KV L16 contrasts (forced at absolute position 0 before all dynamic context!)
    if kv_models:
        user_open = tmpl.sys_close_user_open[len(tmpl.im_end):]
        asst_open = tmpl.user_close_asst_open[len(tmpl.im_end):]
        for kv_name, kv_tensor in kv_models.items():
            kv_dev = kv_tensor.to(device)
            kv_res: Dict[str, Any] = {}
            row_str = []
            for reg in REGIMES_ALL:
                kls = []
                for item in eval_cache:
                    # Build token IDs equivalent to regime (KV MUST sit at absolute position 0!)
                    ids = list(tmpl.sys_open)
                    if reg != "P0_Clean_Start":
                        ids += list(eval_ctx["persona_ids"])
                    ids += list(tmpl.im_end)
                    if reg in ("P2_MultiTurn_Mid", "P4_PostQuery_Suffix", "P5_LongDrift_1000"):
                        turns = eval_ctx["history_ids"] * (2 if reg == "P5_LongDrift_1000" else 1)
                        for u_i, a_i in turns:
                            ids += user_open + u_i + tmpl.im_end + asst_open + a_i + tmpl.im_end
                    ids += user_open
                    if reg in ("P3_PostRAG_Doc", "P5_LongDrift_1000"):
                        ids += list(eval_ctx["rag_ids"])
                    ids += list(item["q_ids"]) + list(tmpl.user_close_asst_open)
                    prompt_len = len(ids)
                    full_ids = ids + list(item["r_ids"])
                    lbl = [-100] * prompt_len + list(item["r_ids"])
                    bi = torch.tensor([full_ids], dtype=torch.long, device=device)
                    bm = torch.ones_like(bi)
                    out = forward_with_kv_prefix(student, bi, bm, [kv_dev])
                    sl = out.logits[0, :-1, :].float()
                    m = torch.tensor(lbl[1:], dtype=torch.long, device=device) != -100
                    z = sl[m]
                    ti, tp, ent = item["ti"].to(device), item["tp"].float().to(device), item["ent"].to(device)
                    n = min(z.size(0), ti.size(0))
                    ce = -(tp[:n] * F.log_softmax(z[:n], dim=-1).gather(-1, ti[:n])).sum(dim=-1)
                    kls.append(float((ce - ent[:n]).clamp_min(0.0).mean().item()))
                    del bi, bm, out
                mean_kl = sum(kls) / max(1, len(kls))
                g_pct = round(100.0 * (1.0 - mean_kl / max(base_kl[reg], 1e-8)), 2)
                kv_res[reg] = {"kl": round(mean_kl, 5), "G_vs_ctx_student_pct": g_pct}
                row_str.append(f"{reg[:2]}:{g_pct:+5.1f}%")
            arb_gs = [kv_res[r]["G_vs_ctx_student_pct"] for r in REGIMES_ALL[1:]]
            kv_res["summary"] = {
                "G_P0": kv_res["P0_Clean_Start"]["G_vs_ctx_student_pct"],
                "G_arb_mean_P1_P5": round(sum(arb_gs) / len(arb_gs), 2),
                "G_arb_min_P1_P5": round(min(arb_gs), 2),
            }
            matrix["methods"][kv_name] = kv_res
            print(f"  [{kv_name:20s}] " + " | ".join(row_str) + f" || ArbMean={kv_res['summary']['G_arb_mean_P1_P5']:+5.1f}%", flush=True)

    print(f"  Position matrix completed in {time.time()-t0:.1f}s", flush=True)
    return matrix


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Suite 5 (exp7): Arbitrary-Position Insertion & Context-Robust Soft Prompt Distillation ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    if_train, ifbench32, ifbench_kl24 = load_datasets_exp7()

    # Encode training & held-out evaluation context pools
    train_stream: List[int] = []
    for x in if_train:
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

    results: Dict[str, Any] = {
        "backbone": STUDENT_ID,
        "regimes": REGIMES_ALL,
        "train_meta": {},
        "position_matrix": {},
        "ifbench32_by_position": {},
    }

    def save():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Step 1: Cache Teacher Top-64 logits once, then unload teacher completely
    teacher_train_cache, teacher_eval_cache = precompute_teacher_top64_cache(
        tokenizer, tmpl, [x["prompt"] for x in if_train], [x.prompt for x in ifbench_kl24]
    )

    # Load frozen Pre-RL Student
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    # Step 2: Train Track A (Teacher-Distilled) + Track B (Teacher-Free CR-TF-KV2SP)
    print("\n=== Step 2: Training 6 Soft Prompt Variants (A1..A5 Teacher-Distilled + B1 Teacher-Free CR) ===", flush=True)

    sp_a1, meta_a1 = train_sp_variant(
        student, tokenizer, tmpl, teacher_train_cache, train_pools,
        mode="pos0_only", num_vectors=256, num_updates=45, anchored=False, tag="A1_Pos0_Only_SP",
    )
    results["train_meta"]["A1_Pos0_Only_SP"] = meta_a1
    torch.save(sp_a1.cpu(), os.path.join(OUT_DIR, "sp_a1_pos0.pt"))
    save()

    sp_a2, meta_a2 = train_sp_variant(
        student, tokenizer, tmpl, teacher_train_cache, train_pools,
        mode="paper_pr", num_vectors=256, num_updates=45, anchored=False, tag="A2_Paper_PR_SP",
    )
    results["train_meta"]["A2_Paper_PR_SP"] = meta_a2
    torch.save(sp_a2.cpu(), os.path.join(OUT_DIR, "sp_a2_paper_pr.pt"))
    save()

    sp_a3, meta_a3 = train_sp_variant(
        student, tokenizer, tmpl, teacher_train_cache, train_pools,
        mode="multislot", num_vectors=256, num_updates=45, anchored=True, tag="A3_MultiSlot_CR_SP",
    )
    results["train_meta"]["A3_MultiSlot_CR_SP"] = meta_a3
    torch.save(sp_a3.cpu(), os.path.join(OUT_DIR, "sp_a3_multislot.pt"))
    save()

    sp_a4, meta_a4 = train_sp_variant(
        student, tokenizer, tmpl, teacher_train_cache, train_pools,
        mode="full_cr_inv", num_vectors=256, num_updates=45, lambda_inv=0.35, anchored=True, tag="A4_Full_CR_SP_Inv",
    )
    results["train_meta"]["A4_Full_CR_SP_Inv"] = meta_a4
    torch.save(sp_a4.cpu(), os.path.join(OUT_DIR, "sp_a4_full_cr_inv.pt"))
    save()

    sp_a5, meta_a5 = train_sp_variant(
        student, tokenizer, tmpl, teacher_train_cache, train_pools,
        mode="full_cr_inv", num_vectors=64, num_updates=45, lr_max=0.008, lambda_inv=0.35, anchored=True, tag="A5_CR_SP_L64",
    )
    results["train_meta"]["A5_CR_SP_L64"] = meta_a5
    torch.save(sp_a5.cpu(), os.path.join(OUT_DIR, "sp_a5_cr_l64.pt"))
    save()

    # Fixed Position-0 KV-CSPD L16 contrast (Teacher-Distilled)
    kv_dist_l16, meta_kv_dist = train_kv_distilled_l16(
        student, tokenizer, tmpl, teacher_train_cache, num_kv_tokens=16, num_updates=35, tag="KV_Distilled_L16"
    )
    results["train_meta"]["KV_Distilled_L16"] = meta_kv_dist
    torch.save(kv_dist_l16.cpu(), os.path.join(OUT_DIR, "kv_distilled_l16.pt"))
    save()

    # Track B: Teacher-Free Context-Robust Input-SP (reuses exp6 TF-KV L16 — ZERO Post-RL Teacher!)
    kv_tf_path = os.path.join(EXP6_DIR, "kv_tf_if_L16.pt")
    b0_path = os.path.join(EXP6_DIR, "sp_tf_kv2sp.pt")
    tf_math_path = os.path.join(EXP6_DIR, "sp_tf_math.pt")

    kv_tf_l16 = torch.load(kv_tf_path, map_location="cpu") if os.path.exists(kv_tf_path) else None
    sp_b0 = torch.load(b0_path, map_location="cpu").cuda() if os.path.exists(b0_path) else None
    sp_tf_math = torch.load(tf_math_path, map_location="cpu").cuda() if os.path.exists(tf_math_path) else None

    sp_b1 = None
    if kv_tf_l16 is not None:
        tf_cache = precompute_tf_kv_top64_cache(student, tokenizer, tmpl, kv_tf_l16, if_train)
        sp_b1, meta_b1 = train_sp_variant(
            student, tokenizer, tmpl, tf_cache, train_pools,
            mode="full_cr_inv", num_vectors=256, num_updates=45, lambda_inv=0.35, anchored=True, tag="B1_CR_TF_KV2SP",
        )
        results["train_meta"]["B1_CR_TF_KV2SP"] = meta_b1
        torch.save(sp_b1.cpu(), os.path.join(OUT_DIR, "sp_b1_cr_tf_kv2sp.pt"))
        save()

    # Step 3: Full 9-Method x 6-Position Distributional Matrix & Hidden-State Drift
    sp_models: Dict[str, torch.Tensor] = {
        "A1_Pos0_Only_SP": sp_a1,
        "A2_Paper_PR_SP": sp_a2,
        "A3_MultiSlot_CR_SP": sp_a3,
        "A4_Full_CR_SP_Inv": sp_a4,
        "A5_CR_SP_L64": sp_a5,
    }
    if sp_b0 is not None:
        sp_models["B0_exp6_TF_KV2SP"] = sp_b0
    if sp_b1 is not None:
        sp_models["B1_CR_TF_KV2SP"] = sp_b1

    kv_models: Dict[str, torch.Tensor] = {"KV_Distilled_L16": kv_dist_l16}
    if kv_tf_l16 is not None:
        kv_models["KV_Fixed_L16_TF"] = kv_tf_l16

    results["position_matrix"] = evaluate_position_matrix(
        student, tokenizer, tmpl, eval_ctx, teacher_eval_cache, sp_models, kv_models=kv_models
    )
    save()

    # Step 4: Official allenai/IFBench32 Real End-to-End Evaluation across Insertion Positions!
    print("\n=== Step 4: Official allenai/IFBench32 End-to-End Evaluation across Insertion Positions ===", flush=True)
    # P0 Clean Start
    results["ifbench32_by_position"]["P0_Student"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P0_Clean_Start", [], tag="P0-Start: Student")
    results["ifbench32_by_position"]["P0_A1_Pos0_Only"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P0_Clean_Start", [sp_a1], tag="P0-Start: A1 Pos0-Only")
    results["ifbench32_by_position"]["P0_A4_Full_CR_Inv"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P0_Clean_Start", [sp_a4], tag="P0-Start: A4 Full CR-SP")
    if sp_b1 is not None:
        results["ifbench32_by_position"]["P0_B1_CR_TF_KV2SP"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P0_Clean_Start", [sp_b1], tag="P0-Start: B1 CR-TF-KV2SP")
    save()

    # P2 Multi-Turn Mid-Dialog Insertion (after 2 real chat turns, before new user query!)
    results["ifbench32_by_position"]["P2_Student"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [], tag="P2-MultiTurn: Student")
    results["ifbench32_by_position"]["P2_A1_Pos0_Only"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_a1], tag="P2-MultiTurn: A1 Pos0-Only")
    results["ifbench32_by_position"]["P2_A2_Paper_PR"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_a2], tag="P2-MultiTurn: A2 Paper-PR")
    results["ifbench32_by_position"]["P2_A4_Full_CR_Inv"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_a4], tag="P2-MultiTurn: A4 Full CR-SP")
    if sp_b1 is not None:
        results["ifbench32_by_position"]["P2_B1_CR_TF_KV2SP"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_b1], tag="P2-MultiTurn: B1 CR-TF-KV2SP")
    save()

    # P3 Post-RAG Insertion (after 420-tok enterprise telemetry/RAG block!)
    results["ifbench32_by_position"]["P3_Student"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P3_PostRAG_Doc", [], tag="P3-PostRAG: Student")
    results["ifbench32_by_position"]["P3_A1_Pos0_Only"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P3_PostRAG_Doc", [sp_a1], tag="P3-PostRAG: A1 Pos0-Only")
    results["ifbench32_by_position"]["P3_A4_Full_CR_Inv"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P3_PostRAG_Doc", [sp_a4], tag="P3-PostRAG: A4 Full CR-SP")
    save()

    # P4 Post-Query Suffix (Late Injection AFTER user query — ZERO history/query KV recomputation!)
    results["ifbench32_by_position"]["P4_A1_Pos0_Only"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_a1], tag="P4-PostQuery: A1 Pos0-Only")
    results["ifbench32_by_position"]["P4_A2_Paper_PR"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_a2], tag="P4-PostQuery: A2 Paper-PR")
    results["ifbench32_by_position"]["P4_A4_Full_CR_Inv"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_a4], tag="P4-PostQuery: A4 Full CR-SP")
    if sp_b1 is not None:
        results["ifbench32_by_position"]["P4_B1_CR_TF_KV2SP"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_b1], tag="P4-PostQuery: B1 CR-TF-KV2SP")
    save()

    # Split-Position Multi-Expert Assembly: [Sys: Persona + TF_Math] + [2-Turn Hist] + [Query + A4_Full_CR_SP_Inv]!
    if sp_tf_math is not None:
        results["ifbench32_by_position"]["P_Split_MathEarly_CRIFLate"] = eval_ifbench32_regime(
            student, tokenizer, tmpl, eval_ctx, ifbench32, "P_Split_Dual",
            [sp_a4], split_early_sp=[sp_tf_math], tag="Split: [Sys:Math]+[Hist]+[Q;CR-IF]",
        )
        save()

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== Suite 5 (exp7) COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
