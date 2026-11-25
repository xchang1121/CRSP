"""
Paper Reference: Section 4.1–4.2 (Stage I Input-Space CSPD Single/Multi-Expert Baseline).
Stage A Experiment: Faithful Reproduction of Composable Soft-Prompt Distillation (CSPD)
Paper: "Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training" (2026)

Uses the EXACT Table 2 / Section 3.3 4B checkpoints from HuggingFace:
  - Frozen Pre-RL Student: Jackwang111/M2RL-SFT (Qwen3-4B)
  - Math RL Expert Teacher: Jackwang111/M2RL-RL_Math
  - IF RL Expert Teacher:   Jackwang111/M2RL-RL_IF

Reproduces:
  1. Single-expert RL distillation (Math & Instruction Following) with L=256 soft prompts
  2. Zero-training concatenation composition [P_math; P_if] & reverse [P_if; P_math]
  3. Continual post-training addition (Student -> +P_if -> +[P_math; P_if]) & leave-one-out removal
  4. Key ablations (Table 5): Prefix-Robust Training (on vs off) & Optimizer (Muon vs AdamW)
  5. Out-of-domain preservation check
  6. Section 4 & Appendix G Mechanistic Analysis:
     - Eq. (5) exact full-vocab Teacher-Student KL gap closed G(A): Soft prompt vs written Check/Verify instructions
     - Fig. 6b / 11c: Most raised & lowered tokens
     - Appendix G.2: Embedding geometry (nearest vocab token unchanged %, median cosine sim, effective rank)
     - Eq. (6) Attention pathway ablation K(A): Full / Response-only / Input-only / Blocked
"""

import collections
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

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import (
    SUPPORTED_IFEVAL_IDS,
    batch_generate_with_soft_prompts,
    extract_boxed,
    strip_thinking,
    verify_ifeval_response,
    verify_math_response,
)
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

STUDENT_ID = "Jackwang111/M2RL-SFT"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"

OUT_DIR = "/content/cspd/exp1_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load Strictly Disjoint Verifiable Datasets (Math & IFEval)
# ---------------------------------------------------------------------------

def load_exp1_datasets(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)

    # Math Train: DigitalLearningGmbH/MATH-lighteval (train split, Level 3-4)
    ds_math_tr = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    math_train_pool = []
    for row in ds_math_tr:
        lvl = str(row.get("level", ""))
        if any(k in lvl for k in ["3", "4"]):
            ans = extract_boxed(row["solution"])
            if ans:
                math_train_pool.append({"problem": row["problem"], "answer": ans, "level": lvl})
    rng.shuffle(math_train_pool)

    # Math Held-Out Test: HuggingFaceH4/MATH-500 (24 Level 4-5 problems, max_new_tokens=3072)
    ds_math_te = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_lvl = {4: [], 5: []}
    for row in ds_math_te:
        lv = int(row["level"])
        if lv in by_lvl:
            by_lvl[lv].append({"problem": row["problem"], "answer": str(row["answer"]), "level": lv})
    math_test = []
    for lv, cnt in [(4, 14), (5, 10)]:
        rng.shuffle(by_lvl[lv])
        math_test.extend(by_lvl[lv][:cnt])

    # Remove any accidental duplicate problem string from train pool
    test_prob_set = {x["problem"].strip() for x in math_test}
    math_train = [x for x in math_train_pool if x["problem"].strip() not in test_prob_set][:48]

    # IFEval: google/IFEval filtered to our deterministic SUPPORTED_IFEVAL_IDS
    ds_if = load_dataset("google/IFEval", split="train")
    if_supported = []
    for row in ds_if:
        ids = row["instruction_id_list"]
        if ids and all(i in SUPPORTED_IFEVAL_IDS for i in ids):
            if_supported.append({
                "key": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": ids,
                "kwargs": row["kwargs"],
            })
    rng.shuffle(if_supported)
    if_test = if_supported[:48]
    if_train = if_supported[48:144]

    print(
        f"Datasets ready | Math train={len(math_train)}, test={len(math_test)} | "
        f"IFEval train={len(if_train)}, test={len(if_test)} (from {len(if_supported)} supported)",
        flush=True,
    )
    return {
        "math_train": math_train,
        "math_test": math_test,
        "if_train": if_train,
        "if_test": if_test,
    }


def format_math_prompt(problem: str) -> str:
    return problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."


# ---------------------------------------------------------------------------
# 2. Evaluation Functions (Math & IFEval)
# ---------------------------------------------------------------------------

def eval_math_set(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    math_test: List[Dict[str, Any]],
    soft_prompts: List[torch.Tensor],
    batch_size: int = 24,
    max_new_tokens: int = 3072,
    temperature: float = 0.6,
    tag: str = "",
) -> Tuple[float, List[Dict[str, Any]]]:
    torch.manual_seed(42)
    qs = [format_math_prompt(x["problem"]) for x in math_test]
    t0 = time.time()
    outs = batch_generate_with_soft_prompts(
        model, tokenizer, tmpl, qs, soft_prompts,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=0.95, top_k=20, enable_thinking=True,
    )
    records = []
    n_ok = 0
    for item, resp in zip(math_test, outs):
        ok = verify_math_response(resp, item["answer"])
        n_ok += int(ok)
        records.append({
            "problem": item["problem"],
            "gold": item["answer"],
            "level": item["level"],
            "correct": ok,
            "response": resp,
            "resp_len_chars": len(resp),
        })
    acc = 100.0 * n_ok / len(math_test)
    n_closed = sum(1 for r in records if "</think>" in r["response"])
    print(f"  [Eval Math | {tag}] acc={acc:.2f}% ({n_ok}/{len(math_test)}) | closed_think={n_closed}/{len(math_test)} in {time.time()-t0:.1f}s", flush=True)
    return round(acc, 2), records


def eval_if_set(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    if_test: List[Dict[str, Any]],
    soft_prompts: List[torch.Tensor],
    batch_size: int = 48,
    max_new_tokens: int = 1536,
    temperature: float = 0.6,
    tag: str = "",
) -> Tuple[float, float, List[Dict[str, Any]]]:
    torch.manual_seed(42)
    qs = [x["prompt"] for x in if_test]
    t0 = time.time()
    outs = batch_generate_with_soft_prompts(
        model, tokenizer, tmpl, qs, soft_prompts,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=0.95, top_k=20, enable_thinking=True,
    )
    records = []
    n_strict = 0
    inst_rates = []
    for item, resp in zip(if_test, outs):
        strict_ok, inst_rate = verify_ifeval_response(resp, item["instruction_id_list"], item["kwargs"])
        n_strict += int(strict_ok)
        inst_rates.append(inst_rate)
        records.append({
            "key": item["key"],
            "prompt": item["prompt"],
            "strict_ok": strict_ok,
            "inst_rate": inst_rate,
            "response": resp,
        })
    strict_acc = 100.0 * n_strict / len(if_test)
    inst_acc = 100.0 * sum(inst_rates) / len(inst_rates)
    print(
        f"  [Eval IFEval | {tag}] strict={strict_acc:.2f}% ({n_strict}/{len(if_test)}) | "
        f"inst={inst_acc:.2f}% in {time.time()-t0:.1f}s",
        flush=True,
    )
    return round(strict_acc, 2), round(inst_acc, 2), records


# ---------------------------------------------------------------------------
# 3. Phase 1: Teacher Evaluation & Verifier-Filtered Trajectory Generation
# ---------------------------------------------------------------------------

def run_phase1_teachers(data: Dict[str, Any], tokenizer, tmpl: ChatTemplateTokens) -> Dict[str, Any]:
    cache_path = os.path.join(OUT_DIR, "teacher_data.json")
    if os.path.exists(cache_path):
        print(f"Loading cached teacher data from {cache_path}", flush=True)
        return json.load(open(cache_path))

    out: Dict[str, Any] = {}

    # --- 1A. Math RL Teacher ---
    print(f"\n=== Phase 1A: Loading Math RL Teacher ({MATH_TEACHER_ID}) ===", flush=True)
    m_teacher = AutoModelForCausalLM.from_pretrained(
        MATH_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    m_teacher.eval()

    math_t_acc, math_test_recs = eval_math_set(
        m_teacher, tokenizer, tmpl, data["math_test"], [], batch_size=24, max_new_tokens=3072, tag="Teacher Math"
    )
    out["teacher_math_test_acc"] = math_t_acc
    # Save up to 20 correct held-out teacher traces for Section 4 mechanistic KL analysis!
    out["mech_math_traces"] = [
        {"question": format_math_prompt(r["problem"]), "response": r["response"]}
        for r in math_test_recs if r["correct"] and "</think>" in r["response"]
    ][:20]

    print("Sampling verifier-filtered Math teacher training trajectories...", flush=True)
    t0 = time.time()
    tr_qs = [format_math_prompt(x["problem"]) for x in data["math_train"]]
    tr_outs = batch_generate_with_soft_prompts(
        m_teacher, tokenizer, tmpl, tr_qs, [], batch_size=24, max_new_tokens=2560,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    math_verified = []
    for item, resp in zip(data["math_train"], tr_outs):
        if verify_math_response(resp, item["answer"]):
            r_ids = tokenizer.encode(resp, add_special_tokens=False)
            if 30 <= len(r_ids) <= 2200:
                math_verified.append({"question": format_math_prompt(item["problem"]), "response": resp})
    print(
        f"  Math verified traces: {len(math_verified)}/{len(data['math_train'])} in {time.time()-t0:.1f}s",
        flush=True,
    )
    out["math_train_verified"] = math_verified

    del m_teacher
    gc.collect()
    torch.cuda.empty_cache()

    # --- 1B. IF RL Teacher ---
    print(f"\n=== Phase 1B: Loading IF RL Teacher ({IF_TEACHER_ID}) ===", flush=True)
    if_teacher = AutoModelForCausalLM.from_pretrained(
        IF_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    if_teacher.eval()

    if_t_strict, if_t_inst, _ = eval_if_set(
        if_teacher, tokenizer, tmpl, data["if_test"], [], batch_size=48, max_new_tokens=1536, tag="Teacher IF"
    )
    out["teacher_if_test_strict"] = if_t_strict
    out["teacher_if_test_inst"] = if_t_inst

    print("Sampling verifier-filtered IF teacher training trajectories...", flush=True)
    t0 = time.time()
    if_tr_qs = [x["prompt"] for x in data["if_train"]]
    if_tr_outs = batch_generate_with_soft_prompts(
        if_teacher, tokenizer, tmpl, if_tr_qs, [], batch_size=48, max_new_tokens=1536,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    if_verified = []
    for item, resp in zip(data["if_train"], if_tr_outs):
        strict_ok, _ = verify_ifeval_response(resp, item["instruction_id_list"], item["kwargs"])
        if strict_ok:
            r_ids = tokenizer.encode(resp, add_special_tokens=False)
            if 20 <= len(r_ids) <= 1600:
                if_verified.append({"question": item["prompt"], "response": resp})
    print(
        f"  IF verified traces: {len(if_verified)}/{len(data['if_train'])} in {time.time()-t0:.1f}s",
        flush=True,
    )
    out["if_train_verified"] = if_verified

    del if_teacher
    gc.collect()
    torch.cuda.empty_cache()

    with open(cache_path, "w") as f:
        json.dump(out, f)
    return out


# ---------------------------------------------------------------------------
# 4. Phase 2: Independent Soft-Prompt Distillation (Exact CSPD Recipe)
# ---------------------------------------------------------------------------

def distill_rl_expert_sp(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    traces: List[Dict[str, str]],
    num_vectors: int = 256,
    num_updates: int = 80,
    batch_size: int = 16,
    micro_batch: int = 2,
    optimizer_type: str = "muon",  # "muon" or "adamw"
    prefix_robust: bool = True,
    seed: int = 42,
    tag: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Distill one RL expert from verifier-filtered trajectories into P_j in R^{L x d} (FP32)
    on frozen BF16 student, following Section 2.1, 2.2, and Appendix A.
    """
    t0 = time.time()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = embed_layer(torch.tensor([pad_id], device=device)).detach().to(dtype)

    tokenized = []
    prompt_stream: List[int] = []
    for tr in traces:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False) + tmpl.im_end
        tokenized.append((tr["question"], q_ids, r_ids))
        prompt_stream.extend(tmpl.sys_close_user_open + q_ids + tmpl.user_close_asst_open)

    sp = init_soft_prompt_from_vocab(embed_layer.weight, num_vectors=num_vectors, seed=seed)
    init_sp = sp.detach().clone()

    if optimizer_type == "muon":
        lr_max, lr_min = 0.03, 0.003
        optimizer = MuonSoftPrompt([sp], lr=lr_max, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)
    elif optimizer_type == "adamw":
        lr_max, lr_min = 0.01, 0.001
        optimizer = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    else:
        raise ValueError(optimizer_type)

    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = random.Random(seed)
    losses = []
    num_mb = max(1, batch_size // micro_batch)

    for step in range(num_updates):
        progress = step / max(1, num_updates - 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        batch = [tokenized[rng.randrange(len(tokenized))] for _ in range(batch_size)]
        # Sort by sequence length inside batch to minimize right-padding waste!
        batch.sort(key=lambda x: len(x[1]) + len(x[2]))
        step_loss = 0.0

        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            embs_l, lbls_l = [], []
            for q_text, q_ids, r_ids in sub:
                prefix_F = None
                if prefix_robust and should_add_prefix(q_text):
                    prefix_F = sample_random_prefix(prompt_stream, rng, min_len=128, max_len=256)
                emb, labels, _ = build_soft_prompt_inputs(
                    embed_layer=embed_layer, tmpl=tmpl, soft_prompts=[sp],
                    question_ids=q_ids, response_ids=r_ids,
                    system_text_ids=None, prefix_F_ids=prefix_F,
                    device=device, model_dtype=dtype,
                )
                embs_l.append(emb)
                lbls_l.append(labels)

            b_emb, b_mask, b_lbl = pad_and_stack_train_batch(embs_l, lbls_l, pad_vec)
            out = model(inputs_embeds=b_emb, attention_mask=b_mask)
            loss = length_normalized_nll_loss(out.logits, b_lbl)
            step_loss += loss.item()
            (loss / num_mb).backward()
            del b_emb, b_mask, b_lbl, out

        torch.nn.utils.clip_grad_norm_([sp], max_norm=1.0)
        optimizer.step()
        avg_loss = step_loss / num_mb
        losses.append(round(avg_loss, 4))
        if (step + 1) % 20 == 0 or step == 0 or step == num_updates - 1:
            print(
                f"  [{tag}] step {step+1:3d}/{num_updates} | nll={avg_loss:.4f} | "
                f"lr={lr:.5f} | elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True

    # Appendix G.2 Embedding Geometry Analysis
    with torch.no_grad():
        final_sp = sp.detach().clone()
        W = embed_layer.weight.detach().float()
        # Normalize for nearest-neighbor cosine lookup
        W_norm = F.normalize(W, dim=-1)
        init_nn = (F.normalize(init_sp, dim=-1) @ W_norm.T).argmax(dim=-1)
        final_nn = (F.normalize(final_sp, dim=-1) @ W_norm.T).argmax(dim=-1)
        nn_unchanged_pct = 100.0 * (init_nn == final_nn).float().mean().item()
        med_cos = float(F.cosine_similarity(final_sp, init_sp, dim=-1).median().item())

        delta = (final_sp - init_sp).float()
        sv = torch.linalg.svdvals(delta)
        p_sv = (sv ** 2) / (sv ** 2).sum().clamp_min(1e-12)
        eff_rank = float(torch.exp(-(p_sv * p_sv.clamp_min(1e-12).log()).sum()).item())

    meta = {
        "tag": tag,
        "optimizer": optimizer_type,
        "prefix_robust": prefix_robust,
        "num_vectors": num_vectors,
        "num_updates": num_updates,
        "num_traces": len(traces),
        "train_seconds": round(time.time() - t0, 1),
        "initial_nll": losses[0],
        "final_nll": losses[-1],
        "nn_token_unchanged_pct": round(nn_unchanged_pct, 2),
        "median_cosine_to_init": round(med_cos, 4),
        "effective_rank_delta": round(eff_rank, 2),
    }
    print(
        f"  [{tag} Geometry] NN unchanged={meta['nn_token_unchanged_pct']:.1f}% | "
        f"median cos={meta['median_cosine_to_init']:.3f} | eff_rank={meta['effective_rank_delta']:.1f}/{num_vectors}",
        flush=True,
    )
    return final_sp, init_sp, meta


# ---------------------------------------------------------------------------
# 5. Phase 3: Section 4 & Appendix G Mechanistic Analysis (Eq. 5 & Eq. 6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_mechanistic_analysis(
    student_model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    mech_traces: List[Dict[str, str]],
    sp_muon: torch.Tensor,
    sp_init: torch.Tensor,
    sp_adamw: torch.Tensor,
    max_resp_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Reproduce Section 4 / Appendix G:
      - Exact full-vocab Teacher-Student KL gap closed G(A) (Eq. 5)
      - Soft prompt vs written Verify/Check/Neutral instructions
      - Top raised and lowered tokens (Fig. 6b / 11c)
      - Attention pathway ablation K(A) (Eq. 6): Full / Response-only / Input-only / Blocked
    """
    print("\n=== Phase 3: Mechanistic Analysis (Section 4 & Appendix G) ===", flush=True)
    traces = mech_traces[:20]
    if not traces:
        return {"error": "No valid traces"}

    # Load Math RL Teacher alongside Student (2 x 8GB = 16GB fits easily on 40GB A100!)
    t_model = AutoModelForCausalLM.from_pretrained(
        MATH_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    t_model.eval()
    t_embed = t_model.get_input_embeddings()
    s_embed = student_model.get_input_embeddings()

    # Written instructions directly from Appendix G.3
    INSTRUCTIONS = {
        "check_instruction": "Do not conclude too quickly. After each intermediate result, question it and check it before moving on.",
        "verify_instruction": "Do not conclude too quickly. After each intermediate result, question it and check it before moving on. Double-check results with another method and consider edge cases before giving your final answer.",
        "neutral_instruction": "Read the problem carefully and answer.",
    }
    inst_ids_map = {k: tokenizer.encode(v, add_special_tokens=False) for k, v in INSTRUCTIONS.items()}

    kl_accum = collections.defaultdict(list)
    tok_deltas = []  # (delta_logp, token_str)

    for idx, tr in enumerate(traces):
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:max_resp_tokens]
        if len(r_ids) < 32:
            continue

        t_lp, _ = compute_response_logprobs(t_model, t_embed, tmpl, [], q_ids, r_ids, attention_mode="full")
        s_lp, _ = compute_response_logprobs(student_model, s_embed, tmpl, [], q_ids, r_ids, attention_mode="full")
        sp_lp, _ = compute_response_logprobs(student_model, s_embed, tmpl, [sp_muon], q_ids, r_ids, attention_mode="full")
        init_lp, _ = compute_response_logprobs(student_model, s_embed, tmpl, [sp_init], q_ids, r_ids, attention_mode="full")
        aw_lp, _ = compute_response_logprobs(student_model, s_embed, tmpl, [sp_adamw], q_ids, r_ids, attention_mode="full")

        kl_accum["student"].append(mean_kl_divergence(t_lp, s_lp))
        kl_accum["sp_muon"].append(mean_kl_divergence(t_lp, sp_lp))
        kl_accum["sp_init"].append(mean_kl_divergence(t_lp, init_lp))
        kl_accum["sp_adamw"].append(mean_kl_divergence(t_lp, aw_lp))

        # Written instructions
        for name, s_ids in inst_ids_map.items():
            i_lp, _ = compute_response_logprobs(
                student_model, s_embed, tmpl, [], q_ids, r_ids, system_text_ids=s_ids, attention_mode="full"
            )
            kl_accum[name].append(mean_kl_divergence(t_lp, i_lp))

        # Attention pathway masks (Eq. 6)
        for mode in ["response_only", "input_only", "blocked"]:
            m_lp, _ = compute_response_logprobs(student_model, s_embed, tmpl, [sp_muon], q_ids, r_ids, attention_mode=mode)
            kl_accum[f"attn_{mode}"].append(mean_kl_divergence(t_lp, m_lp))

        # Token probability changes on the actual trace tokens (Fig. 6b / 11c)
        n = min(len(r_ids), s_lp.size(0), sp_lp.size(0))
        r_tensor = torch.tensor(r_ids[:n], device=s_lp.device)
        s_tok_lp = s_lp[:n].gather(1, r_tensor.unsqueeze(1)).squeeze(1)
        sp_tok_lp = sp_lp[:n].gather(1, r_tensor.unsqueeze(1)).squeeze(1)
        diff = (sp_tok_lp - s_tok_lp).cpu().tolist()
        for tok_id, d_val in zip(r_ids[:n], diff):
            tok_deltas.append((d_val, tokenizer.decode([tok_id])))

        del t_lp, s_lp, sp_lp, init_lp, aw_lp

    del t_model
    gc.collect()
    torch.cuda.empty_cache()

    mean_kls = {k: sum(v) / len(v) for k, v in kl_accum.items()}
    kl_S = mean_kls["student"]
    kl_SP = mean_kls["sp_muon"]

    def gap_closed_G(kl_A: float) -> float:
        return round(100.0 * (1.0 - kl_A / max(kl_S, 1e-8)), 2)

    def pathway_kept_K(kl_A: float) -> float:
        return round(100.0 * (kl_S - kl_A) / max(kl_S - kl_SP, 1e-8), 2)

    # Top raised & lowered tokens among top-500 most-changed positions
    tok_deltas.sort(key=lambda x: x[0])
    k_top = min(500, len(tok_deltas) // 4)
    lowered_counts = collections.Counter(t for _, t in tok_deltas[:k_top]).most_common(12)
    raised_counts = collections.Counter(t for _, t in tok_deltas[-k_top:]).most_common(12)

    mech_res = {
        "num_traces": len(kl_accum["student"]),
        "mean_kl": {k: round(v, 5) for k, v in mean_kls.items()},
        "kl_gap_closed_G_pct": {
            "student": 0.0,
            "sp_init": gap_closed_G(mean_kls["sp_init"]),
            "sp_muon": gap_closed_G(mean_kls["sp_muon"]),
            "sp_adamw": gap_closed_G(mean_kls["sp_adamw"]),
            "verify_instruction": gap_closed_G(mean_kls["verify_instruction"]),
            "check_instruction": gap_closed_G(mean_kls["check_instruction"]),
            "neutral_instruction": gap_closed_G(mean_kls["neutral_instruction"]),
        },
        "attention_pathway_kept_K_pct": {
            "full": 100.0,
            "response_only": pathway_kept_K(mean_kls["attn_response_only"]),
            "input_only": pathway_kept_K(mean_kls["attn_input_only"]),
            "blocked": pathway_kept_K(mean_kls["attn_blocked"]),
        },
        "top_raised_tokens": raised_counts,
        "top_lowered_tokens": lowered_counts,
    }
    print(f"  KL(T||S)={kl_S:.4f} -> KL(T||S+SP_Muon)={kl_SP:.4f}", flush=True)
    print(f"  G(A) KL gap closed (%): {mech_res['kl_gap_closed_G_pct']}", flush=True)
    print(f"  K(A) Attention pathways (%): {mech_res['attention_pathway_kept_K_pct']}", flush=True)
    print(f"  Top raised tokens:  {raised_counts[:8]}", flush=True)
    print(f"  Top lowered tokens: {lowered_counts[:8]}", flush=True)
    return mech_res


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Stage A: Reproducing Composable Soft-Prompt Distillation (CSPD) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    data = load_exp1_datasets(seed=42)

    results: Dict[str, Any] = {
        "student_model": STUDENT_ID,
        "math_teacher": MATH_TEACHER_ID,
        "if_teacher": IF_TEACHER_ID,
        "evals": {},
        "sp_meta": {},
    }

    def save_results():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Phase 1: Teachers
    t_data = run_phase1_teachers(data, tokenizer, tmpl)
    results["evals"]["teacher_oracle"] = {
        "math": t_data["teacher_math_test_acc"],
        "if_strict": t_data["teacher_if_test_strict"],
        "if_inst": t_data["teacher_if_test_inst"],
        "avg": round((t_data["teacher_math_test_acc"] + t_data["teacher_if_test_strict"]) / 2.0, 2),
    }
    save_results()

    # Phase 2: Load Frozen Student ONCE
    print(f"\n=== Phase 2: Loading Frozen Student ({STUDENT_ID}) ===", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    # Student baseline
    s_math, _ = eval_math_set(student, tokenizer, tmpl, data["math_test"], [], tag="Student (no SP)")
    s_if_s, s_if_i, _ = eval_if_set(student, tokenizer, tmpl, data["if_test"], [], tag="Student (no SP)")
    results["evals"]["student_no_sp"] = {
        "math": s_math, "if_strict": s_if_s, "if_inst": s_if_i,
        "avg": round((s_math + s_if_s) / 2.0, 2),
    }
    save_results()

    # Distill 6 soft prompts (L=256):
    #   (1) Muon + PR (CSPD main): P_math, P_if
    #   (2) Muon w/o PR (ablation): P_math_nopr, P_if_nopr
    #   (3) AdamW + PR (ablation): P_math_adamw, P_if_adamw
    sp_math, sp_math_init, meta_m = distill_rl_expert_sp(
        student, tokenizer, tmpl, t_data["math_train_verified"],
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="muon", prefix_robust=True, seed=42, tag="P_math_Muon_PR"
    )
    sp_if, _, meta_i = distill_rl_expert_sp(
        student, tokenizer, tmpl, t_data["if_train_verified"],
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="muon", prefix_robust=True, seed=43, tag="P_if_Muon_PR"
    )
    sp_math_nopr, _, meta_m_nopr = distill_rl_expert_sp(
        student, tokenizer, tmpl, t_data["math_train_verified"],
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="muon", prefix_robust=False, seed=42, tag="P_math_NoPR"
    )
    sp_if_nopr, _, meta_i_nopr = distill_rl_expert_sp(
        student, tokenizer, tmpl, t_data["if_train_verified"],
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="muon", prefix_robust=False, seed=43, tag="P_if_NoPR"
    )
    sp_math_aw, _, meta_m_aw = distill_rl_expert_sp(
        student, tokenizer, tmpl, t_data["math_train_verified"],
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="adamw", prefix_robust=True, seed=42, tag="P_math_AdamW_PR"
    )
    sp_if_aw, _, meta_i_aw = distill_rl_expert_sp(
        student, tokenizer, tmpl, t_data["if_train_verified"],
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="adamw", prefix_robust=True, seed=43, tag="P_if_AdamW_PR"
    )

    for name, tensor, meta in [
        ("P_math_muon_pr", sp_math, meta_m),
        ("P_if_muon_pr", sp_if, meta_i),
        ("P_math_nopr", sp_math_nopr, meta_m_nopr),
        ("P_if_nopr", sp_if_nopr, meta_i_nopr),
        ("P_math_adamw_pr", sp_math_aw, meta_m_aw),
        ("P_if_adamw_pr", sp_if_aw, meta_i_aw),
    ]:
        torch.save(tensor.cpu(), os.path.join(OUT_DIR, f"{name}.pt"))
        results["sp_meta"][name] = meta
    save_results()

    # Phase 3 FIRST (~90s): Mechanistic Analysis (Section 4 & Appendix G) so results are saved early!
    results["mechanistic"] = run_mechanistic_analysis(
        student, tokenizer, tmpl, t_data["mech_math_traces"], sp_math, sp_math_init, sp_math_aw
    )
    save_results()

    # Phase 4: Evaluate Single-Domain, Continual Addition, Order Invariance, and Ablations
    eval_configs = [
        ("single_P_math", [sp_math]),
        ("single_P_if", [sp_if]),
        ("cspd_concat_Math_IF", [sp_math, sp_if]),
        ("cspd_concat_IF_Math", [sp_if, sp_math]),
        ("ablation_nopr_Math_IF", [sp_math_nopr, sp_if_nopr]),
        ("ablation_adamw_Math_IF", [sp_math_aw, sp_if_aw]),
    ]

    A_S_avg = results["evals"]["student_no_sp"]["avg"]
    A_T_avg = results["evals"]["teacher_oracle"]["avg"]
    A_S_m, A_T_m = results["evals"]["student_no_sp"]["math"], results["evals"]["teacher_oracle"]["math"]
    A_S_i, A_T_i = results["evals"]["student_no_sp"]["if_strict"], results["evals"]["teacher_oracle"]["if_strict"]

    def rec(val, s, t):
        return round(100.0 * (val - s) / (t - s), 2) if abs(t - s) > 1e-6 else 0.0

    for cfg_name, sps in eval_configs:
        m_acc, _ = eval_math_set(student, tokenizer, tmpl, data["math_test"], sps, tag=cfg_name)
        i_s, i_i, _ = eval_if_set(student, tokenizer, tmpl, data["if_test"], sps, tag=cfg_name)
        avg = round((m_acc + i_s) / 2.0, 2)
        results["evals"][cfg_name] = {
            "math": m_acc,
            "if_strict": i_s,
            "if_inst": i_i,
            "avg": avg,
            "recovery_math_pct": rec(m_acc, A_S_m, A_T_m),
            "recovery_if_pct": rec(i_s, A_S_i, A_T_i),
            "recovery_avg_pct": rec(avg, A_S_avg, A_T_avg),
        }
        save_results()

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save_results()
    print(f"\n=== Stage A COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
