"""
Stage A (v2 Clean Reproduction): Faithful Reproduction of Composable Soft-Prompt Distillation (CSPD)
Paper: "Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training" (2026)

Uses the EXACT Paper Table 2 4B checkpoints from HuggingFace:
  - Frozen Pre-RL Student: Jackwang111/M2RL-SFT (Qwen3-4B)
  - IF RL Expert Teacher:   Jackwang111/M2RL-RL_IF
  - Math RL Expert Teacher: Jackwang111/M2RL-RL_Math

And the EXACT IFBench evaluator (/content/IFBench/data/IFBench_test.jsonl)!

Key Scientific Upgrade over Stage A v1 (30-epoch 1-hot NLL overfitting on 32 traces):
  CSPD Eq. (1) optimizes E_{y ~ p_T}[ -1/|y| sum_t log p_S(y_t | I(P,x), y_{<t}) ].
  By the Law of Iterated Expectations, its exact Rao-Blackwellized conditional expectation
  at each response token is E_{y_t ~ p_T(.|x, y_{<t})}[-log p_S(y_t | I(P,x), y_{<t})],
  which optimizes the EXACT same population objective as Eq. (1) while eliminating 1-sample
  Monte Carlo variance under single-GPU trace budgets (~70-120 traces instead of 28,946).

Reproduces:
  1. Single-expert RL distillation (P_if, P_math; L=256, Muon + Prefix-Robust Training)
  2. Training dynamics G(A) across updates 0, 10, 20, 40, 60 (Fig. 11a)
  3. Zero-training concatenation composition [P_math; P_if] & reverse [P_if; P_math] (Table 2)
  4. Table 5 Ablations: Prefix-Robust Training (on vs off) & Optimizer (Muon vs AdamW)
  5. IFBench end-to-end generation accuracy & recovery (Table 2 benchmark!)
  6. Section 4 & Appendix G Mechanistic Analysis:
     - Eq. (5) exact full-vocab Teacher-Student KL gap closed G(A): SP vs written instructions
     - Fig. 6b / 11c: Most raised & lowered tokens + reasoning/checking token shifts
     - Appendix G.2: Embedding geometry (NN unchanged %, median cosine sim, effective rank)
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
sys.path.insert(0, "/content/IFBench")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluation_lib as ifbench_eval

from benchmarks import (
    SUPPORTED_IFEVAL_IDS,
    batch_generate_with_soft_prompts,
    extract_boxed,
    strip_thinking,
)
from cspd_core import (
    ChatTemplateTokens,
    MuonSoftPrompt,
    build_soft_prompt_inputs,
    compute_response_logprobs,
    get_qwen_template_tokens,
    init_soft_prompt_from_vocab,
    mean_kl_divergence,
    pad_and_stack_train_batch,
    sample_random_prefix,
    should_add_prefix,
)

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"

OUT_DIR = "/content/cspd/exp1b_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Official IFBench Evaluation Helper
# ---------------------------------------------------------------------------

def load_ifbench_subset(n: int = 40, seed: int = 42) -> List[Any]:
    all_inps = ifbench_eval.read_prompt_list("/content/IFBench/data/IFBench_test.jsonl")
    # Stratify evenly across the 300 prompts so all constraint families are represented
    rng = random.Random(seed)
    idxs = list(range(len(all_inps)))
    rng.shuffle(idxs)
    chosen = [all_inps[i] for i in sorted(idxs[:n])]
    print(f"Loaded {len(chosen)}/{len(all_inps)} official IFBench_test prompts.", flush=True)
    return chosen


def eval_ifbench(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    ifbench_inps: List[Any],
    soft_prompts: List[torch.Tensor],
    batch_size: int = 40,
    max_new_tokens: int = 3072,
    temperature: float = 0.6,
    tag: str = "",
) -> Dict[str, Any]:
    torch.manual_seed(42)
    t0 = time.time()
    prompts = [inp.prompt for inp in ifbench_inps]
    outs = batch_generate_with_soft_prompts(
        model, tokenizer, tmpl, prompts, soft_prompts,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=0.95, top_k=20, enable_thinking=True,
    )
    prompt_to_resp = {}
    n_closed = 0
    total_chars = 0
    for inp, raw_resp in zip(ifbench_inps, outs):
        if "</think>" in raw_resp:
            n_closed += 1
        total_chars += len(raw_resp)
        prompt_to_resp[inp.prompt] = strip_thinking(raw_resp)

    strict_ok, loose_ok = 0, 0
    inst_strict_list, inst_loose_list = [], []
    for inp in ifbench_inps:
        r_s = ifbench_eval.test_instruction_following_strict(inp, prompt_to_resp)
        r_l = ifbench_eval.test_instruction_following_loose(inp, prompt_to_resp)
        strict_ok += int(r_s.follow_all_instructions)
        loose_ok += int(r_l.follow_all_instructions)
        inst_strict_list.extend(r_s.follow_instruction_list)
        inst_loose_list.extend(r_l.follow_instruction_list)

    n = len(ifbench_inps)
    res = {
        "prompt_strict": round(100.0 * strict_ok / n, 2),
        "prompt_loose": round(100.0 * loose_ok / n, 2),
        "inst_strict": round(100.0 * sum(inst_strict_list) / max(1, len(inst_strict_list)), 2),
        "inst_loose": round(100.0 * sum(inst_loose_list) / max(1, len(inst_loose_list)), 2),
        "n_strict": strict_ok,
        "n_total": n,
        "closed_think": n_closed,
        "avg_chars": round(total_chars / n, 1),
        "eval_seconds": round(time.time() - t0, 1),
    }
    print(
        f"  [IFBench | {tag}] strict={res['prompt_strict']:.2f}% ({strict_ok}/{n}) | "
        f"loose={res['prompt_loose']:.2f}% | inst_strict={res['inst_strict']:.2f}% | "
        f"closed_think={n_closed}/{n} | avg_chars={res['avg_chars']:.0f} in {res['eval_seconds']:.1f}s",
        flush=True,
    )
    return res


# ---------------------------------------------------------------------------
# 2. Phase 1: Collect Diverse Teacher Traces & Precompute Exact Top-K Logprobs
# ---------------------------------------------------------------------------

def format_math_prompt(problem: str) -> str:
    return problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."


def collect_teacher_traces_and_eval_teachers(
    tokenizer,
    tmpl: ChatTemplateTokens,
    ifbench_inps: List[Any],
) -> Dict[str, Any]:
    cache_path = os.path.join(OUT_DIR, "teacher_traces_v2.json")
    if os.path.exists(cache_path):
        print(f"Loading cached v2 traces from {cache_path}", flush=True)
        return json.load(open(cache_path))

    # Load any existing traces from exp1_results/teacher_data.json to save time
    old_cache = "/content/cspd/exp1_results/teacher_data.json"
    old_data = json.load(open(old_cache)) if os.path.exists(old_cache) else {}

    rng = random.Random(42)

    # --- IF prompts from google/IFEval (disjoint from IFBench_test!) ---
    ds_if = load_dataset("google/IFEval", split="train")
    if_prompts_all = [r["prompt"] for r in ds_if]
    rng.shuffle(if_prompts_all)
    seen_if_q = {x["question"] for x in old_data.get("if_train_verified", [])}
    extra_if_qs = [p for p in if_prompts_all if p not in seen_if_q][:96]

    # --- Math prompts from DigitalLearningGmbH/MATH-lighteval (Level 3-4) ---
    ds_math_tr = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    math_pool = []
    for row in ds_math_tr:
        lvl = str(row.get("level", ""))
        if any(k in lvl for k in ["3", "4"]):
            math_pool.append(format_math_prompt(row["problem"]))
    rng.shuffle(math_pool)
    seen_m_q = {x["question"] for x in old_data.get("math_train_verified", [])}
    seen_m_q |= {x["question"] for x in old_data.get("mech_math_traces", [])}
    extra_math_qs = [q for q in math_pool if q not in seen_m_q][:48]

    out: Dict[str, Any] = {}

    # === 1A. IF Teacher: Official IFBench eval + 96 new IF traces ===
    print(f"\n=== Phase 1A: Loading IF RL Teacher ({IF_TEACHER_ID}) ===", flush=True)
    t_if = AutoModelForCausalLM.from_pretrained(
        IF_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    t_if.eval()
    out["teacher_if_ifbench"] = eval_ifbench(
        t_if, tokenizer, tmpl, ifbench_inps, [], batch_size=40, max_new_tokens=3072, tag="Teacher M2RL-RL_IF"
    )

    print(f"Generating {len(extra_if_qs)} extra IF teacher traces...", flush=True)
    t0 = time.time()
    torch.manual_seed(123)
    new_if_resps = batch_generate_with_soft_prompts(
        t_if, tokenizer, tmpl, extra_if_qs, [], batch_size=48, max_new_tokens=1536,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    if_traces_all = list(old_data.get("if_train_verified", []))
    for q, r in zip(extra_if_qs, new_if_resps):
        r_ids = tokenizer.encode(r, add_special_tokens=False)
        if len(r_ids) >= 40:
            if_traces_all.append({"question": q, "response": r})
    rng.shuffle(if_traces_all)
    out["if_val_traces"] = if_traces_all[:20]
    out["if_train_traces"] = if_traces_all[20:140]
    print(f"  IF traces ready: train={len(out['if_train_traces'])}, heldout_val={len(out['if_val_traces'])} ({time.time()-t0:.1f}s)", flush=True)

    del t_if
    gc.collect()
    torch.cuda.empty_cache()

    # === 1B. Math Teacher: 48 extra MATH traces ===
    print(f"\n=== Phase 1B: Loading Math RL Teacher ({MATH_TEACHER_ID}) ===", flush=True)
    t_math = AutoModelForCausalLM.from_pretrained(
        MATH_TEACHER_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    t_math.eval()

    print(f"Generating {len(extra_math_qs)} extra Math teacher traces...", flush=True)
    t0 = time.time()
    torch.manual_seed(124)
    new_m_resps = batch_generate_with_soft_prompts(
        t_math, tokenizer, tmpl, extra_math_qs, [], batch_size=48, max_new_tokens=2048,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    math_train_all = list(old_data.get("math_train_verified", []))
    for q, r in zip(extra_math_qs, new_m_resps):
        r_ids = tokenizer.encode(r, add_special_tokens=False)
        if len(r_ids) >= 60:
            math_train_all.append({"question": q, "response": r})
    out["math_train_traces"] = math_train_all[:80]
    # Strictly held-out MATH-500 test traces from Phase 1A of exp1!
    out["math_val_traces"] = old_data.get("mech_math_traces", [])[:12]
    print(f"  Math traces ready: train={len(out['math_train_traces'])}, heldout_val={len(out['math_val_traces'])} ({time.time()-t0:.1f}s)", flush=True)

    del t_math
    gc.collect()
    torch.cuda.empty_cache()

    with open(cache_path, "w") as f:
        json.dump(out, f)
    return out


@torch.no_grad()
def precompute_teacher_targets(
    teacher_model_id: str,
    tokenizer,
    tmpl: ChatTemplateTokens,
    train_traces: List[Dict[str, str]],
    val_traces: List[Dict[str, str]],
    max_train_resp: int = 768,
    max_val_resp: int = 1024,
    topk: int = 64,
    tag: str = "",
) -> Tuple[List[Tuple], List[Tuple], List[int]]:
    """
    Precompute:
      - Train: top-64 exact teacher distribution (indices, probs) per response token (>99.7% mass)
      - Val: full-vocab teacher logprobs for exact Eq. (5) & Eq. (6) KL evaluation
    """
    t0 = time.time()
    print(f"Precomputing exact teacher targets for {tag} ({teacher_model_id})...", flush=True)
    t_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    t_model.eval()
    t_embed = t_model.get_input_embeddings()

    tr_cache = []
    prompt_stream: List[int] = []
    for tr in train_traces:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:max_train_resp]
        if len(r_ids) < 24:
            continue
        t_lp, _ = compute_response_logprobs(t_model, t_embed, tmpl, [], q_ids, r_ids)
        topv, topi = t_lp.topk(topk, dim=-1)
        topp = F.softmax(topv, dim=-1)
        tr_cache.append((tr["question"], q_ids, r_ids, topi.cpu(), topp.to(torch.bfloat16).cpu()))
        prompt_stream.extend(tmpl.sys_close_user_open + q_ids + tmpl.user_close_asst_open)

    val_cache = []
    for tr in val_traces:
        q_ids = tokenizer.encode(tr["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(tr["response"], add_special_tokens=False)[:max_val_resp]
        if len(r_ids) < 24:
            continue
        t_lp, _ = compute_response_logprobs(t_model, t_embed, tmpl, [], q_ids, r_ids)
        val_cache.append((q_ids, r_ids, t_lp.cpu()))

    del t_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  [{tag}] Cached {len(tr_cache)} train + {len(val_cache)} val teacher targets in {time.time()-t0:.1f}s", flush=True)
    return tr_cache, val_cache, prompt_stream


# ---------------------------------------------------------------------------
# 3. Exact Held-Out KL / G(A) / K(A) Evaluators
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_heldout_kl(
    student,
    s_embed,
    tmpl: ChatTemplateTokens,
    val_cache: List[Tuple],
    soft_prompts: List[torch.Tensor],
    system_text_ids: Optional[List[int]] = None,
    attention_mode: str = "full",
) -> float:
    kls = []
    for q_ids, r_ids, t_lp in val_cache:
        lp, _ = compute_response_logprobs(
            student, s_embed, tmpl, soft_prompts, q_ids, r_ids,
            system_text_ids=system_text_ids, attention_mode=attention_mode,
        )
        kls.append(mean_kl_divergence(t_lp.cuda(), lp))
    return float(sum(kls) / max(1, len(kls)))


# ---------------------------------------------------------------------------
# 4. Phase 2: Rao-Blackwellized CSPD Soft-Prompt Distillation
# ---------------------------------------------------------------------------

def distill_sp_rb(
    student,
    tokenizer,
    tmpl: ChatTemplateTokens,
    tr_cache: List[Tuple],
    val_cache: List[Tuple],
    prompt_stream: List[int],
    kl_student_base: float,
    num_vectors: int = 256,
    num_updates: int = 60,
    batch_size: int = 16,
    micro_batch: int = 2,
    optimizer_type: str = "muon",
    prefix_robust: bool = True,
    lr_max: float = 0.025,
    seed: int = 42,
    track_curve: bool = False,
    tag: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_embed = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = s_embed(torch.tensor([pad_id], device=device)).detach().to(dtype)

    sp = init_soft_prompt_from_vocab(s_embed.weight, num_vectors=num_vectors, seed=seed)
    init_sp = sp.detach().clone()

    lr_min = lr_max * 0.1
    if optimizer_type == "muon":
        optimizer = MuonSoftPrompt([sp], lr=lr_max, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    else:
        raise ValueError(optimizer_type)

    curve: List[Dict[str, Any]] = []
    if track_curve:
        kl0 = eval_heldout_kl(student, s_embed, tmpl, val_cache, [sp])
        g0 = 100.0 * (1.0 - kl0 / kl_student_base)
        curve.append({"step": 0, "heldout_kl": round(kl0, 5), "G_pct": round(g0, 2)})
        print(f"  [{tag}] step   0/{num_updates} | heldout_KL={kl0:.5f} | G={g0:+.2f}%", flush=True)

    rng = random.Random(seed)
    num_mb = max(1, batch_size // micro_batch)
    warmup_steps = 4
    losses = []

    for step in range(1, num_updates + 1):
        if step <= warmup_steps:
            lr = lr_max * (0.25 + 0.75 * (step / warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, num_updates - warmup_steps)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        student.config.use_cache = False
        student.train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer.zero_grad()

        batch = [tr_cache[rng.randrange(len(tr_cache))] for _ in range(batch_size)]
        batch.sort(key=lambda x: len(x[1]) + len(x[2]))
        step_loss = 0.0

        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            el, ll, top_is, top_ps = [], [], [], []
            for qt, qi, ri, ti, tp in sub:
                pf = sample_random_prefix(prompt_stream, rng, 128, 256) if (prefix_robust and should_add_prefix(qt)) else None
                e, l, _ = build_soft_prompt_inputs(s_embed, tmpl, [sp], qi, ri, None, pf, device, dtype)
                el.append(e); ll.append(l); top_is.append(ti); top_ps.append(tp)

            be, bm, bl = pad_and_stack_train_batch(el, ll, pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm)
            shift_logits = out.logits[:, :-1, :]
            shift_labels = bl[:, 1:]
            per_sample = []
            for i in range(len(sub)):
                mask = shift_labels[i] != -100
                s_resp = shift_logits[i][mask].float()
                ti = top_is[i].to(device)
                tp = top_ps[i].float().to(device)
                n = min(s_resp.size(0), ti.size(0))
                s_lp = F.log_softmax(s_resp[:n], dim=-1)
                s_top_lp = s_lp.gather(-1, ti[:n])
                # Exact conditional expectation of Eq. (1): -sum_v p_T(v) log p_S(v)
                ce = -(tp[:n] * s_top_lp).sum(dim=-1).mean()
                per_sample.append(ce)
            loss = torch.stack(per_sample).mean()
            step_loss += loss.item()
            (loss / num_mb).backward()
            del be, bm, bl, out

        torch.nn.utils.clip_grad_norm_([sp], max_norm=1.0)
        optimizer.step()

        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True

        avg_loss = step_loss / num_mb
        losses.append(round(avg_loss, 4))

        if track_curve and step in [10, 20, 40, num_updates]:
            kl_v = eval_heldout_kl(student, s_embed, tmpl, val_cache, [sp])
            g_v = 100.0 * (1.0 - kl_v / kl_student_base)
            curve.append({"step": step, "train_rb_ce": round(avg_loss, 4), "heldout_kl": round(kl_v, 5), "G_pct": round(g_v, 2)})
            print(
                f"  [{tag}] step {step:3d}/{num_updates} | rb_ce={avg_loss:.4f} | "
                f"heldout_KL={kl_v:.5f} | G={g_v:+.2f}% | elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
        elif (not track_curve) and (step in [1, 30, num_updates]):
            print(
                f"  [{tag}] step {step:3d}/{num_updates} | rb_ce={avg_loss:.4f} | "
                f"lr={lr:.5f} | elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    # Appendix G.2 Embedding Geometry Analysis
    with torch.no_grad():
        final_sp = sp.detach().clone()
        W = s_embed.weight.detach().float()
        W_norm = F.normalize(W, dim=-1)
        init_nn = (F.normalize(init_sp, dim=-1) @ W_norm.T).argmax(dim=-1)
        final_nn = (F.normalize(final_sp, dim=-1) @ W_norm.T).argmax(dim=-1)
        nn_unchanged_pct = 100.0 * (init_nn == final_nn).float().mean().item()
        med_cos = float(F.cosine_similarity(final_sp, init_sp, dim=-1).median().item())
        rms_final = float(final_sp.pow(2).mean().sqrt().item())
        rms_init = float(init_sp.pow(2).mean().sqrt().item())

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
        "num_train_traces": len(tr_cache),
        "train_seconds": round(time.time() - t0, 1),
        "initial_rb_ce": losses[0],
        "final_rb_ce": losses[-1],
        "nn_token_unchanged_pct": round(nn_unchanged_pct, 2),
        "median_cosine_to_init": round(med_cos, 4),
        "effective_rank_delta": round(eff_rank, 2),
        "rms_init": round(rms_init, 5),
        "rms_final": round(rms_final, 5),
        "curve": curve,
    }
    print(
        f"  [{tag} Geometry] NN unchanged={meta['nn_token_unchanged_pct']:.1f}% | "
        f"med_cos={meta['median_cosine_to_init']:.3f} | eff_rank={meta['effective_rank_delta']:.1f}/{num_vectors} | "
        f"rms {meta['rms_init']:.4f}->{meta['rms_final']:.4f}",
        flush=True,
    )
    return final_sp, init_sp, meta


# ---------------------------------------------------------------------------
# 5. Phase 3: Full Mechanistic & Compositional Analysis (Section 4 & Table 2/5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_token_shifts(
    student, s_embed, tokenizer, tmpl, val_cache: List[Tuple], sp: torch.Tensor
) -> Dict[str, Any]:
    tok_deltas = []
    # Also track specific reasoning/verification discourse tokens from Fig. 6b / 11b
    WATCH = ["?", "But", " But", "Wait", " Wait", "Check", " check", "Let", " Let", "So", " So", "Thus", " Thus", ".\n\n", "However", " However"]
    watch_ids = {}
    for w in WATCH:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            watch_ids[w] = ids[0]
    watch_s_probs = collections.defaultdict(list)
    watch_sp_probs = collections.defaultdict(list)
    watch_t_probs = collections.defaultdict(list)

    for q_ids, r_ids, t_lp in val_cache:
        s_lp, _ = compute_response_logprobs(student, s_embed, tmpl, [], q_ids, r_ids)
        sp_lp, _ = compute_response_logprobs(student, s_embed, tmpl, [sp], q_ids, r_ids)
        n = min(len(r_ids), s_lp.size(0), sp_lp.size(0), t_lp.size(0))
        r_tensor = torch.tensor(r_ids[:n], device=s_lp.device)
        s_tok = s_lp[:n].gather(1, r_tensor.unsqueeze(1)).squeeze(1)
        sp_tok = sp_lp[:n].gather(1, r_tensor.unsqueeze(1)).squeeze(1)
        diff = (sp_tok - s_tok).cpu().tolist()
        for tok_id, d_val in zip(r_ids[:n], diff):
            tok_deltas.append((d_val, tokenizer.decode([tok_id])))

        # Track mean probabilities on discourse tokens (Fig. 11b style)
        t_lp_c = t_lp[:n].cuda()
        for w, tid in watch_ids.items():
            watch_t_probs[w].append(t_lp_c[:, tid].exp().mean().item())
            watch_s_probs[w].append(s_lp[:n, tid].exp().mean().item())
            watch_sp_probs[w].append(sp_lp[:n, tid].exp().mean().item())

    tok_deltas.sort(key=lambda x: x[0])
    k_top = min(600, max(50, len(tok_deltas) // 6))
    lowered = collections.Counter(t for _, t in tok_deltas[:k_top]).most_common(12)
    raised = collections.Counter(t for _, t in tok_deltas[-k_top:]).most_common(12)

    discourse = {}
    for w in watch_ids:
        t_m = sum(watch_t_probs[w]) / len(watch_t_probs[w])
        s_m = sum(watch_s_probs[w]) / len(watch_s_probs[w])
        sp_m = sum(watch_sp_probs[w]) / len(watch_sp_probs[w])
        if t_m > 1e-6:
            discourse[w] = {
                "student_rel_to_teacher": round(s_m / t_m, 3),
                "sp_rel_to_teacher": round(sp_m / t_m, 3),
            }
    return {
        "top_raised_tokens": raised,
        "top_lowered_tokens": lowered,
        "discourse_prob_rel_teacher": discourse,
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Stage A (v2 Clean): Composable Soft-Prompt Distillation Reproduction ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    ifbench_inps = load_ifbench_subset(n=40, seed=42)

    results: Dict[str, Any] = {
        "student_model": STUDENT_ID,
        "if_teacher": IF_TEACHER_ID,
        "math_teacher": MATH_TEACHER_ID,
        "ifbench_evals": {},
        "distributional_G": {},
        "mechanistic_IF": {},
        "mechanistic_Math": {},
        "sp_meta": {},
    }

    def save_results():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Phase 1: Traces + Official IFBench Teacher Score
    traces = collect_teacher_traces_and_eval_teachers(tokenizer, tmpl, ifbench_inps)
    results["ifbench_evals"]["teacher_M2RL_RL_IF"] = traces["teacher_if_ifbench"]
    save_results()

    # Precompute Exact Teacher Top-64 Train & Full-Vocab Val Logprobs
    if_tr_cache, if_val_cache, if_stream = precompute_teacher_targets(
        IF_TEACHER_ID, tokenizer, tmpl, traces["if_train_traces"], traces["if_val_traces"],
        max_train_resp=768, max_val_resp=1024, topk=64, tag="IF",
    )
    math_tr_cache, math_val_cache, math_stream = precompute_teacher_targets(
        MATH_TEACHER_ID, tokenizer, tmpl, traces["math_train_traces"], traces["math_val_traces"],
        max_train_resp=768, max_val_resp=1024, topk=64, tag="Math",
    )

    # Load Frozen Student ONCE
    print(f"\n=== Phase 2: Loading Frozen Student ({STUDENT_ID}) ===", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad = False
    s_embed = student.get_input_embeddings()

    # Baseline Held-Out KL(T || S)
    kl_S_if = eval_heldout_kl(student, s_embed, tmpl, if_val_cache, [])
    kl_S_math = eval_heldout_kl(student, s_embed, tmpl, math_val_cache, [])
    print(f"Held-out Student baselines: KL(T_IF||S)={kl_S_if:.5f} | KL(T_Math||S)={kl_S_math:.5f}", flush=True)

    # Distill 6 Soft Prompts (L=256, 60 updates, Rao-Blackwellized Eq. 1)
    sp_if, sp_if_init, meta_if = distill_sp_rb(
        student, tokenizer, tmpl, if_tr_cache, if_val_cache, if_stream, kl_S_if,
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="muon",
        prefix_robust=True, lr_max=0.025, seed=42, track_curve=True, tag="P_if_Muon_PR",
    )
    sp_math, sp_math_init, meta_math = distill_sp_rb(
        student, tokenizer, tmpl, math_tr_cache, math_val_cache, math_stream, kl_S_math,
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="muon",
        prefix_robust=True, lr_max=0.020, seed=43, track_curve=True, tag="P_math_Muon_PR",
    )
    sp_if_nopr, _, meta_if_nopr = distill_sp_rb(
        student, tokenizer, tmpl, if_tr_cache, if_val_cache, if_stream, kl_S_if,
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="muon",
        prefix_robust=False, lr_max=0.025, seed=42, track_curve=False, tag="P_if_NoPR",
    )
    sp_math_nopr, _, meta_math_nopr = distill_sp_rb(
        student, tokenizer, tmpl, math_tr_cache, math_val_cache, math_stream, kl_S_math,
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="muon",
        prefix_robust=False, lr_max=0.020, seed=43, track_curve=False, tag="P_math_NoPR",
    )
    sp_if_aw, _, meta_if_aw = distill_sp_rb(
        student, tokenizer, tmpl, if_tr_cache, if_val_cache, if_stream, kl_S_if,
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="adamw",
        prefix_robust=True, lr_max=0.008, seed=42, track_curve=False, tag="P_if_AdamW_PR",
    )
    sp_math_aw, _, meta_math_aw = distill_sp_rb(
        student, tokenizer, tmpl, math_tr_cache, math_val_cache, math_stream, kl_S_math,
        num_vectors=256, num_updates=60, batch_size=16, optimizer_type="adamw",
        prefix_robust=True, lr_max=0.007, seed=43, track_curve=False, tag="P_math_AdamW_PR",
    )

    for name, ten, meta in [
        ("P_if_muon_pr", sp_if, meta_if),
        ("P_math_muon_pr", sp_math, meta_math),
        ("P_if_nopr", sp_if_nopr, meta_if_nopr),
        ("P_math_nopr", sp_math_nopr, meta_math_nopr),
        ("P_if_adamw_pr", sp_if_aw, meta_if_aw),
        ("P_math_adamw_pr", sp_math_aw, meta_math_aw),
    ]:
        torch.save(ten.cpu(), os.path.join(OUT_DIR, f"{name}.pt"))
        results["sp_meta"][name] = meta
    save_results()

    # === Phase 3: Compositional & Mechanistic Analysis (Section 4 & Table 2/5) ===
    print("\n=== Phase 3: Compositional G(A) & Section 4 Mechanistic Analysis ===", flush=True)

    def g_pct(kl_a: float, kl_s: float) -> float:
        return round(100.0 * (1.0 - kl_a / max(kl_s, 1e-8)), 2)

    comp_configs = [
        ("student_no_sp", []),
        ("sp_init", [sp_if_init]),
        ("single_P_if", [sp_if]),
        ("single_P_math", [sp_math]),
        ("cspd_concat_Math_IF", [sp_math, sp_if]),
        ("cspd_concat_IF_Math", [sp_if, sp_math]),
        ("ablation_nopr_Math_IF", [sp_math_nopr, sp_if_nopr]),
        ("ablation_nopr_IF_Math", [sp_if_nopr, sp_math_nopr]),
        ("ablation_adamw_Math_IF", [sp_math_aw, sp_if_aw]),
        ("single_P_if_adamw", [sp_if_aw]),
        ("single_P_math_adamw", [sp_math_aw]),
    ]

    for cname, sps in comp_configs:
        kl_i = eval_heldout_kl(student, s_embed, tmpl, if_val_cache, sps)
        kl_m = eval_heldout_kl(student, s_embed, tmpl, math_val_cache, sps)
        gi, gm = g_pct(kl_i, kl_S_if), g_pct(kl_m, kl_S_math)
        results["distributional_G"][cname] = {
            "kl_if": round(kl_i, 5),
            "kl_math": round(kl_m, 5),
            "G_if_pct": gi,
            "G_math_pct": gm,
            "G_avg_pct": round((gi + gm) / 2.0, 2),
        }
        print(f"  [G(A) | {cname:24s}] G_IF={gi:+6.2f}% (KL={kl_i:.4f}) | G_Math={gm:+6.2f}% (KL={kl_m:.4f}) | Avg={(gi+gm)/2:+6.2f}%", flush=True)

    # Written instructions vs Soft Prompt & Attention Pathway Ablation K(A) for BOTH IF and Math
    IF_INSTS = {
        "verify_instruction": "Before writing your final response, carefully list every formatting, length, keyword, punctuation, and structural constraint in the user's request, draft your answer, and double-check that every single constraint is strictly satisfied.",
        "check_instruction": "Follow all user instructions and formatting constraints strictly. Double-check every constraint before concluding.",
        "neutral_instruction": "Read the request carefully and answer.",
    }
    MATH_INSTS = {
        "verify_instruction": "Do not conclude too quickly. After each intermediate result, question it and check it before moving on. Double-check results with another method and consider edge cases before giving your final answer.",
        "check_instruction": "Do not conclude too quickly. After each intermediate result, question it and check it before moving on.",
        "neutral_instruction": "Read the problem carefully and answer.",
    }

    for dom_name, val_c, kl_base, sp_main, sp_aw, sp_ini, inst_map in [
        ("mechanistic_IF", if_val_cache, kl_S_if, sp_if, sp_if_aw, sp_if_init, IF_INSTS),
        ("mechanistic_Math", math_val_cache, kl_S_math, sp_math, sp_math_aw, sp_math_init, MATH_INSTS),
    ]:
        kl_full = eval_heldout_kl(student, s_embed, tmpl, val_c, [sp_main], attention_mode="full")
        kl_aw = eval_heldout_kl(student, s_embed, tmpl, val_c, [sp_aw], attention_mode="full")
        kl_ini = eval_heldout_kl(student, s_embed, tmpl, val_c, [sp_ini], attention_mode="full")
        inst_kls = {}
        for k, txt in inst_map.items():
            s_ids = tokenizer.encode(txt, add_special_tokens=False)
            inst_kls[k] = eval_heldout_kl(student, s_embed, tmpl, val_c, [], system_text_ids=s_ids)

        kl_resp = eval_heldout_kl(student, s_embed, tmpl, val_c, [sp_main], attention_mode="response_only")
        kl_inp = eval_heldout_kl(student, s_embed, tmpl, val_c, [sp_main], attention_mode="input_only")
        kl_blk = eval_heldout_kl(student, s_embed, tmpl, val_c, [sp_main], attention_mode="blocked")

        def k_rel(kl_a: float, ref: float) -> float:
            denom = ref - kl_full
            return round(100.0 * (ref - kl_a) / denom, 2) if abs(denom) > 1e-6 else 0.0

        tok_info = analyze_token_shifts(student, s_embed, tokenizer, tmpl, val_c, sp_main)

        m_dict = {
            "num_traces": len(val_c),
            "mean_kl": {
                "student": round(kl_base, 5),
                "sp_muon": round(kl_full, 5),
                "sp_adamw": round(kl_aw, 5),
                "sp_init": round(kl_ini, 5),
                **{k: round(v, 5) for k, v in inst_kls.items()},
                "attn_response_only": round(kl_resp, 5),
                "attn_input_only": round(kl_inp, 5),
                "attn_blocked": round(kl_blk, 5),
            },
            "kl_gap_closed_G_pct": {
                "student": 0.0,
                "sp_muon": g_pct(kl_full, kl_base),
                "sp_adamw": g_pct(kl_aw, kl_base),
                "sp_init": g_pct(kl_ini, kl_base),
                **{k: g_pct(v, kl_base) for k, v in inst_kls.items()},
            },
            "attention_pathway_K_vs_student_pct": {
                "full": 100.0,
                "response_only": k_rel(kl_resp, kl_base),
                "input_only": k_rel(kl_inp, kl_base),
                "blocked": k_rel(kl_blk, kl_base),
            },
            "attention_pathway_K_vs_blocked_pct": {
                "full": 100.0,
                "response_only": k_rel(kl_resp, kl_blk),
                "input_only": k_rel(kl_inp, kl_blk),
                "blocked": 0.0,
            },
            **tok_info,
        }
        results[dom_name] = m_dict
        print(f"\n  [{dom_name}] G(A): {m_dict['kl_gap_closed_G_pct']}", flush=True)
        print(f"  [{dom_name}] K_vs_student: {m_dict['attention_pathway_K_vs_student_pct']}", flush=True)
        print(f"  [{dom_name}] K_vs_blocked: {m_dict['attention_pathway_K_vs_blocked_pct']}", flush=True)
        print(f"  [{dom_name}] Top raised:  {tok_info['top_raised_tokens'][:6]}", flush=True)
        print(f"  [{dom_name}] Top lowered: {tok_info['top_lowered_tokens'][:6]}", flush=True)
    save_results()

    # === Phase 4: IFBench End-to-End Task Generation ===
    print("\n=== Phase 4: IFBench Task Generation (40 prompts, max_new=3072) ===", flush=True)
    ifbench_configs = [
        ("student_no_sp", []),
        ("single_P_if", [sp_if]),
        ("cspd_concat_Math_IF", [sp_math, sp_if]),
        ("cspd_concat_IF_Math", [sp_if, sp_math]),
        ("ablation_nopr_Math_IF", [sp_math_nopr, sp_if_nopr]),
        ("ablation_adamw_Math_IF", [sp_math_aw, sp_if_aw]),
    ]
    for cname, sps in ifbench_configs:
        results["ifbench_evals"][cname] = eval_ifbench(
            student, tokenizer, tmpl, ifbench_inps, sps, batch_size=40, max_new_tokens=3072, tag=cname
        )
        save_results()

    # Compute Recovery % on IFBench relative to Student -> Teacher M2RL-RL_IF
    s_ifb = results["ifbench_evals"]["student_no_sp"]["prompt_strict"]
    t_ifb = results["ifbench_evals"]["teacher_M2RL_RL_IF"]["prompt_strict"]
    for cname, _ in ifbench_configs:
        v = results["ifbench_evals"][cname]["prompt_strict"]
        rec = round(100.0 * (v - s_ifb) / (t_ifb - s_ifb), 2) if abs(t_ifb - s_ifb) > 1e-6 else 0.0
        results["ifbench_evals"][cname]["recovery_strict_pct"] = rec

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save_results()
    print(f"\n=== Stage A v2 COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
