"""
Suite 6 (exp8): KV-Prefix Inversion to Context-Robust Soft Prompts (Q1) &
                Direct Teacher-Free Context-Robust Soft Prompt Construction (Q2)
================================================================================
Directly investigates the user's two core research questions on frozen M2RL-SFT (Qwen3-4B):

  Q1: Can an RL-trained or RL-distilled KV-based soft prompt K* be inverted (`inv`) into
      a genuinely Context-Robust Input-Space Soft Prompt P?
      We test BOTH KV sources:
        - Source 1A: KV_Distilled_L16 (distilled from M2RL-RL_IF dense Top-64 logits)
        - Source 1B: KV_GRPO_TF_L16   (trained directly via Online On-Policy GRPO in KV space,
                                       100% ZERO Post-RL teacher!)
      And compare 4 mathematical inversion operators I(K*) -> P:
        - Inv1a_GeoKV_DataFree    : Zero-data direct geometric KV pre-image (L=16, MSE on K,V across 36 layers)
        - Inv1b_CR_GeoKV_DataFree : Zero-data context-robust geometric pre-image (L=16, across pre-contexts)
        - Inv2_LogitCR_L256       : Multi-slot functional logit inversion + L_inv (throws away RL teacher!)
        - Inv3_FuncOp_L256 & L64  : FULL Multi-Layer Functional Operator Inversion (output hybrid KL +
                                    deep residual-stream hidden alignment at layers {12,24,36} + L_inv!)

  Q2: Can high-performing Context-Robust Soft Prompts be constructed DIRECTLY without ANY Post-RL model?
      (Strictly executed BEFORE M2RL-RL_IF is ever loaded on GPU!)
      We construct and compare 4 fundamentally distinct Teacher-Free CR methods:
        - TF1_CR_KVGRPO_Inv       : Online KV-Space GRPO -> Inv-3 Functional Inversion -> CR-SP (unifies Q1 & Q2!)
        - TF2_CR_Direct_GRPO      : Direct Multi-Slot Online GRPO on Layer-0 Input-SP (L=64, P0..P4 rollouts)
        - TF3_CR_Anchored_SelfBoot: Multi-Context Verified Self-Bootstrap + Bare-Student KL Anchor + L_inv
        - TF4_CR_SelfRubric       : Zero-RL Rubric-Prompted Self-Teacher + Verifier Filter -> Inv-3 CR-SP

Evaluates:
  - Layer-by-layer geometric inversion error profile (layers 0..35) for Inv1a & Inv1b
  - Full 11-Method x 6-Position (P0..P5) Distributional & Hidden-State Drift Matrix
  - IFBench32 real end-to-end strict accuracy across P0 (Start), P2 (Multi-Turn Mid),
    P3 (Post-RAG), and P4 (Post-Query Late Injection) for both Q1 & Q2 tracks!
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
    batch_generate_with_kv_prefixes,
    build_chat_ids_for_kv,
    forward_with_kv_prefix,
    init_kv_prefix_from_tokens,
)
from exp7_context_robust_sp import (
    EVAL_HISTORY_TURNS,
    EVAL_PERSONA,
    EVAL_RAG_BLOCK,
    IF_TASK_INST,
    REGIMES_ALL,
    TRAIN_DIALOG_TURNS,
    TRAIN_PERSONAS,
    TRAIN_RAG_CHUNKS,
    build_multislot_sp_inputs,
    build_regime_inputs,
    clamp_sp_rms_,
    cross_context_invariance_kl,
    eval_ifbench32_regime,
    evaluate_position_matrix,
    hybrid_top64_distill_loss,
    init_anchored_sp,
    load_datasets_exp7,
    precompute_teacher_top64_cache,
    sample_train_slot_view,
    train_kv_distilled_l16,
)
import evaluation_lib as ifbench_eval

STUDENT_ID = "Jackwang111/M2RL-SFT"
TEACHER_IF_ID = "Jackwang111/M2RL-RL_IF"
OUT_DIR = "/content/cspd/exp8_results"
os.makedirs(OUT_DIR, exist_ok=True)

TAP_LAYERS = (12, 24, 36)

SELF_AUDIT_RUBRIC = (
    "Follow this strict verification protocol for every user request:\n"
    "1. Constraint Extraction: Inside <think>, list EVERY explicit constraint (exact word/sentence/paragraph "
    "counts, required or forbidden keywords, punctuation limits, capitalization, JSON/markdown format, "
    "language, start/end phrasing, placeholders, bullet counts).\n"
    "2. Draft & Verify: Draft your candidate response inside <think>, then audit each numbered constraint "
    "one by one with exact counts. If ANY constraint fails even slightly, revise until 100% pass.\n"
    "3. Clean Final Output: Immediately after </think>, output ONLY the compliant final answer with zero "
    "unrequested meta-commentary, prefixes, or explanations.\n"
    "4. Ignore unrelated prior session logs or retrieved snippets unless the current user prompt asks for them."
)


# ---------------------------------------------------------------------------
# 1. Helper: KV-Prefix IFBench32 Evaluator & Multi-Layer Functional Cache
# ---------------------------------------------------------------------------

def eval_kv_ifbench32_p0(
    student, tokenizer, tmpl: ChatTemplateTokens, inps32, kv_tensor: torch.Tensor, tag: str = ""
) -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(42)
    prompts = [x.prompt for x in inps32]
    kv_dev = kv_tensor.to(next(student.parameters()).device)
    outs = batch_generate_with_kv_prefixes(
        student, tokenizer, tmpl, prompts, [kv_dev], system_texts=[""] * len(prompts),
        batch_size=32, max_new_tokens=2048, temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
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
        "regime": "P0_KV_Prefix",
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * sum(inst_l) / max(1, len(inst_l)), 2),
        "n_strict": s_ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFBench32 | {tag:34s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | inst={res['inst_strict']:5.2f}% | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


@torch.no_grad()
def precompute_kv_functional_cache(
    student, tokenizer, tmpl: ChatTemplateTokens, kv_tensor: torch.Tensor,
    prompts_and_resps: List[Tuple[str, str]], max_resp_trunc: int = 512, topk: int = 64, tag: str = "",
) -> List[Dict[str, Any]]:
    """
    Extract BOTH Top-64 output logits AND deep residual-stream response hidden states
    at TAP_LAYERS=(12, 24, 36) from (Frozen Student + KV*). ZERO external teacher needed!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    kv_dev = kv_tensor.to(device)
    student.eval()
    student.config.use_cache = True

    cache = []
    for p, r in prompts_and_resps:
        qi = tokenizer.encode(p, add_special_tokens=False)
        ri = tokenizer.encode(r, add_special_tokens=False)[:max_resp_trunc]
        if len(ri) < 16:
            continue
        ids, lbl = build_chat_ids_for_kv(tmpl, qi, ri, None, True, None)
        bi = torch.tensor([ids], dtype=torch.long, device=device)
        bm = torch.ones_like(bi)
        # Forward with ReadOnlyPrefixCache + output_hidden_states=True!
        from kv_prefix_core import make_readonly_kv_cache
        cache_obj = make_readonly_kv_cache([kv_dev], 1, device, next(student.parameters()).dtype)
        L_kv = cache_obj.get_seq_length()
        T = bi.size(1)
        cache_pos = torch.arange(L_kv, L_kv + T, dtype=torch.long, device=device)
        # Build exact 4D additive causal mask [1, 1, T, L_kv + T] matching forward_with_kv_prefix!
        min_val = torch.finfo(next(student.parameters()).dtype).min
        causal = torch.triu(torch.full((T, T), min_val, device=device, dtype=next(student.parameters()).dtype), diagonal=1)
        prefix_zeros = torch.zeros((T, L_kv), device=device, dtype=next(student.parameters()).dtype)
        mask4d = torch.cat([prefix_zeros, causal], dim=-1).unsqueeze(0).unsqueeze(0)

        out = student(
            input_ids=bi, attention_mask=mask4d,
            past_key_values=cache_obj, cache_position=cache_pos,
            use_cache=True, output_hidden_states=True,
        )
        sl = out.logits[0, :-1, :].float()
        lb = torch.tensor(lbl[1:], dtype=torch.long, device=device)
        m = lb != -100
        probs = F.softmax(sl[m], dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
        # Tap response hidden states aligned with shifted logits positions ([:-1][m])
        h_taps = {}
        for ly in TAP_LAYERS:
            h_taps[ly] = out.hidden_states[ly][0, :-1, :][m].to(torch.float16).cpu()
        cache.append({
            "q_ids": qi, "r_ids": ri,
            "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "ent": ent.cpu(),
            "h_taps": h_taps,
        })
        del bi, bm, out, mask4d
    print(f"  [FuncCache {tag:16s}] Extracted {len(cache)} (Top-64 + L12/24/36 hidden) targets in {time.time()-t0:.1f}s", flush=True)
    return cache


# ---------------------------------------------------------------------------
# 2. Q1 Part A: Geometric Data-Free KV Pre-Image Inversion (Inv1a & Inv1b)
# ---------------------------------------------------------------------------

def invert_kv_geometric_datafree(
    student, tokenizer, kv_target: torch.Tensor, train_pools: Optional[Dict[str, Any]] = None,
    context_robust: bool = False, num_steps: int = 160, lr_max: float = 0.025,
    max_rms: float = 0.060, seed: int = 42, tag: str = "Inv1a_GeoKV",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    100% DATA-FREE Geometric Inversion:
      Finds P in R^{L_kv x d} whose frozen-backbone forward KV trajectory Phi_{theta_0}(P)
      best matches K* across all 36 layers!
      - Inv1a (context_robust=False): matches (K, V) at position 0..L_kv-1 directly.
      - Inv1b (context_robust=True):  matches (K, V) at pos 0 PLUS RoPE-free V states across
                                      random preceding contexts C_pre!
    Also logs exact layer-by-layer relative reconstruction error eps_l across all 36 layers!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    L_kv = int(kv_target.size(2))
    num_layers = int(kv_target.size(0))
    kv_t = kv_target.to(device=device, dtype=torch.float32)  # [36, 2, L_kv, H_kv, D_h]

    sp = init_anchored_sp(s_emb, tokenizer, IF_TASK_INST, num_vectors=L_kv, anchored=False)
    opt = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)

    student.gradient_checkpointing_disable()
    student.eval()
    student.config.use_cache = True
    log_hist = {}

    for step in range(1, num_steps + 1):
        prog = (step - 1) / max(1, num_steps - 1)
        lr = lr_max * 0.08 + 0.5 * (lr_max * 0.92) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad()

        # Clean pos-0 geometric KV matching (exact RoPE alignment for both K and V)
        emb0 = sp.to(dtype).unsqueeze(0)  # [1, L_kv, d]
        out0 = student(inputs_embeds=emb0, use_cache=True)
        pkv0 = out0.past_key_values
        loss_k, loss_v = 0.0, 0.0
        for ly in range(num_layers):
            k_cur = pkv0.layers[ly].keys[0].permute(1, 0, 2).float()
            v_cur = pkv0.layers[ly].values[0].permute(1, 0, 2).float()
            k_ref = kv_t[ly, 0]
            v_ref = kv_t[ly, 1]
            # Normalized relative MSE per layer so every layer has equal scale
            loss_k = loss_k + F.mse_loss(k_cur, k_ref) / (k_ref.pow(2).mean().clamp_min(1e-4))
            loss_v = loss_v + F.mse_loss(v_cur, v_ref) / (v_ref.pow(2).mean().clamp_min(1e-4))
        loss_clean = (loss_k + loss_v) / (2.0 * num_layers)

        loss_ctx_v = torch.tensor(0.0, device=device)
        if context_robust and train_pools is not None:
            # Match RoPE-invariant Value states V^(l) when P is preceded by random context C_pre!
            pre_ids = rng.choice(train_pools["personas"])
            if rng.random() < 0.5:
                pre_ids = pre_ids + sample_random_prefix(train_pools["stream"], rng, 32, 128)
            pre_e = s_emb(torch.tensor(pre_ids, dtype=torch.long, device=device)).to(dtype)
            emb_c = torch.cat([pre_e, sp.to(dtype)], dim=0).unsqueeze(0)
            m0 = pre_e.size(0)
            out_c = student(inputs_embeds=emb_c, use_cache=True)
            pkv_c = out_c.past_key_values
            lv_c = 0.0
            for ly in range(num_layers):
                v_cur_c = pkv_c.layers[ly].values[0, :, m0 : m0 + L_kv, :].permute(1, 0, 2).float()
                v_ref = kv_t[ly, 1]
                lv_c = lv_c + F.mse_loss(v_cur_c, v_ref) / (v_ref.pow(2).mean().clamp_min(1e-4))
            loss_ctx_v = lv_c / num_layers
            del out_c, pkv_c

        loss = loss_clean + (0.6 * loss_ctx_v if context_robust else 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)
        del out0, pkv0

        if step in {1, 40, 100, num_steps}:
            log_hist[str(step)] = {"rel_kv_mse": round(float(loss_clean.item()), 4), "ctx_v_mse": round(float(loss_ctx_v.item()), 4)}
            print(f"  [{tag:22s}] step {step:3d}/{num_steps} | rel_kv_mse={loss_clean.item():.4f} | ctx_v={loss_ctx_v.item():.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    # Measure final exact layer-by-layer relative reconstruction error profile (0..35)
    with torch.no_grad():
        out_f = student(inputs_embeds=sp.detach().to(dtype).unsqueeze(0), use_cache=True)
        pkv_f = out_f.past_key_values
        layer_err_k, layer_err_v = [], []
        for ly in range(num_layers):
            k_cur = pkv_f.layers[ly].keys[0].permute(1, 0, 2).float()
            v_cur = pkv_f.layers[ly].values[0].permute(1, 0, 2).float()
            ek = float((k_cur - kv_t[ly, 0]).norm() / kv_t[ly, 0].norm().clamp_min(1e-6))
            ev = float((v_cur - kv_t[ly, 1]).norm() / kv_t[ly, 1].norm().clamp_min(1e-6))
            layer_err_k.append(round(ek, 4))
            layer_err_v.append(round(ev, 4))

    meta = {
        "tag": tag, "method": "geometric_kv_preimage", "context_robust": context_robust,
        "num_vectors": L_kv, "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "mean_rel_err_K": round(sum(layer_err_k) / num_layers, 4),
        "mean_rel_err_V": round(sum(layer_err_v) / num_layers, 4),
        "layer0_err_KV": [layer_err_k[0], layer_err_v[0]],
        "layer18_err_KV": [layer_err_k[18], layer_err_v[18]],
        "layer35_err_KV": [layer_err_k[35], layer_err_v[35]],
        "layer_err_k": layer_err_k, "layer_err_v": layer_err_v,
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    print(f"  [{tag:22s}] Layerwise RelErr K/V: L0={meta['layer0_err_KV']} | L18={meta['layer18_err_KV']} | L35={meta['layer35_err_KV']} | MeanK={meta['mean_rel_err_K']} MeanV={meta['mean_rel_err_V']}", flush=True)
    return sp.detach().clone(), meta


# ---------------------------------------------------------------------------
# 3. Q1 Part B: Functional Operator & Logit Inversion (Inv2 & Inv3!)
# ---------------------------------------------------------------------------

def invert_functional_cr_sp(
    student, tokenizer, tmpl: ChatTemplateTokens, func_cache: List[Dict[str, Any]],
    train_pools: Dict[str, Any], num_vectors: int = 256, num_updates: int = 38,
    batch_size: int = 10, lr_max: float = 0.006, use_hidden_align: bool = True,
    lambda_feat: float = 0.80, lambda_inv: float = 0.35, max_rms: float = 0.055,
    seed: int = 42, tag: str = "Inv3_FuncOp",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Functional Operator Inversion (Inv-3) & Logit-Only Inversion (Inv-2):
      Compiles a KV-prefix (or Self-Rubric teacher) into a Context-Robust Input-Space SP
      via Dual-View Multi-Slot Sampling + Hybrid Top-64 KL + Deep Residual Hidden Alignment
      at layers {12, 24, 36} + Cross-Context Behavioral Invariance L_inv!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    sp = init_anchored_sp(s_emb, tokenizer, IF_TASK_INST, num_vectors=num_vectors, anchored=True)
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

        batch = [func_cache[rng.randrange(len(func_cache))] for _ in range(batch_size)]
        tot_dist, tot_feat, tot_inv = 0.0, 0.0, 0.0

        for item in batch:
            qi, ri = item["q_ids"], item["r_ids"]
            ti = item["ti"].to(device)
            tp = item["tp"].float().to(device)

            e0, l0, _ = build_multislot_sp_inputs(
                s_emb, tmpl, qi, ri, sp_sys_start=[sp], device=device, model_dtype=dtype
            )
            slot_kw = sample_train_slot_view("full_cr_inv", rng, train_pools, sp)
            ec, lc, _ = build_multislot_sp_inputs(
                s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **slot_kw
            )
            be, bm, bl = pad_and_stack_train_batch([e0, ec], [l0, lc], pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm, output_hidden_states=use_hidden_align)
            sl = out.logits[:, :-1, :]
            lb = bl[:, 1:]
            m0 = lb[0] != -100
            mc = lb[1] != -100
            z0 = sl[0][m0].float()
            zc = sl[1][mc].float()

            loss_c = hybrid_top64_distill_loss(zc, ti, tp)
            loss_0 = hybrid_top64_distill_loss(z0, ti, tp)
            loss_inv = cross_context_invariance_kl(z0, zc, topk=32)

            loss_feat = torch.tensor(0.0, device=device)
            if use_hidden_align and "h_taps" in item:
                fl = []
                for ly in TAP_LAYERS:
                    h_ref = item["h_taps"][ly].float().to(device)
                    hc = out.hidden_states[ly][1, :-1, :][mc].float()
                    n_h = min(hc.size(0), h_ref.size(0))
                    fl.append((1.0 - F.cosine_similarity(hc[:n_h], h_ref[:n_h], dim=-1)).mean())
                loss_feat = torch.stack(fl).mean()

            loss = (loss_c + 0.35 * loss_0 + lambda_inv * loss_inv + (lambda_feat * loss_feat if use_hidden_align else 0.0)) / batch_size
            loss.backward()
            tot_dist += float(loss_c.item()) / batch_size
            tot_feat += float(loss_feat.item()) / batch_size
            tot_inv += float(loss_inv.item()) / batch_size
            del be, bm, bl, out, z0, zc

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)
        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True

        if step in {1, 15, 28, num_updates}:
            log_hist[str(step)] = {"dist": round(tot_dist, 4), "feat": round(tot_feat, 4), "inv": round(tot_inv, 4)}
            print(f"  [{tag:22s}] step {step:2d}/{num_updates} | dist={tot_dist:.4f} | feat={tot_feat:.4f} | inv={tot_inv:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "method": "func_op_inv" if use_hidden_align else "logit_cr_inv",
        "num_vectors": num_vectors, "use_hidden_align": use_hidden_align,
        "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


# ---------------------------------------------------------------------------
# 4. Q2: Four Teacher-Free Context-Robust Construction Engines (0 Post-RL!)
# ---------------------------------------------------------------------------

def train_online_kv_grpo(
    student, tokenizer, tmpl: ChatTemplateTokens, if_train: List[Dict[str, Any]],
    num_kv_tokens: int = 16, num_steps: int = 14, prompts_per_step: int = 4, group_size: int = 6,
    max_new_tokens: int = 768, lr_max: float = 0.012, beta_kl: float = 0.08, max_d_rms: float = 0.22,
    seed: int = 42, tag: str = "KV_GRPO_TF",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    TRUE Online On-Policy GRPO directly in 16-token Layerwise KV Space!
    ZERO Post-RL teacher! ZERO gradient checkpointing overhead! Tight Schulman Reverse-KL trust region
    + delta_rms bound keeps logits clean (avoiding off-policy DPO distortion).
    """
    t0 = time.time()
    device = next(student.parameters()).device
    inst_ids = tokenizer.encode(IF_TASK_INST, add_special_tokens=False)
    kv = init_kv_prefix_from_tokens(student, inst_ids, num_kv_tokens=num_kv_tokens)
    init_kv = kv.detach().clone()

    opt = torch.optim.AdamW([kv], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)
    online_curve = []

    for step in range(1, num_steps + 1):
        prog = (step - 1) / max(1, num_steps - 1)
        lr = lr_max * 0.2 + 0.5 * (lr_max * 0.8) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        step_prompts = [if_train[rng.randrange(len(if_train))] for _ in range(prompts_per_step)]
        flat_q = []
        for pm in step_prompts:
            flat_q.extend([pm["prompt"]] * group_size)

        student.eval()
        student.config.use_cache = True
        torch.manual_seed(seed * 100 + step)
        outs = batch_generate_with_kv_prefixes(
            student, tokenizer, tmpl, flat_q, [kv.detach()], system_texts=[""] * len(flat_q),
            batch_size=prompts_per_step * group_size, max_new_tokens=max_new_tokens,
            temperature=0.85, top_p=0.95, top_k=20, enable_thinking=True,
        )

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
                r_ids = tokenizer.encode(resp, add_special_tokens=False)[:640]
                if len(r_ids) >= 12 and abs(adv) > 0.05:
                    samples_to_opt.append((q_ids, r_ids, adv))

        mean_kl = 0.0
        if samples_to_opt:
            # Bare-student reference token log-probs (KV=None)
            ref_lps = []
            with torch.no_grad():
                for qi, ri, _ in samples_to_opt:
                    ids, lbl = build_chat_ids_for_kv(tmpl, qi, ri, None, True, None)
                    bi = torch.tensor([ids], dtype=torch.long, device=device)
                    out_r = student(input_ids=bi, use_cache=False)
                    sl = out_r.logits[0, :-1, :].float()
                    lb = torch.tensor(lbl[1:], dtype=torch.long, device=device)
                    m = lb != -100
                    lp = F.log_softmax(sl[m], dim=-1).gather(-1, lb[m].unsqueeze(-1)).squeeze(-1)
                    ref_lps.append(lp.detach())
                    del bi, out_r

            student.gradient_checkpointing_disable()
            student.eval()
            student.config.use_cache = True
            opt.zero_grad()
            n_opt = len(samples_to_opt)
            for (qi, ri, adv), r_lp in zip(samples_to_opt, ref_lps):
                ids, lbl = build_chat_ids_for_kv(tmpl, qi, ri, None, True, None)
                bi = torch.tensor([ids], dtype=torch.long, device=device)
                bm = torch.ones_like(bi)
                out_c = forward_with_kv_prefix(student, bi, bm, [kv])
                sl = out_c.logits[0, :-1, :].float()
                lb = torch.tensor(lbl[1:], dtype=torch.long, device=device)
                m = lb != -100
                cur_lp = F.log_softmax(sl[m], dim=-1).gather(-1, lb[m].unsqueeze(-1)).squeeze(-1)
                n_tok = min(cur_lp.size(0), r_lp.size(0))
                log_ratio = (r_lp[:n_tok] - cur_lp[:n_tok]).clamp(-10.0, 10.0)
                kl_est = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
                pg_loss = -(adv * cur_lp[:n_tok].mean()) + beta_kl * kl_est
                (pg_loss / n_opt).backward()
                mean_kl += float(kl_est.item()) / n_opt
                del bi, bm, out_c

            torch.nn.utils.clip_grad_norm_([kv], 1.0)
            opt.step()
            with torch.no_grad():
                delta = kv - init_kv
                dr = delta.pow(2).mean().sqrt().clamp_min(1e-8)
                if dr > max_d_rms:
                    kv.copy_(init_kv + delta * (max_d_rms / dr))

        rec = {
            "step": step,
            "mean_reward": round(tot_r / max(1, len(flat_q)), 4),
            "strict_pct": round(100.0 * tot_strict / max(1, len(flat_q)), 1),
            "closed_pct": round(100.0 * tot_closed / max(1, len(flat_q)), 1),
            "kl_to_ref": round(mean_kl, 4),
            "d_rms": round(float((kv.detach() - init_kv).pow(2).mean().sqrt().item()), 4),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        online_curve.append(rec)
        if step in {1, 5, 10, num_steps}:
            print(f"  [{tag:22s}] step {step:2d}/{num_steps} | rew={rec['mean_reward']:.3f} | strict={rec['strict_pct']:.1f}% | kl={rec['kl_to_ref']:.4f} | d_rms={rec['d_rms']:.4f} | t={rec['elapsed_sec']}s", flush=True)

    meta = {
        "tag": tag, "method": "online_kv_grpo", "num_kv_tokens": num_kv_tokens,
        "final_d_rms": round(float((kv.detach() - init_kv).pow(2).mean().sqrt().item()), 4),
        "train_sec": round(time.time() - t0, 1), "online_curve": online_curve,
    }
    return kv.detach().clone(), meta


def train_direct_multislot_input_grpo(
    student, tokenizer, tmpl: ChatTemplateTokens, if_train: List[Dict[str, Any]],
    train_pools: Dict[str, Any], num_vectors: int = 64, num_steps: int = 12,
    prompts_per_step: int = 4, group_size: int = 6, max_new_tokens: int = 768,
    lr_max: float = 0.0025, beta_kl: float = 0.05, lambda_inv: float = 0.25,
    max_rms: float = 0.055, seed: int = 108, tag: str = "TF2_CR_Direct_GRPO",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Q2 Method TF-2: Direct Multi-Slot Online On-Policy GRPO on compact Layer-0 Input-SP (L=64)!
    ZERO Post-RL teacher! ZERO KV scaffold!
    Samples online rollouts WITH REAL MULTI-TURN / RAG / POST-QUERY CONTEXTS during RL itself,
    so policy gradient directly rewards context-resilient instruction following!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    sp = init_anchored_sp(s_emb, tokenizer, IF_TASK_INST, num_vectors=num_vectors, anchored=True)
    opt = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(seed)
    online_curve = []

    for step in range(1, num_steps + 1):
        prog = (step - 1) / max(1, num_steps - 1)
        lr = lr_max * 0.2 + 0.5 * (lr_max * 0.8) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        step_prompts = [if_train[rng.randrange(len(if_train))] for _ in range(prompts_per_step)]
        # Sample one arbitrary context slot per prompt in the step!
        step_slots = [sample_train_slot_view("full_cr_inv", rng, train_pools, sp.detach()) for _ in step_prompts]

        # Batched multi-slot generation for all 24 rollouts
        student.eval()
        student.config.use_cache = True
        seq_embeds = []
        for pm, skw in zip(step_prompts, step_slots):
            q_ids = tokenizer.encode(pm["prompt"], add_special_tokens=False)
            e, _, _ = build_multislot_sp_inputs(s_emb, tmpl, q_ids, None, device=device, model_dtype=dtype, **skw)
            for _ in range(group_size):
                seq_embeds.append(e.squeeze(0))
        max_len = max(e.size(0) for e in seq_embeds)
        bs = len(seq_embeds)
        padded = pad_vec.expand(bs, max_len, seq_embeds[0].size(1)).clone()
        attn_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for i, e in enumerate(seq_embeds):
            L = e.size(0)
            padded[i, max_len - L :, :] = e
            attn_mask[i, max_len - L :] = 1

        torch.manual_seed(seed * 100 + step)
        with torch.no_grad():
            out_ids = student.generate(
                inputs_embeds=padded, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True, do_sample=True, temperature=0.85, top_p=0.95, top_k=20,
            )
        outs = [tokenizer.decode(row, skip_special_tokens=True) for row in out_ids]
        del padded, attn_mask, out_ids

        samples_to_opt = []
        tot_r, tot_strict, tot_closed = 0.0, 0, 0
        for p_i, (pm, skw) in enumerate(zip(step_prompts, step_slots)):
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
            # Re-bind slot kwargs without detached sp tensor
            skw_clean = {k: v for k, v in skw.items() if not k.startswith("sp_")}
            sp_key = [k for k in skw.keys() if k.startswith("sp_")][0]
            for resp, rew in zip(grp, rewards):
                adv = (rew - mean_r) / (std_r + 1e-4)
                r_ids = tokenizer.encode(resp, add_special_tokens=False)[:640]
                if len(r_ids) >= 12 and abs(adv) > 0.05:
                    samples_to_opt.append((q_ids, r_ids, adv, skw_clean, sp_key))

        mean_kl = 0.0
        if samples_to_opt:
            # Bare-student reference token log-probs under the SAME context (no SP)
            ref_lps = []
            with torch.no_grad():
                for qi, ri, _, skw_c, _ in samples_to_opt:
                    e_r, l_r, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **skw_c)
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
            for (qi, ri, adv, skw_c, sp_key), r_lp in zip(samples_to_opt, ref_lps):
                kw_cur = dict(skw_c)
                kw_cur[sp_key] = [sp]
                ec, lc, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **kw_cur)
                e0, l0, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, sp_sys_start=[sp], device=device, model_dtype=dtype)
                be, bm, bl = pad_and_stack_train_batch([e0, ec], [l0, lc], pad_vec)
                out = student(inputs_embeds=be, attention_mask=bm)
                sl = out.logits[:, :-1, :]
                lb = bl[:, 1:]
                m0 = lb[0] != -100
                mc = lb[1] != -100
                z0 = sl[0][m0].float()
                zc = sl[1][mc].float()
                cur_lp = F.log_softmax(zc, dim=-1).gather(-1, lb[1][mc].unsqueeze(-1)).squeeze(-1)
                n_tok = min(cur_lp.size(0), r_lp.size(0))
                log_ratio = (r_lp[:n_tok] - cur_lp[:n_tok]).clamp(-10.0, 10.0)
                kl_est = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
                l_inv = cross_context_invariance_kl(z0, zc, topk=32)
                pg_loss = -(adv * cur_lp[:n_tok].mean()) + beta_kl * kl_est + lambda_inv * l_inv
                (pg_loss / n_opt).backward()
                mean_kl += float(kl_est.item()) / n_opt
                del be, bm, bl, out, z0, zc

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
        if step in {1, 4, 8, num_steps}:
            print(f"  [{tag:22s}] step {step:2d}/{num_steps} | rew={rec['mean_reward']:.3f} | strict={rec['strict_pct']:.1f}% | kl={rec['kl_to_ref']:.4f} | rms={rec['rms']:.4f} | t={rec['elapsed_sec']}s", flush=True)

    meta = {
        "tag": tag, "method": "direct_multislot_grpo", "num_vectors": num_vectors,
        "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "online_curve": online_curve,
    }
    return sp.detach().clone(), meta


def train_tf_anchored_selfboot_cr(
    student, tokenizer, tmpl: ChatTemplateTokens, if_train: List[Dict[str, Any]],
    train_pools: Dict[str, Any], num_vectors: int = 256, num_updates: int = 38,
    batch_size: int = 10, lr_max: float = 0.005, lambda_anc: float = 0.55, lambda_inv: float = 0.35,
    max_rms: float = 0.055, seed: int = 202, tag: str = "TF3_CR_Anchored_SelfBoot",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Q2 Method TF-3: Context-Robust KL-Anchored Verified Self-Bootstrap!
    ZERO Post-RL teacher! ZERO policy-gradient variance!
    1. Harvests verified strict+closed winning traces from bare Student (3 rollouts/prompt).
    2. Precomputes bare-student clean Top-32 trust-region anchor distributions on winning traces.
    3. Optimizes Multi-Slot Verified-Win NLL + Bare-Student KL Trust-Region Anchor (prevents RFT overfitting!)
       + Cross-Context Behavioral Invariance L_inv across P0..P4!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    # Harvest 3 rollouts per prompt from bare frozen Student
    flat_q = []
    for pm in if_train:
        flat_q.extend([pm["prompt"]] * 3)
    student.eval()
    student.config.use_cache = True
    torch.manual_seed(seed)
    outs = batch_generate_with_soft_prompts(
        student, tokenizer, tmpl, flat_q, [], batch_size=32, max_new_tokens=768,
        temperature=0.85, top_p=0.95, top_k=20, enable_thinking=True,
    )
    wins = []
    for i, pm in enumerate(if_train):
        grp = outs[3 * i : 3 * i + 3]
        passed = []
        for resp in grp:
            if "</think>" in resp:
                strict, _ = verify_ifeval_response(resp, pm["iids"], pm["kwargs"])
                if strict:
                    passed.append(resp)
        if passed:
            w_val = 1.0 / max(1, len(passed))  # Upweight harder prompts (solved 1/3 > 3/3)
            for resp in passed[:2]:
                qi = tokenizer.encode(pm["prompt"], add_special_tokens=False)
                ri = tokenizer.encode(resp, add_special_tokens=False)[:512]
                if len(ri) >= 16:
                    wins.append({"q_ids": qi, "r_ids": ri, "weight": w_val})

    # Precompute bare-student clean Top-32 anchor probabilities on wins
    with torch.no_grad():
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
    print(f"  [{tag:22s}] Harvested {len(wins)} verified wins + bare-student anchors in {time.time()-t0:.1f}s", flush=True)

    sp = init_anchored_sp(s_emb, tokenizer, IF_TASK_INST, num_vectors=num_vectors, anchored=True)
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
        tot_nll, tot_anc, tot_inv = 0.0, 0.0, 0.0
        for item in batch:
            qi, ri = item["q_ids"], item["r_ids"]
            e0, l0, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, sp_sys_start=[sp], device=device, model_dtype=dtype)
            slot_kw = sample_train_slot_view("full_cr_inv", rng, train_pools, sp)
            ec, lc, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, device=device, model_dtype=dtype, **slot_kw)
            be, bm, bl = pad_and_stack_train_batch([e0, ec], [l0, lc], pad_vec)
            out = student(inputs_embeds=be, attention_mask=bm)
            sl = out.logits[:, :-1, :]
            lb = bl[:, 1:]
            m0 = lb[0] != -100
            mc = lb[1] != -100
            z0 = sl[0][m0].float()
            zc = sl[1][mc].float()

            nll_c = F.cross_entropy(zc, lb[1][mc], reduction="mean")
            nll_0 = F.cross_entropy(z0, lb[0][m0], reduction="mean")
            # Bare-student KL trust-region anchor (prevents RFT overfitting!)
            ati = item["anc_ti"].to(device)
            atp = item["anc_tp"].float().to(device)
            aent = item["anc_ent"].to(device)
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

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        clamp_sp_rms_(sp, max_rms)
        student.gradient_checkpointing_disable()
        student.eval()
        student.config.use_cache = True

        if step in {1, 15, 28, num_updates}:
            log_hist[str(step)] = {"nll": round(tot_nll, 4), "anc_kl": round(tot_anc, 4), "inv": round(tot_inv, 4)}
            print(f"  [{tag:22s}] step {step:2d}/{num_updates} | nll={tot_nll:.4f} | anc_kl={tot_anc:.4f} | inv={tot_inv:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "method": "cr_anchored_selfboot", "num_vectors": num_vectors, "num_wins": len(wins),
        "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


@torch.no_grad()
def precompute_self_rubric_functional_cache(
    student, tokenizer, tmpl: ChatTemplateTokens, if_train: List[Dict[str, Any]],
    max_resp_gen: int = 768, max_resp_trunc: int = 512, topk: int = 64,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Q2 Method TF-4 (Stage 1): Zero-RL, Zero-Post-RL Self-Context Teacher!
    Conditions the EXACT SAME frozen Student on SELF_AUDIT_RUBRIC, harvests verified wins,
    and extracts dense Top-64 logits + L12/24/36 hidden states on its own smooth manifold!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    rubric_ids = tokenizer.encode(SELF_AUDIT_RUBRIC, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype)

    # Generate 2 rollouts per prompt with (Frozen Student + SELF_AUDIT_RUBRIC)
    flat_items = []
    for pm in if_train:
        flat_items.extend([pm, pm])
    outs = []
    student.eval()
    student.config.use_cache = True
    torch.manual_seed(303)
    for start in range(0, len(flat_items), 32):
        sub = flat_items[start : start + 32]
        seq_e = []
        for pm in sub:
            qi = tokenizer.encode(pm["prompt"], add_special_tokens=False)
            e, _, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, None, sys_text_ids=rubric_ids, device=device, model_dtype=dtype)
            seq_e.append(e.squeeze(0))
        ml = max(x.size(0) for x in seq_e)
        bs = len(seq_e)
        padded = pad_vec.expand(bs, ml, seq_e[0].size(1)).clone()
        am = torch.zeros((bs, ml), dtype=torch.long, device=device)
        for i, x in enumerate(seq_e):
            L = x.size(0)
            padded[i, ml - L :, :] = x
            am[i, ml - L :] = 1
        oids = student.generate(
            inputs_embeds=padded, attention_mask=am, max_new_tokens=max_resp_gen,
            pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True, do_sample=True, temperature=0.75, top_p=0.95, top_k=20,
        )
        for row in oids:
            outs.append(tokenizer.decode(row, skip_special_tokens=True))
        del padded, am, oids

    selected = []
    n_strict = 0
    for i, pm in enumerate(if_train):
        best_r, best_sc, best_st = None, -1.0, False
        for resp in outs[2 * i : 2 * i + 2]:
            closed = "</think>" in resp
            strict, inst_r = verify_ifeval_response(resp, pm["iids"], pm["kwargs"])
            sc = (2.0 if (strict and closed) else 0.0) + inst_r + (0.2 if closed else 0.0)
            if sc > best_sc:
                best_sc, best_r, best_st = sc, resp, (strict and closed)
        if best_r is not None and best_sc >= 0.8:
            n_strict += int(best_st)
            selected.append((pm["prompt"], best_r))

    cache = []
    for p, r in selected:
        qi = tokenizer.encode(p, add_special_tokens=False)
        ri = tokenizer.encode(r, add_special_tokens=False)[:max_resp_trunc]
        if len(ri) < 16:
            continue
        e, l, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, sys_text_ids=rubric_ids, device=device, model_dtype=dtype)
        out = student(inputs_embeds=e, use_cache=False, output_hidden_states=True)
        sl = out.logits[0, :-1, :].float()
        m = l[0, 1:] != -100
        probs = F.softmax(sl[m], dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
        h_taps = {ly: out.hidden_states[ly][0, :-1, :][m].to(torch.float16).cpu() for ly in TAP_LAYERS}
        cache.append({"q_ids": qi, "r_ids": ri, "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "ent": ent.cpu(), "h_taps": h_taps})
        del e, l, out

    stats = {"rubric_tokens": len(rubric_ids), "selected": len(cache), "strict_wins": n_strict, "prep_sec": round(time.time() - t0, 1)}
    print(f"  [SelfRubric Cache      ] {stats}", flush=True)
    return cache, stats


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Suite 6 (exp8): KV-Inversion to CR-SP (Q1) & Direct Teacher-Free CR-SP (Q2) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    if_train, ifbench32, ifbench_kl24 = load_datasets_exp7()

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
        "q2_teacher_free_meta": {},
        "q1_kv_inversion_meta": {},
        "position_matrix": {},
        "ifbench32_evals": {},
    }

    def save():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # =========================================================================
    # PHASE 1 (Q2): 100% TEACHER-FREE CR-SP CONSTRUCTIONS (0 Post-RL Models!)
    # =========================================================================
    print("\n=== Phase 1 (Q2): 4 Teacher-Free Context-Robust Constructions (ZERO Post-RL Model Loaded!) ===", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    # TF-1: Online On-Policy GRPO in 16-tok KV Space -> Functional Operator Inversion (Inv-3) -> CR-SP!
    kv_grpo_tf, meta_kv_grpo = train_online_kv_grpo(student, tokenizer, tmpl, if_train, num_steps=14, tag="KV_GRPO_TF_L16")
    results["q2_teacher_free_meta"]["KV_GRPO_TF_L16"] = meta_kv_grpo
    torch.save(kv_grpo_tf.cpu(), os.path.join(OUT_DIR, "kv_grpo_tf_l16.pt"))
    save()

    # Harvest self-generated verified wins from (Frozen Student + KV_GRPO_TF) & invert via Inv-3!
    print("  --- Generating self-rollouts from (Student + KV_GRPO_TF_L16) for Inv-3 compilation ---", flush=True)
    flat_q_kv = []
    for pm in if_train:
        flat_q_kv.extend([pm["prompt"], pm["prompt"]])
    torch.manual_seed(404)
    outs_kv = batch_generate_with_kv_prefixes(
        student, tokenizer, tmpl, flat_q_kv, [kv_grpo_tf], system_texts=[""] * len(flat_q_kv),
        batch_size=32, max_new_tokens=768, temperature=0.75, top_p=0.95, top_k=20, enable_thinking=True,
    )
    kv_grpo_pairs = []
    for i, pm in enumerate(if_train):
        best_r, best_sc = None, -1.0
        for resp in outs_kv[2 * i : 2 * i + 2]:
            closed = "</think>" in resp
            strict, inst_r = verify_ifeval_response(resp, pm["iids"], pm["kwargs"])
            sc = (2.0 if (strict and closed) else 0.0) + inst_r + (0.2 if closed else 0.0)
            if sc > best_sc:
                best_sc, best_r = sc, resp
        if best_r is not None and best_sc >= 0.8:
            kv_grpo_pairs.append((pm["prompt"], best_r))
    func_cache_kvgrpo = precompute_kv_functional_cache(student, tokenizer, tmpl, kv_grpo_tf, kv_grpo_pairs, tag="KV_GRPO_TF")
    sp_tf1, meta_tf1 = invert_functional_cr_sp(
        student, tokenizer, tmpl, func_cache_kvgrpo, train_pools,
        num_vectors=256, num_updates=38, use_hidden_align=True, tag="TF1_CR_KVGRPO_Inv",
    )
    results["q2_teacher_free_meta"]["TF1_CR_KVGRPO_Inv"] = meta_tf1
    torch.save(sp_tf1.cpu(), os.path.join(OUT_DIR, "sp_tf1_kvgrpo_inv.pt"))
    save()

    # TF-2: Direct Multi-Slot Online GRPO on Layer-0 Input-SP (L=64)
    sp_tf2, meta_tf2 = train_direct_multislot_input_grpo(
        student, tokenizer, tmpl, if_train, train_pools, num_vectors=64, num_steps=12, tag="TF2_CR_Direct_GRPO"
    )
    results["q2_teacher_free_meta"]["TF2_CR_Direct_GRPO"] = meta_tf2
    torch.save(sp_tf2.cpu(), os.path.join(OUT_DIR, "sp_tf2_direct_grpo.pt"))
    save()

    # TF-3: Context-Robust KL-Anchored Verified Self-Bootstrap
    sp_tf3, meta_tf3 = train_tf_anchored_selfboot_cr(
        student, tokenizer, tmpl, if_train, train_pools, num_vectors=256, num_updates=38, tag="TF3_CR_Anchored_SelfBoot"
    )
    results["q2_teacher_free_meta"]["TF3_CR_Anchored_SelfBoot"] = meta_tf3
    torch.save(sp_tf3.cpu(), os.path.join(OUT_DIR, "sp_tf3_anchored_selfboot.pt"))
    save()

    # TF-4: Zero-RL Rubric-Prompted Self-Teacher + Verified Filter -> Inv-3 CR-SP
    rubric_cache, rubric_stats = precompute_self_rubric_functional_cache(student, tokenizer, tmpl, if_train)
    results["q2_teacher_free_meta"]["SelfRubric_Cache_Stats"] = rubric_stats
    sp_tf4, meta_tf4 = invert_functional_cr_sp(
        student, tokenizer, tmpl, rubric_cache, train_pools,
        num_vectors=256, num_updates=38, use_hidden_align=True, tag="TF4_CR_SelfRubric",
    )
    results["q2_teacher_free_meta"]["TF4_CR_SelfRubric"] = meta_tf4
    torch.save(sp_tf4.cpu(), os.path.join(OUT_DIR, "sp_tf4_self_rubric.pt"))
    save()

    # =========================================================================
    # PHASE 2 (Q1): KV-CSPD DISTILLATION & 4 MATHEMATICAL KV INVERSION OPERATORS
    # =========================================================================
    print("\n=== Phase 2 (Q1): Distill KV_L16 from M2RL-RL_IF, Throw Away Teacher, & Run 4 KV-Inversion Operators ===", flush=True)
    # Move student to CPU briefly while caching teacher logits once
    student.cpu()
    torch.cuda.empty_cache()
    teacher_train_cache, teacher_eval_cache = precompute_teacher_top64_cache(
        tokenizer, tmpl, [x["prompt"] for x in if_train], [x.prompt for x in ifbench_kl24]
    )
    student.to("cuda")
    student.eval()

    # Train strong KV_Distilled_L16 (45 steps, ~90s), then NEVER touch teacher_train_cache again!
    kv_dist_l16, meta_kv_dist = train_kv_distilled_l16(
        student, tokenizer, tmpl, teacher_train_cache, num_kv_tokens=16, num_updates=45, tag="KV_Distilled_L16"
    )
    results["q1_kv_inversion_meta"]["KV_Distilled_L16"] = meta_kv_dist
    torch.save(kv_dist_l16.cpu(), os.path.join(OUT_DIR, "kv_distilled_l16.pt"))
    save()

    # Inv-1a: 100% Data-Free Geometric KV Pre-Image (L=16)
    sp_inv1a, meta_inv1a = invert_kv_geometric_datafree(
        student, tokenizer, kv_dist_l16, train_pools=None, context_robust=False, num_steps=160, tag="Inv1a_GeoKV_DataFree"
    )
    results["q1_kv_inversion_meta"]["Inv1a_GeoKV_DataFree"] = meta_inv1a
    torch.save(sp_inv1a.cpu(), os.path.join(OUT_DIR, "sp_inv1a_geokv.pt"))
    save()

    # Inv-1b: 100% Data-Free Context-Robust Geometric Pre-Image (L=16)
    sp_inv1b, meta_inv1b = invert_kv_geometric_datafree(
        student, tokenizer, kv_dist_l16, train_pools=train_pools, context_robust=True, num_steps=160, tag="Inv1b_CR_GeoKV_DF"
    )
    results["q1_kv_inversion_meta"]["Inv1b_CR_GeoKV_DataFree"] = meta_inv1b
    torch.save(sp_inv1b.cpu(), os.path.join(OUT_DIR, "sp_inv1b_cr_geokv.pt"))
    save()

    # Self-generate rollouts from (Frozen Student + KV_Distilled_L16) ONLY (ZERO RL teacher queried!)
    print("  --- Self-generating 64 rollouts from (Frozen Student + KV_Distilled_L16) ONLY (Teacher Discarded!) ---", flush=True)
    torch.manual_seed(606)
    kv_dist_resps = batch_generate_with_kv_prefixes(
        student, tokenizer, tmpl, [x["prompt"] for x in if_train], [kv_dist_l16],
        system_texts=[""] * len(if_train), batch_size=32, max_new_tokens=768,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    kv_dist_func_cache = precompute_kv_functional_cache(
        student, tokenizer, tmpl, kv_dist_l16, list(zip([x["prompt"] for x in if_train], kv_dist_resps)), tag="KV_Dist_SelfGen"
    )

    # Inv-2: Multi-Slot Functional Logit Inversion (no hidden alignment ablation, L=256)
    sp_inv2, meta_inv2 = invert_functional_cr_sp(
        student, tokenizer, tmpl, kv_dist_func_cache, train_pools,
        num_vectors=256, num_updates=38, use_hidden_align=False, tag="Inv2_LogitCR_L256",
    )
    results["q1_kv_inversion_meta"]["Inv2_LogitCR_L256"] = meta_inv2
    torch.save(sp_inv2.cpu(), os.path.join(OUT_DIR, "sp_inv2_logit_cr.pt"))
    save()

    # Inv-3 (L=256): Full Multi-Layer Functional Operator Inversion (Logits + L12/24/36 Hidden + L_inv!)
    sp_inv3_256, meta_inv3_256 = invert_functional_cr_sp(
        student, tokenizer, tmpl, kv_dist_func_cache, train_pools,
        num_vectors=256, num_updates=38, use_hidden_align=True, tag="Inv3_FuncOp_L256",
    )
    results["q1_kv_inversion_meta"]["Inv3_FuncOp_L256"] = meta_inv3_256
    torch.save(sp_inv3_256.cpu(), os.path.join(OUT_DIR, "sp_inv3_funcop_256.pt"))
    save()

    # Inv-3 (L=64): Compact 4x Shorter Functional Operator Inversion!
    sp_inv3_64, meta_inv3_64 = invert_functional_cr_sp(
        student, tokenizer, tmpl, kv_dist_func_cache, train_pools,
        num_vectors=64, num_updates=38, lr_max=0.008, use_hidden_align=True, tag="Inv3_FuncOp_L64",
    )
    results["q1_kv_inversion_meta"]["Inv3_FuncOp_L64"] = meta_inv3_64
    torch.save(sp_inv3_64.cpu(), os.path.join(OUT_DIR, "sp_inv3_funcop_64.pt"))
    save()

    # =========================================================================
    # PHASE 3: FULL 11-METHOD x 6-POSITION DISTRIBUTIONAL & HIDDEN-DRIFT MATRIX
    # =========================================================================
    sp_models = {
        "Inv1a_GeoKV_DF": sp_inv1a,
        "Inv1b_CR_GeoKV_DF": sp_inv1b,
        "Inv2_LogitCR_L256": sp_inv2,
        "Inv3_FuncOp_L256": sp_inv3_256,
        "Inv3_FuncOp_L64": sp_inv3_64,
        "TF1_CR_KVGRPO_Inv": sp_tf1,
        "TF2_CR_Direct_GRPO": sp_tf2,
        "TF3_CR_Anchored_SelfBoot": sp_tf3,
        "TF4_CR_SelfRubric": sp_tf4,
    }
    kv_models = {
        "KV_Distilled_L16": kv_dist_l16,
        "KV_GRPO_TF_L16": kv_grpo_tf,
    }
    results["position_matrix"] = evaluate_position_matrix(
        student, tokenizer, tmpl, eval_ctx, teacher_eval_cache, sp_models, kv_models=kv_models
    )
    save()

    # =========================================================================
    # PHASE 4: IFBench32 REAL END-TO-END ACROSS P0/P2/P3/P4
    # =========================================================================
    print("\n=== Phase 4: IFBench32 End-to-End Evaluation (Q1 KV-Inv & Q2 Teacher-Free CR) ===", flush=True)

    # --- Q1 Track: KV_Distilled BEFORE vs AFTER Geometric & Functional Inversion ---
    results["ifbench32_evals"]["P0_KV_Distilled_L16"] = eval_kv_ifbench32_p0(student, tokenizer, tmpl, ifbench32, kv_dist_l16, tag="P0-KV: KV_Distilled_L16")
    results["ifbench32_evals"]["P0_Inv1a_GeoKV_DF"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P0_Clean_Start", [sp_inv1a], tag="P0-Start: Inv1a GeoKV (L16)")
    results["ifbench32_evals"]["P0_Inv3_FuncOp_L256"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P0_Clean_Start", [sp_inv3_256], tag="P0-Start: Inv3 FuncOp L256")
    results["ifbench32_evals"]["P2_Inv3_FuncOp_L256"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_inv3_256], tag="P2-MultiTurn: Inv3 FuncOp L256")
    results["ifbench32_evals"]["P2_Inv3_FuncOp_L64"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_inv3_64], tag="P2-MultiTurn: Inv3 FuncOp L64")
    results["ifbench32_evals"]["P3_Inv3_FuncOp_L256"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P3_PostRAG_Doc", [sp_inv3_256], tag="P3-PostRAG: Inv3 FuncOp L256")
    results["ifbench32_evals"]["P4_Inv3_FuncOp_L256"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_inv3_256], tag="P4-PostQuery: Inv3 FuncOp L256")
    results["ifbench32_evals"]["P4_Inv3_FuncOp_L64"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_inv3_64], tag="P4-PostQuery: Inv3 FuncOp L64")
    save()

    # --- Q2 Track: All 4 Zero-Post-RL Teacher-Free CR Constructions ---
    results["ifbench32_evals"]["P0_KV_GRPO_TF_L16"] = eval_kv_ifbench32_p0(student, tokenizer, tmpl, ifbench32, kv_grpo_tf, tag="P0-KV: KV_GRPO_TF_L16 (0-Tch)")
    results["ifbench32_evals"]["P0_TF1_CR_KVGRPO_Inv"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P0_Clean_Start", [sp_tf1], tag="P0-Start: TF1 KVGRPO->Inv3")
    results["ifbench32_evals"]["P2_TF1_CR_KVGRPO_Inv"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_tf1], tag="P2-MultiTurn: TF1 KVGRPO->Inv3")
    results["ifbench32_evals"]["P4_TF1_CR_KVGRPO_Inv"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_tf1], tag="P4-PostQuery: TF1 KVGRPO->Inv3")
    save()

    results["ifbench32_evals"]["P2_TF2_CR_Direct_GRPO"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_tf2], tag="P2-MultiTurn: TF2 Direct GRPO")
    results["ifbench32_evals"]["P4_TF2_CR_Direct_GRPO"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_tf2], tag="P4-PostQuery: TF2 Direct GRPO")
    results["ifbench32_evals"]["P2_TF3_CR_Anchored_SelfBoot"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_tf3], tag="P2-MultiTurn: TF3 AnchSelfBoot")
    results["ifbench32_evals"]["P4_TF3_CR_Anchored_SelfBoot"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_tf3], tag="P4-PostQuery: TF3 AnchSelfBoot")
    save()

    results["ifbench32_evals"]["P0_TF4_CR_SelfRubric"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P0_Clean_Start", [sp_tf4], tag="P0-Start: TF4 SelfRubric->Inv3")
    results["ifbench32_evals"]["P2_TF4_CR_SelfRubric"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P2_MultiTurn_Mid", [sp_tf4], tag="P2-MultiTurn: TF4 SelfRubric")
    results["ifbench32_evals"]["P4_TF4_CR_SelfRubric"] = eval_ifbench32_regime(student, tokenizer, tmpl, eval_ctx, ifbench32, "P4_PostQuery_Suffix", [sp_tf4], tag="P4-PostQuery: TF4 SelfRubric")
    save()

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== Suite 6 (exp8) COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
