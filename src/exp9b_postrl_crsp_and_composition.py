"""
Paper Reference: Section 6.2–6.3, Tables 7–9, Fig. 10–11 (Scaled IFBench N=64, Post-RAG Noise & 3-Expert Split).
Suite 7B (exp9b): Post-RL Distilled & KV-Inverted (Alg A: Inv-2) CR-SPs —
                  3-Seed Validation, Scaled IFBench64 (incl. PostRAG Post-RAG), &
                  3-Expert Simultaneous Split-Slot Composition (Head-to-Head vs Zero-Post-RL exp9)
===================================================================================================
Runs on the EXACT SAME datasets, held-out splits, and multi-turn/RAG evaluation contexts as exp9:
  - IFEval96 (96 held-out), Official IFBench64 (64 stratified), MATH48 (48 Level 3-5), HeliosPay Doc36 (36)
  - Reuses E3_Doc_CSCR (HeliosPay v4.2, L=64) from exp9_results/sp_doc_cscr.pt!

Covers BOTH Post-RL CR-SP acquisition pathways with Co-Presence Distractor Aug + L_null:
  1) Direct Post-RL Distillation (M2RL-RL_IF & M2RL-RL_Math -> CS-CR L=64 & L=256, 3 seeds for IF L=64!)
  2) Algorithm A FuncCR-Inv (Inv-2, lambda_feat=0, NO intermediate-layer over-constraining!):
       Post-RL Teacher -> KV_L16 -> Discard Teacher -> Self-Gen Rollouts -> Inv-2 CS-CR (L=256 & L=64)!
  3) Grand 3-Way Head-to-Head across Single, PostRAG Post-RAG, Permutation Invariance, AND
     3-Expert Simultaneous Split-Slot Composition ([Doc@PostSys + Math@PreQuery + IF@PostQuery], 192 vecs total!):
       Zero-Post-RL Alg B (exp9)  vs  Post-RL KV-Inv2 Alg A (exp9b)  vs  Direct Post-RL CS-CR (exp9b)!
"""

import gc
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content/cspd")
sys.path.insert(0, "/content/IFBench")

from cspd_core import ChatTemplateTokens, MuonSoftPrompt, get_qwen_template_tokens, pad_and_stack_train_batch
from benchmarks import batch_generate_with_soft_prompts, strip_thinking
from kv_prefix_core import (
    batch_generate_with_kv_prefixes,
    build_chat_ids_for_kv,
    forward_with_kv_prefix,
    init_kv_prefix_from_tokens,
    make_readonly_kv_cache,
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
    DOC_A_TEXT,
    DOC_TASK_INST,
    MATH_TASK_INST,
    batch_generate_slots,
    build_foreign_distractor_pool,
    compute_interference_matrix,
    eval_doc36_slots,
    eval_ifbench64_slots,
    eval_ifeval96_slots,
    eval_math48_slots,
    harvest_verified_wins_with_anchors,
    load_datasets_exp9,
    precompute_doc_self_distill_cache,
    sample_slot_view_cscr,
    train_doc_cr_sp,
)
import evaluation_lib as ifbench_eval

STUDENT_ID = "Jackwang111/M2RL-SFT"
IF_TEACHER_ID = "Jackwang111/M2RL-RL_IF"
MATH_TEACHER_ID = "Jackwang111/M2RL-RL_Math"

EXP9_DIR = "/content/cspd/exp9_results"
OUT_DIR = "/content/cspd/exp9b_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Post-RL Teacher Dense Top-64 Caching (Load Once -> Cache -> Unload)
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_postrl_teacher_top64(
    model_id: str, tokenizer, tmpl: ChatTemplateTokens, prompts: List[str],
    max_resp_gen: int = 768, max_resp_trunc: int = 512, topk: int = 64, seed: int = 777, tag: str = "Teacher",
) -> List[Dict[str, Any]]:
    t0 = time.time()
    print(f"  [Cache {tag:14s}] Loading {model_id} for {len(prompts)} prompts...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    teacher.eval()
    device = next(teacher.parameters()).device
    dtype = next(teacher.parameters()).dtype
    t_emb = teacher.get_input_embeddings()

    torch.manual_seed(seed)
    resps = batch_generate_with_soft_prompts(
        teacher, tokenizer, tmpl, prompts, [], batch_size=32, max_new_tokens=max_resp_gen,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    cache = []
    for p, r in zip(prompts, resps):
        qi = tokenizer.encode(p, add_special_tokens=False)
        ri = tokenizer.encode(r, add_special_tokens=False)[:max_resp_trunc]
        if len(ri) < 16:
            continue
        e, l, _ = build_multislot_sp_inputs(t_emb, tmpl, qi, ri, device=device, model_dtype=dtype)
        out = teacher(inputs_embeds=e, use_cache=False)
        sl = out.logits[0, :-1, :].float()
        m = l[0, 1:] != -100
        probs = F.softmax(sl[m], dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
        cache.append({"q_ids": qi, "r_ids": ri, "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "ent": ent.cpu()})
        del e, l, out

    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  [Cache {tag:14s}] Cached {len(cache)} dense Top-{topk} targets & freed GPU in {time.time()-t0:.1f}s", flush=True)
    return cache


def train_kv_distilled_l16_custom(
    student, tokenizer, tmpl: ChatTemplateTokens, train_cache: List[Dict[str, Any]],
    inst_text: str = IF_TASK_INST, num_kv_tokens: int = 16, num_updates: int = 45,
    batch_size: int = 12, lr_max: float = 0.025, seed: int = 42, tag: str = "KV_Distilled_L16",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    device = next(student.parameters()).device
    inst_ids = tokenizer.encode(inst_text, add_special_tokens=False)
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
            loss = hybrid_top64_distill_loss(sl[m], item["ti"].to(device), item["tp"].float().to(device))
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
        if step in {1, 20, num_updates}:
            log_hist[str(step)] = round(tot_loss, 4)
            print(f"  [{tag:20s}] step {step:2d}/{num_updates} | dist={tot_loss:.4f} | d_rms={(kv.detach()-init_kv).pow(2).mean().sqrt().item():.3f} | t={time.time()-t0:.1f}s", flush=True)
    meta = {
        "tag": tag, "num_kv_tokens": num_kv_tokens,
        "delta_rms": round(float((kv.detach() - init_kv).pow(2).mean().sqrt().item()), 4),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return kv.detach().clone(), meta


@torch.no_grad()
def cache_kv_pure_logit_targets(
    student, tokenizer, tmpl: ChatTemplateTokens, kv_tensor: torch.Tensor,
    prompts: List[str], max_resp_gen: int = 768, max_resp_trunc: int = 512,
    topk: int = 64, seed: int = 606, tag: str = "KV_SelfGen",
) -> List[Dict[str, Any]]:
    """
    Algorithm A (FuncCR-Inv / Inv-2) Stage 1:
      Self-generate rollouts from (Frozen Student + KV*) ONLY (Post-RL teacher discarded!),
      then extract PURE output Top-64 logits (output_hidden_states=False -> zero over-constraining!).
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    kv_dev = kv_tensor.to(device)
    student.eval()
    student.config.use_cache = True

    torch.manual_seed(seed)
    resps = batch_generate_with_kv_prefixes(
        student, tokenizer, tmpl, prompts, [kv_dev], system_texts=[""] * len(prompts),
        batch_size=32, max_new_tokens=max_resp_gen, temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )

    cache = []
    min_val = torch.finfo(dtype).min
    for p, r in zip(prompts, resps):
        qi = tokenizer.encode(p, add_special_tokens=False)
        ri = tokenizer.encode(r, add_special_tokens=False)[:max_resp_trunc]
        if len(ri) < 16:
            continue
        ids, lbl = build_chat_ids_for_kv(tmpl, qi, ri, None, True, None)
        bi = torch.tensor([ids], dtype=torch.long, device=device)
        cache_obj = make_readonly_kv_cache([kv_dev], 1, device, dtype)
        L_kv = cache_obj.get_seq_length()
        T = bi.size(1)
        cache_pos = torch.arange(L_kv, L_kv + T, dtype=torch.long, device=device)
        causal = torch.triu(torch.full((T, T), min_val, device=device, dtype=dtype), diagonal=1)
        mask4d = torch.cat([torch.zeros((T, L_kv), device=device, dtype=dtype), causal], dim=-1).unsqueeze(0).unsqueeze(0)
        out = student(
            input_ids=bi, attention_mask=mask4d, past_key_values=cache_obj,
            cache_position=cache_pos, use_cache=True, output_hidden_states=False,
        )
        sl = out.logits[0, :-1, :].float()
        lb = torch.tensor(lbl[1:], dtype=torch.long, device=device)
        m = lb != -100
        probs = F.softmax(sl[m], dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        ent = -(tp * tp.clamp_min(1e-9).log()).sum(dim=-1)
        cache.append({"q_ids": qi, "r_ids": ri, "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "ent": ent.cpu()})
        del bi, mask4d, out

    print(f"  [AlgA {tag:15s}] Self-generated & cached {len(cache)} pure-logit targets in {time.time()-t0:.1f}s", flush=True)
    return cache


# ---------------------------------------------------------------------------
# 2. Unified Post-RL CS-CR / Algorithm A (Inv-2) Compiler (with Co-Presence + L_null)
# ---------------------------------------------------------------------------

def train_postrl_cscr_sp(
    student, tokenizer, tmpl: ChatTemplateTokens, target_cache: List[Dict[str, Any]],
    train_pools: Dict[str, Any], task_inst: str, num_vectors: int = 64,
    num_updates: int = 45, batch_size: int = 10, lr_max: float = 0.007,
    lambda_inv: float = 0.35, max_rms: float = 0.050,
    foreign_pool: Optional[List[torch.Tensor]] = None, co_prob: float = 0.60,
    null_pool: Optional[List[Dict[str, Any]]] = None, lambda_null: float = 0.0,
    seed: int = 42, tag: str = "PostRL_CSCR",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compiles EITHER Direct Post-RL Teacher Top-64 targets OR Algorithm A (Inv-2) KV self-gen
    pure-logit targets into a Co-Presence Context-Robust Input-Space Soft Prompt!
    Zero intermediate-layer over-constraining (lambda_feat = 0)!
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

        batch = [target_cache[rng.randrange(len(target_cache))] for _ in range(batch_size)]
        tot_dist, tot_inv, tot_null = 0.0, 0.0, 0.0

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

            l_c = hybrid_top64_distill_loss(zc, ti, tp)
            l_0 = hybrid_top64_distill_loss(z0, ti, tp)
            l_inv = cross_context_invariance_kl(z0, zc, topk=32)
            loss = (l_c + 0.35 * l_0 + lambda_inv * l_inv) / batch_size
            loss.backward()
            tot_dist += float(l_c.item()) / batch_size
            tot_inv += float(l_inv.item()) / batch_size
            del be, bm, bl, out, z0, zc

        if null_pool and lambda_null > 0.0:
            n_null = 2
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
            log_hist[str(step)] = {"dist": round(tot_dist, 4), "inv": round(tot_inv, 4), "null": round(tot_null, 4)}
            print(f"  [{tag:26s}] step {step:2d}/{num_updates} | dist={tot_dist:.4f} | inv={tot_inv:.4f} | null={tot_null:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "num_vectors": num_vectors, "co_presence": foreign_pool is not None,
        "lambda_null": lambda_null, "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


@torch.no_grad()
def precompute_doc_cache_eos(
    student, tokenizer, tmpl: ChatTemplateTokens, doc_text: str, doc_train: List[Dict[str, Any]], topk: int = 64
) -> List[Dict[str, Any]]:
    """EOS-terminated (+ tmpl.im_end) In-Context Self-Distillation cache for HeliosPay v4.2!"""
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
        ri = nt_ids + tokenizer.encode(item["gold_answer"], add_special_tokens=False) + list(tmpl.im_end)
        e, l, _ = build_multislot_sp_inputs(s_emb, tmpl, qi, ri, sys_text_ids=doc_ids, device=device, model_dtype=dtype)
        out = student(inputs_embeds=e, use_cache=False)
        sl = out.logits[0, :-1, :].float()
        m = l[0, 1:] != -100
        probs = F.softmax(sl[m], dim=-1)
        tp, ti = torch.topk(probs, topk, dim=-1)
        tp = tp / tp.sum(dim=-1, keepdim=True)
        cache.append({"q_ids": qi, "r_ids": ri, "ti": ti.cpu(), "tp": tp.to(torch.float16).cpu(), "kind": item["kind"]})
        del e, l, out
    print(f"  [DocSelfCache-EOS        ] Cached {len(cache)} EOS-terminated Top-64 targets in {time.time()-t0:.1f}s", flush=True)
    return cache


def train_doc_cscr_muon_l128(
    student, tokenizer, tmpl: ChatTemplateTokens, doc_cache: List[Dict[str, Any]],
    train_pools: Dict[str, Any], num_vectors: int = 128, num_updates: int = 120,
    batch_size: int = 10, lr_max: float = 0.045, lambda_nll: float = 0.85,
    lambda_inv: float = 0.25, max_rms: float = 0.14,
    foreign_pool: Optional[List[torch.Tensor]] = None, co_prob: float = 0.65,
    null_pool: Optional[List[Dict[str, Any]]] = None, lambda_null: float = 0.75,
    seed: int = 42, tag: str = "E3_Doc_CSCR_MuonL128",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    High-capacity Muon + EOS + Co-Presence + L_null Declarative Domain-Doc CR-SP:
    Obeys the Declarative Capacity / Norm Law (L=128, RMS<=0.14, Muon) while enforcing
    Content-Addressable Identity Preservation (L_null) so off-domain IF/Math stays clean!
    """
    t0 = time.time()
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype
    s_emb = student.get_input_embeddings()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_vec = s_emb(torch.tensor([pad_id], device=device)).to(dtype).detach()

    sp = init_anchored_sp(s_emb, tokenizer, DOC_TASK_INST, num_vectors=num_vectors, anchored=True)
    opt = MuonSoftPrompt([sp], lr=lr_max, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)
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
            tgt = lb[1][mc]

            l_dist = hybrid_top64_distill_loss(zc, ti, tp)
            l_nll = F.cross_entropy(zc, tgt) + 0.35 * F.cross_entropy(z0, lb[0][m0])
            l_inv = cross_context_invariance_kl(z0, zc, topk=32)
            loss = (0.55 * l_dist + lambda_nll * l_nll + lambda_inv * l_inv) / batch_size
            loss.backward()
            tot_dist += float(l_dist.item()) / batch_size
            tot_nll += float(F.cross_entropy(zc, tgt).item()) / batch_size
            tot_inv += float(l_inv.item()) / batch_size
            del be, bm, bl, out, z0, zc

        if null_pool and lambda_null > 0.0:
            n_null = 2
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

        if step in {1, 40, 80, num_updates}:
            log_hist[str(step)] = {"dist": round(tot_dist, 4), "nll": round(tot_nll, 4), "inv": round(tot_inv, 4), "null": round(tot_null, 4)}
            print(f"  [{tag:26s}] step {step:3d}/{num_updates} | dist={tot_dist:.4f} | nll={tot_nll:.4f} | inv={tot_inv:.4f} | null={tot_null:.4f} | rms={sp.detach().pow(2).mean().sqrt().item():.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "num_vectors": num_vectors, "co_presence": foreign_pool is not None,
        "lambda_null": lambda_null, "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_sec": round(time.time() - t0, 1), "log_hist": log_hist,
    }
    return sp.detach().clone(), meta


def eval_ifbench64_post_rag(model, tokenizer, tmpl, eval_ctx, inps64, sp_pre_query=None, tag: str = "") -> Dict[str, Any]:
    """Evaluate official IFBench64 under PostRAG_Doc (Persona + 420-tok RAG + [SP] + Query)."""
    t0 = time.time()
    outs = batch_generate_slots(
        model, tokenizer, tmpl, [x.prompt for x in inps64],
        sys_text_ids=eval_ctx["persona_ids"], rag_ids=eval_ctx["rag_ids"],
        sp_pre_query=sp_pre_query, batch_size=32, max_new_tokens=2048, seed=42,
    )
    pre_query_res = {}
    closed = 0
    for inp, r in zip(inps64, outs):
        closed += int("</think>" in r)
        pre_query_res[inp.prompt] = strip_thinking(r)
    s_ok = sum(int(ifbench_eval.test_instruction_following_strict(inp, pre_query_res).follow_all_instructions) for inp in inps64)
    inst_l = []
    for inp in inps64:
        inst_l.extend(ifbench_eval.test_instruction_following_strict(inp, pre_query_res).follow_instruction_list)
    n = len(inps64)
    res = {
        "regime": "PostRAG_Doc",
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * sum(inst_l) / max(1, len(inst_l)), 2),
        "n_strict": s_ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFBench64-PostRAG | {tag:33s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | inst={res['inst_strict']:5.2f}% | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


def eval_ifbench64_kv_start(model, tokenizer, tmpl, kv_tensor, inps64, tag: str = "") -> Dict[str, Any]:
    """Evaluate Post-RL KV_L16 directly at Start on official IFBench64 to measure exact Alg A inversion retention."""
    t0 = time.time()
    torch.manual_seed(42)
    outs = batch_generate_with_kv_prefixes(
        model, tokenizer, tmpl, [x.prompt for x in inps64], [kv_tensor.to(next(model.parameters()).device)],
        system_texts=[""] * len(inps64), batch_size=32, max_new_tokens=2048,
        temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True,
    )
    pre_query_res = {}
    closed = 0
    for inp, r in zip(inps64, outs):
        closed += int("</think>" in r)
        pre_query_res[inp.prompt] = strip_thinking(r)
    s_ok = sum(int(ifbench_eval.test_instruction_following_strict(inp, pre_query_res).follow_all_instructions) for inp in inps64)
    inst_l = []
    for inp in inps64:
        inst_l.extend(ifbench_eval.test_instruction_following_strict(inp, pre_query_res).follow_instruction_list)
    n = len(inps64)
    res = {
        "regime": "Start_KV_Prefix",
        "prompt_strict": round(100.0 * s_ok / n, 2),
        "inst_strict": round(100.0 * sum(inst_l) / max(1, len(inst_l)), 2),
        "n_strict": s_ok, "n_total": n,
        "closed_think": closed,
        "closed_acc": round(100.0 * s_ok / max(1, closed), 2),
        "eval_sec": round(time.time() - t0, 1),
    }
    print(f"  [IFBench64-KV | {tag:33s}] strict={res['prompt_strict']:5.2f}% ({s_ok:2d}/{n}) | inst={res['inst_strict']:5.2f}% | closed={closed:2d}/{n} ({res['closed_acc']:5.1f}%) | t={res['eval_sec']}s", flush=True)
    return res


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Suite 7B (exp9b): Post-RL Direct CS-CR & Algorithm A (Inv-2) CR-SP Comparison ===", flush=True)
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

    results: Dict[str, Any] = {
        "backbone": STUDENT_ID,
        "teachers": {"if": IF_TEACHER_ID, "math": MATH_TEACHER_ID},
        "train_meta": {},
        "ifeval96_multiseed_direct_l64": {},
        "ifeval96_multiseed_summary": {},
        "ifeval96_evals": {},
        "ifbench64_evals": {},
        "math48_evals": {},
        "doc36_evals": {},
        "interference_kl": {},
    }

    def save():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # =========================================================================
    # STEP 1: CACHE POST-RL TEACHERS ONCE (M2RL-RL_IF & M2RL-RL_Math) THEN UNLOAD
    # =========================================================================
    print("\n=== Step 1: Caching Post-RL Teachers (M2RL-RL_IF on 96 IF + M2RL-RL_Math on 64 MATH) ===", flush=True)
    if_teacher_cache = cache_postrl_teacher_top64(
        IF_TEACHER_ID, tokenizer, tmpl, [x["prompt"] for x in if_train96],
        max_resp_gen=768, max_resp_trunc=512, topk=64, seed=777, tag="M2RL-RL_IF",
    )
    math_teacher_cache = cache_postrl_teacher_top64(
        MATH_TEACHER_ID, tokenizer, tmpl, [x["prompt"] for x in math_train64],
        max_resp_gen=960, max_resp_trunc=512, topk=64, seed=888, tag="M2RL-RL_Math",
    )

    # Now load Frozen Student ONLY for all remaining training & evaluation!
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad = False
    s_emb = student.get_input_embeddings()
    foreign_pool = build_foreign_distractor_pool(s_emb, tokenizer, num_vectors=64, seed=77)

    # Build null-task anchor pools on 20 IF + 16 Math prompts (protects BOTH IF & Math <think> chains!) + train EOS Doc CS-CR (Muon L=128)
    null_if_wins, _ = harvest_verified_wins_with_anchors(
        student, tokenizer, tmpl, if_train96[:24], task="if", n_samples=2, max_new_tokens=640, seed=111, tag="NullAnchor_IF24"
    )
    null_math_wins, _ = harvest_verified_wins_with_anchors(
        student, tokenizer, tmpl, math_train64[:20], task="math", n_samples=2, max_new_tokens=768, seed=112, tag="NullAnchor_MATH20"
    )
    null_for_doc = null_if_wins[:16] + null_math_wins[:16]
    doc_cache_eos = precompute_doc_cache_eos(student, tokenizer, tmpl, DOC_A_TEXT, doc_train, topk=64)
    e3_doc_cscr, meta_doc_l128 = train_doc_cscr_muon_l128(
        student, tokenizer, tmpl, doc_cache_eos, train_pools, num_vectors=128, num_updates=120,
        batch_size=10, lr_max=0.045, lambda_nll=0.85, lambda_inv=0.25, max_rms=0.14,
        foreign_pool=foreign_pool, co_prob=0.65, null_pool=null_for_doc, lambda_null=0.85,
        seed=42, tag="E3_Doc_CSCR_MuonL128",
    )
    results["train_meta"]["E3_Doc_CSCR_MuonL128"] = meta_doc_l128
    torch.save(e3_doc_cscr.cpu(), os.path.join(OUT_DIR, "sp_doc_cscr_muon_l128.pt"))
    save()

    # =========================================================================
    # STEP 2: DIRECT POST-RL -> CS-CR (3 SEEDS FOR IF L=64 + IF L=256 + MATH L=64)
    # =========================================================================
    print("\n=== Step 2: Direct Post-RL Teacher -> CS-CR Distillation (Co-Presence + L_inv + L_null) ===", flush=True)
    seeds = [42, 108, 202]
    direct_if_l64_by_seed = {}
    for sd in seeds:
        sp_d, meta_d = train_postrl_cscr_sp(
            student, tokenizer, tmpl, if_teacher_cache, train_pools, IF_TASK_INST,
            num_vectors=64, num_updates=45, lr_max=0.008, max_rms=0.050,
            foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_math_wins, lambda_null=0.20,
            seed=sd, tag=f"PostRL_Direct_IF_L64_s{sd}",
        )
        direct_if_l64_by_seed[sd] = sp_d
        results["train_meta"][f"PostRL_Direct_IF_L64_s{sd}"] = meta_d
        torch.save(sp_d.cpu(), os.path.join(OUT_DIR, f"sp_postrl_direct_if_l64_s{sd}.pt"))

    sp_direct_if_l64 = direct_if_l64_by_seed[42]
    sp_direct_if_l256, meta_d256 = train_postrl_cscr_sp(
        student, tokenizer, tmpl, if_teacher_cache, train_pools, IF_TASK_INST,
        num_vectors=256, num_updates=45, lr_max=0.006, max_rms=0.050,
        foreign_pool=foreign_pool, co_prob=0.60, seed=42, tag="PostRL_Direct_IF_L256",
    )
    sp_direct_math_l64, meta_dm64 = train_postrl_cscr_sp(
        student, tokenizer, tmpl, math_teacher_cache, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, lr_max=0.008, max_rms=0.050,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_if_wins, lambda_null=0.35,
        seed=42, tag="PostRL_Direct_Math_L64",
    )
    results["train_meta"]["PostRL_Direct_IF_L256"] = meta_d256
    results["train_meta"]["PostRL_Direct_Math_L64"] = meta_dm64
    torch.save(sp_direct_if_l256.cpu(), os.path.join(OUT_DIR, "sp_postrl_direct_if_l256.pt"))
    torch.save(sp_direct_math_l64.cpu(), os.path.join(OUT_DIR, "sp_postrl_direct_math_l64.pt"))
    save()

    # =========================================================================
    # STEP 3: ALGORITHM A (FuncCR-Inv / Inv-2, lambda_feat=0!) ON BOTH IF & MATH
    # =========================================================================
    print("\n=== Step 3: Post-RL KV_L16 -> Discard Teacher -> Algorithm A (Inv-2, lambda_feat=0) CS-CR ===", flush=True)
    kv_if_l16, meta_kv_if = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, if_teacher_cache, inst_text=IF_TASK_INST, num_kv_tokens=16, num_updates=45, tag="KV_PostRL_IF_L16"
    )
    kv_math_l16, meta_kv_math = train_kv_distilled_l16_custom(
        student, tokenizer, tmpl, math_teacher_cache, inst_text=MATH_TASK_INST, num_kv_tokens=16, num_updates=45, tag="KV_PostRL_Math_L16"
    )
    results["train_meta"]["KV_PostRL_IF_L16"] = meta_kv_if
    results["train_meta"]["KV_PostRL_Math_L16"] = meta_kv_math

    # Teacher caches no longer needed! Invert solely from (Frozen Student + KV*) via Algorithm A (Inv-2)!
    kv_if_logit_cache = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_if_l16, [x["prompt"] for x in if_train96],
        max_resp_gen=768, max_resp_trunc=512, topk=64, seed=606, tag="KV_IF_Inv2",
    )
    kv_math_logit_cache = cache_kv_pure_logit_targets(
        student, tokenizer, tmpl, kv_math_l16, [x["prompt"] for x in math_train64],
        max_resp_gen=960, max_resp_trunc=512, topk=64, seed=607, tag="KV_Math_Inv2",
    )

    sp_kvinv2_if_l256, meta_k256 = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_if_logit_cache, train_pools, IF_TASK_INST,
        num_vectors=256, num_updates=45, lr_max=0.006, max_rms=0.050,
        foreign_pool=foreign_pool, co_prob=0.60, seed=42, tag="PostRL_KVInv2_IF_L256",
    )
    sp_kvinv2_if_l64, meta_k64 = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_if_logit_cache, train_pools, IF_TASK_INST,
        num_vectors=64, num_updates=45, lr_max=0.008, max_rms=0.050,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_math_wins, lambda_null=0.20,
        seed=42, tag="PostRL_KVInv2_IF_L64",
    )
    sp_kvinv2_math_l64, meta_km64 = train_postrl_cscr_sp(
        student, tokenizer, tmpl, kv_math_logit_cache, train_pools, MATH_TASK_INST,
        num_vectors=64, num_updates=45, lr_max=0.008, max_rms=0.050,
        foreign_pool=foreign_pool, co_prob=0.60, null_pool=null_if_wins, lambda_null=0.35,
        seed=42, tag="PostRL_KVInv2_Math_L64",
    )
    results["train_meta"]["PostRL_KVInv2_IF_L256"] = meta_k256
    results["train_meta"]["PostRL_KVInv2_IF_L64"] = meta_k64
    results["train_meta"]["PostRL_KVInv2_Math_L64"] = meta_km64
    torch.save(sp_kvinv2_if_l256.cpu(), os.path.join(OUT_DIR, "sp_postrl_kvinv2_if_l256.pt"))
    torch.save(sp_kvinv2_if_l64.cpu(), os.path.join(OUT_DIR, "sp_postrl_kvinv2_if_l64.pt"))
    torch.save(sp_kvinv2_math_l64.cpu(), os.path.join(OUT_DIR, "sp_postrl_kvinv2_math_l64.pt"))
    save()

    # =========================================================================
    # STEP 4: EXACT CROSS-EXPERT INTERFERENCE MATRIX (POST-RL DIRECT & KV-INV2)
    # =========================================================================
    print("\n=== Step 4: Cross-Expert KL Interference Matrix (Post-RL Direct & Alg A KV-Inv2) ===", flush=True)
    doc_probe_cache = precompute_doc_self_distill_cache(student, tokenizer, tmpl, DOC_A_TEXT, doc_train[:10], topk=32)
    probes = {
        "if": [(x["q_ids"], x["r_ids"][:256]) for x in if_teacher_cache[:10]],
        "math": [(x["q_ids"], x["r_ids"][:256]) for x in math_teacher_cache[:10]],
        "doc": [(x["q_ids"], x["r_ids"][:256]) for x in doc_probe_cache[:10]],
    }
    results["interference_kl"]["PostRL_Direct_CSCR"] = compute_interference_matrix(
        student, tokenizer, tmpl, eval_ctx, probes,
        {"if": sp_direct_if_l64, "math": sp_direct_math_l64, "doc": e3_doc_cscr}, tag="PR-Dir",
    )
    results["interference_kl"]["PostRL_KVInv2_CSCR"] = compute_interference_matrix(
        student, tokenizer, tmpl, eval_ctx, probes,
        {"if": sp_kvinv2_if_l64, "math": sp_kvinv2_math_l64, "doc": e3_doc_cscr}, tag="PR-Inv2",
    )
    save()

    # =========================================================================
    # STEP 5: FULL SCALED EVALUATIONS (IFEval96 + IFBench64 incl. PostRAG + MATH48 + Doc36)
    # =========================================================================
    print("\n=== Step 5A: 3-Seed Post-RL Direct L=64 & Algorithm A (Inv-2) on Held-Out IFEval96 ===", flush=True)
    for sd in seeds:
        r_sd = {}
        r_sd["PreQuery"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[direct_if_l64_by_seed[sd]], tag=f"s{sd} PreQuery PostRL_Dir_L64")
        r_sd["PostQuery"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_post_query=[direct_if_l64_by_seed[sd]], tag=f"s{sd} PostQuery PostRL_Dir_L64")
        results["ifeval96_multiseed_direct_l64"][f"seed_{sd}"] = r_sd
        save()
    for pos in ["PreQuery", "PostQuery"]:
        vals = [results["ifeval96_multiseed_direct_l64"][f"seed_{sd}"][pos]["prompt_strict"] for sd in seeds]
        cls = [results["ifeval96_multiseed_direct_l64"][f"seed_{sd}"][pos]["closed_acc"] for sd in seeds]
        results["ifeval96_multiseed_summary"][f"{pos}_PostRL_Direct_L64"] = {
            "strict_mean": round(statistics.mean(vals), 2),
            "strict_std": round(statistics.stdev(vals), 2),
            "strict_per_seed": vals,
            "closed_acc_mean": round(statistics.mean(cls), 2),
        }
        s = results["ifeval96_multiseed_summary"][f"{pos}_PostRL_Direct_L64"]
        print(f"    {pos}_PostRL_Direct_L64: strict = {s['strict_mean']:5.2f} +/- {s['strict_std']:4.2f}% {vals} | closed_acc = {s['closed_acc_mean']:5.2f}%", flush=True)

    ie = results["ifeval96_evals"]
    ie["PostQuery_PostRL_Direct_L256"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_post_query=[sp_direct_if_l256], tag="PostQuery PostRL_Direct_L256")
    ie["PreQuery_PostRL_KVInv2_L256"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[sp_kvinv2_if_l256], tag="PreQuery PostRL_KVInv2_L256")
    ie["PostQuery_PostRL_KVInv2_L256"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_post_query=[sp_kvinv2_if_l256], tag="PostQuery PostRL_KVInv2_L256")
    ie["PostQuery_PostRL_KVInv2_L64"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_post_query=[sp_kvinv2_if_l64], tag="PostQuery PostRL_KVInv2_L64")
    ie["Dual_PostRL_Adj_IF_Math_PreQuery"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[sp_direct_if_l64, sp_direct_math_l64], tag="Dual PostRL [IF;Math]@PreQuery")
    ie["Dual_PostRL_Adj_Math_IF_PreQuery"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_pre_query=[sp_direct_math_l64, sp_direct_if_l64], tag="Dual PostRL [Math;IF]@PreQuery")
    ie["Triple_PostRL_Direct_Split"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_sys_end=[e3_doc_cscr], sp_pre_query=[sp_direct_math_l64], sp_post_query=[sp_direct_if_l64], tag="Triple PostRL_Direct [PostSys+PreQuery+PostQuery]")
    ie["Triple_PostRL_KVInv2_Split"] = eval_ifeval96_slots(student, tokenizer, tmpl, eval_ctx, ifeval_test96, sp_sys_end=[e3_doc_cscr], sp_pre_query=[sp_kvinv2_math_l64], sp_post_query=[sp_kvinv2_if_l64], tag="Triple PostRL_KVInv2 [PostSys+PreQuery+PostQuery]")
    save()

    print("\n=== Step 5B: Scaled Official IFBench64 (Incl. Start KV Ceiling, PostRAG Post-RAG & 3-Expert Composition!) ===", flush=True)
    be = results["ifbench64_evals"]
    be["Start_KV_PostRL_IF_L16"] = eval_ifbench64_kv_start(student, tokenizer, tmpl, kv_if_l16, ifbench64, tag="Start KV_PostRL_IF_L16 (Ceiling)")
    be["PostRAG_Bare_Student"] = eval_ifbench64_post_rag(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_pre_query=None, tag="PostRAG Bare Student")
    exp9_algb_path = os.path.join(EXP9_DIR, "sp_if_algb_seed42.pt")
    if os.path.exists(exp9_algb_path):
        sp_exp9_algb = torch.load(exp9_algb_path, map_location="cuda").float()
        be["PostRAG_ZeroPostRL_AlgB_L64"] = eval_ifbench64_post_rag(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_pre_query=[sp_exp9_algb], tag="PostRAG ZeroPostRL_AlgB_L64")
        del sp_exp9_algb
    be["PreQuery_PostRL_Direct_L64"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_pre_query=[sp_direct_if_l64], tag="PreQuery PostRL_Direct_L64")
    be["PostQuery_PostRL_Direct_L64"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_post_query=[sp_direct_if_l64], tag="PostQuery PostRL_Direct_L64")
    be["PostRAG_PostRL_Direct_L256"] = eval_ifbench64_post_rag(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_pre_query=[sp_direct_if_l256], tag="PostRAG PostRL_Direct_L256")
    be["PostQuery_PostRL_Direct_L256"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_post_query=[sp_direct_if_l256], tag="PostQuery PostRL_Direct_L256")
    be["PreQuery_PostRL_KVInv2_L256"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_pre_query=[sp_kvinv2_if_l256], tag="PreQuery AlgA KVInv2_L256")
    be["PostRAG_PostRL_KVInv2_L256"] = eval_ifbench64_post_rag(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_pre_query=[sp_kvinv2_if_l256], tag="PostRAG AlgA KVInv2_L256")
    be["PostQuery_PostRL_KVInv2_L256"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_post_query=[sp_kvinv2_if_l256], tag="PostQuery AlgA KVInv2_L256")
    be["PostQuery_PostRL_KVInv2_L64"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_post_query=[sp_kvinv2_if_l64], tag="PostQuery AlgA KVInv2_L64")
    be["Triple_PostRL_Direct_Split"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_sys_end=[e3_doc_cscr], sp_pre_query=[sp_direct_math_l64], sp_post_query=[sp_direct_if_l64], tag="Triple PostRL_Direct [PostSys+PreQuery+PostQuery]")
    be["Triple_PostRL_KVInv2_Split"] = eval_ifbench64_slots(student, tokenizer, tmpl, eval_ctx, ifbench64, sp_sys_end=[e3_doc_cscr], sp_pre_query=[sp_kvinv2_math_l64], sp_post_query=[sp_kvinv2_if_l64], tag="Triple PostRL_KVInv2 [PostSys+PreQuery+PostQuery]")
    save()

    print("\n=== Step 5C: MATH48 & HeliosPay Doc36 Post-RL Composition Evaluations ===", flush=True)
    me = results["math48_evals"]
    me["PreQuery_PostRL_Direct_Math_L64"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_pre_query=[sp_direct_math_l64], tag="PreQuery PostRL_Direct_Math_L64")
    me["PostQuery_PostRL_Direct_Math_L64"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_post_query=[sp_direct_math_l64], tag="PostQuery PostRL_Direct_Math_L64")
    me["PreQuery_PostRL_KVInv2_Math_L64"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_pre_query=[sp_kvinv2_math_l64], tag="PreQuery AlgA KVInv2_Math_L64")
    me["Dual_PostRL_Adj_IF_Math_PreQuery"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_pre_query=[sp_direct_if_l64, sp_direct_math_l64], tag="Dual PostRL [IF;Math]@PreQuery")
    me["Dual_PostRL_Adj_Math_IF_PreQuery"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_pre_query=[sp_direct_math_l64, sp_direct_if_l64], tag="Dual PostRL [Math;IF]@PreQuery")
    me["Triple_PostRL_Direct_Split"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_sys_end=[e3_doc_cscr], sp_pre_query=[sp_direct_math_l64], sp_post_query=[sp_direct_if_l64], tag="Triple PostRL_Direct [PostSys+PreQuery+PostQuery]")
    me["Triple_PostRL_KVInv2_Split"] = eval_math48_slots(student, tokenizer, tmpl, eval_ctx, math_test48, sp_sys_end=[e3_doc_cscr], sp_pre_query=[sp_kvinv2_math_l64], sp_post_query=[sp_kvinv2_if_l64], tag="Triple PostRL_KVInv2 [PostSys+PreQuery+PostQuery]")
    save()

    de = results["doc36_evals"]
    de["Single_Doc_CSCR_MuonL128_PostSys"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_cscr], tag="Single Doc_MuonL128 @ PostSys")
    de["Single_Doc_CSCR_MuonL128_PreQuery"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_pre_query=[e3_doc_cscr], tag="Single Doc_MuonL128 @ PreQuery")
    de["Single_Doc_CSCR_MuonL128_PostQuery"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_post_query=[e3_doc_cscr], tag="Single Doc_MuonL128 @ PostQuery")
    exp9_m_path = os.path.join(EXP9_DIR, "sp_math_cscr.pt")
    if os.path.exists(exp9_algb_path) and os.path.exists(exp9_m_path):
        sp_e9_if = torch.load(exp9_algb_path, map_location="cuda").float()
        sp_e9_ma = torch.load(exp9_m_path, map_location="cuda").float()
        de["Triple_ZeroPostRL_FixedDoc_Split"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_cscr], sp_pre_query=[sp_e9_ma], sp_post_query=[sp_e9_if], tag="Triple ZeroPostRL+Doc128 [PostSys+PreQuery+PostQuery]")
        del sp_e9_if, sp_e9_ma
    de["Triple_PostRL_Direct_Split"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_cscr], sp_pre_query=[sp_direct_math_l64], sp_post_query=[sp_direct_if_l64], tag="Triple PostRL_Direct [PostSys+PreQuery+PostQuery]")
    de["Triple_PostRL_KVInv2_Split"] = eval_doc36_slots(student, tokenizer, tmpl, eval_ctx, doc_test_rules, doc_test_facts, sp_sys_end=[e3_doc_cscr], sp_pre_query=[sp_kvinv2_math_l64], sp_post_query=[sp_kvinv2_if_l64], tag="Triple PostRL_KVInv2 [PostSys+PreQuery+PostQuery]")
    save()

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== Suite 7B (exp9b) COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
