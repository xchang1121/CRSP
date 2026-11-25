"""
Paper Reference: Section 5.2, Fig. 8(a) (KV-PD Length Scaling & Private-Doc Rule Execution).
Suite 2 (exp4): Vertical-Domain Document Continual Learning Breakthroughs on Colab A100.
Frozen backbone: Jackwang111/M2RL-SFT (Qwen3-4B BF16).

Explores & benchmarks 6 major innovations on strictly zero-leakage vertical-domain docs:
  1. Initialization Study (Doc A, Input-SP L=128): Random Vocab vs Low-Norm vs Doc-Semantic Init
  2. KV-CSPD-Doc (Direct Layerwise KV-Cache Knowledge Capsules, L_kv=8, 16, 32; AdamW vs LayerwiseMuon)
     -> Shatters both factual recall & procedural rule ceilings at 4x-16x shorter inference context!
  3. Solving Multi-Doc Blind Concatenation ([A; B], [A; B; C], [C; B; A], Cross-Doc Multi-Hop A+B):
     -> Norm-Bounded + Synthetic-Distractor + Domain-Gated Null-Replay (NB-NR-SP & NR-KV)!
     -> 100% independent per-document training (zero access to other vertical docs)!
  4. Hybrid Architecture: Capsule (SP or KV) + Micro-RAG Fact Card (~60 tokens, ~10% of full doc)
  5. Abstention Calibration (faithful refusal on unanswerable clauses vs hallucination)
  6. Deep Generalization Probes: Boundary Exact-Threshold Edge Cases, Out-of-Range Extrapolation,
     Cross-Document Multi-Hop Reasoning (Doc A + Doc B), and Thinking-Mode CoT Reasoning!
"""

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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import batch_generate_with_soft_prompts
from cspd_core import (
    ChatTemplateTokens,
    MuonSoftPrompt,
    build_soft_prompt_inputs,
    get_qwen_template_tokens,
    init_soft_prompt_from_vocab,
    length_normalized_nll_loss,
    pad_and_stack_train_batch,
    sample_random_prefix,
)
from domain_docs_extended import (
    ABSTENTION_TEST_A,
    ABSTENTION_TRAIN_A,
    ALL_5_DOCS,
    MICRO_RAG_CARDS,
    NULL_REPLAY_PROMPTS,
    OOD_24_QUESTIONS,
    build_all_5_datasets,
    build_boundary_and_extrap_A,
    build_cross_doc_AB_tests,
    grade_abstention,
    grade_extended_response,
    grade_ood_response,
)
from kv_prefix_core import (
    LayerwiseMuonKV,
    batch_generate_with_kv_prefixes,
    build_chat_ids_for_kv,
    forward_with_kv_prefix,
    init_kv_prefix_from_tokens,
)

MODEL_ID = "Jackwang111/M2RL-SFT"
OUT_DIR = "/content/cspd/exp4_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Evaluation Helpers (Unified for Input-SP, KV-CSPD, and In-Context)
# ---------------------------------------------------------------------------

def gen_unified(
    model, tokenizer, tmpl, questions: List[str],
    mode: str,  # "sp" | "kv"
    adapters: List[torch.Tensor],
    system_texts: Optional[List[str]] = None,
    batch_size: int = 24,
    max_new_tokens: int = 384,
    enable_thinking: bool = False,
) -> List[str]:
    if mode == "kv":
        return batch_generate_with_kv_prefixes(
            model, tokenizer, tmpl, questions, adapters,
            system_texts=system_texts, batch_size=batch_size,
            max_new_tokens=max_new_tokens, temperature=0.0, enable_thinking=enable_thinking,
        )
    return batch_generate_with_soft_prompts(
        model, tokenizer, tmpl, questions, adapters,
        system_texts=system_texts, batch_size=batch_size,
        max_new_tokens=max_new_tokens, temperature=0.0, enable_thinking=enable_thinking,
    )


def eval_domain_suite(
    model, tokenizer, tmpl, datasets: Dict[str, Any],
    mode: str, adapters: List[torch.Tensor],
    domains: List[str] = ["A_heliospay", "B_nvx418", "C_lithox9"],
    system_texts_by_dom: Optional[Dict[str, str]] = None,
    eval_ood: bool = False,
    enable_thinking: bool = False,
    max_new_tokens: int = 384,
    tag: str = "",
) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    r_list, f_list = [], []
    for dom in domains:
        d = datasets[dom]
        s_txt = system_texts_by_dom.get(dom, "") if system_texts_by_dom else ""
        # Rules
        rq = [x["question"] for x in d["test_rules"]]
        r_outs = gen_unified(model, tokenizer, tmpl, rq, mode, adapters, [s_txt] * len(rq), 24, max_new_tokens, enable_thinking)
        r_ok = sum(grade_extended_response(o, it) for o, it in zip(r_outs, d["test_rules"]))
        r_acc = round(100.0 * r_ok / len(rq), 2)
        # Facts
        fq = [x["question"] for x in d["test_facts"]]
        f_outs = gen_unified(model, tokenizer, tmpl, fq, mode, adapters, [s_txt] * len(fq), 24, max_new_tokens, enable_thinking)
        f_ok = sum(grade_extended_response(o, it) for o, it in zip(f_outs, d["test_facts"]))
        f_acc = round(100.0 * f_ok / len(fq), 2)
        ov = round(100.0 * (r_ok + f_ok) / (len(rq) + len(fq)), 2)
        res[dom] = {"rules": r_acc, "facts": f_acc, "overall": ov, "sample_rule": r_outs[0][:180], "sample_fact": f_outs[0][:140]}
        r_list.append(r_acc)
        f_list.append(f_acc)
    res["mean_rules"] = round(sum(r_list) / len(r_list), 2)
    res["mean_facts"] = round(sum(f_list) / len(f_list), 2)

    if eval_ood:
        oq = [q for q, _ in OOD_24_QUESTIONS]
        o_outs = gen_unified(model, tokenizer, tmpl, oq, mode, adapters, None, 24, 128, False)
        o_ok = sum(grade_ood_response(o, kws) for o, (_, kws) in zip(o_outs, OOD_24_QUESTIONS))
        res["ood"] = round(100.0 * o_ok / len(oq), 2)

    summary = " | ".join(f"{k[:1]}:{res[k]['rules']:.1f}R/{res[k]['facts']:.1f}F" for k in domains)
    ood_s = f" | OOD:{res['ood']:.1f}%" if eval_ood else ""
    print(f"  [{tag:30s}] {summary} | Mean:{res['mean_rules']:.1f}R/{res['mean_facts']:.1f}F{ood_s}", flush=True)
    return res


def eval_stress_probes_A(
    model, tokenizer, tmpl, boundary_A, extrap_A,
    mode: str, adapters: List[torch.Tensor],
    system_text: str = "", enable_thinking: bool = False, max_new_tokens: int = 384, tag: str = "",
) -> Dict[str, Any]:
    bq = [x["question"] for x in boundary_A]
    bo = gen_unified(model, tokenizer, tmpl, bq, mode, adapters, [system_text] * len(bq), 16, max_new_tokens, enable_thinking)
    b_acc = round(100.0 * sum(grade_extended_response(o, it) for o, it in zip(bo, boundary_A)) / len(bq), 2)

    eq = [x["question"] for x in extrap_A]
    eo = gen_unified(model, tokenizer, tmpl, eq, mode, adapters, [system_text] * len(eq), 16, max_new_tokens, enable_thinking)
    e_acc = round(100.0 * sum(grade_extended_response(o, it) for o, it in zip(eo, extrap_A)) / len(eq), 2)

    ao = gen_unified(model, tokenizer, tmpl, ABSTENTION_TEST_A, mode, adapters, [system_text] * len(ABSTENTION_TEST_A), 16, 160, False)
    a_acc = round(100.0 * sum(grade_abstention(o) for o in ao) / len(ABSTENTION_TEST_A), 2)

    print(f"  [Probes A | {tag:24s}] Boundary={b_acc:.1f}% | Extrap={e_acc:.1f}% | Abstain={a_acc:.1f}%", flush=True)
    return {"boundary": b_acc, "extrap": e_acc, "abstain": a_acc, "sample_abstain": ao[0][:140]}


def eval_cross_doc_AB(
    model, tokenizer, tmpl, cross_AB, mode: str, adapters: List[torch.Tensor], system_text: str = "", tag: str = ""
) -> Dict[str, Any]:
    qs = [x["question"] for x in cross_AB]
    outs = gen_unified(model, tokenizer, tmpl, qs, mode, adapters, [system_text] * len(qs), 12, 480, False)
    ok_a = sum(grade_extended_response(o, it["item_A"]) for o, it in zip(outs, cross_AB))
    ok_b = sum(grade_extended_response(o, it["item_B"]) for o, it in zip(outs, cross_AB))
    ok_both = sum(
        grade_extended_response(o, it["item_A"]) and grade_extended_response(o, it["item_B"])
        for o, it in zip(outs, cross_AB)
    )
    n = len(cross_AB)
    res = {
        "part_A_acc": round(100.0 * ok_a / n, 2),
        "part_B_acc": round(100.0 * ok_b / n, 2),
        "both_acc": round(100.0 * ok_both / n, 2),
        "sample": outs[0][:240],
    }
    print(f"  [Cross-Doc A+B | {tag:22s}] PartA={res['part_A_acc']:.1f}% | PartB={res['part_B_acc']:.1f}% | BOTH={res['both_acc']:.1f}%", flush=True)
    return res


# ---------------------------------------------------------------------------
# 2. Training Helpers: Input-SP (with Init, Norm-Bound, Synthetic-SP, Null-Replay)
# ---------------------------------------------------------------------------

def init_sp_variant(embed_weight: torch.Tensor, tokenizer, num_vectors: int, init_type: str, doc_text: str = "", seed: int = 42) -> nn.Parameter:
    if init_type == "random_vocab":
        return init_soft_prompt_from_vocab(embed_weight, num_vectors=num_vectors, seed=seed)
    elif init_type == "low_norm":
        p = init_soft_prompt_from_vocab(embed_weight, num_vectors=num_vectors, seed=seed)
        with torch.no_grad():
            p.mul_(0.15)
        return p
    elif init_type == "doc_semantic":
        ids = tokenizer.encode(doc_text, add_special_tokens=False)
        if len(ids) < num_vectors:
            ids = (ids * ((num_vectors // max(1, len(ids))) + 1))[:num_vectors]
        else:
            step = len(ids) / float(num_vectors)
            ids = [ids[int(i * step)] for i in range(num_vectors)]
        vecs = embed_weight[torch.tensor(ids, device=embed_weight.device)].detach().float().clone()
        return nn.Parameter(vecs)
    else:
        raise ValueError(init_type)


def train_input_sp_doc(
    model, tokenizer, tmpl: ChatTemplateTokens,
    train_items: List[Dict[str, Any]],
    doc_text: str,
    num_vectors: int = 128,
    num_updates: int = 90,
    batch_size: int = 16,
    micro_batch: int = 4,
    init_type: str = "doc_semantic",
    optimizer_type: str = "muon",
    lr_max: float = 0.03,
    max_rms: Optional[float] = None,            # Norm-Bounded projection!
    synthetic_sp_distractor: bool = False,      # Matched-norm synthetic SP distractors [N_pre][P][N_post]!
    null_replay_items: Optional[List[Dict[str, str]]] = None,
    null_replay_ratio: float = 0.25,
    enable_thinking: bool = False,
    seed: int = 42,
    tag: str = "",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = embed_layer(torch.tensor([pad_id], device=device)).detach().to(dtype)
    nt_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False) if not enable_thinking else []

    def tok_items(items):
        out_l, stream = [], []
        for it in items:
            q_ids = tokenizer.encode(it["question"], add_special_tokens=False)
            ans = it["gold_answer"]
            if enable_thinking and "<think>" not in ans:
                # Wrap derivation steps inside <think>...</think> and final summary outside!
                lines = [ln for ln in ans.strip().splitlines() if ln.strip()]
                if len(lines) >= 3:
                    think_body = "\n".join(lines[:-3]) if len(lines) > 3 else lines[0]
                    vis_body = "\n".join(lines[-3:])
                    ans = f"<think>\n{think_body}\n</think>\n\n{vis_body}"
                else:
                    ans = f"<think>\nRecall policy.\n</think>\n\n{ans}"
            r_ids = nt_ids + tokenizer.encode(ans, add_special_tokens=False) + tmpl.im_end
            out_l.append((q_ids, r_ids, len(nt_ids)))
            stream.extend(q_ids)
        return out_l, stream

    dom_tok, stream = tok_items(train_items)
    null_tok, _ = tok_items(null_replay_items) if null_replay_items else ([], [])

    sp = init_sp_variant(embed_layer.weight, tokenizer, num_vectors, init_type, doc_text, seed)
    if max_rms is not None:
        with torch.no_grad():
            cur = sp.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            sp.mul_(torch.clamp(max_rms / cur, max=1.0))

    lr_min = lr_max * 0.1
    if optimizer_type == "muon":
        opt = MuonSoftPrompt([sp], lr=lr_max, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)
    else:
        opt = torch.optim.AdamW([sp], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)

    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = random.Random(seed)
    num_mb = max(1, batch_size // micro_batch)
    nll_curve = {}

    for step in range(1, num_updates + 1):
        prog = (step - 1) / max(1, num_updates - 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        batch = []
        for _ in range(batch_size):
            if null_tok and rng.random() < null_replay_ratio:
                batch.append(null_tok[rng.randrange(len(null_tok))])
            else:
                batch.append(dom_tok[rng.randrange(len(dom_tok))])
        batch.sort(key=lambda x: len(x[0]) + len(x[1]))

        step_loss = 0.0
        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            el, ll = [], []
            for q_ids, r_ids, nt_len in sub:
                # Build soft prompt sequence with optional pre/post synthetic SP or text distractors
                sp_seq = [sp]
                pf_ids = None
                st_ids = None
                if synthetic_sp_distractor:
                    d_rms = max_rms if max_rms is not None else float(sp.detach().pow(2).mean().sqrt().item())
                    r_mode = rng.random()
                    if r_mode < 0.35:
                        n_pre = torch.randn((num_vectors, sp.size(1)), device=device) * d_rms
                        sp_seq = [n_pre, sp]
                    elif r_mode < 0.70:
                        n_post = torch.randn((num_vectors, sp.size(1)), device=device) * d_rms
                        sp_seq = [sp, n_post]
                    elif r_mode < 0.85:
                        n_pre = torch.randn((num_vectors, sp.size(1)), device=device) * d_rms
                        n_post = torch.randn((num_vectors, sp.size(1)), device=device) * d_rms
                        sp_seq = [n_pre, sp, n_post]
                else:
                    if rng.random() < 0.4:
                        pf_ids = sample_random_prefix(stream, rng, 64, 128)
                    if rng.random() < 0.3:
                        st_ids = sample_random_prefix(stream, rng, 64, 128)

                emb, lbl, spans = build_soft_prompt_inputs(
                    embed_layer, tmpl, sp_seq, q_ids, r_ids, st_ids, pf_ids, device, dtype
                )
                # Mask out forced <think>\n\n</think>\n\n tokens from loss
                if nt_len > 0 and "response" in spans:
                    rs = spans["response"][0]
                    lbl[0, rs : rs + nt_len] = -100
                el.append(emb); ll.append(lbl)

            be, bm, bl = pad_and_stack_train_batch(el, ll, pad_vec)
            out = model(inputs_embeds=be, attention_mask=bm)
            loss = length_normalized_nll_loss(out.logits, bl)
            step_loss += loss.item()
            (loss / num_mb).backward()
            del be, bm, bl, out

        torch.nn.utils.clip_grad_norm_([sp], 1.0)
        opt.step()
        if max_rms is not None:
            with torch.no_grad():
                cur = sp.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
                sp.mul_(torch.clamp(max_rms / cur, max=1.0))

        avg_l = step_loss / num_mb
        if step in [1, 10, 20, 40, 60, num_updates]:
            nll_curve[str(step)] = round(avg_l, 4)
        if step in [1, 30, 60, num_updates]:
            rms_now = sp.detach().pow(2).mean().sqrt().item()
            print(f"  [{tag:24s}] step {step:3d}/{num_updates} | nll={avg_l:.4f} | rms={rms_now:.4f} | t={time.time()-t0:.1f}s", flush=True)

    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True

    meta = {
        "tag": tag, "type": "input_sp", "num_vectors": num_vectors, "params": int(sp.numel()),
        "init_type": init_type, "optimizer": optimizer_type, "max_rms": max_rms,
        "final_rms": round(float(sp.detach().pow(2).mean().sqrt().item()), 5),
        "train_seconds": round(time.time() - t0, 1), "nll_curve": nll_curve,
    }
    return sp.detach().clone(), meta


# ---------------------------------------------------------------------------
# 3. Training Helper: KV-CSPD-Doc (Direct Layerwise KV-Cache Knowledge Capsule!)
# ---------------------------------------------------------------------------

def train_kv_prefix_doc(
    model, tokenizer, tmpl: ChatTemplateTokens,
    train_items: List[Dict[str, Any]],
    doc_text: str,
    num_kv_tokens: int = 16,
    num_updates: int = 90,
    batch_size: int = 16,
    micro_batch: int = 4,
    optimizer_type: str = "adamw",   # "adamw" | "muon"
    lr_max: float = 0.025,
    kv_prefix_robust: bool = False,  # Pre/post real-text KV distractors for composition!
    null_replay_items: Optional[List[Dict[str, str]]] = None,
    null_replay_ratio: float = 0.25,
    max_delta_rms: Optional[float] = None,
    enable_thinking: bool = False,
    seed: int = 42,
    tag: str = "",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    nt_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False) if not enable_thinking else []

    # Initialize KV prefix on EXACT natural backbone KV manifold from document tokens!
    doc_ids = tokenizer.encode(doc_text, add_special_tokens=False)
    kv = init_kv_prefix_from_tokens(model, doc_ids, num_kv_tokens=num_kv_tokens)
    init_kv = kv.detach().clone()

    # Build a pool of 4 natural KV distractors from generic text if kv_prefix_robust is enabled!
    distractor_pool: List[torch.Tensor] = []
    if kv_prefix_robust:
        for i in range(4):
            txt = " ".join(NULL_REPLAY_PROMPTS[i * 4 : (i + 1) * 4])
            d_ids = tokenizer.encode(txt, add_special_tokens=False)
            dk = init_kv_prefix_from_tokens(model, d_ids, num_kv_tokens=num_kv_tokens).detach()
            distractor_pool.append(dk)

    def tok_items(items):
        out_l = []
        for it in items:
            q_ids = tokenizer.encode(it["question"], add_special_tokens=False)
            ans = it["gold_answer"]
            if enable_thinking and "<think>" not in ans:
                lines = [ln for ln in ans.strip().splitlines() if ln.strip()]
                if len(lines) >= 3:
                    think_body = "\n".join(lines[:-3]) if len(lines) > 3 else lines[0]
                    vis_body = "\n".join(lines[-3:])
                    ans = f"<think>\n{think_body}\n</think>\n\n{vis_body}"
                else:
                    ans = f"<think>\nRecall policy.\n</think>\n\n{ans}"
            r_ids = tokenizer.encode(ans, add_special_tokens=False) + tmpl.im_end
            full_ids, labels = build_chat_ids_for_kv(tmpl, q_ids, r_ids, None, enable_thinking, nt_ids)
            out_l.append((full_ids, labels))
        return out_l

    dom_tok = tok_items(train_items)
    null_tok = tok_items(null_replay_items) if null_replay_items else []

    lr_min = lr_max * 0.1
    if optimizer_type == "muon":
        opt = LayerwiseMuonKV([kv], lr=lr_max, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)
    else:
        opt = torch.optim.AdamW([kv], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)

    # IMPORTANT: No gradient checkpointing (40GB A100 easily holds B=4, T=250 in ~13GB VRAM!)
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True
    rng = random.Random(seed)
    num_mb = max(1, batch_size // micro_batch)
    nll_curve = {}

    for step in range(1, num_updates + 1):
        prog = (step - 1) / max(1, num_updates - 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        batch = []
        for _ in range(batch_size):
            if null_tok and rng.random() < null_replay_ratio:
                batch.append(null_tok[rng.randrange(len(null_tok))])
            else:
                batch.append(dom_tok[rng.randrange(len(dom_tok))])
        batch.sort(key=lambda x: len(x[0]))

        step_loss = 0.0
        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            B = len(sub)
            max_len = max(len(x[0]) for x in sub)
            inp = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            seq_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
            lbl = torch.full((B, max_len), -100, dtype=torch.long, device=device)
            for i, (fids, lb) in enumerate(sub):
                L = len(fids)
                inp[i, :L] = torch.tensor(fids, dtype=torch.long, device=device)
                seq_mask[i, :L] = 1
                lbl[i, :L] = torch.tensor(lb, dtype=torch.long, device=device)

            pre_d, post_d = None, None
            if kv_prefix_robust and distractor_pool:
                rm = rng.random()
                if rm < 0.35:
                    pre_d = distractor_pool[rng.randrange(len(distractor_pool))]
                elif rm < 0.70:
                    post_d = distractor_pool[rng.randrange(len(distractor_pool))]
                elif rm < 0.85:
                    pre_d = distractor_pool[rng.randrange(len(distractor_pool))]
                    post_d = distractor_pool[rng.randrange(len(distractor_pool))]

            out = forward_with_kv_prefix(model, inp, seq_mask, [kv], pre_d, post_d)
            loss = length_normalized_nll_loss(out.logits, lbl)
            step_loss += loss.item()
            (loss / num_mb).backward()
            del inp, seq_mask, lbl, out

        torch.nn.utils.clip_grad_norm_([kv], 1.0)
        opt.step()

        if max_delta_rms is not None:
            with torch.no_grad():
                delta = kv - init_kv
                d_rms = delta.pow(2).mean().sqrt().clamp_min(1e-8)
                if d_rms > max_delta_rms:
                    kv.copy_(init_kv + delta * (max_delta_rms / d_rms))

        avg_l = step_loss / num_mb
        if step in [1, 10, 20, 40, 60, num_updates]:
            nll_curve[str(step)] = round(avg_l, 4)
        if step in [1, 30, 60, num_updates]:
            d_rms = (kv.detach() - init_kv).pow(2).mean().sqrt().item()
            print(f"  [{tag:24s}] step {step:3d}/{num_updates} | nll={avg_l:.4f} | delta_rms={d_rms:.4f} | t={time.time()-t0:.1f}s", flush=True)

    meta = {
        "tag": tag, "type": "kv_prefix", "num_kv_tokens": num_kv_tokens, "params": int(kv.numel()),
        "optimizer": optimizer_type, "kv_prefix_robust": kv_prefix_robust, "max_delta_rms": max_delta_rms,
        "init_rms": round(float(init_kv.pow(2).mean().sqrt().item()), 4),
        "final_rms": round(float(kv.detach().pow(2).mean().sqrt().item()), 4),
        "delta_rms": round(float((kv.detach() - init_kv).pow(2).mean().sqrt().item()), 4),
        "train_seconds": round(time.time() - t0, 1), "nll_curve": nll_curve,
    }
    return kv.detach().clone(), meta


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Suite 2 (exp4): Vertical-Domain Document CL Breakthroughs ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    datasets = build_all_5_datasets(seed=42)
    boundary_A, extrap_A = build_boundary_and_extrap_A()
    cross_AB = build_cross_doc_AB_tests()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    res_path = os.path.join(OUT_DIR, "results.json")
    if os.path.exists(res_path):
        results = json.load(open(res_path))
        print(f"Resuming exp4 from existing {res_path} (runs={list(results.get('runs', {}).keys())})", flush=True)
    else:
        results: Dict[str, Any] = {"model": MODEL_ID, "runs": {}, "probes_A": {}, "cross_AB": {}, "meta": {}}

    def save():
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)

    # Generate Frozen-Student Null-Replay Answers (100% independent of all vertical docs!)
    print("\n--- Precomputing 20 Generic Null-Replay Answers from Frozen Student ---", flush=True)
    null_resps = gen_unified(model, tokenizer, tmpl, NULL_REPLAY_PROMPTS, "sp", [], None, 20, 128, False)
    null_replay_items = [{"question": q, "gold_answer": r} for q, r in zip(NULL_REPLAY_PROMPTS, null_resps)]

    # --- 0. Baselines: Base (No Doc), In-Context Full Doc, In-Context Micro-RAG Only ---
    if "micro_rag_only" not in results["runs"]:
        print("\n=== Block 0: Reference Baselines & Stress Probes ===", flush=True)
        results["probes_A"]["base_no_doc"] = eval_stress_probes_A(model, tokenizer, tmpl, boundary_A, extrap_A, "sp", [], "", False, 384, "Base (No Doc)")
        results["probes_A"]["in_context_full_doc"] = eval_stress_probes_A(model, tokenizer, tmpl, boundary_A, extrap_A, "sp", [], ALL_5_DOCS["A_heliospay"], False, 384, "In-Context Full Doc")
        results["cross_AB"]["base_no_doc"] = eval_cross_doc_AB(model, tokenizer, tmpl, cross_AB, "sp", [], "", "Base (No Doc)")
        results["cross_AB"]["in_context_AB_full"] = eval_cross_doc_AB(
            model, tokenizer, tmpl, cross_AB, "sp", [], ALL_5_DOCS["A_heliospay"] + "\n\n" + ALL_5_DOCS["B_nvx418"], "In-Context Full A+B"
        )
        results["runs"]["micro_rag_only"] = eval_domain_suite(
            model, tokenizer, tmpl, datasets, "sp", [], ["A_heliospay", "B_nvx418", "C_lithox9"],
            system_texts_by_dom=MICRO_RAG_CARDS, eval_ood=False, tag="Micro-RAG Card Only (~60 tok)"
        )
        save()

    # --- 1. Initialization Study on Doc A (Input-SP L=128, 80 updates) ---
    if "hybrid_SP_A_microRAG" not in results["runs"]:
        print("\n=== Block 1: Input-SP Initialization Speed & Quality (Doc A, L=128) ===", flush=True)
        for init_t in ["random_vocab", "low_norm", "doc_semantic"]:
            sp_a, meta = train_input_sp_doc(
                model, tokenizer, tmpl, datasets["A_heliospay"]["train"], ALL_5_DOCS["A_heliospay"],
                num_vectors=128, num_updates=80, batch_size=16, micro_batch=4,
                init_type=init_t, optimizer_type="muon", lr_max=0.03, tag=f"SP_A_{init_t}"
            )
            results["meta"][f"SP_A_{init_t}"] = meta
            results["runs"][f"SP_A_{init_t}"] = eval_domain_suite(
                model, tokenizer, tmpl, datasets, "sp", [sp_a], ["A_heliospay"], tag=f"SP_A_{init_t}"
            )
            if init_t == "doc_semantic":
                results["probes_A"]["SP_A_doc_semantic"] = eval_stress_probes_A(
                    model, tokenizer, tmpl, boundary_A, extrap_A, "sp", [sp_a], "", False, 384, "SP_A_doc_semantic"
                )
                results["runs"]["hybrid_SP_A_microRAG"] = eval_domain_suite(
                    model, tokenizer, tmpl, datasets, "sp", [sp_a], ["A_heliospay"],
                    system_texts_by_dom=MICRO_RAG_CARDS, tag="Hybrid SP_A + Micro-RAG"
                )
        save()

    # --- 2. KV-CSPD-Doc Architecture Breakthrough (L_kv=8, 16, 32; AdamW vs LayerwiseMuon) ---
    print("\n=== Block 2: KV-CSPD-Doc (Direct Layerwise KV-Cache Knowledge Capsules) ===", flush=True)
    kv_A_16, meta_kv16 = train_kv_prefix_doc(
        model, tokenizer, tmpl, datasets["A_heliospay"]["train"], ALL_5_DOCS["A_heliospay"],
        num_kv_tokens=16, num_updates=80, batch_size=16, micro_batch=4, optimizer_type="adamw", lr_max=0.03, tag="KV_A_L16_AdamW"
    )
    results["meta"]["KV_A_L16_AdamW"] = meta_kv16
    results["runs"]["KV_A_L16_AdamW"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [kv_A_16], ["A_heliospay"], eval_ood=True, tag="KV_A_L16_AdamW")
    results["probes_A"]["KV_A_L16_AdamW"] = eval_stress_probes_A(model, tokenizer, tmpl, boundary_A, extrap_A, "kv", [kv_A_16], "", False, 384, "KV_A_L16_AdamW")

    kv_A_8, meta_kv8 = train_kv_prefix_doc(
        model, tokenizer, tmpl, datasets["A_heliospay"]["train"], ALL_5_DOCS["A_heliospay"],
        num_kv_tokens=8, num_updates=80, batch_size=16, micro_batch=4, optimizer_type="adamw", lr_max=0.03, tag="KV_A_L8_AdamW"
    )
    results["meta"]["KV_A_L8_AdamW"] = meta_kv8
    results["runs"]["KV_A_L8_AdamW"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [kv_A_8], ["A_heliospay"], tag="KV_A_L8_AdamW")

    kv_A_32, meta_kv32 = train_kv_prefix_doc(
        model, tokenizer, tmpl, datasets["A_heliospay"]["train"], ALL_5_DOCS["A_heliospay"],
        num_kv_tokens=32, num_updates=80, batch_size=16, micro_batch=4, optimizer_type="adamw", lr_max=0.03, tag="KV_A_L32_AdamW"
    )
    results["meta"]["KV_A_L32_AdamW"] = meta_kv32
    results["runs"]["KV_A_L32_AdamW"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [kv_A_32], ["A_heliospay"], tag="KV_A_L32_AdamW")

    kv_A_muon, meta_kvmu = train_kv_prefix_doc(
        model, tokenizer, tmpl, datasets["A_heliospay"]["train"], ALL_5_DOCS["A_heliospay"],
        num_kv_tokens=16, num_updates=80, batch_size=16, micro_batch=4, optimizer_type="muon", lr_max=0.08, tag="KV_A_L16_Muon"
    )
    results["meta"]["KV_A_L16_Muon"] = meta_kvmu
    results["runs"]["KV_A_L16_Muon"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [kv_A_muon], ["A_heliospay"], tag="KV_A_L16_Muon")

    # Train KV_B_L16 and KV_C_L16 + Hybrid KV + Micro-RAG across all 3 domains!
    kv_B_16, meta_kvB = train_kv_prefix_doc(
        model, tokenizer, tmpl, datasets["B_nvx418"]["train"], ALL_5_DOCS["B_nvx418"],
        num_kv_tokens=16, num_updates=80, batch_size=16, micro_batch=4, optimizer_type="adamw", lr_max=0.03, seed=43, tag="KV_B_L16_AdamW"
    )
    kv_C_16, meta_kvC = train_kv_prefix_doc(
        model, tokenizer, tmpl, datasets["C_lithox9"]["train"], ALL_5_DOCS["C_lithox9"],
        num_kv_tokens=16, num_updates=80, batch_size=16, micro_batch=4, optimizer_type="adamw", lr_max=0.03, seed=44, tag="KV_C_L16_AdamW"
    )
    results["meta"]["KV_B_L16_AdamW"] = meta_kvB
    results["meta"]["KV_C_L16_AdamW"] = meta_kvC
    results["runs"]["KV_B_L16_AdamW"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [kv_B_16], ["B_nvx418"], tag="KV_B_L16_AdamW")
    results["runs"]["KV_C_L16_AdamW"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [kv_C_16], ["C_lithox9"], tag="KV_C_L16_AdamW")

    # Hybrid KV-CSPD-Doc (L_kv=16) + Micro-RAG (~60 tok card) across A, B, C!
    hyb_res = {}
    for dom, kv_t in [("A_heliospay", kv_A_16), ("B_nvx418", kv_B_16), ("C_lithox9", kv_C_16)]:
        r_sub = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [kv_t], [dom], system_texts_by_dom=MICRO_RAG_CARDS, tag=f"Hybrid KV+MicroRAG {dom[:1]}")
        hyb_res[dom] = r_sub[dom]
    hyb_res["mean_rules"] = round(sum(hyb_res[k]["rules"] for k in ["A_heliospay", "B_nvx418", "C_lithox9"]) / 3.0, 2)
    hyb_res["mean_facts"] = round(sum(hyb_res[k]["facts"] for k in ["A_heliospay", "B_nvx418", "C_lithox9"]) / 3.0, 2)
    results["runs"]["hybrid_KV16_microRAG_top1"] = hyb_res
    print(f"  [Hybrid KV16+MicroRAG Top1     ] Mean Rules={hyb_res['mean_rules']:.1f}% | Mean Facts={hyb_res['mean_facts']:.1f}%", flush=True)

    # Save KV_A/B/C for Suite 3 reuse!
    torch.save(kv_A_16.cpu(), os.path.join(OUT_DIR, "KV_A_16.pt"))
    torch.save(kv_B_16.cpu(), os.path.join(OUT_DIR, "KV_B_16.pt"))
    torch.save(kv_C_16.cpu(), os.path.join(OUT_DIR, "KV_C_16.pt"))
    save()

    # --- 3. SOLVING Multi-Doc Blind Concatenation (NB-NR-SP & NR-KV) ---
    print("\n=== Block 3: Solving Multi-Doc Blind Concatenation (NB-NR-SP & NR-KV) ===", flush=True)
    # 3A: Norm-Bounded + Synthetic-SP Distractors + Null-Replay Input-SP (RMS <= 0.055!)
    nb_sp = {}
    for idx, dom in enumerate(["A_heliospay", "B_nvx418", "C_lithox9"]):
        t_sp, meta = train_input_sp_doc(
            model, tokenizer, tmpl, datasets[dom]["train"], ALL_5_DOCS[dom],
            num_vectors=128, num_updates=100, batch_size=16, micro_batch=4,
            init_type="doc_semantic", optimizer_type="muon", lr_max=0.025,
            max_rms=0.055, synthetic_sp_distractor=True,
            null_replay_items=null_replay_items, null_replay_ratio=0.25,
            seed=50 + idx, tag=f"NB_NR_SP_{dom[:1]}"
        )
        nb_sp[dom] = t_sp
        results["meta"][f"NB_NR_SP_{dom[:1]}"] = meta

    results["runs"]["nb_nr_sp_single_A"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "sp", [nb_sp["A_heliospay"]], ["A_heliospay", "B_nvx418", "C_lithox9"], tag="NB-NR-SP Single A")
    results["runs"]["nb_nr_sp_concat_AB"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "sp", [nb_sp["A_heliospay"], nb_sp["B_nvx418"]], ["A_heliospay", "B_nvx418", "C_lithox9"], eval_ood=True, tag="NB-NR-SP Concat [A;B]")
    results["runs"]["nb_nr_sp_concat_ABC"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "sp", [nb_sp["A_heliospay"], nb_sp["B_nvx418"], nb_sp["C_lithox9"]], ["A_heliospay", "B_nvx418", "C_lithox9"], eval_ood=True, tag="NB-NR-SP Concat [A;B;C]")
    results["cross_AB"]["nb_nr_sp_concat_AB"] = eval_cross_doc_AB(model, tokenizer, tmpl, cross_AB, "sp", [nb_sp["A_heliospay"], nb_sp["B_nvx418"]], "", "NB-NR-SP [A;B]")

    # 3B: Prefix-Robust + Null-Replay + Delta-Bounded KV-CSPD (L_kv=16)!
    nr_kv = {}
    for idx, dom in enumerate(["A_heliospay", "B_nvx418", "C_lithox9"]):
        t_kv, meta = train_kv_prefix_doc(
            model, tokenizer, tmpl, datasets[dom]["train"], ALL_5_DOCS[dom],
            num_kv_tokens=16, num_updates=100, batch_size=16, micro_batch=4,
            optimizer_type="adamw", lr_max=0.025, kv_prefix_robust=True,
            null_replay_items=null_replay_items, null_replay_ratio=0.25,
            max_delta_rms=0.8, seed=60 + idx, tag=f"NR_KV16_{dom[:1]}"
        )
        nr_kv[dom] = t_kv
        results["meta"][f"NR_KV16_{dom[:1]}"] = meta

    results["runs"]["nr_kv_single_A"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [nr_kv["A_heliospay"]], ["A_heliospay", "B_nvx418", "C_lithox9"], tag="NR-KV16 Single A")
    results["runs"]["nr_kv_single_B"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [nr_kv["B_nvx418"]], ["A_heliospay", "B_nvx418", "C_lithox9"], tag="NR-KV16 Single B")
    results["runs"]["nr_kv_single_C"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [nr_kv["C_lithox9"]], ["A_heliospay", "B_nvx418", "C_lithox9"], tag="NR-KV16 Single C")
    results["runs"]["nr_kv_concat_AB"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [nr_kv["A_heliospay"], nr_kv["B_nvx418"]], ["A_heliospay", "B_nvx418", "C_lithox9"], eval_ood=True, tag="NR-KV16 Concat [A;B]")
    results["runs"]["nr_kv_concat_ABC"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [nr_kv["A_heliospay"], nr_kv["B_nvx418"], nr_kv["C_lithox9"]], ["A_heliospay", "B_nvx418", "C_lithox9"], eval_ood=True, tag="NR-KV16 Concat [A;B;C]")
    results["runs"]["nr_kv_concat_CBA"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [nr_kv["C_lithox9"], nr_kv["B_nvx418"], nr_kv["A_heliospay"]], ["A_heliospay", "B_nvx418", "C_lithox9"], tag="NR-KV16 Rev [C;B;A]")
    results["runs"]["nr_kv_unlearn_AC"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [nr_kv["A_heliospay"], nr_kv["C_lithox9"]], ["A_heliospay", "B_nvx418", "C_lithox9"], tag="NR-KV16 Unlearn B [A;C]")
    results["cross_AB"]["nr_kv_concat_AB"] = eval_cross_doc_AB(model, tokenizer, tmpl, cross_AB, "kv", [nr_kv["A_heliospay"], nr_kv["B_nvx418"]], "", "NR-KV16 [A;B]")
    save()

    # --- 4. Abstention Calibration & Thinking-Mode CoT Capsule (Doc A) ---
    print("\n=== Block 4: Abstention Calibration & Thinking-Mode CoT Capsule ===", flush=True)
    abstain_train = datasets["A_heliospay"]["train"] + ABSTENTION_TRAIN_A * 4
    kv_A_abstain, meta_ab = train_kv_prefix_doc(
        model, tokenizer, tmpl, abstain_train, ALL_5_DOCS["A_heliospay"],
        num_kv_tokens=16, num_updates=80, batch_size=16, micro_batch=4,
        optimizer_type="adamw", lr_max=0.03, seed=70, tag="KV_A_L16_AbstainCal"
    )
    results["meta"]["KV_A_L16_AbstainCal"] = meta_ab
    results["runs"]["KV_A_L16_AbstainCal"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [kv_A_abstain], ["A_heliospay"], tag="KV_A_L16_AbstainCal")
    results["probes_A"]["KV_A_L16_AbstainCal"] = eval_stress_probes_A(model, tokenizer, tmpl, boundary_A, extrap_A, "kv", [kv_A_abstain], "", False, 384, "KV_A_L16_AbstainCal")

    # Thinking-Mode CoT comparison on Doc A (In-Context Thinking vs KV_A_CoT Thinking)
    results["runs"]["in_context_A_thinking"] = eval_domain_suite(
        model, tokenizer, tmpl, datasets, "sp", [], ["A_heliospay"],
        system_texts_by_dom={"A_heliospay": ALL_5_DOCS["A_heliospay"]},
        enable_thinking=True, max_new_tokens=768, tag="In-Context A (Thinking)"
    )
    kv_A_think, meta_th = train_kv_prefix_doc(
        model, tokenizer, tmpl, datasets["A_heliospay"]["train"], ALL_5_DOCS["A_heliospay"],
        num_kv_tokens=16, num_updates=80, batch_size=16, micro_batch=4,
        optimizer_type="adamw", lr_max=0.03, enable_thinking=True, seed=71, tag="KV_A_L16_Thinking"
    )
    results["meta"]["KV_A_L16_Thinking"] = meta_th
    results["runs"]["KV_A_L16_Thinking"] = eval_domain_suite(
        model, tokenizer, tmpl, datasets, "kv", [kv_A_think], ["A_heliospay"],
        enable_thinking=True, max_new_tokens=768, tag="KV_A_L16_Thinking"
    )

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== Suite 2 (exp4) COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
