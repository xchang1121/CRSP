"""
Paper Reference: Section 4.2 (Input-Space CSPD Capacity & Concatenation Limit on Private Docs).
Stage B Experiment: Continual Learning of Vertical-Domain Documents via
Context-Distilled Composable Soft-Prompt Capsules (CSPD-Doc).

Evaluates on 3 completely synthetic, zero-pretraining-leakage enterprise documents:
  - Doc A: HeliosPay v4.2 FinTech Compliance & Settlement Spec
  - Doc B: NVX-418 Phase IIb Clinical Trial SOP
  - Doc C: Litho-X9 EUV Lithography Yield Diagnostics Manual

Key experiments:
  1. Zero-leakage Base vs In-Context Full Document upper bound
  2. Continual Learning Trajectory: Base -> P_A -> [P_A; P_B] -> [P_A; P_B; P_C] & Reverse [P_C; P_B; P_A]
  3. Exact Unlearning: Drop P_B -> [P_A; P_C]
  4. Retrieved Top-1 Capsule vs Full Concatenation (quantifying cross-doc interference)
  5. Prefix-Robust Training ablation for multi-document composition
  6. Objective Ablation on Doc A: Sequence-level NLL only vs Context Distillation (NLL + In-Context Teacher Forward KL)
  7. Capacity Boundary Sweep on Doc A: L in {32, 128, 256} (Procedural Rules vs Exact Facts)
  8. Sequential LoRA Baseline (A -> B -> C): measuring catastrophic forgetting vs CSPD
  9. Out-of-Domain (OOD) general reasoning/knowledge preservation
"""

import gc
import json
import math
import os
import random
import re
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
from peft import LoraConfig, get_peft_model

from cspd_core import (
    ChatTemplateTokens,
    MuonSoftPrompt,
    build_soft_prompt_inputs,
    get_qwen_template_tokens,
    init_soft_prompt_from_vocab,
    length_normalized_nll_loss,
    pad_and_stack_train_batch,
    sample_random_prefix,
    should_add_prefix,
)
from domain_docs_data import DOCS, build_domain_datasets, grade_domain_response


MODEL_ID = "Jackwang111/M2RL-SFT"
OUT_DIR = "/content/cspd/exp2_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Out-of-Domain General Knowledge / Reasoning Preservation Mini-Benchmark (24 Qs)
# ---------------------------------------------------------------------------
OOD_QUESTIONS = [
    ("What is the chemical symbol for gold?", ["au"]),
    ("What is 14 * 17? Give the exact number.", ["238"]),
    ("Which planet in our solar system has the most prominent ring system?", ["saturn"]),
    ("Who wrote the play 'Hamlet'?", ["shakespeare"]),
    ("What is the capital city of Australia?", ["canberra"]),
    ("Compute the derivative of x^3 + 2x with respect to x.", ["3x^2 + 2", "3x^2+2"]),
    ("In computer science, what is the worst-case time complexity of binary search on a sorted array of size n?", ["o(log n)", "log n", "log(n)"]),
    ("What is the boiling point of water at standard atmospheric pressure in degrees Celsius?", ["100"]),
    ("Which gas makes up approximately 78% of Earth's atmosphere?", ["nitrogen"]),
    ("What is the square root of 1,764?", ["42"]),
    ("In what year did humans first land on the Moon (Apollo 11)?", ["1969"]),
    ("What organelle is known as the powerhouse of the eukaryotic cell?", ["mitochondri"]),
    ("If a fair six-sided die is rolled once, what is the probability of rolling a prime number?", ["1/2", "0.5", "50%"]),
    ("What is the SI unit of electrical resistance?", ["ohm"]),
    ("Which country is home to the ancient city of Machu Picchu?", ["peru"]),
    ("What is 2^10?", ["1024", "1,024"]),
    ("What programming language was created by Guido van Rossum?", ["python"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("How many sides does a regular dodecagon have?", ["12"]),
    ("What is the pH of pure neutral water at 25 degrees Celsius?", ["7"]),
    ("Which scientist formulated the three laws of motion?", ["newton"]),
    ("What is the sum of interior angles of a pentagon in degrees?", ["540"]),
    ("What HTTP status code means 'Not Found'?", ["404"]),
    ("Which blood type is commonly known as the universal red blood cell donor?", ["o negative", "o-"]),
]


def grade_ood(resp: str, kws: List[str]) -> bool:
    vis = resp.split("</think>")[-1].strip() if "</think>" in resp else resp.strip()
    low = vis.lower()
    return any(k in low for k in kws)


def make_nothink_template(tokenizer) -> ChatTemplateTokens:
    base = get_qwen_template_tokens(tokenizer)
    no_think = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    return ChatTemplateTokens(
        sys_open=base.sys_open,
        sys_close_user_open=base.sys_close_user_open,
        user_close_asst_open=base.user_close_asst_open + no_think,
        im_end=base.im_end,
    )


@torch.no_grad()
def batch_generate_nothink(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    questions: List[str],
    soft_prompts: List[torch.Tensor],
    system_text: str = "",
    batch_size: int = 24,
    max_new_tokens: int = 220,
) -> List[str]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = embed_layer(torch.tensor([pad_id], device=device)).to(dtype)
    s_ids = tokenizer.encode(system_text, add_special_tokens=False) if system_text else None

    outputs_all: List[str] = []
    for start in range(0, len(questions), batch_size):
        batch_q = questions[start : start + batch_size]
        seq_embs = []
        for q in batch_q:
            q_ids = tokenizer.encode(q, add_special_tokens=False)
            emb, _, _ = build_soft_prompt_inputs(
                embed_layer=embed_layer,
                tmpl=tmpl,
                soft_prompts=soft_prompts,
                question_ids=q_ids,
                response_ids=None,
                system_text_ids=s_ids,
                prefix_F_ids=None,
                device=device,
                model_dtype=dtype,
            )
            seq_embs.append(emb.squeeze(0))

        max_len = max(e.size(0) for e in seq_embs)
        bs = len(seq_embs)
        d = seq_embs[0].size(1)
        padded = pad_vec.expand(bs, max_len, d).clone()
        attn_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for i, e in enumerate(seq_embs):
            L = e.size(0)
            padded[i, max_len - L :, :] = e
            attn_mask[i, max_len - L :] = 1

        out_ids = model.generate(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        for row in out_ids:
            outputs_all.append(tokenizer.decode(row, skip_special_tokens=True))
    return outputs_all


def eval_all_domains(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    datasets: Dict[str, Dict[str, List[Dict[str, Any]]]],
    soft_prompts: List[torch.Tensor],
    system_texts_by_domain: Optional[Dict[str, str]] = None,
    eval_ood: bool = True,
    tag: str = "",
) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    rule_accs, fact_accs = [], []

    for dom in ["A_heliospay", "B_nvx418", "C_lithox9"]:
        sys_txt = system_texts_by_domain.get(dom, "") if system_texts_by_domain else ""
        # Rules
        r_items = datasets[dom]["test_rules"]
        r_preds = batch_generate_nothink(
            model, tokenizer, tmpl, [it["question"] for it in r_items], soft_prompts, system_text=sys_txt, max_new_tokens=384
        )
        r_ok = [grade_domain_response(p, it) for p, it in zip(r_preds, r_items)]
        r_acc = 100.0 * sum(r_ok) / len(r_ok)

        # Facts
        f_items = datasets[dom]["test_facts"]
        f_preds = batch_generate_nothink(
            model, tokenizer, tmpl, [it["question"] for it in f_items], soft_prompts, system_text=sys_txt, max_new_tokens=120
        )
        f_ok = [grade_domain_response(p, it) for p, it in zip(f_preds, f_items)]
        f_acc = 100.0 * sum(f_ok) / len(f_ok)

        comb = 100.0 * (sum(r_ok) + sum(f_ok)) / (len(r_ok) + len(f_ok))
        res[dom] = {
            "rules": round(r_acc, 2),
            "facts": round(f_acc, 2),
            "overall": round(comb, 2),
            "sample_rule_pred": r_preds[0][:240],
            "sample_fact_pred": f_preds[0][:160],
        }
        rule_accs.append(r_acc)
        fact_accs.append(f_acc)

    res["mean_rules"] = round(sum(rule_accs) / 3.0, 2)
    res["mean_facts"] = round(sum(fact_accs) / 3.0, 2)
    res["mean_all"] = round((res["mean_rules"] + res["mean_facts"]) / 2.0, 2)

    if eval_ood:
        ood_preds = batch_generate_nothink(
            model, tokenizer, tmpl, [q for q, _ in OOD_QUESTIONS], soft_prompts, system_text="", max_new_tokens=96
        )
        ood_ok = [grade_ood(p, kws) for p, (_, kws) in zip(ood_preds, OOD_QUESTIONS)]
        res["ood"] = round(100.0 * sum(ood_ok) / len(ood_ok), 2)

    print(
        f"[{tag}] A(R={res['A_heliospay']['rules']:.1f}, F={res['A_heliospay']['facts']:.1f}) | "
        f"B(R={res['B_nvx418']['rules']:.1f}, F={res['B_nvx418']['facts']:.1f}) | "
        f"C(R={res['C_lithox9']['rules']:.1f}, F={res['C_lithox9']['facts']:.1f}) | "
        f"Mean(R={res['mean_rules']:.1f}, F={res['mean_facts']:.1f})"
        + (f" | OOD={res['ood']:.1f}" if eval_ood else ""),
        flush=True,
    )
    return res


def distill_doc_capsule(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    domain_name: str,
    doc_text: str,
    train_items: List[Dict[str, Any]],
    num_vectors: int = 128,
    num_updates: int = 80,
    batch_size: int = 16,
    micro_batch: int = 8,
    lr_max: float = 0.03,
    lr_min: float = 0.003,
    kl_weight: float = 0.5,
    prefix_robust: bool = True,
    seed: int = 42,
    tag: str = "",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Distill a single vertical-domain document C_j into an independent soft prompt capsule P_j
    using cached Top-64 In-Context Teacher Forward KL + Length-Normalized NLL + Muon + Prefix-Robust Training.
    """
    t0 = time.time()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = embed_layer(torch.tensor([pad_id], device=device)).detach().to(dtype)

    doc_ids = tokenizer.encode(doc_text, add_special_tokens=False)
    tokenized = []
    prompt_stream: List[int] = []
    for it in train_items:
        q_ids = tokenizer.encode(it["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(it["gold_answer"], add_special_tokens=False) + tmpl.im_end
        tokenized.append((it["question"], q_ids, r_ids))
        prompt_stream.extend(tmpl.sys_close_user_open + q_ids + tmpl.user_close_asst_open)

    # Pre-cache Top-64 In-Context Teacher distribution once per unique question (~2s total!)
    teacher_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    if kl_weight > 0.0:
        unique_examples = {}
        for q_text, q_ids, r_ids in tokenized:
            if q_text not in unique_examples:
                unique_examples[q_text] = (q_ids, r_ids)
        u_keys = list(unique_examples.keys())
        with torch.no_grad():
            for st in range(0, len(u_keys), 8):
                chunk = u_keys[st : st + 8]
                embs_l, lbls_l = [], []
                for k in chunk:
                    qi, ri = unique_examples[k]
                    te, tl, _ = build_soft_prompt_inputs(
                        embed_layer=embed_layer, tmpl=tmpl, soft_prompts=[],
                        question_ids=qi, response_ids=ri, system_text_ids=doc_ids,
                        device=device, model_dtype=dtype,
                    )
                    embs_l.append(te)
                    lbls_l.append(tl)
                b_emb, b_mask, b_lbl = pad_and_stack_train_batch(embs_l, lbls_l, pad_vec)
                t_out = model(inputs_embeds=b_emb, attention_mask=b_mask)
                shift_logits = t_out.logits[:, :-1, :]
                shift_labels = b_lbl[:, 1:]
                for i, k in enumerate(chunk):
                    rmask = shift_labels[i] != -100
                    resp_logits = shift_logits[i][rmask].float()
                    top_vals, top_idx = torch.topk(resp_logits, k=64, dim=-1)
                    top_logp = F.log_softmax(top_vals, dim=-1)
                    teacher_cache[k] = (top_idx.cpu(), top_logp.cpu())
                del b_emb, b_mask, b_lbl, t_out

    sp = init_soft_prompt_from_vocab(embed_layer.weight, num_vectors=num_vectors, seed=seed)
    init_sp = sp.detach().clone()
    optimizer = MuonSoftPrompt([sp], lr=lr_max, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)

    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = random.Random(seed)
    loss_hist = []
    num_mb = max(1, batch_size // micro_batch)

    for step in range(num_updates):
        progress = step / max(1, num_updates - 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        batch = [tokenized[rng.randrange(len(tokenized))] for _ in range(batch_size)]
        step_nll = 0.0
        step_kl = 0.0

        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            embs_l, lbls_l, q_keys = [], [], []
            for q_text, q_ids, r_ids in sub:
                prefix_F = None
                if prefix_robust and should_add_prefix(q_text):
                    prefix_F = sample_random_prefix(prompt_stream, rng, min_len=128, max_len=256)
                se, sl, _ = build_soft_prompt_inputs(
                    embed_layer=embed_layer, tmpl=tmpl, soft_prompts=[sp],
                    question_ids=q_ids, response_ids=r_ids,
                    system_text_ids=None, prefix_F_ids=prefix_F,
                    device=device, model_dtype=dtype,
                )
                embs_l.append(se)
                lbls_l.append(sl)
                q_keys.append(q_text)

            b_emb, b_mask, b_lbl = pad_and_stack_train_batch(embs_l, lbls_l, pad_vec)
            s_out = model(inputs_embeds=b_emb, attention_mask=b_mask)
            nll = length_normalized_nll_loss(s_out.logits, b_lbl)

            if kl_weight > 0.0:
                shift_logits = s_out.logits[:, :-1, :]
                shift_labels = b_lbl[:, 1:]
                kl_terms = []
                for i, qk in enumerate(q_keys):
                    rmask = shift_labels[i] != -100
                    s_resp = shift_logits[i][rmask].float()
                    t_idx, t_lp = teacher_cache[qk]
                    t_idx = t_idx.to(device)
                    t_lp = t_lp.to(device)
                    n = min(s_resp.size(0), t_idx.size(0))
                    s_full_lp = F.log_softmax(s_resp[:n], dim=-1)
                    s_sel_lp = s_full_lp.gather(-1, t_idx[:n])
                    kl_i = (t_lp[:n].exp() * (t_lp[:n] - s_sel_lp)).sum(dim=-1).mean()
                    kl_terms.append(kl_i)
                kl = torch.stack(kl_terms).mean()
                loss = (1.0 - kl_weight) * nll + kl_weight * kl
                step_kl += kl.item()
            else:
                loss = nll

            step_nll += nll.item()
            (loss / num_mb).backward()
            del b_emb, b_mask, b_lbl, s_out

        torch.nn.utils.clip_grad_norm_([sp], max_norm=1.0)
        optimizer.step()

        avg_nll = step_nll / num_mb
        avg_kl = step_kl / num_mb
        loss_hist.append({"step": step + 1, "nll": round(avg_nll, 4), "kl": round(avg_kl, 4), "lr": round(lr, 5)})
        if (step + 1) % 20 == 0 or step == 0 or step == num_updates - 1:
            print(
                f"  [{tag or domain_name}] step {step+1:3d}/{num_updates} | "
                f"nll={avg_nll:.4f} | kl={avg_kl:.4f} | lr={lr:.5f} | elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True

    with torch.no_grad():
        final_sp = sp.detach().clone()
        cos_sim = F.cosine_similarity(final_sp, init_sp, dim=-1)
        delta = (final_sp - init_sp).float()
        sv = torch.linalg.svdvals(delta)
        p_sv = (sv ** 2) / (sv ** 2).sum().clamp_min(1e-12)
        eff_rank = float(torch.exp(-(p_sv * (p_sv.clamp_min(1e-12)).log()).sum()).item())

    meta = {
        "domain": domain_name,
        "num_vectors": num_vectors,
        "num_updates": num_updates,
        "kl_weight": kl_weight,
        "prefix_robust": prefix_robust,
        "train_seconds": round(time.time() - t0, 1),
        "final_nll": loss_hist[-1]["nll"],
        "final_kl": loss_hist[-1]["kl"],
        "median_cos_to_init": round(float(cos_sim.median().item()), 4),
        "effective_rank_delta": round(eff_rank, 2),
    }
    return final_sp, meta


def train_lora_on_domain(
    peft_model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    train_items: List[Dict[str, Any]],
    num_updates: int = 80,
    batch_size: int = 16,
    micro_batch: int = 8,
    lr_max: float = 2e-4,
    seed: int = 42,
    tag: str = "",
):
    """Train a LoRA adapter on one domain sequentially (same step budget as soft prompts)."""
    t0 = time.time()
    device = next(peft_model.parameters()).device
    dtype = next(peft_model.parameters()).dtype
    embed_layer = peft_model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = embed_layer(torch.tensor([pad_id], device=device)).detach().to(dtype)

    tokenized = []
    for it in train_items:
        q_ids = tokenizer.encode(it["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(it["gold_answer"], add_special_tokens=False) + tmpl.im_end
        tokenized.append((q_ids, r_ids))

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr_max, weight_decay=0.0)
    peft_model.config.use_cache = False
    peft_model.train()
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = random.Random(seed)
    num_mb = max(1, batch_size // micro_batch)

    for step in range(num_updates):
        progress = step / max(1, num_updates - 1)
        lr = (lr_max * 0.1) + 0.5 * (lr_max * 0.9) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad()
        batch = [tokenized[rng.randrange(len(tokenized))] for _ in range(batch_size)]
        step_loss = 0.0
        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            embs_l, lbls_l = [], []
            for q_ids, r_ids in sub:
                emb, labels, _ = build_soft_prompt_inputs(
                    embed_layer=embed_layer, tmpl=tmpl, soft_prompts=[],
                    question_ids=q_ids, response_ids=r_ids,
                    device=device, model_dtype=dtype,
                )
                embs_l.append(emb)
                lbls_l.append(labels)
            b_emb, b_mask, b_lbl = pad_and_stack_train_batch(embs_l, lbls_l, pad_vec)
            b_emb = b_emb.requires_grad_(True)
            out = peft_model(inputs_embeds=b_emb, attention_mask=b_mask)
            loss = length_normalized_nll_loss(out.logits, b_lbl)
            step_loss += loss.item()
            (loss / num_mb).backward()
            del b_emb, b_mask, b_lbl, out
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if (step + 1) % 20 == 0 or step == num_updates - 1:
            print(f"  [LoRA {tag}] step {step+1:3d}/{num_updates} | nll={step_loss/num_mb:.4f} | elapsed={time.time()-t0:.1f}s", flush=True)

    peft_model.gradient_checkpointing_disable()
    peft_model.eval()
    peft_model.config.use_cache = True


def main():
    t_start = time.time()
    print(f"=== Stage B: Vertical-Domain Document Continual Learning (CSPD-Doc) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tmpl = make_nothink_template(tokenizer)
    datasets = build_domain_datasets(seed=42)

    results: Dict[str, Any] = {"model": MODEL_ID, "runs": {}, "capsule_meta": {}}

    def save_results():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # 0. Baselines: Base (no doc) & In-Context Full Document (Upper Bound)
    # ------------------------------------------------------------------
    print("\n--- Step 0: Baselines (Base Zero-Shot & In-Context Upper Bound) ---", flush=True)
    results["runs"]["base_no_doc"] = eval_all_domains(model, tokenizer, tmpl, datasets, [], eval_ood=True, tag="Base (No Doc)")
    results["runs"]["in_context_doc_upper_bound"] = eval_all_domains(
        model, tokenizer, tmpl, datasets, [], system_texts_by_domain=DOCS, eval_ood=False, tag="In-Context Full Doc"
    )
    save_results()

    # ------------------------------------------------------------------
    # 1. Independently Distill 3 Domain Document Capsules (L=128, Muon+PR+CtxKL)
    # ------------------------------------------------------------------
    print("\n--- Step 1: Independent Context-Distilled Soft-Prompt Capsules (L=128, Muon+PR+CtxKL) ---", flush=True)
    capsules: Dict[str, torch.Tensor] = {}
    for idx, dom in enumerate(["A_heliospay", "B_nvx418", "C_lithox9"]):
        sp, meta = distill_doc_capsule(
            model, tokenizer, tmpl, dom, DOCS[dom], datasets[dom]["train"],
            num_vectors=128, num_updates=80, batch_size=16, micro_batch=4,
            lr_max=0.03, lr_min=0.003, kl_weight=0.5, prefix_robust=True,
            seed=42 + idx, tag=f"Capsule_{dom}_L128_CtxKL"
        )
        capsules[dom] = sp
        results["capsule_meta"][dom] = meta
        torch.save(sp.cpu(), os.path.join(OUT_DIR, f"sp_{dom}_L128_ctxkl.pt"))
    save_results()

    # ------------------------------------------------------------------
    # 2. Continual Learning Trajectory + Order Invariance + Exact Unlearning + Routing
    # ------------------------------------------------------------------
    print("\n--- Step 2: Continual Addition, Exact Unlearning & Retrieved Single Capsule ---", flush=True)
    pA, pB, pC = capsules["A_heliospay"], capsules["B_nvx418"], capsules["C_lithox9"]

    results["runs"]["stage1_PA"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pA], eval_ood=True, tag="Stage1: [P_A]")
    results["runs"]["stage2_PA_PB"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pA, pB], eval_ood=True, tag="Stage2: [P_A; P_B]")
    results["runs"]["stage3_PA_PB_PC"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pA, pB, pC], eval_ood=True, tag="Stage3: [P_A; P_B; P_C]")
    results["runs"]["stage3_rev_PC_PB_PA"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pC, pB, pA], eval_ood=False, tag="Stage3-Rev: [P_C; P_B; P_A]")
    results["runs"]["stage4_unlearn_drop_PB"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pA, pC], eval_ood=False, tag="Unlearn(-B): [P_A; P_C]")

    # Single-domain capsule (oracle / RAG-over-Soft-Prompts top-1 retrieval)
    sB = eval_all_domains(model, tokenizer, tmpl, datasets, [pB], eval_ood=False, tag="Single: [P_B]")
    sC = eval_all_domains(model, tokenizer, tmpl, datasets, [pC], eval_ood=False, tag="Single: [P_C]")
    results["runs"]["single_PB"] = sB
    results["runs"]["single_PC"] = sC
    results["runs"]["retrieved_top1_summary"] = {
        "A_heliospay": results["runs"]["stage1_PA"]["A_heliospay"],
        "B_nvx418": sB["B_nvx418"],
        "C_lithox9": sC["C_lithox9"],
        "mean_rules": round((results["runs"]["stage1_PA"]["A_heliospay"]["rules"] + sB["B_nvx418"]["rules"] + sC["C_lithox9"]["rules"]) / 3.0, 2),
        "mean_facts": round((results["runs"]["stage1_PA"]["A_heliospay"]["facts"] + sB["B_nvx418"]["facts"] + sC["C_lithox9"]["facts"]) / 3.0, 2),
    }
    save_results()

    # ------------------------------------------------------------------
    # 3. Ablation: Prefix-Robust Training (No Prefix F) for 3-Doc Composition
    # ------------------------------------------------------------------
    print("\n--- Step 3: Ablation — Without Prefix-Robust Training ---", flush=True)
    caps_nopr = {}
    for idx, dom in enumerate(["A_heliospay", "B_nvx418", "C_lithox9"]):
        sp, meta = distill_doc_capsule(
            model, tokenizer, tmpl, dom, DOCS[dom], datasets[dom]["train"],
            num_vectors=128, num_updates=80, batch_size=16, micro_batch=4,
            lr_max=0.03, lr_min=0.003, kl_weight=0.5, prefix_robust=False,
            seed=42 + idx, tag=f"NoPR_{dom}"
        )
        caps_nopr[dom] = sp
        results["capsule_meta"][f"{dom}_nopr"] = meta
    results["runs"]["ablation_nopr_concat_ABC"] = eval_all_domains(
        model, tokenizer, tmpl, datasets,
        [caps_nopr["A_heliospay"], caps_nopr["B_nvx418"], caps_nopr["C_lithox9"]],
        eval_ood=False, tag="NoPR Concat [A;B;C]"
    )
    save_results()

    # ------------------------------------------------------------------
    # 4. Ablations on Doc A: (a) Pure NLL vs Context KL, (b) Capacity L in {32, 128, 256}
    # ------------------------------------------------------------------
    print("\n--- Step 4: Doc A Ablations (Pure NLL vs Context KL; Capacity L=32/128/256) ---", flush=True)
    sp_a_nll, meta_nll = distill_doc_capsule(
        model, tokenizer, tmpl, "A_heliospay", DOCS["A_heliospay"], datasets["A_heliospay"]["train"],
        num_vectors=128, num_updates=80, batch_size=16, micro_batch=4,
        lr_max=0.03, lr_min=0.003, kl_weight=0.0, prefix_robust=True,
        seed=42, tag="DocA_L128_NLL_only"
    )
    results["capsule_meta"]["A_nll_only"] = meta_nll
    results["runs"]["ablation_A_L128_nll_only"] = eval_all_domains(model, tokenizer, tmpl, datasets, [sp_a_nll], eval_ood=False, tag="DocA L128 NLL-only")

    for L_val in [32, 256]:
        sp_L, meta_L = distill_doc_capsule(
            model, tokenizer, tmpl, "A_heliospay", DOCS["A_heliospay"], datasets["A_heliospay"]["train"],
            num_vectors=L_val, num_updates=80, batch_size=16, micro_batch=4,
            lr_max=0.03, lr_min=0.003, kl_weight=0.5, prefix_robust=True,
            seed=42, tag=f"DocA_L{L_val}_CtxKL"
        )
        results["capsule_meta"][f"A_L{L_val}"] = meta_L
        results["runs"][f"ablation_A_L{L_val}_ctxkl"] = eval_all_domains(model, tokenizer, tmpl, datasets, [sp_L], eval_ood=False, tag=f"DocA L={L_val} CtxKL")
    save_results()

    # ------------------------------------------------------------------
    # 5. Baseline: Sequential LoRA Continual Learning (A -> B -> C)
    # ------------------------------------------------------------------
    print("\n--- Step 5: Sequential LoRA Baseline (Catastrophic Forgetting Comparison) ---", flush=True)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()

    for stage_idx, (dom, stage_name) in enumerate([
        ("A_heliospay", "lora_seq_stage1_A"),
        ("B_nvx418", "lora_seq_stage2_AB"),
        ("C_lithox9", "lora_seq_stage3_ABC"),
    ]):
        train_lora_on_domain(
            peft_model, tokenizer, tmpl, datasets[dom]["train"],
            num_updates=80, batch_size=16, micro_batch=4, lr_max=2e-4, seed=42 + stage_idx, tag=stage_name
        )
        results["runs"][stage_name] = eval_all_domains(
            peft_model, tokenizer, tmpl, datasets, [], eval_ood=True, tag=f"SeqLoRA {stage_name}"
        )
        save_results()

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save_results()
    print(f"\n=== Stage B COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
