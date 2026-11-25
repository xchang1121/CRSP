"""
Paper Reference: Section 4.2, Fig. 7–8 (Private-Doc Length Scaling & Sequential LoRA Baseline).
Stage B-v2: Vertical-Domain Document Continual Learning with Verified Sequence-Level NLL
(CSPD Eq. 1) + Bidirectional Prefix/Suffix-Robust Training.

Building on Stage B-v1's discovery that noisy weak In-Context Teacher KL (beta=0.5)
stalled NLL at 0.23 while pure verified Sequence-Level NLL reached 0.0163 and BEAT
In-Context Full Document on unseen procedural rules (50.0% vs 37.5%):

Tests:
  1. Pure Verified NLL + Bidirectional Context-Robust Training ([F_pre][P_j][F_post]) for P_A, P_B, P_C (L=128, 120 updates)
  2. Full Continual Learning Trajectory: Base -> P_A -> [P_A; P_B] -> [P_A; P_B; P_C] & Reverse [P_C; P_B; P_A]
  3. Exact Unlearning: Drop P_B -> [P_A; P_C]
  4. Retrieved Top-1 Capsule (P_A, P_B, P_C) vs Full 3-Doc Concatenation [P_A; P_B; P_C]
  5. Capacity Sweep under Pure NLL on Doc A: L in {32, 128, 256} (Rules vs Exact Facts!)
  6. OOD preservation across continual addition
"""

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

from cspd_core import (
    ChatTemplateTokens,
    MuonSoftPrompt,
    build_soft_prompt_inputs,
    init_soft_prompt_from_vocab,
    length_normalized_nll_loss,
    pad_and_stack_train_batch,
    sample_random_prefix,
)
from domain_docs_data import DOCS, build_domain_datasets
from exp2_domain_doc_cl import (
    MODEL_ID,
    eval_all_domains,
    make_nothink_template,
)

OUT_DIR = "/content/cspd/exp2b_results"
os.makedirs(OUT_DIR, exist_ok=True)


def build_bidir_inputs(
    embed_layer,
    tmpl: ChatTemplateTokens,
    sp: torch.Tensor,
    question_ids: List[int],
    response_ids: List[int],
    prefix_pre_ids: Optional[List[int]],
    prefix_post_ids: Optional[List[int]],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    <|im_start|>system\n [F_pre] [P_j] [F_post] <|im_end|>\n<|im_start|>user\n [q] <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n [y]
    Trains P_j to be robust whether other capsules appear BEFORE it, AFTER it, or BOTH!
    """
    parts = []
    cursor = 0

    def add_ids(ids: Optional[List[int]]):
        nonlocal cursor
        if not ids:
            return
        t = torch.tensor(ids, dtype=torch.long, device=device)
        parts.append(embed_layer(t).to(model_dtype))
        cursor += len(ids)

    add_ids(tmpl.sys_open)
    add_ids(prefix_pre_ids)
    parts.append(sp.to(device=device, dtype=model_dtype))
    cursor += sp.size(0)
    add_ids(prefix_post_ids)
    add_ids(tmpl.sys_close_user_open)
    add_ids(question_ids)
    add_ids(tmpl.user_close_asst_open)

    prompt_len = cursor
    add_ids(response_ids)
    labels = torch.full((1, cursor), -100, dtype=torch.long, device=device)
    labels[0, prompt_len:cursor] = torch.tensor(response_ids, dtype=torch.long, device=device)
    return torch.cat(parts, dim=0).unsqueeze(0), labels


def distill_doc_nll_bidir(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    domain_name: str,
    train_items: List[Dict[str, Any]],
    num_vectors: int = 128,
    num_updates: int = 120,
    batch_size: int = 16,
    micro_batch: int = 4,
    lr_max: float = 0.03,
    lr_min: float = 0.003,
    bidir_robust: bool = True,
    seed: int = 42,
    tag: str = "",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = embed_layer(torch.tensor([pad_id], device=device)).detach().to(dtype)

    tokenized = []
    prompt_stream: List[int] = []
    for it in train_items:
        q_ids = tokenizer.encode(it["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(it["gold_answer"], add_special_tokens=False) + tmpl.im_end
        tokenized.append((q_ids, r_ids))
        prompt_stream.extend(tmpl.sys_close_user_open + q_ids + tmpl.user_close_asst_open)

    sp = init_soft_prompt_from_vocab(embed_layer.weight, num_vectors=num_vectors, seed=seed)
    init_sp = sp.detach().clone()
    optimizer = MuonSoftPrompt([sp], lr=lr_max, momentum=0.95, nesterov=True, ns_steps=5, rms_scale=0.2)

    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = random.Random(seed)
    num_mb = max(1, batch_size // micro_batch)
    losses = []

    for step in range(num_updates):
        progress = step / max(1, num_updates - 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        batch = [tokenized[rng.randrange(len(tokenized))] for _ in range(batch_size)]
        step_nll = 0.0

        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            embs_l, lbls_l = [], []
            for q_ids, r_ids in sub:
                f_pre, f_post = None, None
                if bidir_robust:
                    u = rng.random()
                    # 30% clean, 25% pre-only, 25% post-only, 20% both pre & post (128-256 tok each)
                    if u < 0.25:
                        f_pre = sample_random_prefix(prompt_stream, rng, 128, 256)
                    elif u < 0.50:
                        f_post = sample_random_prefix(prompt_stream, rng, 128, 256)
                    elif u < 0.70:
                        f_pre = sample_random_prefix(prompt_stream, rng, 128, 256)
                        f_post = sample_random_prefix(prompt_stream, rng, 128, 256)
                se, sl = build_bidir_inputs(
                    embed_layer, tmpl, sp, q_ids, r_ids, f_pre, f_post, device, dtype
                )
                embs_l.append(se)
                lbls_l.append(sl)

            b_emb, b_mask, b_lbl = pad_and_stack_train_batch(embs_l, lbls_l, pad_vec)
            out = model(inputs_embeds=b_emb, attention_mask=b_mask)
            nll = length_normalized_nll_loss(out.logits, b_lbl)
            step_nll += nll.item()
            (nll / num_mb).backward()
            del b_emb, b_mask, b_lbl, out

        torch.nn.utils.clip_grad_norm_([sp], max_norm=1.0)
        optimizer.step()
        avg_nll = step_nll / num_mb
        losses.append(round(avg_nll, 4))
        if (step + 1) % 30 == 0 or step == 0 or step == num_updates - 1:
            print(
                f"  [{tag or domain_name}] step {step+1:3d}/{num_updates} | "
                f"nll={avg_nll:.4f} | lr={lr:.5f} | elapsed={time.time()-t0:.1f}s",
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
        eff_rank = float(torch.exp(-(p_sv * p_sv.clamp_min(1e-12).log()).sum()).item())

    meta = {
        "domain": domain_name,
        "num_vectors": num_vectors,
        "num_updates": num_updates,
        "bidir_robust": bidir_robust,
        "train_seconds": round(time.time() - t0, 1),
        "final_nll": losses[-1],
        "median_cos_to_init": round(float(cos_sim.median().item()), 4),
        "effective_rank_delta": round(eff_rank, 2),
    }
    return final_sp, meta


def main():
    t_start = time.time()
    print("=== Stage B-v2: Pure Verified NLL + Bidirectional Robust Vertical-Doc Continual Learning ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
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

    # 1. Distill P_A, P_B, P_C with Pure Verified NLL + Bidirectional Context-Robustness (L=128, 120 updates)
    caps = {}
    for idx, dom in enumerate(["A_heliospay", "B_nvx418", "C_lithox9"]):
        sp, meta = distill_doc_nll_bidir(
            model, tokenizer, tmpl, dom, datasets[dom]["train"],
            num_vectors=128, num_updates=120, batch_size=16, micro_batch=4,
            bidir_robust=True, seed=42 + idx, tag=f"NLL_BiDir_{dom}_L128"
        )
        caps[dom] = sp
        results["capsule_meta"][dom] = meta
        torch.save(sp.cpu(), os.path.join(OUT_DIR, f"sp_{dom}_L128_nll_bidir.pt"))
    save_results()

    pA, pB, pC = caps["A_heliospay"], caps["B_nvx418"], caps["C_lithox9"]

    # 2. Continual Learning Trajectory + Exact Unlearning + Retrieved Single Capsule
    print("\n--- Evaluating Continual Addition, Order Invariance, Exact Unlearning & Retrieved Top-1 ---", flush=True)
    results["runs"]["stage1_PA"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pA], eval_ood=True, tag="Stage1: [P_A]")
    results["runs"]["stage2_PA_PB"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pA, pB], eval_ood=True, tag="Stage2: [P_A; P_B]")
    results["runs"]["stage3_PA_PB_PC"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pA, pB, pC], eval_ood=True, tag="Stage3: [P_A; P_B; P_C]")
    results["runs"]["stage3_rev_PC_PB_PA"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pC, pB, pA], eval_ood=False, tag="Stage3-Rev: [P_C; P_B; P_A]")
    results["runs"]["stage4_unlearn_drop_PB"] = eval_all_domains(model, tokenizer, tmpl, datasets, [pA, pC], eval_ood=False, tag="Unlearn(-B): [P_A; P_C]")

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
        "mean_overall": round((results["runs"]["stage1_PA"]["A_heliospay"]["overall"] + sB["B_nvx418"]["overall"] + sC["C_lithox9"]["overall"]) / 3.0, 2),
    }
    save_results()

    # 3. Capacity Sweep under Pure NLL on Doc A: L = 32, 256 (L=128 already above!)
    print("\n--- Capacity Sweep under Pure NLL (Doc A: L=32 vs L=128 vs L=256) ---", flush=True)
    for L_val in [32, 256]:
        sp_L, meta_L = distill_doc_nll_bidir(
            model, tokenizer, tmpl, "A_heliospay", datasets["A_heliospay"]["train"],
            num_vectors=L_val, num_updates=120, batch_size=16, micro_batch=4,
            bidir_robust=True, seed=42, tag=f"NLL_BiDir_DocA_L{L_val}"
        )
        results["capsule_meta"][f"A_L{L_val}"] = meta_L
        results["runs"][f"cap_A_L{L_val}"] = eval_all_domains(model, tokenizer, tmpl, datasets, [sp_L], eval_ood=False, tag=f"DocA NLL L={L_val}")
        save_results()

    # 4. Native PyTorch Sequential LoRA Baseline (A -> B -> C) — zero peft/torchao conflicts!
    print("\n--- Step 4: Native PyTorch Sequential LoRA Baseline (r=16, alpha=32, A -> B -> C) ---", flush=True)
    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.scaling = alpha / r
            dev = base.weight.device
            self.lora_A = nn.Parameter(torch.empty((r, base.in_features), dtype=torch.float32, device=dev))
            self.lora_B = nn.Parameter(torch.zeros((base.out_features, r), dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base(x)
            delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
            return base_out + delta.to(base_out.dtype)

    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target_suffixes and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=16, alpha=32.0))
                replaced += 1
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Injected LoRALinear into {replaced} projections | trainable params: {n_trainable:,}", flush=True)

    embed_layer = model.get_input_embeddings()
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = embed_layer(torch.tensor([pad_id], device=device)).detach().to(dtype)

    for st_idx, (dom, stage_key) in enumerate([
        ("A_heliospay", "lora_seq_stage1_A"),
        ("B_nvx418", "lora_seq_stage2_AB"),
        ("C_lithox9", "lora_seq_stage3_ABC"),
    ]):
        t_lora = time.time()
        tokenized = [
            (tokenizer.encode(it["question"], add_special_tokens=False),
             tokenizer.encode(it["gold_answer"], add_special_tokens=False) + tmpl.im_end)
            for it in datasets[dom]["train"]
        ]
        lr_max = 2e-4
        opt = torch.optim.AdamW(trainable_params, lr=lr_max, weight_decay=0.0)
        model.config.use_cache = False
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        rng = random.Random(42 + st_idx)
        num_updates, batch_size, micro_batch = 120, 16, 4
        num_mb = batch_size // micro_batch
        for step in range(num_updates):
            progress = step / max(1, num_updates - 1)
            lr = (lr_max * 0.1) + 0.5 * (lr_max * 0.9) * (1.0 + math.cos(math.pi * progress))
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad()
            batch = [tokenized[rng.randrange(len(tokenized))] for _ in range(batch_size)]
            step_loss = 0.0
            for mb_i in range(0, batch_size, micro_batch):
                sub = batch[mb_i : mb_i + micro_batch]
                embs_l, lbls_l = [], []
                for qi, ri in sub:
                    emb, lbl, _ = build_soft_prompt_inputs(
                        embed_layer, tmpl, [], qi, ri, device=device, model_dtype=dtype
                    )
                    embs_l.append(emb)
                    lbls_l.append(lbl)
                b_emb, b_mask, b_lbl = pad_and_stack_train_batch(embs_l, lbls_l, pad_vec)
                b_emb = b_emb.requires_grad_(True)
                out = model(inputs_embeds=b_emb, attention_mask=b_mask)
                loss = length_normalized_nll_loss(out.logits, b_lbl)
                step_loss += loss.item()
                (loss / num_mb).backward()
                del b_emb, b_mask, b_lbl, out
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            if (step + 1) % 40 == 0 or step == num_updates - 1:
                print(f"  [SeqLoRA {stage_key}] step {step+1:3d}/{num_updates} | nll={step_loss/num_mb:.4f} | elapsed={time.time()-t_lora:.1f}s", flush=True)
        model.gradient_checkpointing_disable()
        model.eval()
        model.config.use_cache = True
        results["runs"][stage_key] = eval_all_domains(model, tokenizer, tmpl, datasets, [], eval_ood=True, tag=f"SeqLoRA {stage_key}")
        save_results()

    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save_results()
    print(f"\n=== Stage B-v2 COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
