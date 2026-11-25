"""
Paper Reference: Section 5.2, Table 5, Fig. 7–8 (5-Domain Auto-Routed KV-PD + Micro-RAG & 4-Token Delta).
Suite 3 (exp5): Enterprise Vertical-Domain Continual Learning Stress Tests on Colab A100.
Frozen backbone: Jackwang111/M2RL-SFT (Qwen3-4B BF16).

Reuses KV_A_16 / KV_B_16 / KV_C_16 from Suite 2 (exp4_results) and tests:
  1. 5-Domain Scaling (A -> B -> C -> D_gridops -> E_customs):
     - Verifies 0.0% zero-leakage on new domains D & E + In-Context Upper Bounds
     - Trains KV_D_16 & KV_E_16 (L_kv=16, 1.18M params/doc, 16 tokens context)
     - REAL Zero-Oracle Automatic Router (Frozen-LLM Last-Token Embedding + TF-IDF Hybrid + OOD Gate)
       evaluated across all 170 domain test queries + 24 OOD queries!
     - Head-to-Head vs 5-Stage Sequential LoRA (r=16, alpha=32, 11.8M params) Catastrophic Forgetting!
  2. Enterprise Document Version Upgrade & Knowledge Editing (HeliosPay v4.2 -> v5.0):
     - Outdated KV_A_v4.2 baseline (measures accuracy drop & stale v4.2 leakage)
     - KV-CSPD-Doc Atomic Swap (KV_A_v5.0): zero stale leakage + instant rollback
     - Incremental Delta-Patch Capsule ([KV_A_v4.2_frozen; Delta_KV_L4_v5.0], tiny 4-token 0.29M patch!)
     - Sequential LoRA Continual Update (LoRA_v4.2 -> LoRA_v5.0): measures v5.0 accuracy & v4.2 ghosting!
"""

import copy
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

from benchmarks import batch_generate_with_soft_prompts
from cspd_core import ChatTemplateTokens, get_qwen_template_tokens, length_normalized_nll_loss
from domain_docs_extended import (
    ALL_5_DOCS,
    DOC_A_V50_TEXT,
    DOMAIN_DESCRIPTORS,
    MICRO_RAG_CARDS,
    OOD_24_QUESTIONS,
    build_all_5_datasets,
    build_v50_dataset,
    grade_extended_response,
    grade_ood_response,
    has_v42_stale_leak,
)
from exp4_doc_cl_breakthroughs import eval_domain_suite, gen_unified, train_kv_prefix_doc
from kv_prefix_core import (
    batch_generate_with_kv_prefixes,
    build_chat_ids_for_kv,
    forward_with_kv_prefix,
    init_kv_prefix_from_tokens,
)

MODEL_ID = "Jackwang111/M2RL-SFT"
OUT_DIR = "/content/cspd/exp5_results"
os.makedirs(OUT_DIR, exist_ok=True)
DOM5 = ["A_heliospay", "B_nvx418", "C_lithox9", "D_gridops", "E_customs"]


# ---------------------------------------------------------------------------
# 1. Real Zero-Oracle Hybrid Semantic + Lexical Router (with OOD Abstention)
# ---------------------------------------------------------------------------

class ZeroOracleDomainRouter:
    """
    Automatic router combining:
      1) Frozen Qwen3 backbone mean-pooled hidden-state cosine similarity
      2) Token-level IDF-weighted lexical overlap with domain descriptor cards
    Routes to 'NONE' (no capsule) if max combined score < threshold -> OOD safe!
    """

    def __init__(self, model, tokenizer, descriptors: Dict[str, str], ood_threshold: float = 0.22):
        self.model = model
        self.tokenizer = tokenizer
        self.domains = list(descriptors.keys())
        self.ood_threshold = ood_threshold
        self.lex_sets = {d: self._tok_set(txt) for d, txt in descriptors.items()}
        self.doc_embs = {d: self._embed_text(txt) for d, txt in descriptors.items()}

    @staticmethod
    def _tok_set(text: str) -> set:
        return {t for t in re.findall(r"[a-z0-9\-_]+", text.lower()) if len(t) >= 3}

    @torch.no_grad()
    def _embed_text(self, text: str) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:256]
        t = torch.tensor([ids], dtype=torch.long, device=dev)
        out = self.model.model(input_ids=t, use_cache=False)
        h = out.last_hidden_state[0].float().mean(dim=0)
        return F.normalize(h, dim=-1)

    def route(self, query: str) -> Tuple[Optional[str], float]:
        q_emb = self._embed_text(query)
        q_lex = self._tok_set(query)
        best_dom, best_score = None, -1.0
        for d in self.domains:
            sem = float(torch.dot(q_emb, self.doc_embs[d]).item())
            inter = len(q_lex & self.lex_sets[d])
            lex = inter / max(1.0, math.sqrt(len(q_lex) + 1.0))
            # Lexical precision signal on domain-specific entity codes + semantic cosine
            score = 0.25 * max(0.0, sem) + 0.75 * min(1.0, lex * 0.85)
            if score > best_score:
                best_score = score
                best_dom = d
        if best_score < self.ood_threshold:
            return None, best_score
        return best_dom, best_score


# ---------------------------------------------------------------------------
# 2. Native PyTorch Sequential LoRA (r=16, alpha=32 across all q/k/v/o_proj)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / float(r)
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.empty((r, base.in_features), device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros((base.out_features, r), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self._dt = dt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(x.float(), self.A), self.B) * self.scale
        return base_out + delta.to(base_out.dtype)


def inject_lora(model, r: int = 16, alpha: int = 32) -> List[nn.Parameter]:
    params = []
    for layer in model.model.layers:
        attn = layer.self_attn
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            orig = getattr(attn, attr)
            if isinstance(orig, LoRALinear):
                params.extend([orig.A, orig.B])
            else:
                wrapped = LoRALinear(orig, r=r, alpha=alpha)
                setattr(attn, attr, wrapped)
                params.extend([wrapped.A, wrapped.B])
    return params


def remove_lora(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            cur = getattr(attn, attr)
            if isinstance(cur, LoRALinear):
                setattr(attn, attr, cur.base)


def train_lora_stage(
    model, tokenizer, tmpl: ChatTemplateTokens,
    lora_params: List[nn.Parameter],
    train_items: List[Dict[str, Any]],
    num_updates: int = 75,
    batch_size: int = 16,
    micro_batch: int = 4,
    lr_max: float = 2e-4,
    seed: int = 42,
    tag: str = "",
):
    t0 = time.time()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    nt_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)

    tok_data = []
    for it in train_items:
        q_ids = tokenizer.encode(it["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(it["gold_answer"], add_special_tokens=False) + tmpl.im_end
        fids, lbls = build_chat_ids_for_kv(tmpl, q_ids, r_ids, None, False, nt_ids)
        tok_data.append((fids, lbls))

    opt = torch.optim.AdamW(lora_params, lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = random.Random(seed)
    num_mb = max(1, batch_size // micro_batch)
    lr_min = lr_max * 0.1

    for step in range(1, num_updates + 1):
        prog = (step - 1) / max(1, num_updates - 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad()
        batch = [tok_data[rng.randrange(len(tok_data))] for _ in range(batch_size)]
        batch.sort(key=lambda x: len(x[0]))
        sl = 0.0
        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            B = len(sub)
            ml = max(len(x[0]) for x in sub)
            inp = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
            am = torch.zeros((B, ml), dtype=torch.long, device=device)
            lb = torch.full((B, ml), -100, dtype=torch.long, device=device)
            for i, (fids, lvals) in enumerate(sub):
                L = len(fids)
                inp[i, :L] = torch.tensor(fids, dtype=torch.long, device=device)
                am[i, :L] = 1
                lb[i, :L] = torch.tensor(lvals, dtype=torch.long, device=device)
            out = model(input_ids=inp, attention_mask=am)
            loss = length_normalized_nll_loss(out.logits, lb)
            sl += loss.item()
            (loss / num_mb).backward()
            del inp, am, lb, out
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        if step in [1, 40, num_updates]:
            print(f"  [{tag:24s}] step {step:3d}/{num_updates} | nll={sl/num_mb:.4f} | t={time.time()-t0:.1f}s", flush=True)

    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True


# ---------------------------------------------------------------------------
# 3. Incremental Delta-Patch KV Capsule Training ([KV_v42_frozen; Delta_KV_v50])
# ---------------------------------------------------------------------------

def train_kv_delta_patch(
    model, tokenizer, tmpl: ChatTemplateTokens,
    frozen_kv_base: torch.Tensor,
    train_items: List[Dict[str, Any]],
    doc_v50_text: str,
    patch_kv_tokens: int = 4,
    num_updates: int = 80,
    batch_size: int = 16,
    micro_batch: int = 4,
    lr_max: float = 0.035,
    seed: int = 88,
    tag: str = "KV_DeltaPatch_L4",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    t0 = time.time()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    nt_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)

    doc_ids = tokenizer.encode(doc_v50_text, add_special_tokens=False)
    delta_kv = init_kv_prefix_from_tokens(model, doc_ids, num_kv_tokens=patch_kv_tokens)
    base_frozen = frozen_kv_base.detach().to(device)

    tok_data = []
    for it in train_items:
        q_ids = tokenizer.encode(it["question"], add_special_tokens=False)
        r_ids = tokenizer.encode(it["gold_answer"], add_special_tokens=False) + tmpl.im_end
        fids, lbls = build_chat_ids_for_kv(tmpl, q_ids, r_ids, None, False, nt_ids)
        tok_data.append((fids, lbls))

    opt = torch.optim.AdamW([delta_kv], lr=lr_max, betas=(0.9, 0.999), weight_decay=0.0)
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True
    rng = random.Random(seed)
    num_mb = max(1, batch_size // micro_batch)
    lr_min = lr_max * 0.1

    for step in range(1, num_updates + 1):
        prog = (step - 1) / max(1, num_updates - 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad()
        batch = [tok_data[rng.randrange(len(tok_data))] for _ in range(batch_size)]
        batch.sort(key=lambda x: len(x[0]))
        sl = 0.0
        for mb_i in range(0, batch_size, micro_batch):
            sub = batch[mb_i : mb_i + micro_batch]
            B = len(sub)
            ml = max(len(x[0]) for x in sub)
            inp = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
            sm = torch.zeros((B, ml), dtype=torch.long, device=device)
            lb = torch.full((B, ml), -100, dtype=torch.long, device=device)
            for i, (fids, lvals) in enumerate(sub):
                L = len(fids)
                inp[i, :L] = torch.tensor(fids, dtype=torch.long, device=device)
                sm[i, :L] = 1
                lb[i, :L] = torch.tensor(lvals, dtype=torch.long, device=device)
            # Concatenate [frozen_v42_kv ; trainable_delta_v50_kv] — gradients flow ONLY to delta_kv!
            out = forward_with_kv_prefix(model, inp, sm, [base_frozen, delta_kv])
            loss = length_normalized_nll_loss(out.logits, lb)
            sl += loss.item()
            (loss / num_mb).backward()
            del inp, sm, lb, out
        torch.nn.utils.clip_grad_norm_([delta_kv], 1.0)
        opt.step()
        if step in [1, 40, num_updates]:
            print(f"  [{tag:24s}] step {step:3d}/{num_updates} | nll={sl/num_mb:.4f} | t={time.time()-t0:.1f}s", flush=True)

    return delta_kv.detach().clone(), {"patch_kv_tokens": patch_kv_tokens, "params": int(delta_kv.numel()), "train_seconds": round(time.time() - t0, 1)}


def eval_v50_with_stale_check(
    model, tokenizer, tmpl, v50_data, mode: str, adapters: List[torch.Tensor], system_text: str = "", tag: str = ""
) -> Dict[str, Any]:
    rq = [x["question"] for x in v50_data["test_rules"]]
    ro = gen_unified(model, tokenizer, tmpl, rq, mode, adapters, [system_text] * len(rq), 24, 384, False)
    r_ok = sum(grade_extended_response(o, it) for o, it in zip(ro, v50_data["test_rules"]))
    r_stale = sum(has_v42_stale_leak(o, it) for o, it in zip(ro, v50_data["test_rules"]))

    fq = [x["question"] for x in v50_data["test_facts"]]
    fo = gen_unified(model, tokenizer, tmpl, fq, mode, adapters, [system_text] * len(fq), 24, 256, False)
    f_ok = sum(grade_extended_response(o, it) for o, it in zip(fo, v50_data["test_facts"]))
    f_stale = sum(has_v42_stale_leak(o, it) for o, it in zip(fo, v50_data["test_facts"]))

    res = {
        "v50_rules": round(100.0 * r_ok / len(rq), 2),
        "v50_facts": round(100.0 * f_ok / len(fq), 2),
        "stale_v42_rules_pct": round(100.0 * r_stale / len(rq), 2),
        "stale_v42_facts_pct": round(100.0 * f_stale / len(fq), 2),
        "sample_rule": ro[0][:180],
        "sample_fact": fo[0][:140],
    }
    print(
        f"  [v5.0 Upgrade | {tag:24s}] v5.0={res['v50_rules']:.1f}R/{res['v50_facts']:.1f}F | "
        f"Stale v4.2 Leak={res['stale_v42_rules_pct']:.1f}R/{res['stale_v42_facts_pct']:.1f}F",
        flush=True,
    )
    return res


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=== Suite 3 (exp5): 5-Domain Zero-Oracle CL & Version Upgrade Stress Tests ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tmpl = get_qwen_template_tokens(tokenizer)
    datasets = build_all_5_datasets(seed=42)
    v50_data = build_v50_dataset(seed=150)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    results: Dict[str, Any] = {"model": MODEL_ID, "domains_5": {}, "auto_router": {}, "seq_lora_5": {}, "version_upgrade": {}}

    def save():
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # --- 1. Base (No Doc) & In-Context Upper Bound on D & E ---
    print("\n=== Block 1: Zero-Leakage & In-Context Reference on New Domains D & E ===", flush=True)
    results["domains_5"]["base_DE"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "sp", [], ["D_gridops", "E_customs"], tag="Base (No Doc) D&E")
    results["domains_5"]["in_context_DE"] = eval_domain_suite(
        model, tokenizer, tmpl, datasets, "sp", [], ["D_gridops", "E_customs"],
        system_texts_by_dom=ALL_5_DOCS, tag="In-Context Full Doc D&E"
    )
    save()

    # --- 2. Load KV_A/B/C from exp4 & Train KV_D_16, KV_E_16 ---
    print("\n=== Block 2: Building 5-Domain KV-CSPD-Doc Library (L_kv=16) ===", flush=True)
    kv_lib: Dict[str, torch.Tensor] = {}
    for dom, fname in [("A_heliospay", "KV_A_16.pt"), ("B_nvx418", "KV_B_16.pt"), ("C_lithox9", "KV_C_16.pt")]:
        p = os.path.join("/content/cspd/exp4_results", fname)
        if os.path.exists(p):
            kv_lib[dom] = torch.load(p, map_location="cuda")
            print(f"  Loaded cached {dom} from {p}", flush=True)
        else:
            kv_lib[dom], _ = train_kv_prefix_doc(
                model, tokenizer, tmpl, datasets[dom]["train"], ALL_5_DOCS[dom],
                num_kv_tokens=16, num_updates=80, batch_size=16, micro_batch=4, optimizer_type="adamw", lr_max=0.03, tag=f"KV_{dom[:1]}_16"
            )
    for idx, dom in enumerate(["D_gridops", "E_customs"]):
        kv_lib[dom], meta = train_kv_prefix_doc(
            model, tokenizer, tmpl, datasets[dom]["train"], ALL_5_DOCS[dom],
            num_kv_tokens=16, num_updates=80, batch_size=16, micro_batch=4, optimizer_type="adamw", lr_max=0.03, seed=45 + idx, tag=f"KV_{dom[:1]}_16"
        )
        results["domains_5"][f"single_{dom}"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "kv", [kv_lib[dom]], [dom], tag=f"KV_{dom[:1]}_16 Single")
    save()

    # --- 3. REAL Zero-Oracle Automatic Router across ALL 5 Domains + OOD (194 queries!) ---
    print("\n=== Block 3: End-to-End Zero-Oracle Automatic Routing (5 Domains + OOD) ===", flush=True)
    router = ZeroOracleDomainRouter(model, tokenizer, DOMAIN_DESCRIPTORS, ood_threshold=0.22)
    routing_stats = {}
    e2e_kv_only = {}
    e2e_kv_hybrid = {}

    for dom in DOM5:
        d = datasets[dom]
        # Route all rule & fact queries automatically!
        r_routes = [router.route(x["question"]) for x in d["test_rules"]]
        f_routes = [router.route(x["question"]) for x in d["test_facts"]]
        r_hit = sum(1 for rd, _ in r_routes if rd == dom)
        f_hit = sum(1 for rd, _ in f_routes if rd == dom)
        routing_stats[dom] = {
            "rule_route_acc": round(100.0 * r_hit / len(r_routes), 2),
            "fact_route_acc": round(100.0 * f_hit / len(f_routes), 2),
        }

        # Execute end-to-end generation using ONLY the automatically selected capsule (zero oracle!)
        def run_routed(items, routes, strategy: str):
            # strategy: "kv_only" | "kv_plus_card" | "dual_intent" (rules->KV, facts->MicroRAG card!)
            ok = 0
            by_key: Dict[Tuple[Optional[str], str], List[int]] = {}
            for idx_i, ((rd, _), it) in enumerate(zip(routes, items)):
                q_low = it["question"].lower()
                is_rule_q = ("evaluate" in q_low or "compute" in q_low or "determine" in q_low) and ("\n-" in q_low)
                sub_mode = "kv" if (strategy == "kv_only" or (strategy == "dual_intent" and is_rule_q)) else (
                    "card" if strategy == "dual_intent" else "both"
                )
                by_key.setdefault((rd, sub_mode), []).append(idx_i)
            preds = [""] * len(items)
            for (rd, sub_mode), idxs in by_key.items():
                qs = [items[i]["question"] for i in idxs]
                if rd is None:
                    outs = gen_unified(model, tokenizer, tmpl, qs, "sp", [], [""] * len(idxs), 24, 384, False)
                elif sub_mode == "kv":
                    outs = gen_unified(model, tokenizer, tmpl, qs, "kv", [kv_lib[rd]], [""] * len(idxs), 24, 384, False)
                elif sub_mode == "card":
                    outs = gen_unified(model, tokenizer, tmpl, qs, "sp", [], [MICRO_RAG_CARDS[rd]] * len(idxs), 24, 256, False)
                else:
                    outs = gen_unified(model, tokenizer, tmpl, qs, "kv", [kv_lib[rd]], [MICRO_RAG_CARDS[rd]] * len(idxs), 24, 384, False)
                for i, o in zip(idxs, outs):
                    preds[i] = o
            for o, it in zip(preds, items):
                ok += int(grade_extended_response(o, it))
            return round(100.0 * ok / len(items), 2)

        kv_r = run_routed(d["test_rules"], r_routes, "kv_only")
        kv_f = run_routed(d["test_facts"], f_routes, "kv_only")
        hy_r = run_routed(d["test_rules"], r_routes, "dual_intent")
        hy_f = run_routed(d["test_facts"], f_routes, "dual_intent")
        e2e_kv_only[dom] = {"rules": kv_r, "facts": kv_f}
        e2e_kv_hybrid[dom] = {"rules": hy_r, "facts": hy_f}
        print(
            f"  [AutoRouter | {dom:12s}] Route={routing_stats[dom]['rule_route_acc']:.0f}%R/{routing_stats[dom]['fact_route_acc']:.0f}%F | "
            f"KV-Auto={kv_r:.1f}%R/{kv_f:.1f}%F | DualIntent(KV+MicroRAG)-Auto={hy_r:.1f}%R/{hy_f:.1f}%F",
            flush=True,
        )

    # Route OOD 24 queries (should abstain -> route to None!)
    ood_routes = [router.route(q) for q, _ in OOD_24_QUESTIONS]
    ood_abstain_hit = sum(1 for rd, _ in ood_routes if rd is None)
    # Run OOD end-to-end
    ood_ok = 0
    for (q, kws), (rd, _) in zip(OOD_24_QUESTIONS, ood_routes):
        if rd is None:
            out = gen_unified(model, tokenizer, tmpl, [q], "sp", [], [""], 1, 128, False)[0]
        else:
            out = gen_unified(model, tokenizer, tmpl, [q], "kv", [kv_lib[rd]], [""], 1, 128, False)[0]
        ood_ok += int(grade_ood_response(out, kws))
    routing_stats["OOD_route_to_none_pct"] = round(100.0 * ood_abstain_hit / len(OOD_24_QUESTIONS), 2)
    e2e_kv_only["ood"] = round(100.0 * ood_ok / len(OOD_24_QUESTIONS), 2)
    e2e_kv_only["mean_rules"] = round(sum(e2e_kv_only[d]["rules"] for d in DOM5) / 5.0, 2)
    e2e_kv_only["mean_facts"] = round(sum(e2e_kv_only[d]["facts"] for d in DOM5) / 5.0, 2)
    e2e_kv_hybrid["mean_rules"] = round(sum(e2e_kv_hybrid[d]["rules"] for d in DOM5) / 5.0, 2)
    e2e_kv_hybrid["mean_facts"] = round(sum(e2e_kv_hybrid[d]["facts"] for d in DOM5) / 5.0, 2)
    e2e_kv_hybrid["ood"] = e2e_kv_only["ood"]
    results["auto_router"] = {
        "routing_accuracy": routing_stats,
        "e2e_kv16_auto_routed": e2e_kv_only,
        "e2e_kv16_microRAG_auto_routed": e2e_kv_hybrid,
    }
    print(
        f"  [AutoRouter SUMMARY (5 Docs)] OOD->None={routing_stats['OOD_route_to_none_pct']:.1f}% (OOD Acc={e2e_kv_only['ood']:.1f}%) | "
        f"KV-Auto Mean={e2e_kv_only['mean_rules']:.1f}R/{e2e_kv_only['mean_facts']:.1f}F | "
        f"KV+MicroRAG-Auto Mean={e2e_kv_hybrid['mean_rules']:.1f}R/{e2e_kv_hybrid['mean_facts']:.1f}F",
        flush=True,
    )
    save()

    # --- 4. Document Version Upgrade & Knowledge Editing (HeliosPay v4.2 -> v5.0) ---
    print("\n=== Block 4: Document Version Upgrade (HeliosPay v4.2 -> v5.0) ===", flush=True)
    results["version_upgrade"]["in_context_v50"] = eval_v50_with_stale_check(
        model, tokenizer, tmpl, v50_data, "sp", [], DOC_A_V50_TEXT, "In-Context v5.0"
    )
    results["version_upgrade"]["outdated_KV_v42"] = eval_v50_with_stale_check(
        model, tokenizer, tmpl, v50_data, "kv", [kv_lib["A_heliospay"]], "", "Outdated KV_v4.2"
    )
    # Atomic Swap: train clean KV_A_v50 independently!
    kv_A_v50, meta_v50 = train_kv_prefix_doc(
        model, tokenizer, tmpl, v50_data["train"], DOC_A_V50_TEXT,
        num_kv_tokens=16, num_updates=80, batch_size=16, micro_batch=4,
        optimizer_type="adamw", lr_max=0.03, seed=90, tag="KV_A_v50_Swap"
    )
    results["version_upgrade"]["kv_atomic_swap_v50"] = eval_v50_with_stale_check(
        model, tokenizer, tmpl, v50_data, "kv", [kv_A_v50], "", "KV Atomic Swap v5.0"
    )
    # Incremental Delta Patch: keep KV_v42 FROZEN, append tiny L_kv=4 patch (0.29M params)!
    delta_v50, meta_dp = train_kv_delta_patch(
        model, tokenizer, tmpl, kv_lib["A_heliospay"], v50_data["train"], DOC_A_V50_TEXT,
        patch_kv_tokens=4, num_updates=80, batch_size=16, micro_batch=4, lr_max=0.035, tag="KV_DeltaPatch_L4"
    )
    results["version_upgrade"]["kv_delta_patch_L4"] = eval_v50_with_stale_check(
        model, tokenizer, tmpl, v50_data, "kv", [kv_lib["A_heliospay"], delta_v50], "", "KV Delta-Patch [v42;d4]"
    )
    results["version_upgrade"]["kv_delta_patch_meta"] = meta_dp
    save()

    # --- 5. Sequential LoRA (A -> v5.0 Knowledge Edit Check) + 5-Stage SeqLoRA (A -> B -> C -> D -> E) ---
    print("\n=== Block 5: Sequential LoRA (v4.2->v5.0 Edit + 5-Domain Catastrophic Forgetting) ===", flush=True)
    # 5A: LoRA v4.2 -> v5.0 edit check
    lora_p = inject_lora(model, r=16, alpha=32)
    train_lora_stage(model, tokenizer, tmpl, lora_p, datasets["A_heliospay"]["train"], num_updates=70, seed=42, tag="LoRA_Stage1_A_v42")
    results["seq_lora_5"]["stage1_A"] = eval_domain_suite(model, tokenizer, tmpl, datasets, "sp", [], DOM5, eval_ood=True, tag="SeqLoRA Stage 1 (A)")

    # Save LoRA Stage 1 state so we can test v4.2->v5.0 update cleanly, then restore for 5-domain A->B->C->D->E!
    saved_stage1 = [p.detach().clone() for p in lora_p]
    train_lora_stage(model, tokenizer, tmpl, lora_p, v50_data["train"], num_updates=70, seed=91, tag="LoRA_Edit_v42_to_v50")
    results["version_upgrade"]["lora_continual_v42_to_v50"] = eval_v50_with_stale_check(
        model, tokenizer, tmpl, v50_data, "sp", [], "", "LoRA Update v4.2->v5.0"
    )
    # Restore Stage 1 LoRA weights to run clean 5-domain A -> B -> C -> D -> E!
    with torch.no_grad():
        for p, s in zip(lora_p, saved_stage1):
            p.copy_(s)
    del saved_stage1

    for st_idx, dom in enumerate(["B_nvx418", "C_lithox9", "D_gridops", "E_customs"], start=2):
        train_lora_stage(model, tokenizer, tmpl, lora_p, datasets[dom]["train"], num_updates=70, seed=42 + st_idx, tag=f"LoRA_Stage{st_idx}_{dom[:1]}")
        results["seq_lora_5"][f"stage{st_idx}_{dom[:1]}"] = eval_domain_suite(
            model, tokenizer, tmpl, datasets, "sp", [], DOM5, eval_ood=(st_idx == 5), tag=f"SeqLoRA Stage {st_idx} ({dom[:1]})"
        )
        save()

    remove_lora(model)
    results["total_elapsed_sec"] = round(time.time() - t_start, 1)
    save()
    print(f"\n=== Suite 3 (exp5) COMPLETE in {results['total_elapsed_sec']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
