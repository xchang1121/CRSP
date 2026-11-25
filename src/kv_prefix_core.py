"""
Paper Reference: Section 2.3 (Eqs. 7–10), Section 5 (Layerwise KV-PD & Delta Patch).
Direct Layerwise KV-Cache Prefix Distillation (KV-CSPD / Cartridge-Style Prefix)
for Frozen Qwen3 (and compatible HuggingFace Transformers 5.x CausalLMs).

Key Properties:
  - Direct gradient highway to every attention layer's K/V space (faster convergence than layer-0 SP)
  - 8x-32x shorter inference context overhead (e.g. L_kv=8 or 16 vs L=256 input SP)
  - Natural-manifold initialization by extracting real KV states from text/document tokens
  - Zero-training concatenation composition: KV_{1:m} = cat([KV_1, ..., KV_m], dim=seq)
  - Supports both AdamW and Layerwise-Muon optimization + Norm-Bounded projection
  - Non-mutating ReadOnlyPrefixCache makes it 100% safe with gradient checkpointing!
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from cspd_core import ChatTemplateTokens, newton_schulz_5


class ReadOnlyPrefixCache(DynamicCache):
    """
    Non-mutating KV prefix cache for training & logprob evaluation.
    Safe under torch.utils.checkpoint recomputation because update() never mutates self._pk/_pv.
    """

    def __init__(self, prefix_k: torch.Tensor, prefix_v: torch.Tensor):
        # prefix_k, prefix_v: [num_layers, B, H_kv, L_kv, D_h]
        super().__init__()
        self._pk = prefix_k
        self._pv = prefix_v
        self._seq_len = int(prefix_k.size(-2))

    def __len__(self) -> int:
        return int(self._pk.size(0))

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs):
        k_pre = self._pk[layer_idx].to(device=key_states.device, dtype=key_states.dtype)
        v_pre = self._pv[layer_idx].to(device=value_states.device, dtype=value_states.dtype)
        return torch.cat([k_pre, key_states], dim=-2), torch.cat([v_pre, value_states], dim=-2)


def make_mutable_kv_cache(kv_list: List[torch.Tensor], batch_size: int, device, dtype) -> DynamicCache:
    """
    Build a standard mutable HF DynamicCache for model.generate() from 1 or more
    KV-prefix tensors, each shaped [num_layers, 2, L_kv, H_kv, D_h] (FP32).
    Zero-training composition = direct concatenation along sequence dim (dim=2)!
    """
    if len(kv_list) == 1:
        cat_kv = kv_list[0]
    else:
        cat_kv = torch.cat(kv_list, dim=2)  # [num_layers, 2, sum(L_kv), H_kv, D_h]

    num_layers = cat_kv.size(0)
    cache = DynamicCache()
    for l_idx in range(num_layers):
        # [L_kv, H_kv, D_h] -> [B, H_kv, L_kv, D_h]
        k = cat_kv[l_idx, 0].detach().to(device=device, dtype=dtype).permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        v = cat_kv[l_idx, 1].detach().to(device=device, dtype=dtype).permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        cache.update(k, v, l_idx)
    return cache


def make_readonly_kv_cache(
    kv_list: List[torch.Tensor],
    batch_size: int,
    device,
    dtype,
    distractor_kv: Optional[torch.Tensor] = None,
    post_distractor_kv: Optional[torch.Tensor] = None,
) -> ReadOnlyPrefixCache:
    """
    Build ReadOnlyPrefixCache preserving full autograd graph back to trainable kv_list tensors.
    Supports optional pre/post KV distractors for Prefix-Robust KV training!
    """
    parts = []
    if distractor_kv is not None:
        parts.append(distractor_kv.to(device=device, dtype=dtype))
    for kv in kv_list:
        parts.append(kv.to(device=device, dtype=dtype))
    if post_distractor_kv is not None:
        parts.append(post_distractor_kv.to(device=device, dtype=dtype))
    cat_kv = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)  # [num_layers, 2, L_tot, H_kv, D_h]

    # Convert to [num_layers, B, H_kv, L_tot, D_h]
    pk = cat_kv[:, 0].permute(0, 2, 1, 3).unsqueeze(1).expand(-1, batch_size, -1, -1, -1)
    pv = cat_kv[:, 1].permute(0, 2, 1, 3).unsqueeze(1).expand(-1, batch_size, -1, -1, -1)
    return ReadOnlyPrefixCache(pk, pv)


@torch.no_grad()
def init_kv_prefix_from_tokens(
    model,
    token_ids: List[int],
    num_kv_tokens: int = 16,
) -> nn.Parameter:
    """
    Initialize KV-prefix [num_layers, 2, L_kv, H_kv, D_h] (FP32) on the frozen backbone's
    EXACT natural KV manifold by running a real token sequence (e.g., document text, task
    instruction, or random vocab tokens) through the frozen backbone once!
    """
    device = next(model.parameters()).device
    if len(token_ids) < num_kv_tokens:
        reps = (num_kv_tokens // max(1, len(token_ids))) + 1
        token_ids = (token_ids * reps)[:num_kv_tokens]
    elif len(token_ids) > num_kv_tokens:
        # Strided sampling across the whole document/text so full coverage is captured!
        step = len(token_ids) / float(num_kv_tokens)
        token_ids = [token_ids[int(i * step)] for i in range(num_kv_tokens)]

    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
    out = model(input_ids=inp, use_cache=True)
    pkv = out.past_key_values
    num_layers = len(pkv.layers)
    stacked = []
    for l_idx in range(num_layers):
        # pkv.layers[l_idx].keys has shape [1, H_kv, L_kv, D_h]
        k = pkv.layers[l_idx].keys[0].permute(1, 0, 2).float()  # [L_kv, H_kv, D_h]
        v = pkv.layers[l_idx].values[0].permute(1, 0, 2).float()
        stacked.append(torch.stack([k, v], dim=0))  # [2, L_kv, H_kv, D_h]
    tensor = torch.stack(stacked, dim=0).contiguous()  # [num_layers, 2, L_kv, H_kv, D_h]
    return nn.Parameter(tensor)


def build_chat_ids_for_kv(
    tmpl: ChatTemplateTokens,
    question_ids: List[int],
    response_ids: Optional[List[int]] = None,
    system_text_ids: Optional[List[int]] = None,
    enable_thinking: bool = True,
    no_think_ids: Optional[List[int]] = None,
) -> Tuple[List[int], Optional[List[int]]]:
    """
    Build token ID sequence that follows a KV prefix (no soft-prompt tokens in input_ids).
    """
    ids: List[int] = []
    if system_text_ids:
        ids.extend(tmpl.sys_open + system_text_ids + tmpl.sys_close_user_open)
    else:
        ids.extend(tmpl.sys_open + tmpl.sys_close_user_open)
    ids.extend(question_ids + tmpl.user_close_asst_open)
    if (not enable_thinking) and no_think_ids:
        ids.extend(no_think_ids)
    prompt_len = len(ids)
    if response_ids is None:
        return ids, None
    full_ids = ids + response_ids
    labels = [-100] * prompt_len + list(response_ids)
    return full_ids, labels


def forward_with_kv_prefix(
    model,
    input_ids: torch.Tensor,          # [B, T]
    seq_attn_mask: torch.Tensor,      # [B, T] (1 for real, 0 for pad)
    kv_list: List[torch.Tensor],      # list of [num_layers, 2, L_kv_i, H_kv, D_h]
    distractor_kv: Optional[torch.Tensor] = None,
    post_distractor_kv: Optional[torch.Tensor] = None,
):
    """
    Run forward pass through frozen backbone with differentiable KV prefix cache.
    Automatically constructs combined attention_mask [B, L_kv + T] and explicit cache_position!
    """
    B, T = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    cache = make_readonly_kv_cache(kv_list, B, device, dtype, distractor_kv, post_distractor_kv)
    L_kv = cache.get_seq_length()
    # Explicit 4D additive causal mask [B, 1, T, L_kv + T]: immune to HF create_causal_mask quirks!
    min_val = torch.finfo(dtype).min
    causal = torch.triu(torch.full((T, T), min_val, dtype=dtype, device=device), diagonal=1)
    prefix_zeros = torch.zeros((T, L_kv), dtype=dtype, device=device)
    mask_4d = torch.cat([prefix_zeros, causal], dim=1).unsqueeze(0).unsqueeze(0).expand(B, 1, T, L_kv + T).clone()
    pad_cols = (seq_attn_mask == 0)
    if pad_cols.any():
        mask_4d[:, :, :, L_kv:].masked_fill_(pad_cols[:, None, None, :], min_val)
    cache_pos = torch.arange(L_kv, L_kv + T, dtype=torch.long, device=device)
    return model(
        input_ids=input_ids,
        attention_mask=mask_4d,
        past_key_values=cache,
        cache_position=cache_pos,
        use_cache=True,
    )


@torch.no_grad()
def batch_generate_with_kv_prefixes(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    questions: List[str],
    kv_list: List[torch.Tensor],
    system_texts: Optional[List[str]] = None,
    batch_size: int = 16,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 20,
    enable_thinking: bool = False,
) -> List[str]:
    """
    Fast left-padded batched generation with 0, 1, or m concatenated KV-prefix capsules.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    nt_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False) if not enable_thinking else None
    if system_texts is None:
        system_texts = [""] * len(questions)

    all_outputs: List[str] = []
    for start in range(0, len(questions), batch_size):
        bq = questions[start : start + batch_size]
        bs_txt = system_texts[start : start + batch_size]
        seqs = []
        for q, s in zip(bq, bs_txt):
            q_ids = tokenizer.encode(q, add_special_tokens=False)
            s_ids = tokenizer.encode(s, add_special_tokens=False) if s else None
            ids, _ = build_chat_ids_for_kv(tmpl, q_ids, None, s_ids, enable_thinking, nt_ids)
            seqs.append(ids)

        B = len(seqs)
        max_len = max(len(x) for x in seqs)
        inp = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        seq_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for i, ids in enumerate(seqs):
            L = len(ids)
            inp[i, max_len - L :] = torch.tensor(ids, dtype=torch.long, device=device)
            seq_mask[i, max_len - L :] = 1

        gen_kwargs = dict(
            input_ids=inp,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if kv_list:
            cache = make_mutable_kv_cache(kv_list, B, device, dtype)
            L_kv = cache.get_seq_length()
            full_mask = torch.cat([torch.ones((B, L_kv), dtype=torch.long, device=device), seq_mask], dim=1)
            gen_kwargs["past_key_values"] = cache
            gen_kwargs["attention_mask"] = full_mask
        else:
            gen_kwargs["attention_mask"] = seq_mask

        if temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            gen_kwargs.update(do_sample=False)

        out_ids = model.generate(**gen_kwargs)
        # model.generate returns [input_ids, new_tokens] -> strip max_len prefix!
        for row in out_ids:
            all_outputs.append(tokenizer.decode(row[max_len:], skip_special_tokens=True))
    return all_outputs


class LayerwiseMuonKV(torch.optim.Optimizer):
    """
    Muon optimizer adapted for 5D KV-prefix tensors [num_layers, 2, L_kv, H_kv, D_h]:
    Applies 5-step Newton-Schulz orthogonalization independently to each [L_kv, H_kv*D_h] 2D slice!
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, rms_scale: float = 0.2):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, rms_scale=rms_scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, nest, ns, rms_s = group["lr"], group["momentum"], group["nesterov"], group["ns_steps"], group["rms_scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mu).add_(g)
                g_eff = g.add(buf, alpha=mu) if nest else buf

                # Reshape [num_layers, 2, L_kv, H_kv, D_h] -> iterate over [L_kv, H_kv*D_h]
                nl, two, lkv, hkv, dh = g_eff.shape
                upd = torch.empty_like(g_eff)
                for li in range(nl):
                    for kv_i in range(two):
                        mat = g_eff[li, kv_i].reshape(lkv, hkv * dh)
                        orth = newton_schulz_5(mat, steps=ns)
                        cur_rms = orth.pow(2).mean().sqrt().clamp_min(1e-8)
                        upd[li, kv_i] = (orth * (rms_s / cur_rms)).reshape(lkv, hkv, dh)
                p.add_(upd.to(p.dtype), alpha=-lr)
        return loss
