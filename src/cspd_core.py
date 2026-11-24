"""
Core implementation of Composable Soft-Prompt Distillation (CSPD)
Paper: "Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training" (2026)

Includes:
- Muon optimizer with 5 Newton-Schulz iterations and Moonlight RMS rescaling (0.2 * lr)
- SoftPromptDistiller wrapping any frozen HuggingFace CausalLM (Qwen3 / Qwen2.5 etc.)
- Exact Qwen3 chat template insertion with 50% content-hash Prefix-Robust Training
- Length-normalized sequence-level NLL loss + optional Context Distillation (Token-level KL)
- Zero-training concatenation composition P_{1:k} = [P_1; ...; P_k]
- Mechanistic analysis: Teacher-Student KL gap closed G(A) (Eq. 5) & Attention Pathway Ablation K(A) (Eq. 6)
"""

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Muon Optimizer (Newton-Schulz-5 + Moonlight RMS=0.2 rescaling)
# ---------------------------------------------------------------------------

def newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Quintic Newton-Schulz iteration approximating the polar factor U V^T of G = U S V^T.
    Matches Moonlight (Liu et al., 2025) / Keller Jordan Muon coefficients.
    """
    assert G.ndim == 2, f"Muon expects 2D parameter, got shape {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MuonSoftPrompt(torch.optim.Optimizer):
    """
    Muon optimizer tailored for 2D soft-prompt matrices P in R^{L x d} (FP32).
    Follows Appendix A of CSPD:
      - Nesterov momentum mu = 0.95
      - 5 Newton-Schulz orthogonalization iterations
      - Moonlight RMS rescaling so RMS(update) = 0.2 * lr
    """

    def __init__(
        self,
        params,
        lr: float = 0.03,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        rms_scale: float = 0.2,
        scaling_mode: str = "rms",  # "rms" (exact RMS=0.2) or "moonlight_maxdim" (0.2*sqrt(max(L,d)))
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            rms_scale=rms_scale,
            scaling_mode=scaling_mode,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            rms_scale = group["rms_scale"]
            scaling_mode = group.get("scaling_mode", "rms")

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                if nesterov:
                    g_eff = g.add(buf, alpha=mu)
                else:
                    g_eff = buf

                orth = newton_schulz_5(g_eff, steps=ns_steps)
                if scaling_mode == "moonlight_maxdim":
                    update = orth * (rms_scale * math.sqrt(max(p.shape)))
                else:
                    # Moonlight RMS rescaling: rescale orthogonalized update to RMS = 0.2
                    cur_rms = orth.pow(2).mean().sqrt().clamp_min(1e-8)
                    update = orth * (rms_scale / cur_rms)
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
# 2. Soft Prompt Module & Template Construction (Appendix A)
# ---------------------------------------------------------------------------

def init_soft_prompt_from_vocab(
    embed_weight: torch.Tensor,
    num_vectors: int = 256,
    seed: int = 42,
    vocab_limit: Optional[int] = None,
    tokenizer=None,
) -> nn.Parameter:
    """
    Initialize soft prompt P in R^{L x d} (FP32) by sampling vocabulary embeddings.
    When ``tokenizer`` is supplied, samples uniformly over all non-special token IDs
    in ``[0, min(len(tokenizer), vocab_size))``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    if tokenizer is not None:
        upper = min(len(tokenizer), embed_weight.size(0))
        special_ids = {int(v) for v in getattr(tokenizer, "all_special_ids", [])}
        allowed = [tid for tid in range(upper) if tid not in special_ids]
        if not allowed:
            raise ValueError("tokenizer has no non-special vocabulary rows to sample")
        pos = torch.randint(0, len(allowed), size=(num_vectors,), generator=gen).tolist()
        indices = torch.tensor([allowed[i] for i in pos], dtype=torch.long)
    else:
        max_id = vocab_limit if vocab_limit is not None else min(embed_weight.size(0), 150000)
        low_id = min(100, max_id // 10)
        indices = torch.randint(low=low_id, high=max_id, size=(num_vectors,), generator=gen)
    init_vecs = embed_weight[indices.to(embed_weight.device)].detach().float().clone()
    return nn.Parameter(init_vecs)


@dataclass
class ChatTemplateTokens:
    sys_open: List[int]        # "<|im_start|>system\n"
    sys_close_user_open: List[int]  # "<|im_end|>\n<|im_start|>user\n"
    user_close_asst_open: List[int] # "<|im_end|>\n<|im_start|>assistant\n"
    im_end: List[int]          # "<|im_end|>\n"


def get_qwen_template_tokens(tokenizer) -> ChatTemplateTokens:
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    return ChatTemplateTokens(
        sys_open=enc("<|im_start|>system\n"),
        sys_close_user_open=enc("<|im_end|>\n<|im_start|>user\n"),
        user_close_asst_open=enc("<|im_end|>\n<|im_start|>assistant\n"),
        im_end=enc("<|im_end|>\n"),
    )


def should_add_prefix(text: str) -> bool:
    """Deterministic 50% selection by content hash (Appendix A)."""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return (h % 2) == 0


def sample_random_prefix(
    domain_token_stream: List[int],
    rng: random.Random,
    min_len: int = 128,
    max_len: int = 256,
) -> List[int]:
    """Sample 128-256 tokens from prompt-token stream of other examples in same domain."""
    if not domain_token_stream:
        return []
    length = rng.randint(min_len, max_len)
    if len(domain_token_stream) <= length:
        reps = (length // len(domain_token_stream)) + 1
        stream = domain_token_stream * reps
        return stream[:length]
    start = rng.randint(0, len(domain_token_stream) - length)
    return domain_token_stream[start : start + length]


def build_soft_prompt_inputs(
    embed_layer: nn.Embedding,
    tmpl: ChatTemplateTokens,
    soft_prompts: List[torch.Tensor],  # list of [L_i, d] tensors (concatenated left-to-right)
    question_ids: List[int],
    response_ids: Optional[List[int]] = None,
    system_text_ids: Optional[List[int]] = None,
    prefix_F_ids: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    model_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Tuple[int, int]]]:
    """
    Build continuous `inputs_embeds` [1, T, d] and `labels` [1, T] following Appendix A:
      <|im_start|>system\n [F] [P_1; ...; P_k] [s] <|im_end|>\n<|im_start|>user\n [q] <|im_end|>\n<|im_start|>assistant\n [y]
    Returns:
      inputs_embeds: [1, T, d] (cast to model_dtype; preserves grad to soft_prompts)
      labels: [1, T] (-100 everywhere except response tokens) or None if response_ids is None
      spans: dict mapping region name -> (start_idx, end_idx) for attention pathway analysis
    """
    if device is None:
        device = embed_layer.weight.device

    parts = []
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    def add_ids(name: str, ids: List[int]):
        nonlocal cursor
        if not ids:
            return
        t = torch.tensor(ids, dtype=torch.long, device=device)
        e = embed_layer(t).to(model_dtype)
        parts.append(e)
        spans[name] = (cursor, cursor + len(ids))
        cursor += len(ids)

    # 1. System open
    add_ids("sys_open", tmpl.sys_open)

    # 2. Optional prefix F (frozen embeddings, training only)
    if prefix_F_ids:
        add_ids("prefix_F", prefix_F_ids)

    # 3. Concatenated Soft Prompts P_1 ... P_k
    if soft_prompts:
        sp_start = cursor
        for sp in soft_prompts:
            sp_cast = sp.to(device=device, dtype=model_dtype)
            parts.append(sp_cast)
            cursor += sp_cast.size(0)
        spans["soft_prompt"] = (sp_start, cursor)

    # 4. Original system text (if any)
    if system_text_ids:
        add_ids("sys_text", system_text_ids)

    # 5. System close + User open + Question + User close + Assistant open
    add_ids("sys_close_user_open", tmpl.sys_close_user_open)
    add_ids("question", question_ids)
    add_ids("user_close_asst_open", tmpl.user_close_asst_open)

    prompt_len = cursor

    # 6. Optional target response y
    labels = None
    if response_ids is not None:
        add_ids("response", response_ids)
        labels = torch.full((1, cursor), -100, dtype=torch.long, device=device)
        labels[0, prompt_len:cursor] = torch.tensor(response_ids, dtype=torch.long, device=device)

    inputs_embeds = torch.cat(parts, dim=0).unsqueeze(0)  # [1, T, d]
    return inputs_embeds, labels, spans


def length_normalized_nll_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Equation (1): Length-normalized negative log-likelihood over response tokens:
      - (1 / |y|) * sum_{t=1}^{|y|} log p(y_t | I(P_j, x), y_{<t})
    Memory-efficient: computes CE only on valid (non -100) response positions!
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    bs = shift_logits.size(0)
    per_sample = []
    for i in range(bs):
        mask = shift_labels[i] != -100
        if mask.any():
            ce = F.cross_entropy(shift_logits[i][mask].float(), shift_labels[i][mask], reduction="mean")
            per_sample.append(ce)
    return torch.stack(per_sample).mean()


def context_distill_kl_loss(
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_resp_logprobs: torch.Tensor,  # [resp_len, V] log_softmax from teacher on response positions
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Forward KL divergence D_KL(p_teacher || p_student) averaged over response positions.
    Used for Context-to-Soft-Prompt Distillation (teacher sees raw document C_j, student sees only P_j).
    """
    shift_logits = student_logits[0, :-1, :]
    shift_labels = student_labels[0, 1:]
    resp_mask = shift_labels != -100
    s_resp_logits = shift_logits[resp_mask]  # [resp_len, V]
    n = min(s_resp_logits.size(0), teacher_resp_logprobs.size(0))
    s_logp = F.log_softmax(s_resp_logits[:n].float() / temperature, dim=-1)
    t_logp = teacher_resp_logprobs[:n].float().to(s_logp.device)
    t_prob = t_logp.exp()
    # KL(T || S) = sum_v p_T(v) * (log p_T(v) - log p_S(v))
    kl = (t_prob * (t_logp - s_logp)).sum(dim=-1).mean() * (temperature ** 2)
    return kl


def pad_and_stack_train_batch(
    embs_list: List[torch.Tensor],    # list of [1, T_i, d]
    labels_list: List[torch.Tensor],  # list of [1, T_i]
    pad_vec: torch.Tensor,            # [1, d]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Right-pad variable-length training sequences into a single batched tensor
    (inputs_embeds [B, T_max, d], attention_mask [B, T_max], labels [B, T_max]).
    Preserves gradients back to soft prompt parameters cleanly!
    """
    bs = len(embs_list)
    max_len = max(e.size(1) for e in embs_list)
    device = embs_list[0].device
    dtype = embs_list[0].dtype

    padded_embs = []
    attn_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
    padded_labels = torch.full((bs, max_len), -100, dtype=torch.long, device=device)

    for i, (e, lbl) in enumerate(zip(embs_list, labels_list)):
        L = e.size(1)
        attn_mask[i, :L] = 1
        padded_labels[i, :L] = lbl[0]
        if L < max_len:
            pad = pad_vec.to(device=device, dtype=dtype).expand(1, max_len - L, -1)
            padded_embs.append(torch.cat([e, pad], dim=1))
        else:
            padded_embs.append(e)

    return torch.cat(padded_embs, dim=0), attn_mask, padded_labels


def batched_context_distill_kl_loss(
    student_logits: torch.Tensor,           # [B, T_s, V]
    student_labels: torch.Tensor,           # [B, T_s]
    teacher_resp_logprobs_list: List[torch.Tensor],  # B tensors each [resp_len_i, V]
) -> torch.Tensor:
    """Per-sample length-normalized Forward KL averaged across a batch."""
    bs = student_logits.size(0)
    shift_logits = student_logits[:, :-1, :]
    shift_labels = student_labels[:, 1:]
    kls = []
    for i in range(bs):
        mask = shift_labels[i] != -100
        s_resp = shift_logits[i][mask]
        t_lp = teacher_resp_logprobs_list[i].float().to(s_resp.device)
        n = min(s_resp.size(0), t_lp.size(0))
        s_lp = F.log_softmax(s_resp[:n].float(), dim=-1)
        kl_i = (t_lp[:n].exp() * (t_lp[:n] - s_lp)).sum(dim=-1).mean()
        kls.append(kl_i)
    return torch.stack(kls).mean()


# ---------------------------------------------------------------------------
# 3. Generation Helper with Concatenated Soft Prompts
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_soft_prompts(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    question: str,
    soft_prompts: List[torch.Tensor],
    system_text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95,
    enable_thinking: bool = False,
) -> str:
    """Generate response from frozen model with 0, 1, or k concatenated soft prompts."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()

    q_ids = tokenizer.encode(question, add_special_tokens=False)
    s_ids = tokenizer.encode(system_text, add_special_tokens=False) if system_text else None
    # For Qwen3 non-thinking mode if desired, prepend empty think tags in assistant turn
    inputs_embeds, _, _ = build_soft_prompt_inputs(
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
    if not enable_thinking:
        no_think_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
        nt_emb = embed_layer(torch.tensor(no_think_ids, device=device)).to(dtype).unsqueeze(0)
        inputs_embeds = torch.cat([inputs_embeds, nt_emb], dim=1)

    attn_mask = torch.ones((1, inputs_embeds.size(1)), dtype=torch.long, device=device)
    gen_kwargs = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0.0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    out_ids = model.generate(**gen_kwargs)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 4. Mechanistic Analysis: KL Gap G(A) (Eq. 5) & Attention Pathways K(A) (Eq. 6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_response_logprobs(
    model,
    embed_layer: nn.Embedding,
    tmpl: ChatTemplateTokens,
    soft_prompts: List[torch.Tensor],
    question_ids: List[int],
    response_ids: List[int],
    system_text_ids: Optional[List[int]] = None,
    attention_mode: str = "full",  # "full" | "response_only" | "input_only" | "blocked"
) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int]]]:
    """
    Compute per-position log-probabilities over response tokens, supporting
    custom 4D causal attention masks for Section 4 / Appendix G.4 pathway ablation.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs_embeds, labels, spans = build_soft_prompt_inputs(
        embed_layer=embed_layer,
        tmpl=tmpl,
        soft_prompts=soft_prompts,
        question_ids=question_ids,
        response_ids=response_ids,
        system_text_ids=system_text_ids,
        prefix_F_ids=None,
        device=device,
        model_dtype=dtype,
    )
    T = inputs_embeds.size(1)

    if attention_mode == "full" or "soft_prompt" not in spans:
        outputs = model(inputs_embeds=inputs_embeds)
    else:
        # Build explicit additive 4D causal mask [1, 1, T, T]
        min_val = torch.finfo(dtype).min
        mask = torch.triu(torch.full((T, T), min_val, dtype=dtype, device=device), diagonal=1)
        sp_s, sp_e = spans["soft_prompt"]
        resp_s, resp_e = spans["response"]
        # Positions predicting response tokens are [resp_s - 1, resp_e - 1)
        # Input tokens after soft prompt are [sp_e, resp_s - 1)
        if attention_mode == "response_only":
            # Block input tokens (between soft_prompt and first response prediction) from attending to soft_prompt
            mask[sp_e : resp_s - 1, sp_s:sp_e] = min_val
        elif attention_mode == "input_only":
            # Block response positions from attending directly to soft_prompt
            mask[resp_s - 1 : resp_e, sp_s:sp_e] = min_val
        elif attention_mode == "blocked":
            # Block all tokens after soft_prompt from attending to soft_prompt
            mask[sp_e:, sp_s:sp_e] = min_val
        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=mask.unsqueeze(0).unsqueeze(0))

    shift_logits = outputs.logits[0, :-1, :]
    shift_labels = labels[0, 1:]
    resp_mask = shift_labels != -100
    resp_logprobs = F.log_softmax(shift_logits[resp_mask].float(), dim=-1)
    return resp_logprobs, spans


def mean_kl_divergence(teacher_logprobs: torch.Tensor, model_logprobs: torch.Tensor) -> float:
    """Average D_KL(T || A) over response token positions."""
    n = min(teacher_logprobs.size(0), model_logprobs.size(0))
    t_lp = teacher_logprobs[:n].float()
    a_lp = model_logprobs[:n].float()
    kl = (t_lp.exp() * (t_lp - a_lp)).sum(dim=-1).mean().item()
    return float(kl)


# ---------------------------------------------------------------------------
# 5. Core Distillation & Verification Utilities: Response-Only lm_head, EOS/Think Gates,
#    Deterministic SHA-256 Prefix Sampling, Scaffold Audit & Safety Wrapper
# ---------------------------------------------------------------------------

QWEN_END_TOKENS: Tuple[int, ...] = (151645, 151643)  # <|im_end|>, <|endoftext|>


def verify_qwen_system_scaffold(tokenizer) -> Tuple[List[int], List[int]]:
    """
    Verify exact Qwen3 empty-system scaffold invariance before training/evaluation:
    adding an empty system turn must equal header + close + user_prompt token-for-token.
    """
    header = tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
    close = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    example = [{"role": "user", "content": "scaffold-check"}]
    kw = dict(tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)
    original = list(tokenizer.apply_chat_template(example, **kw))
    empty = list(tokenizer.apply_chat_template([{"role": "system", "content": ""}, *example], **kw))
    if empty != header + close + original:
        raise ValueError("empty system scaffold does not match the tokenizer chat template")
    return header, close


def deterministic_prefix_slice(
    input_ids: List[int],
    prefix_pool: List[List[int]],
    min_len: int = 128,
    max_len: int = 256,
) -> Optional[List[int]]:
    """
    Deterministic SHA-256 content-hash prefix selection:
    selects 50% of examples, exact length in [min_len, max_len], and deterministic pool offset.
    """
    if not prefix_pool:
        return None
    raw = ",".join(str(int(x)) for x in input_ids).encode("ascii")
    h = int(hashlib.sha256(raw).hexdigest(), 16)
    if (h % 2) == 0:
        return None
    span = max(1, max_len - min_len + 1)
    length = min_len + ((h // 2) % span)
    ids: List[int] = []
    j = (h // 512) % len(prefix_pool)
    while len(ids) < length:
        ids.extend(prefix_pool[j])
        j = (j + 1) % len(prefix_pool)
    return ids[:length]


def final_answer_strict(text: str) -> str:
    """
    Extract text after a single well-formed ``</think>`` block.
    Returns empty string if thinking tags are unclosed, duplicated, or out of order.
    """
    openings = text.count("<think>")
    closings = text.count("</think>")
    if closings == 1 and openings <= 1 and ("<think>" not in text or text.index("<think>") < text.index("</think>")):
        return text.split("</think>", 1)[1].strip()
    if openings or closings:
        return ""
    return text.strip()


def verify_complete_eos_trace(
    response_ids: List[int],
    response_text: str,
    require_think_close: bool = True,
    end_tokens: Tuple[int, ...] = QWEN_END_TOKENS,
) -> bool:
    """
    Gate training trajectories so every distilled trace terminates cleanly with EOS
    (``<|im_end|>`` or ``<|endoftext|>``) and, when thinking is enabled, contains a
    single properly closed ``</think>`` block followed by a non-empty final answer.
    """
    if not response_ids or int(response_ids[-1]) not in end_tokens:
        return False
    if require_think_close:
        if "</think>" not in response_text:
            return False
        if not final_answer_strict(response_text):
            return False
    return True


def compute_response_logits_fast(
    model,
    inputs_embeds: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
) -> List[torch.Tensor]:
    """
    Memory-efficient response-only projection:
    runs the frozen Transformer decoder backbone first and applies ``lm_head`` ONLY
    at supervised response positions (``labels[:, 1:] != -100``), avoiding materializing
    ``[B, T_ctx, V]`` logits over thousands of preceding context tokens.

    Returns a list of ``[n_resp_i, V]`` FP32 logits (one tensor per batch row),
    preserving full autograd graph back to ``inputs_embeds`` / ``past_key_values``.
    """
    decoder_name = getattr(model, "base_model_prefix", "model")
    decoder = getattr(model, decoder_name, None)
    output_head = model.get_output_embeddings()
    if decoder is None or output_head is None:
        # Fallback for non-standard architectures
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=False,
        )
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        return [shift_logits[i][shift_labels[i] != -100].float() for i in range(labels.size(0))]

    hidden = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state  # [B, T, d]

    shifted_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    out_list: List[torch.Tensor] = []
    for i in range(labels.size(0)):
        mask = shift_labels[i] != -100
        if not bool(mask.any()):
            out_list.append(shifted_hidden.new_empty((0, output_head.weight.size(0)), dtype=torch.float32))
            continue
        resp_h = shifted_hidden[i, mask]
        resp_logits = output_head(resp_h).float()
        out_list.append(resp_logits)
    return out_list


def cosine_lr_no_warmup(step_0based: int, lr_max: float, lr_min: float, total_updates: int) -> float:
    """Cosine learning-rate schedule without warmup: update 0 -> lr_max, update (N-1) -> lr_min."""
    if total_updates <= 1:
        return lr_max
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * step_0based / (total_updates - 1)))


def alignment_trace_multislot(
    prompt_len: int,
    response_ids: List[int],
    insertions: List[Tuple[int, int, str]],  # sorted list of (insert_pos_in_orig_prompt, sp_len, slot_name)
) -> Tuple[List[int], List[str]]:
    """
    Pure-CPU causal alignment verifier for single-slot and multi-slot soft-prompt insertion.
    Returns (aligned_labels, region_tags) so unit tests can assert exact label shift invariants
    before touching any GPU.
    """
    orig_labels = [-100] * prompt_len + list(response_ids)
    orig_regions = ["prompt"] * prompt_len + ["response"] * len(response_ids)
    ins_sorted = sorted(insertions, key=lambda x: x[0])
    aligned_labels: List[int] = []
    aligned_regions: List[str] = []
    prev = 0
    for pos, sp_len, name in ins_sorted:
        if not (0 <= pos <= prompt_len) or pos < prev or sp_len <= 0:
            raise ValueError(f"Invalid insertion ({pos}, {sp_len}, {name}) for prompt_len={prompt_len}")
        aligned_labels.extend(orig_labels[prev:pos])
        aligned_regions.extend(orig_regions[prev:pos])
        aligned_labels.extend([-100] * sp_len)
        aligned_regions.extend([f"sp:{name}"] * sp_len)
        prev = pos
    aligned_labels.extend(orig_labels[prev:])
    aligned_regions.extend(orig_regions[prev:])
    return aligned_labels, aligned_regions


class FrozenBackboneSoftPromptWrapper:
    """
    Non-``nn.Module`` safety wrapper for frozen-backbone soft-prompt optimization:
    callers cannot accidentally pass frozen backbone parameters to an optimizer via
    ``.parameters()``; the sole trainable tensor is exposed as ``prompt_parameter``.
    """

    def __init__(self, base_model, prompt_embeddings: torch.Tensor):
        if prompt_embeddings.ndim != 2 or prompt_embeddings.size(0) <= 0:
            raise ValueError("prompt_embeddings must have shape [L, hidden]")
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.prompt_parameter = nn.Parameter(prompt_embeddings.detach().float().clone())

    def assert_backbone_frozen(self) -> None:
        if any(p.requires_grad for p in self.base_model.parameters()):
            raise RuntimeError("Frozen backbone invariant violated: a base parameter has requires_grad=True")

