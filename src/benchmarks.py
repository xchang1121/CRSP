"""
Paper Reference: Section 3.2 (MATH, IFEval, IFBench Evaluators & Thinking-Block Parsers).
Verifiers & Batched Soft-Prompt Generation for CSPD Experiments:
- Deterministic Math answer extraction & equivalence checking (nested \boxed{} + sympy)
- Deterministic IFEval constraint verifiers (strips Qwen3 <think>...</think> before verifying)
- Left-padded batched generation with arbitrary concatenated soft prompts [P_1; ...; P_k]
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from cspd_core import ChatTemplateTokens, build_soft_prompt_inputs


# ---------------------------------------------------------------------------
# 1. Strip Qwen3 Thinking Block
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning trace to obtain the final visible answer."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "<think>" in text:
        # Unclosed thinking (hit max_new_tokens) -> return empty or tail
        return ""
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Math Answer Extractor & Verifier
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \boxed{...} supporting arbitrarily nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\fbox{")
        if idx == -1:
            return None
        start = idx + 6
    else:
        start = idx + 7

    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def normalize_math_str(s: str) -> str:
    s = s.strip()
    s = s.replace("\\$", "").replace("$", "").replace(",", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\!", "", s)
    s = re.sub(r"\\,", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s


def extract_math_answer(text: str) -> Optional[str]:
    vis = strip_thinking(text)
    if not vis:
        vis = text
    ans = extract_boxed(vis)
    if ans is not None:
        return ans
    # Fallback: last "#### <ans>" or "The answer is <ans>"
    m = re.findall(r"####\s*([^\n]+)", vis)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer is|answer is)\s*[:$]?\s*([^\n.$]+)", vis, re.IGNORECASE)
    if m:
        return m[-1].strip()
    return None


def math_answers_equal(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    p = normalize_math_str(pred)
    g = normalize_math_str(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # Try numeric comparison
    try:
        def to_float(x: str) -> float:
            m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", x)
            if m:
                sign = -1.0 if m.group(1) == "-" else 1.0
                return sign * float(m.group(2)) / float(m.group(3))
            if "/" in x and re.fullmatch(r"-?\d+/-?\d+", x):
                a, b = x.split("/")
                return float(a) / float(b)
            return float(x)

        pf, gf = to_float(p), to_float(g)
        if abs(pf - gf) <= 1e-4 * max(1.0, abs(gf)):
            return True
    except Exception:
        pass
    # Try sympy if installed
    try:
        import sympy as sp
        from sympy.parsing.latex import parse_latex

        def to_sym(x: str):
            try:
                return parse_latex(x)
            except Exception:
                return sp.sympify(x.replace("^", "**"))

        diff = sp.simplify(to_sym(p) - to_sym(g))
        if diff == 0:
            return True
    except Exception:
        pass
    return False


def verify_math_response(response_text: str, gold_answer: str) -> bool:
    pred = extract_math_answer(response_text)
    return math_answers_equal(pred, gold_answer)


# ---------------------------------------------------------------------------
# 3. Deterministic IFEval Constraint Verifiers
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_sentences(text: str) -> int:
    sents = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", text.strip()) if s.strip()]
    return len(sents)


def _compare_relation(val: int, threshold: int, relation: str) -> bool:
    rel = (relation or "").lower().strip()
    if rel == "at least":
        return val >= threshold
    if rel == "less than":
        return val < threshold
    if rel == "at most":
        return val <= threshold
    if rel == "exact" or rel == "equal to":
        return val == threshold
    return val >= threshold


def check_ifeval_instruction(response: str, inst_id: str, kwargs: Dict[str, Any]) -> bool:
    """Verify a single google/IFEval constraint deterministically on visible response."""
    text = strip_thinking(response)
    if not text:
        return False
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None}

    if inst_id == "keywords:existence":
        keywords = kw.get("keywords", [])
        low = text.lower()
        return all(k.lower() in low for k in keywords)

    if inst_id == "keywords:forbidden_words":
        fwords = kw.get("forbidden_words", [])
        low = text.lower()
        return all(re.search(r"\b" + re.escape(w.lower()) + r"\b", low) is None for w in fwords)

    if inst_id == "keywords:frequency":
        word = kw.get("keyword", "")
        freq = int(kw.get("frequency", 1))
        rel = kw.get("relation", "at least")
        cnt = len(re.findall(r"\b" + re.escape(word.lower()) + r"\b", text.lower()))
        return _compare_relation(cnt, freq, rel)

    if inst_id == "keywords:letter_frequency":
        letter = kw.get("letter", "").lower()
        freq = int(kw.get("let_frequency", 1))
        rel = kw.get("let_relation", "at least")
        cnt = text.lower().count(letter)
        return _compare_relation(cnt, freq, rel)

    if inst_id == "length_constraints:number_words":
        n = int(kw.get("num_words", 0))
        rel = kw.get("relation", "at least")
        return _compare_relation(_count_words(text), n, rel)

    if inst_id == "length_constraints:number_sentences":
        n = int(kw.get("num_sentences", 0))
        rel = kw.get("relation", "at least")
        return _compare_relation(_count_sentences(text), n, rel)

    if inst_id == "length_constraints:number_paragraphs":
        n = int(kw.get("num_paragraphs", 1))
        paras = [p.strip() for p in re.split(r"\s*\*\*\*\s*", text) if p.strip()]
        return len(paras) == n

    if inst_id == "length_constraints:nth_paragraph_first_word":
        n = int(kw.get("num_paragraphs", 1))
        nth = int(kw.get("nth_paragraph", 1))
        first_word = str(kw.get("first_word", "")).lower().strip()
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paras) != n or nth > len(paras):
            return False
        words = re.findall(r"\b\w+\b", paras[nth - 1].lower())
        return bool(words) and words[0] == first_word

    if inst_id == "detectable_format:number_bullet_lists":
        n = int(kw.get("num_bullets", 1))
        bullets = re.findall(r"(?m)^\s*(?:\*|-)\s+\S", text)
        return len(bullets) == n

    if inst_id == "detectable_format:json_format":
        cleaned = text.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
        if m:
            cleaned = m.group(1).strip()
        try:
            json.loads(cleaned)
            return True
        except Exception:
            return False

    if inst_id == "detectable_format:title":
        return bool(re.search(r"<<[^<>\n]+>>", text))

    if inst_id == "detectable_format:number_highlighted_sections":
        n = int(kw.get("num_highlights", 1))
        hl = re.findall(r"\*[^*\n]+\*", text)
        return len(hl) >= n

    if inst_id == "detectable_format:multiple_sections":
        splitter = kw.get("section_spliter", "Section")
        n = int(kw.get("num_sections", 1))
        matches = re.findall(re.escape(splitter) + r"\s*\d+", text, re.IGNORECASE)
        return len(matches) >= n

    if inst_id == "detectable_format:constrained_response":
        opts = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
        return any(o in text for o in opts)

    if inst_id == "detectable_content:number_placeholders":
        n = int(kw.get("num_placeholders", 1))
        phs = re.findall(r"\[[^\[\]\n]+\]", text)
        return len(phs) >= n

    if inst_id == "detectable_content:postscript":
        marker = kw.get("postscript_marker", "P.S.")
        return marker.lower() in text.lower()

    if inst_id == "punctuation:no_comma":
        return "," not in text

    if inst_id == "change_case:english_capital":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    if inst_id == "change_case:english_lowercase":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)

    if inst_id == "change_case:capital_word_frequency":
        n = int(kw.get("capital_frequency", 1))
        rel = kw.get("capital_relation", "at least")
        words = re.findall(r"\b[A-Z]{2,}\b", text)
        return _compare_relation(len(words), n, rel)

    if inst_id == "startend:end_checker":
        end_phrase = str(kw.get("end_phrase", "")).strip()
        return text.rstrip().endswith(end_phrase)

    if inst_id == "startend:quotation":
        s = text.strip()
        return len(s) >= 2 and s[0] == '"' and s[-1] == '"'

    if inst_id == "combination:two_responses":
        parts = [p.strip() for p in text.split("******") if p.strip()]
        return len(parts) == 2

    if inst_id == "combination:repeat_prompt":
        p_repeat = str(kw.get("prompt_to_repeat", "")).strip()
        return text.lower().startswith(p_repeat.lower())

    # Unknown / language-specific constraint (skip by filtering dataset to supported IDs)
    return False


SUPPORTED_IFEVAL_IDS = {
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:letter_frequency",
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:nth_paragraph_first_word",
    "detectable_format:number_bullet_lists",
    "detectable_format:json_format",
    "detectable_format:title",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:constrained_response",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "punctuation:no_comma",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "change_case:capital_word_frequency",
    "startend:end_checker",
    "startend:quotation",
    "combination:two_responses",
    "combination:repeat_prompt",
}


def verify_ifeval_response(
    response: str, instruction_id_list: List[str], kwargs_list: List[Dict[str, Any]]
) -> Tuple[bool, float]:
    """Return (strict_all_pass, instruction_level_pass_rate)."""
    if not instruction_id_list:
        return True, 1.0
    results = [
        check_ifeval_instruction(response, iid, kw)
        for iid, kw in zip(instruction_id_list, kwargs_list)
    ]
    return all(results), sum(results) / len(results)


# ---------------------------------------------------------------------------
# 4. Left-Padded Batched Generation with Concatenated Soft Prompts
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_generate_with_soft_prompts(
    model,
    tokenizer,
    tmpl: ChatTemplateTokens,
    questions: List[str],
    soft_prompts: List[torch.Tensor],
    system_texts: Optional[List[str]] = None,
    batch_size: int = 16,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    enable_thinking: bool = True,
) -> List[str]:
    """
    Fast batched generation on GPU with left-padded inputs_embeds and arbitrary
    concatenated soft prompts [P_1; ...; P_k].
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_vec = embed_layer(torch.tensor([pad_id], device=device)).to(dtype)  # [1, d]

    no_think_emb = None
    if not enable_thinking:
        nt_ids = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
        no_think_emb = embed_layer(torch.tensor(nt_ids, device=device)).to(dtype)  # [L_nt, d]

    all_outputs: List[str] = []
    if system_texts is None:
        system_texts = [""] * len(questions)

    for start in range(0, len(questions), batch_size):
        batch_q = questions[start : start + batch_size]
        batch_s = system_texts[start : start + batch_size]

        seq_embeds: List[torch.Tensor] = []
        for q, s in zip(batch_q, batch_s):
            q_ids = tokenizer.encode(q, add_special_tokens=False)
            s_ids = tokenizer.encode(s, add_special_tokens=False) if s else None
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
            e2d = emb.squeeze(0)  # [T_i, d]
            if no_think_emb is not None:
                e2d = torch.cat([e2d, no_think_emb], dim=0)
            seq_embeds.append(e2d)

        max_len = max(e.size(0) for e in seq_embeds)
        bs = len(seq_embeds)
        d = seq_embeds[0].size(1)

        padded = pad_vec.expand(bs, max_len, d).clone()
        attn_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for i, e in enumerate(seq_embeds):
            L = e.size(0)
            padded[i, max_len - L :, :] = e
            attn_mask[i, max_len - L :] = 1

        gen_kwargs = dict(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            gen_kwargs.update(do_sample=False)

        out_ids = model.generate(**gen_kwargs)
        for row in out_ids:
            all_outputs.append(tokenizer.decode(row, skip_special_tokens=True))
        print(f"    gen batch {min(start + batch_size, len(questions))}/{len(questions)} done (out_len={out_ids.size(1)})", flush=True)

    return all_outputs
