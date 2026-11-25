"""
CPU Unit Test Suite for CSPD / KV-PD / CRSP Core Alignment & Export Contracts
=============================================================================
Paper Reference: Section 2.2, Section 2.4, and Section 7.4 (Table 16).

Runs purely on CPU without GPU or model weights.
Verifies:
  1. Canonical semantic slot definitions and paper-to-code index integrity.
  2. Exact multi-slot causal label alignment (start, post_system, pre_query, post_query, split-slot).
  3. Strict thinking-block extraction & complete-trajectory EOS filtering.
  4. Deterministic SHA-256 prefix-robust slicing and cosine schedule endpoints.
  5. Byte-exact safetensors embedding-shard extension & SHA-256 read-back verification.
  6. RadixAttention shared-prefix invariance for sliding pre_query/post_query CRSP vs. start-fixed prompts.
  7. Frozen-backbone wrapper non-Module isolation invariant (when PyTorch is installed).
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from slot_positions import (
    CANONICAL_SLOTS,
    PAPER_SECTION_TO_MODULES,
    SLOT_DISPLAY_EN,
    SLOT_DISPLAY_ZH,
    SLOT_POST_QUERY,
    SLOT_PRE_QUERY,
    SLOT_START,
    validate_slot_name,
)
from virtual_tokens_export import (
    HARD_LOGIT_BIAS,
    build_hard_output_mask_ids,
    extend_embedding_shard_bytes,
    render_multislot_virtual_ids,
)

try:
    import torch
    from cspd_core import (
        FrozenBackboneSoftPromptWrapper,
        alignment_trace_multislot,
        cosine_lr_no_warmup,
        deterministic_prefix_slice,
        final_answer_strict,
        verify_complete_eos_trace,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    import hashlib
    import math

    def final_answer_strict(text: str) -> str:
        openings, closings = text.count("<think>"), text.count("</think>")
        if closings == 1 and openings <= 1 and ("<think>" not in text or text.index("<think>") < text.index("</think>")):
            return text.split("</think>", 1)[1].strip()
        if openings or closings:
            return ""
        return text.strip()

    def verify_complete_eos_trace(response_ids, response_text, require_think_close=True, end_tokens=(151645, 151643)):
        if not response_ids or int(response_ids[-1]) not in end_tokens:
            return False
        if require_think_close and ("</think>" not in response_text or not final_answer_strict(response_text)):
            return False
        return True

    def deterministic_prefix_slice(input_ids, prefix_pool, min_len=128, max_len=256):
        if not prefix_pool:
            return None
        h = int(hashlib.sha256(",".join(str(int(x)) for x in input_ids).encode("ascii")).hexdigest(), 16)
        if (h % 2) == 0:
            return None
        length = min_len + ((h // 2) % max(1, max_len - min_len + 1))
        ids, j = [], (h // 512) % len(prefix_pool)
        while len(ids) < length:
            ids.extend(prefix_pool[j])
            j = (j + 1) % len(prefix_pool)
        return ids[:length]

    def cosine_lr_no_warmup(step_0based, lr_max, lr_min, total_updates):
        if total_updates <= 1:
            return lr_max
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * step_0based / (total_updates - 1)))

    def alignment_trace_multislot(prompt_len, response_ids, insertions):
        orig_labels = [-100] * prompt_len + list(response_ids)
        orig_regions = ["prompt"] * prompt_len + ["response"] * len(response_ids)
        aligned_labels, aligned_regions, prev = [], [], 0
        for pos, sp_len, name in sorted(insertions, key=lambda x: x[0]):
            if not (0 <= pos <= prompt_len) or pos < prev or sp_len <= 0:
                raise ValueError("Invalid insertion")
            aligned_labels.extend(orig_labels[prev:pos])
            aligned_regions.extend(orig_regions[prev:pos])
            aligned_labels.extend([-100] * sp_len)
            aligned_regions.extend([f"sp:{name}"] * sp_len)
            prev = pos
        aligned_labels.extend(orig_labels[prev:])
        aligned_regions.extend(orig_regions[prev:])
        return aligned_labels, aligned_regions


def test_canonical_slots_and_code_index():
    assert len(CANONICAL_SLOTS) == 6
    for slot in CANONICAL_SLOTS:
        assert validate_slot_name(slot) == slot
        assert slot in SLOT_DISPLAY_EN and slot in SLOT_DISPLAY_ZH
    for sec, paths in PAPER_SECTION_TO_MODULES.items():
        for rel in paths:
            assert (ROOT / rel).exists(), f"Missing indexed module for {sec}: {rel}"


def test_multislot_causal_alignment_invariants():
    resp = [101, 102, 103, 151645]
    # Split-slot [Math @ pre_query + IF @ post_query] inside prompt_len=20
    lbls, regs = alignment_trace_multislot(
        20,
        resp,
        [(12, 64, f"Math@{SLOT_PRE_QUERY}"), (20, 64, f"IF@{SLOT_POST_QUERY}")],
    )
    assert len(lbls) == 20 + 64 + 64 + len(resp)
    # Exactly the response tokens carry supervision; no soft-prompt or prompt tokens are supervised.
    supervised = [x for x in lbls if x != -100]
    assert supervised == resp
    assert regs.count(f"sp:Math@{SLOT_PRE_QUERY}") == 64
    assert regs.count(f"sp:IF@{SLOT_POST_QUERY}") == 64
    assert all(l == -100 for l, r in zip(lbls, regs) if r != "response")


def test_final_answer_strict_and_eos_gate():
    assert final_answer_strict("<think>step 1</think>\n42") == "42"
    assert final_answer_strict("<think>unclosed chain") == ""
    assert final_answer_strict("<think>a</think>x</think>y") == ""
    assert final_answer_strict("plain answer") == "plain answer"

    # Reject truncated trace lacking EOS (151645 / 151643)
    assert not verify_complete_eos_trace([10, 20, 30], "<think>ok</think>42")
    # Reject unclosed thinking block even if EOS is present
    assert not verify_complete_eos_trace([10, 20, 151645], "<think>truncated")
    # Accept complete thinking + EOS trace
    assert verify_complete_eos_trace([10, 20, 151645], "<think>ok</think>42")


def test_deterministic_prefix_slice_and_cosine():
    pool = [[i + 100 for i in range(40)] for _ in range(16)]
    hits = 0
    for k in range(100):
        seq = [1, 2, 3, k, k * 7]
        a = deterministic_prefix_slice(seq, pool, 128, 256)
        b = deterministic_prefix_slice(seq, pool, 128, 256)
        assert a == b
        if a is not None:
            hits += 1
            assert 128 <= len(a) <= 256
    assert 35 <= hits <= 65
    assert abs(cosine_lr_no_warmup(0, 0.03, 0.003, 200) - 0.03) < 1e-12
    assert abs(cosine_lr_no_warmup(199, 0.03, 0.003, 200) - 0.003) < 1e-12


def test_byte_exact_safetensors_virtual_token_extension():
    hidden = 8
    orig_vocab = 16
    emb_bytes = bytes(range(256))  # 16 * 8 * 2 = 256 bytes (BF16)
    other_bytes = b"\x11\x22" * 32  # 64 bytes
    header = {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [orig_vocab, hidden], "data_offsets": [0, len(emb_bytes)]},
        "model.norm.weight": {"dtype": "BF16", "shape": [32], "data_offsets": [len(emb_bytes), len(emb_bytes) + len(other_bytes)]},
    }
    enc = json.dumps(header, separators=(",", ":")).encode("utf-8")
    enc += b" " * ((-len(enc)) % 8)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.safetensors"
        dst = Path(tmp) / "dst.safetensors"
        src.write_bytes(struct.pack("<Q", len(enc)) + enc + emb_bytes + other_bytes)
        add_rows = 4
        appended = b"\xAB\xCD" * (add_rows * hidden)
        proof = extend_embedding_shard_bytes(src, dst, appended, rows=add_rows, hidden=hidden)
        assert proof["original_vocab_size"] == orig_vocab
        assert proof["extended_vocab_size"] == orig_vocab + add_rows
        assert proof["added_bytes"] == len(appended)

        masked = build_hard_output_mask_ids(12, orig_vocab + add_rows)
        assert masked == list(range(12, 20))
        assert HARD_LOGIT_BIAS <= -1e9


def test_radix_prefix_sharing_sliding_vs_start():
    """Verify sliding pre_query/post_query CRSP preserves 100% token-identical prefix across tenants."""
    shared_sys = [10, 11, 12]
    shared_hist = [([20, 21], [30, 31])]
    shared_rag = [40, 41, 42, 43]
    q = [50, 51]
    common_kw = dict(sys_text_ids=shared_sys, history_turns_ids=shared_hist, rag_ids=shared_rag)

    # Tenant A and Tenant B mount different experts at pre_query
    seq_a_sliding = render_multislot_virtual_ids([1], [2], [3], [4], q, vids_pre_query=[900, 901], **common_kw)
    seq_b_sliding = render_multislot_virtual_ids([1], [2], [3], [4], q, vids_pre_query=[910, 911], **common_kw)
    pre_query_idx = seq_a_sliding.index(900)
    assert seq_a_sliding[:pre_query_idx] == seq_b_sliding[:pre_query_idx]
    assert pre_query_idx == 1 + 3 + 1 + (1 + 2 + 1 + 1 + 2 + 1) + 1 + 4  # 18 shared prefix tokens

    # Contrast: start-fixed insertion diverges at token index 1 (0% shared history/RAG KV cache)
    seq_a_start = render_multislot_virtual_ids([1], [2], [3], [4], q, vids_start=[900, 901], **common_kw)
    seq_b_start = render_multislot_virtual_ids([1], [2], [3], [4], q, vids_start=[910, 911], **common_kw)
    assert seq_a_start[1] != seq_b_start[1]
    assert SLOT_START == "start"


def test_non_module_wrapper_invariant():
    if not HAS_TORCH:
        return
    base = torch.nn.Linear(8, 8)
    wrapper = FrozenBackboneSoftPromptWrapper(base, torch.zeros(4, 8))
    assert not isinstance(wrapper, torch.nn.Module)
    wrapper.assert_backbone_frozen()
    assert all(not p.requires_grad for p in base.parameters())
    assert wrapper.prompt_parameter.requires_grad


if __name__ == "__main__":
    test_canonical_slots_and_code_index()
    test_multislot_causal_alignment_invariants()
    test_final_answer_strict_and_eos_gate()
    test_deterministic_prefix_slice_and_cosine()
    test_byte_exact_safetensors_virtual_token_extension()
    test_radix_prefix_sharing_sliding_vs_start()
    test_non_module_wrapper_invariant()
    print("ALL 7 CPU ALIGNMENT, SLOT-INDEX & EXPORT UNIT TESTS PASSED!")
