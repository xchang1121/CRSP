"""
Canonical Insertion Slot Definitions & Paper-to-Code Index
==========================================================
Paper Reference: Section 2.1, Section 2.4, Section 6, Section 7, and Section 8.

Single source of truth for the six semantic soft-prompt insertion positions used across
training, multi-expert composition, long-context evaluation, and virtual-token serving:

  - ``start``       (序列起点)  : Beginning of sequence with no preceding dialogue/context.
  - ``post_system`` (系统提示后): After system persona prompt, before multi-turn history.
  - ``pre_query``   (查询前)    : After multi-turn history, immediately before current user query.
  - ``post_rag``    (检索后)    : After retrieved RAG/tool passages, before current user query.
  - ``post_query``  (查询后)    : Immediately after current user query, before assistant generation.
  - ``long_offset`` (长偏移后)  : After >1000-token combined context, immediately before current query.
"""

from __future__ import annotations

from typing import Dict, Tuple

SLOT_START: str = "start"
SLOT_POST_SYSTEM: str = "post_system"
SLOT_PRE_QUERY: str = "pre_query"
SLOT_POST_RAG: str = "post_rag"
SLOT_POST_QUERY: str = "post_query"
SLOT_LONG_OFFSET: str = "long_offset"

CANONICAL_SLOTS: Tuple[str, ...] = (
    SLOT_START,
    SLOT_POST_SYSTEM,
    SLOT_PRE_QUERY,
    SLOT_POST_RAG,
    SLOT_POST_QUERY,
    SLOT_LONG_OFFSET,
)

NON_START_SLOTS: Tuple[str, ...] = (
    SLOT_POST_SYSTEM,
    SLOT_PRE_QUERY,
    SLOT_POST_RAG,
    SLOT_POST_QUERY,
    SLOT_LONG_OFFSET,
)

SLOT_DISPLAY_EN: Dict[str, str] = {
    SLOT_START: "Start",
    SLOT_POST_SYSTEM: "Post-System",
    SLOT_PRE_QUERY: "Pre-Query",
    SLOT_POST_RAG: "Post-RAG",
    SLOT_POST_QUERY: "Post-Query",
    SLOT_LONG_OFFSET: "Long-Offset",
}

SLOT_DISPLAY_ZH: Dict[str, str] = {
    SLOT_START: "序列起点",
    SLOT_POST_SYSTEM: "系统提示后",
    SLOT_PRE_QUERY: "查询前",
    SLOT_POST_RAG: "检索后",
    SLOT_POST_QUERY: "查询后",
    SLOT_LONG_OFFSET: "长偏移后",
}

# Maps paper sections to their primary implementation modules for programmatic lookup.
PAPER_SECTION_TO_MODULES: Dict[str, Tuple[str, ...]] = {
    "2.2_CSPD": ("src/cspd_core.py",),
    "2.3_KVPD": ("src/kv_prefix_core.py",),
    "2.4_CRSP_and_Export": (
        "src/slot_positions.py",
        "src/virtual_tokens_export.py",
        "src/exp7_context_robust_sp.py",
        "src/exp8_kv_inversion_and_tf_cr_sp.py",
        "src/exp9_multiseed_cr_grpo_and_composition.py",
    ),
    "3.2_Benchmarks": ("src/benchmarks.py", "src/domain_docs_data.py", "src/domain_docs_extended.py"),
    "4_Stage1_CSPD": ("src/exp1_reproduce_cspd.py", "src/exp1b_cspd_final.py", "src/exp2_domain_doc_cl.py", "src/exp2b_doc_nll_cl.py"),
    "5_Stage2_KVPD": ("src/exp3_cspd_acceleration.py", "src/exp4_doc_cl_breakthroughs.py", "src/exp5_doc_cl_stress_tests.py"),
    "6_Stage3_CRSP": (
        "src/exp6_teacherfree_input_sp.py",
        "src/exp7_context_robust_sp.py",
        "src/exp8_kv_inversion_and_tf_cr_sp.py",
        "src/exp9_multiseed_cr_grpo_and_composition.py",
        "src/exp9b_postrl_crsp_and_composition.py",
    ),
    "7_LongContext_Order_Serving": (
        "src/exp10_long_context_attenuation_and_composition.py",
        "src/exp10b_longctx_e2e_and_doc_kvpd.py",
        "src/exp10c_fast_4096_e2e.py",
        "src/exp11_injection_order_ablation.py",
        "src/exp12_official_cspd_and_eos_upgrade.py",
        "tests/test_alignment_and_core.py",
    ),
    "Figures_1_to_15": ("src/plot_paper_style_figures.py",),
}


def validate_slot_name(slot: str) -> str:
    """Validate and return a canonical semantic slot name."""
    if slot not in CANONICAL_SLOTS:
        raise ValueError(f"Unknown slot {slot!r}; expected one of {CANONICAL_SLOTS}")
    return slot
