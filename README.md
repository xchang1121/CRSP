# Modular Soft-Prompt Distillation for Frozen LLMs: CSPD, KV-PD, and CRSP

Code, experimental logs, quantitative results, publication figures, CPU contract unit tests, and full technical paper for a three-stage modular soft-prompt distillation framework on frozen large language models:

1. **Stage I — Input-Space Composable Soft-Prompt Distillation (`CSPD`)**: Independent distillation of RL experts into frozen-backbone input soft prompts with prefix-robust training, continual prepend/append addition and leave-one-out removal, out-of-domain retention, serving efficiency, and response-side attention mechanism analysis.
2. **Stage II — Layerwise KV-Space Prompt Distillation (`KV-PD`)**: Real-text forward KV initialization injected directly across all Transformer layers via `past_key_values` for 2.71× faster distillation and forgetting-free 5-domain private-document continual learning (`KV-PD + Micro-RAG` auto-routing and 4-token KV incremental patches).
3. **Stage III — Context-Robust Arbitrary-Slot Soft Prompts (`CRSP`)**: Arbitrary-slot insertion across six semantic deployment slots (`start`, `post_system`, `pre_query`, `post_rag`, `post_query`, `long_offset`; canonicalized in [`src/slot_positions.py`](src/slot_positions.py)) via multi-slot context randomization, cross-context output consistency ($`\mathcal{L}_{\text{inv}}`$), and symmetric non-target output preservation ($`\mathcal{L}_{\text{null}}`$), with two construction algorithms:
   - **Algorithm 1 (`KV-Inv-CRSP`)**: Pure output-distribution functional inversion from an archived prefix `KV-PD` module without needing the original RL teacher.
   - **Algorithm 2 (`Two-Stage TF-CRSP`)**: Teacher-free construction combining base-KL-constrained multi-slot self-bootstrap warm-start with multi-slot online GRPO.
4. **Long-Context Attenuation, Multi-Expert Composition, Injection-Order & Serving Verification (0 → 8192 Tokens)**: Systematic evaluation across non-repeating `Salesforce/wikitext` + multi-turn history contexts comparing start-fixed `CSPD` (including reference step-200 checkpoints), start-fixed `KV-PD`, and sliding-slot `CRSP` on single-expert attenuation, 9-layer attention-mass attribution, 36-layer cumulative causal direct-path ablation, complete-trajectory EOS distillation, dual-expert split/adjacent swap, three-expert all-6-permutations (`{Doc, Math, IF}`) across `L=0` and `L=4096`, response-only `lm_head` projection, and arbitrary-slot virtual-token `safetensors` export with tied-embedding `-1e9` output masking.

## Full Technical Paper & Figures

- **Paper (Chinese)**: [docs/paper_zh.md](docs/paper_zh.md) (10-section paper with 15 embedded figures, Tables 1a–1d & 2–17, Algorithms 1 & 2, Related Work, 72 hyperlinked references, and direct code links)
- **Publication Figures**: [docs/figures/](docs/figures/) (`pfig01`–`pfig15`, 220 DPI; reproduced by [`src/plot_paper_style_figures.py`](src/plot_paper_style_figures.py))

## Paper-to-Code Index

| Paper Section / Artifact | Primary Module(s) | Key Components & Verified Artifacts |
| :--- | :--- | :--- |
| **Sec. 2.2 (Eqs. 2–6), Sec. 7.4** | [`src/cspd_core.py`](src/cspd_core.py) | Top-$`K`$ forward KL (`RB-CE`), complete-trajectory EOS filtering, SHA-256 deterministic prefix slicing, response-only `lm_head` projection, Muon optimizer |
| **Sec. 2.3 (Eqs. 7–10)** | [`src/kv_prefix_core.py`](src/kv_prefix_core.py) | Real-text forward KV initialization, layerwise `ReadOnlyPrefixCache`, multi-expert KV concat & 4-token incremental Delta patch |
| **Sec. 2.1 & 2.4 (Slots & Export)** | [`src/slot_positions.py`](src/slot_positions.py), [`src/virtual_tokens_export.py`](src/virtual_tokens_export.py) | Canonical 6 semantic insertion slots (`start`, `post_system`, `pre_query`, `post_rag`, `post_query`, `long_offset`), byte-exact `safetensors` virtual-token export & mandatory `-1e9` tied-embedding mask |
| **Sec. 3.2 (Benchmarks & Data)** | [`src/benchmarks.py`](src/benchmarks.py), [`src/domain_docs_data.py`](src/domain_docs_data.py), [`src/domain_docs_extended.py`](src/domain_docs_extended.py) | MATH / IFEval / IFBench evaluators and 5-domain unseen synthetic private-document benchmarks (`HeliosPay v4.2/v5.0`, `NVX-418`, `Litho-X9`, `GridPulse-9`, `Meridian-C3`) |
| **Sec. 4 (Tables 1–3, Figs. 2–5)** | [`src/exp1_reproduce_cspd.py`](src/exp1_reproduce_cspd.py), [`src/exp1b_cspd_final.py`](src/exp1b_cspd_final.py), [`src/exp2_domain_doc_cl.py`](src/exp2_domain_doc_cl.py), [`src/exp2b_doc_nll_cl.py`](src/exp2b_doc_nll_cl.py) | Stage I `CSPD` distillation, response-side attention mechanism ablation, prefix-robust order gap, and private-document capacity limit |
| **Sec. 5 (Tables 4–5, Figs. 6–8)** | [`src/exp3_cspd_acceleration.py`](src/exp3_cspd_acceleration.py), [`src/exp4_doc_cl_breakthroughs.py`](src/exp4_doc_cl_breakthroughs.py), [`src/exp5_doc_cl_stress_tests.py`](src/exp5_doc_cl_stress_tests.py) | Stage II `KV-PD` 2.71× acceleration, 5-domain auto-routed `KV-PD + Micro-RAG`, length scaling, and 4-token version upgrade (`v4.2 -> v5.0`) |
| **Sec. 6.1 (Alg. 1, Table 6, Fig. 9)** | [`src/exp6_teacherfree_input_sp.py`](src/exp6_teacherfree_input_sp.py), [`src/exp7_context_robust_sp.py`](src/exp7_context_robust_sp.py), [`src/exp8_kv_inversion_and_tf_cr_sp.py`](src/exp8_kv_inversion_and_tf_cr_sp.py) | 6-slot `CRSP` benchmark, consistency loss $`\mathcal{L}_{\text{inv}}`$, pure output-distribution KV inversion (Algorithm 1), and teacher-free baselines |
| **Sec. 6.2–6.3 (Alg. 2, Tables 7–9, Figs. 10–11)** | [`src/exp9_multiseed_cr_grpo_and_composition.py`](src/exp9_multiseed_cr_grpo_and_composition.py), [`src/exp9b_postrl_crsp_and_composition.py`](src/exp9b_postrl_crsp_and_composition.py) | 3-seed `IFEval` stability, Two-Stage `TF-CRSP` (Algorithm 2), scaled `IFBench` ($`N=64`$), adversarial `post_rag` noise, 3-expert split-slot composition, and private-doc slot capacity |
| **Sec. 7.1–7.3 (Tables 10, 12, 14–15, Figs. 12–13)** | [`src/exp10_long_context_attenuation_and_composition.py`](src/exp10_long_context_attenuation_and_composition.py), [`src/exp10b_longctx_e2e_and_doc_kvpd.py`](src/exp10b_longctx_e2e_and_doc_kvpd.py), [`src/exp10c_fast_4096_e2e.py`](src/exp10c_fast_4096_e2e.py) | Long-context attenuation (`0 -> 8192`), 9-layer attention-mass attribution, dual-expert long-context composition, `L=4096` end-to-end benchmarks, and private-doc `KV-PD` (`0 -> 4096`) |
| **Sec. 7.2 (Tables 12–14, Fig. 14)** | [`src/exp11_injection_order_ablation.py`](src/exp11_injection_order_ablation.py) | Dual-expert split-slot & same-slot swap (`0 -> 8192`), 3-expert all-6-permutations (`{Doc, Math, IF}`) at `L=0` & `L=4096`, and end-to-end order-swap verification |
| **Sec. 7.1–7.4 (Tables 6, 10–12, 14, 16, Fig. 15)** | [`src/exp12_official_cspd_and_eos_upgrade.py`](src/exp12_official_cspd_and_eos_upgrade.py) | Reference step-200 `CSPD` 6-slot & `0 -> 8192` audit, complete-trajectory EOS upgrade, 36-layer cumulative causal ablation, response-only `lm_head` benchmark, and virtual-token export hazard test |
| **Figures 1–15** | [`src/plot_paper_style_figures.py`](src/plot_paper_style_figures.py) | Reproduces all 15 publication figures (`docs/figures/pfig01..pfig15.png`) directly from [`results/`](results/) |
| **CPU Unit Tests (Sec. 7.4, Table 16)** | [`tests/test_alignment_and_core.py`](tests/test_alignment_and_core.py) | 7 pure-CPU contract tests: canonical slot index, multi-slot causal alignment, complete-EOS gating, SHA-256 prefix slicing, byte-exact `safetensors` export, RadixAttention shared-prefix invariance |

## Quick CPU Verification

```bash
python tests/test_alignment_and_core.py
```
