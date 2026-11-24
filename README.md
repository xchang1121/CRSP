# Modular Soft-Prompt Distillation for Frozen LLMs: CSPD, KV-PD, and CRSP

Code, experimental logs, quantitative results, publication figures, CPU contract unit tests, and full technical paper for a three-stage modular soft-prompt distillation framework on frozen large language models:

1. **Stage I — Input-Space Composable Soft-Prompt Distillation (`CSPD`)**: Independent distillation of RL experts into frozen-backbone input soft prompts with prefix-robust training, continual prepend/append addition and leave-one-out removal, out-of-domain retention, serving efficiency, and response-side attention mechanism analysis.
2. **Stage II — Layerwise KV-Space Prompt Distillation (`KV-PD`)**: Real-text forward KV initialization injected directly across all Transformer layers via `past_key_values` for 2.71× faster distillation and zero-forgetting 5-domain private-document continual learning (`KV-PD + Micro-RAG` auto-routing and 4-token KV Delta version patches).
3. **Stage III — Context-Robust Arbitrary-Slot Soft Prompts (`CRSP`)**: Arbitrary-slot insertion across six deployment contexts (`P0`–`P5`) via multi-slot context randomization, cross-context output consistency ($`\mathcal{L}_{\text{inv}}`$), and symmetric non-target output preservation ($`\mathcal{L}_{\text{null}}`$), with two construction algorithms:
   - **Algorithm 1 (`KV-Inv-CRSP`)**: Pure output-distribution functional inversion from an archived prefix `KV-PD` module without needing the original RL teacher.
   - **Algorithm 2 (`Two-Stage TF-CRSP`)**: Teacher-free construction combining base-KL-constrained multi-slot self-bootstrap warm-start with multi-slot online GRPO.
4. **Long-Context Attenuation, Multi-Expert Composition, Injection-Order & Serving Verification (0 → 8192 Tokens)**: Systematic evaluation across non-repeating `Salesforce/wikitext` + multi-turn history contexts comparing fixed-`P0` `CSPD` (including reference step-200 checkpoints), fixed-`P0` `KV-PD`, and sliding-slot `CRSP` on single-expert attenuation, 9-layer attention-mass attribution, 36-layer cumulative causal direct-path ablation, complete closed-think EOS trajectory distillation, dual-expert split/adjacent swap (`P0`/`P2`/`P4`), three-expert all-6-permutations (`{Doc, Math, IF}`) across `L=0` & `L=4096`, response-only `lm_head` projection, and arbitrary-slot virtual-token `safetensors` export with tied-embedding `-1e9` output masking.

## Full Technical Paper & Figures

- **Paper (Chinese)**: [docs/paper_zh.md](docs/paper_zh.md) (complete 11-section paper with 15 embedded figures, 22 tables, Algorithm 1 & 2, Related Work, and 69 references)
- **Publication Figures**: [docs/figures/](docs/figures/) (`pfig01`–`pfig15`, 220 DPI)

## Repository Layout

| Directory / File | Description |
| :--- | :--- |
| `src/cspd_core.py`, `src/benchmarks.py`, `src/exp1*.py` | Input-space `CSPD` distillation, Top-$`K`$ forward KL alignment, response-only `lm_head` projection, complete-EOS gating, SHA-256 prefix slicing, and attention path ablation |
| `src/virtual_tokens_export.py` | Byte-exact `safetensors` virtual-token exporter for arbitrary-slot (`P0`–`P4`) serving with mandatory tied-embedding `-1e9` output masking |
| `src/domain_docs_data.py`, `src/exp2*.py` | Zero-leakage synthetic domain document continual learning benchmarks (`HeliosPay`, `NVX-418`, `Litho-X9`) |
| `src/kv_prefix_core.py`, `src/domain_docs_extended.py`, `src/exp3*.py`, `src/exp4*.py`, `src/exp5*.py` | Layerwise `KV-PD` acceleration, 5-domain auto-routed `KV-PD + Micro-RAG` continual learning, and 4-token Delta upgrade |
| `src/exp6_teacherfree_input_sp.py` | Single-slot teacher-free input-space soft prompt baselines (RFT, iRFT, DPO, GRPO, KV2SP) |
| `src/exp7_context_robust_sp.py` | Six-slot (`P0`–`P5`) `CRSP` benchmark, positional drift evaluation, and $`\mathcal{L}_{\text{inv}}`$ ablation |
| `src/exp8_kv_inversion_and_tf_cr_sp.py` | Geometric vs. pure output-distribution KV inversion (Algorithm 1) and multi-slot teacher-free pathways |
| `src/exp9_multiseed_cr_grpo_and_composition.py`, `src/exp9b_postrl_crsp_and_composition.py` | 3-seed IFEval stability, Two-Stage TF-CRSP (Algorithm 2), scaled IFBench ($`N=64`$), adversarial `P3` Post-RAG, 3-expert split-slot composition, and zero-leakage doc slot capacity |
| `src/exp10_long_context_attenuation_and_composition.py`, `src/exp10b_longctx_e2e_and_doc_kvpd.py`, `src/exp10c_fast_4096_e2e.py` | Long-context attenuation (`0 -> 8192`), 9-layer attention-mass attribution, dual-expert long-context composition, 4096-token end-to-end `IFEval`/`IFBench`/`MATH`, and zero-leakage doc `KV-PD` (`0 -> 4096`) |
| `src/exp11_injection_order_ablation.py` | Dual-expert split-slot & same-slot `P2`/`P4` swap (`0 -> 8192`), 3-expert all-6-permutations (`{Doc, Math, IF}`) at `L=0` & `L=4096`, and end-to-end order-swap verification (`MATH32`, `IFBench32`, `IFEval48`) |
| `src/exp12_official_cspd_and_eos_upgrade.py` | Reference step-200 `CSPD` 6-slot & `0 -> 8192` audit, complete-EOS closed-think trajectory upgrade, 36-layer cumulative causal pathway ablation, response-only `lm_head` VRAM/latency benchmark, and virtual-token export + tied-embedding logit-hazard verification |
| `src/plot_paper_style_figures.py` | Reproduces all 15 publication figures (`pfig01`–`pfig15`) |
| `tests/test_alignment_and_core.py` | Pure-CPU unit test suite verifying multi-slot causal label alignment, strict `</think>`/EOS gating, deterministic SHA-256 prefix slicing, byte-exact `safetensors` virtual-token export, and RadixAttention shared-prefix invariance |
| `results/` | Complete structured JSON metrics and training/evaluation logs (`exp1`–`exp12`) |
| `scripts/` | Portable GPU launch and log-sync shell/Python helpers |
| `docs/` | Full paper (`paper_zh.md`) and 15 publication figures (`figures/`) |

## Quick CPU Verification

```bash
python tests/test_alignment_and_core.py
```
