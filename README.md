# Modular Soft-Prompt Distillation for Frozen LLMs: CSPD, KV-PD, and CRSP

Code, experimental logs, quantitative results, publication figures, and full technical paper for a three-stage modular soft-prompt distillation framework on frozen large language models:

1. **Stage I — Input-Space Composable Soft-Prompt Distillation (`CSPD`)**: Independent distillation of RL experts into frozen-backbone input soft prompts with prefix-robust training, continual addition/removal, out-of-domain retention, serving efficiency, and response-side attention mechanism analysis.
2. **Stage II — Layerwise KV-Space Prompt Distillation (`KV-PD`)**: Real-text forward KV initialization injected directly across all Transformer layers via `past_key_values` for 2.71× faster distillation and zero-forgetting 5-domain private-document continual learning (`KV-PD + Micro-RAG` auto-routing and 4-token KV Delta version patches).
3. **Stage III — Context-Robust Arbitrary-Slot Soft Prompts (`CRSP`)**: Arbitrary-slot insertion across six deployment contexts (`P0`–`P5`) via multi-slot context randomization, cross-context output consistency ($\mathcal{L}_{\text{inv}}$), and symmetric non-target output preservation ($\mathcal{L}_{\text{null}}$), with two construction algorithms:
   - **Algorithm 1 (`KV-Inv-CRSP`)**: Pure output-distribution functional inversion from an archived prefix `KV-PD` module without needing the original RL teacher.
   - **Algorithm 2 (`Two-Stage TF-CRSP`)**: Teacher-free construction combining base-KL-constrained multi-slot self-bootstrap warm-start with multi-slot online GRPO.

## Full Technical Paper & Figures

- **Paper (Chinese)**: [docs/paper_zh.md](docs/paper_zh.md) (complete 10-section paper with 11 embedded figures, 11 tables, Algorithm 1 & 2, Related Work, and 69 references)
- **Publication Figures**: [docs/figures/](docs/figures/) (`pfig01`–`pfig11`, 220 DPI)

## Repository Layout

| Directory / File | Description |
| :--- | :--- |
| `src/cspd_core.py`, `src/benchmarks.py`, `src/exp1*.py` | Input-space `CSPD` distillation, Top-$K$ forward KL alignment, and attention path ablation |
| `src/domain_docs_data.py`, `src/exp2*.py` | Zero-leakage synthetic domain document continual learning benchmarks (`HeliosPay`, `NVX-418`, `Litho-X9`) |
| `src/kv_prefix_core.py`, `src/domain_docs_extended.py`, `src/exp3*.py`, `src/exp4*.py`, `src/exp5*.py` | Layerwise `KV-PD` acceleration, 5-domain auto-routed `KV-PD + Micro-RAG` continual learning, and 4-token Delta upgrade |
| `src/exp6_teacherfree_input_sp.py` | Single-slot teacher-free input-space soft prompt baselines (RFT, iRFT, DPO, GRPO, KV2SP) |
| `src/exp7_context_robust_sp.py` | Six-slot (`P0`–`P5`) `CRSP` benchmark, positional drift evaluation, and $\mathcal{L}_{\text{inv}}$ ablation |
| `src/exp8_kv_inversion_and_tf_cr_sp.py` | Geometric vs. pure output-distribution KV inversion (Algorithm 1) and multi-slot teacher-free pathways |
| `src/exp9_multiseed_cr_grpo_and_composition.py`, `src/exp9b_postrl_crsp_and_composition.py` | 3-seed IFEval stability, Two-Stage TF-CRSP (Algorithm 2), scaled IFBench ($N=64$), adversarial `P3` Post-RAG, 3-expert split-slot composition, and zero-leakage doc slot capacity |
| `src/plot_paper_style_figures.py` | Reproduces all 11 publication figures (`pfig01`–`pfig11`) |
| `results/` | Complete structured JSON metrics and training/evaluation logs (`exp1`–`exp9b`) |
| `scripts/` | Reproducible Colab A100 launch and log-sync shell/Python helpers |
| `docs/` | Full paper (`paper_zh.md`), 11 figures (`figures/`), and reference curve JSON |
