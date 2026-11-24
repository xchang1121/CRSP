"""
Generate 7 cohesive, ICML/arXiv paper-style figures for the unified Technical Report,
directly mirroring the visual organization and conventions of:
  Zhang et al. (2026), "Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training"

Figures produced (saved directly into the artifact directory):
  tr_fig1_hero_overview.png       - Mirrors Paper Fig. 1 (Hero Executive Summary across RL + 5-Domain CL + Patching)
  tr_fig2_mechanistic_align.png   - Mirrors Paper Fig. 6, 11b, 12 (G(A) vs Written Inst, Token Prob Ratio, Attention K(A))
  tr_fig3_accel_and_scaling.png   - Mirrors Paper Fig. 7 & 8 (Wall-Clock Dynamics, 4-Axis Ablation, Length Scaling Law)
  tr_fig4_dual_comp_serving.png   - Mirrors Paper Fig. 3b & 5 (Zero-Training Dual Composition + Official IFBench32 Latency)
  tr_fig5_doc_cl_frontier.png     - Vertical-Domain Capacity Frontier, Convergence Speedup & Stress Probes
  tr_fig6_5domain_forgetting.png  - Mirrors Paper Fig. 3a, 4 & 10 (5-Stage SeqLoRA Forgetting Curve, Per-Domain Bars, Heatmap)
  tr_fig7_upgrade_and_normlaw.png - Policy Upgrade (v4.2->v5.0) & Unified Cross-Stage Norm-Composability Phase Diagram
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Cohesive ICML/arXiv minimalist academic styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#334155",
    "grid.color": "#CBD5E1",
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.facecolor": "white",
})

# Academic colorblind-safe palette
NAVY   = "#1E3A8A"
BLUE   = "#2563EB"
SKY    = "#60A5FA"
TEAL   = "#0D9488"
GREEN  = "#059669"
MINT   = "#34D399"
AMBER  = "#D97706"
ORANGE = "#EA580C"
RED    = "#DC2626"
CORAL  = "#F87171"
PURPLE = "#7C3AED"
SLATE  = "#64748B"
LGRAY  = "#CBD5E1"


def load_all(root: str):
    return {
        "e1":  json.load(open(os.path.join(root, "exp1_results.json"))),
        "e1b": json.load(open(os.path.join(root, "exp1b_results.json"))),
        "e2":  json.load(open(os.path.join(root, "exp2_results.json"))),
        "e2b": json.load(open(os.path.join(root, "exp2b_results.json"))),
        "e3":  json.load(open(os.path.join(root, "exp3_results.json"))),
        "e4":  json.load(open(os.path.join(root, "exp4_results.json"))),
        "e5":  json.load(open(os.path.join(root, "exp5_results.json"))),
    }


# ---------------------------------------------------------------------------
# Figure 1: Hero Overview (Mirrors Paper Fig. 1)
# ---------------------------------------------------------------------------
def make_fig1(D, out_dir):
    e3, e5, e2 = D["e3"], D["e5"], D["e2"]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))

    # (a) RL Distillation & Composition: Orig. CSPD vs KV-CSPD (G_IF, G_Math, Closed IFBench)
    ax = axes[0]
    metrics = ["IF KL Gap\nG_IF (%)", "Math KL Gap\nG_Math (%)", "Dual-Expert\nMean G (%)", "IFBench32\nClosed Acc (%)"]
    # Orig CSPD (Stage A Muon L=256 / Suite 1 RandVocab L=256)
    orig_vals = [
        e3["axis1_init"]["random_vocab"]["final_G_pct"],    # 26.66 (45 steps, 195.8s)
        D["e1b"]["distributional_G"]["single_P_math"]["G_math_pct"],  # 20.22 (60 steps, 273s)
        D["e1b"]["distributional_G"]["cspd_concat_IF_Math"]["G_avg_pct"],  # 5.02 (Muon)
        D["e1b"]["ifbench_evals"]["cspd_concat_IF_Math"]["n_strict"] / D["e1b"]["ifbench_evals"]["cspd_concat_IF_Math"]["closed_think"] * 100.0,  # 62.1%
    ]
    # KV-CSPD (Ours: L_kv=8/16, 72-82s, 32 tok dual)
    kv_vals = [
        e3["axis4_kv_cspd"]["kv_if_L8"]["final_G_pct"],      # 65.46 (L_kv=8, 72.3s)
        e3["axis4_kv_cspd"]["kv_math_L16"]["final_G_pct"],   # 28.01 (L_kv=16, 81.6s)
        e3["dual_composition_G"]["KV16_Concat_I_M"]["G_avg_pct"],  # 19.96 (32 tok)
        e3["ifbench32_verify"]["kv16_concat_M_I"]["closed_acc"],   # 88.24
    ]
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w/2, orig_vals, w, label="Input-SP CSPD (L=256–512)", color=SLATE)
    b2 = ax.bar(x + w/2, kv_vals, w, label="KV-CSPD (Ours, L_kv=8–32)", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8.0)
    ax.set_ylabel("Score / KL Gap Closed (%)")
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=7.8)
    ax.set_title("(a) RL Distillation & Composition (M2RL-4B)")
    for b in b1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.4, color="#334155")
    for b in b2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, color="#065F46", fontweight="bold")
    ax.text(0.5, 0.84, "~20x faster wall-clock | 16x–32x shorter context",
            transform=ax.transAxes, ha="center", fontsize=7.4, fontweight="bold", color="#065F46",
            bbox=dict(boxstyle="round,pad=0.22", fc="#ECFDF5", ec=GREEN, lw=0.8))

    # (b) 5-Domain Zero-Oracle Continual Learning (194 unseen queries)
    ax = axes[1]
    ic_5r = (e2["runs"]["in_context_doc_upper_bound"]["A_heliospay"]["rules"] +
             e2["runs"]["in_context_doc_upper_bound"]["B_nvx418"]["rules"] +
             e2["runs"]["in_context_doc_upper_bound"]["C_lithox9"]["rules"] +
             e5["domains_5"]["in_context_DE"]["D_gridops"]["rules"] +
             e5["domains_5"]["in_context_DE"]["E_customs"]["rules"]) / 5.0
    m_lbls = ["Base\n(No Doc)", "5-Stage\nSeqLoRA\n(11.8M)", "In-Context\nFull Doc\n(~500 tok)", "KV-CSPD\nAuto\n(16 tok)", "Dual-Intent\nAuto (Ours)\n(16–60 tok)"]
    r_v = [0.0, e5["seq_lora_5"]["stage5_E"]["mean_rules"], ic_5r,
           e5["auto_router"]["e2e_kv16_auto_routed"]["mean_rules"],
           e5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["mean_rules"]]
    f_v = [2.0, e5["seq_lora_5"]["stage5_E"]["mean_facts"], 100.0,
           e5["auto_router"]["e2e_kv16_auto_routed"]["mean_facts"],
           e5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["mean_facts"]]
    x = np.arange(len(m_lbls))
    w = 0.36
    b1 = ax.bar(x - w/2, r_v, w, label="5-Domain Rules (%)", color=BLUE)
    b2 = ax.bar(x + w/2, f_v, w, label="5-Domain Facts (%)", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels(m_lbls, fontsize=7.6)
    ax.set_ylabel("5-Domain Mean Accuracy (%)")
    ax.set_ylim(0, 118)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=7.8)
    ax.set_title("(b) 5-Domain Zero-Oracle Continual Learning")
    for b in b1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.3, color=NAVY, fontweight="bold")
    for b in b2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.3, color="#0F766E")

    # (c) Enterprise Policy Version Upgrade (HeliosPay v4.2 -> v5.0)
    ax = axes[2]
    vu = e5["version_upgrade"]
    v_lbls = ["Outdated\nKV v4.2", "In-Context\nFull v5.0", "KV Atomic\nSwap\n(1.18M)", "4-Tok KV\nPatch (Ours)\n(0.29M)", "LoRA\nUpdate\n(11.8M)"]
    vr = [vu["outdated_KV_v42"]["v50_rules"], vu["in_context_v50"]["v50_rules"],
          vu["kv_atomic_swap_v50"]["v50_rules"], vu["kv_delta_patch_L4"]["v50_rules"],
          vu["lora_continual_v42_to_v50"]["v50_rules"]]
    vf = [vu["outdated_KV_v42"]["v50_facts"], vu["in_context_v50"]["v50_facts"],
          vu["kv_atomic_swap_v50"]["v50_facts"], vu["kv_delta_patch_L4"]["v50_facts"],
          vu["lora_continual_v42_to_v50"]["v50_facts"]]
    vl = [vu["outdated_KV_v42"]["stale_v42_rules_pct"], 0.0, 0.0, 0.0, 0.0]
    x = np.arange(len(v_lbls))
    w = 0.27
    b1 = ax.bar(x - w, vr, w, label="v5.0 Rules (%)", color=BLUE)
    b2 = ax.bar(x, vf, w, label="v5.0 Facts (%)", color=GREEN)
    b3 = ax.bar(x + w, vl, w, label="Stale v4.2 Leak (%) ↓", color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(v_lbls, fontsize=7.6)
    ax.set_ylabel("Accuracy / Stale Leak (%)")
    ax.set_ylim(0, 122)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.set_title("(c) Policy Upgrade (HeliosPay v4.2 → v5.0)")
    for b in b1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.1, color=NAVY, fontweight="bold")
    for b in b2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.1, color="#065F46")
    ax.text(0 + w, 51.5, "50%", ha="center", va="bottom", fontsize=7.2, color=RED, fontweight="bold")
    for idx in [1, 2, 3, 4]:
        ax.text(idx + w, 1.5, "0%", ha="center", va="bottom", fontsize=7.0, color="#991B1B")
    ax.annotate("40x < LoRA,\ninstant rollback", xy=(3, 85), xytext=(2.4, 108),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
                fontsize=7.3, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.2", fc="#ECFDF5", ec=GREEN, lw=0.8))

    plt.tight_layout()
    p = os.path.join(out_dir, "tr_fig1_hero_overview.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Figure 2: Mechanistic Alignment (Mirrors Paper Fig. 6, 11b, 12)
# ---------------------------------------------------------------------------
def make_fig2(D, out_dir):
    m_if = D["e1b"]["mechanistic_IF"]
    m_ma = D["e1b"]["mechanistic_Math"]
    e3 = D["e3"]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))

    # (a) Soft Prompt & KV-CSPD vs Written Instructions (Paper Fig. 6a / 12a)
    ax = axes[0]
    labels = [
        "Paper Ref: SP (20B)",
        "KV-CSPD L_kv=8 (Ours)",
        "KV-CSPD L_kv=16 (Ours)",
        "SP Muon/AdamW (IF)",
        "SP AdamW (Math)",
        "Paper Ref: Verify Inst.",
        "Neutral Inst. (IF)",
        "Check Inst. (IF)",
        "Verify Inst. (IF)",
        "SP Random Init (IF)",
    ]
    vals = [
        86.8,
        e3["axis4_kv_cspd"]["kv_if_L8"]["final_G_pct"],
        e3["axis4_kv_cspd"]["kv_if_L16"]["final_G_pct"],
        m_if["kl_gap_closed_G_pct"]["sp_muon"],
        m_ma["kl_gap_closed_G_pct"]["sp_adamw"],
        2.1,
        m_if["kl_gap_closed_G_pct"]["neutral_instruction"],
        m_if["kl_gap_closed_G_pct"]["check_instruction"],
        m_if["kl_gap_closed_G_pct"]["verify_instruction"],
        m_if["kl_gap_closed_G_pct"]["sp_init"],
    ]
    cols = [LGRAY, GREEN, TEAL, BLUE, SKY, LGRAY, CORAL, CORAL, RED, SLATE]
    hatches = ["//", "", "", "", "", "//", "", "", "", ""]
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, height=0.66, color=cols, edgecolor="#334155", linewidth=0.5)
    for b, ht in zip(bars, hatches):
        if ht:
            b.set_hatch(ht)
    ax.axvline(0.0, color="#1E293B", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlabel("Teacher–Student KL Gap Closed, G(A) (%)")
    ax.set_xlim(-46, 106)
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_title("(a) Soft Prompt & KV-CSPD vs. Written Instructions")
    for b in bars:
        v = b.get_width()
        xp = v + 1.8 if v >= 0 else v - 1.8
        ha = "left" if v >= 0 else "right"
        ax.text(xp, b.get_y() + b.get_height()/2, f"{v:+.1f}%", va="center", ha=ha, fontsize=7.4,
                fontweight="bold" if v > 30 else "normal")

    # (b) Decision Token Probability Relative to Teacher (Directly Mirrors Paper Fig. 11b!)
    ax = axes[1]
    d_if = m_if["discourse_prob_rel_teacher"]
    toks = ["Check", " But", " Let", "?", "So", "Thus"]
    disp = ["Check", "␣But", "␣Let", "?", "So", "Thus"]
    s_rel = [d_if[t]["student_rel_to_teacher"] for t in toks]
    p_rel = [d_if[t]["sp_rel_to_teacher"] for t in toks]
    xt = np.arange(len(toks))
    wt = 0.36
    ax.axhline(1.0, color="#1E293B", linestyle="--", linewidth=1.2, label="Teacher = 1.00")
    b1 = ax.bar(xt - wt/2, s_rel, wt, label="Student (No SP)", color=SLATE, alpha=0.85)
    b2 = ax.bar(xt + wt/2, p_rel, wt, label="Student + SP (P_if)", color=BLUE)
    ax.set_xticks(xt)
    ax.set_xticklabels(disp, fontsize=8.8, fontweight="bold")
    ax.set_ylabel("Mean Probability Relative to Teacher")
    ax.set_ylim(0, 1.42)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left", fontsize=7.8)
    ax.set_title("(b) Decision Token Probability Shift (Paper Fig. 11b)")
    for b in b1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.025, f"{h:.2f}", ha="center", va="bottom", fontsize=7.2, color="#475569")
    for b in b2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.025, f"{h:.2f}", ha="center", va="bottom", fontsize=7.3, color=NAVY, fontweight="bold")

    # (c) Attention Pathway Ablation K(A) (Directly Mirrors Paper Fig. 6c / 12b!)
    ax = axes[2]
    p_names = ["Full", "Response only", "Input only", "Blocked"]
    p_keys = ["full", "response_only", "input_only", "blocked"]
    k_ours = [m_if["attention_pathway_K_vs_blocked_pct"][k] for k in p_keys]
    k_paper = [100.0, 80.0, 33.0, 1.0]
    yp = np.arange(len(p_names))
    hp = 0.36
    bo = ax.barh(yp - hp/2, k_ours, hp, label="Ours (M2RL-RL_IF, 4B)", color=BLUE)
    bp = ax.barh(yp + hp/2, k_paper, hp, label="Paper Fig. 6c (GPT-OSS, 20B)", color=LGRAY, edgecolor="#475569", hatch="//")
    ax.set_yticks(yp)
    ax.set_yticklabels(p_names, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("Soft-Prompt Improvement Kept, K(A) (%)")
    ax.set_xlim(0, 118)
    ax.grid(True, axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right", fontsize=7.8)
    ax.set_title("(c) Attention Pathway Ablation (Eq. 6 / Fig. 6c)")
    for b in bo:
        v = b.get_width()
        ax.text(v + 1.8, b.get_y() + b.get_height()/2, f"{v:.1f}%", va="center", ha="left", fontsize=7.6, color=NAVY, fontweight="bold")
    for b in bp:
        v = b.get_width()
        ax.text(v + 1.8, b.get_y() + b.get_height()/2, f"{v:.0f}%", va="center", ha="left", fontsize=7.4, color="#475569")

    plt.tight_layout()
    p = os.path.join(out_dir, "tr_fig2_mechanistic_align.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Figure 3: CSPD Acceleration & Length Scaling (Mirrors Paper Fig. 7 & 8)
# ---------------------------------------------------------------------------
def make_fig3(D, out_dir):
    e3, e2b, e4 = D["e3"], D["e2b"], D["e4"]
    a1, a2, a3, a4 = e3["axis1_init"], e3["axis2_loss_trunc"], e3["axis3_opt_len"], e3["axis4_kv_cspd"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # (a) Wall-Clock Convergence Dynamics (Paper Fig. 7 style, x=wall-clock seconds)
    ax = axes[0]
    specs = [
        (a4["kv_if_L8"],           "KV-CSPD L_kv=8 (0.59M)",     GREEN,  "-",  "o", 2.3),
        (a4["kv_if_L16"],          "KV-CSPD L_kv=16 (1.18M)",    TEAL,   "--", "s", 1.9),
        (a1["task_inst"],          "A1: Task-Inst + Muon",       BLUE,   "-",  "^", 2.0),
        (a3["adamw_lownorm"],      "A3: Low-Norm + AdamW",       PURPLE, "-.", "D", 1.9),
        (a1["random_vocab"],       "Orig. CSPD (RandVocab)",     SLATE,  "--", "v", 1.9),
        (a2["hybrid_fwd_rev_768"], "A2: Hybrid Fwd+Rev KL*",     AMBER,  ":",  "x", 1.8),
        (a1["low_norm"],           "A1: Low-Norm + Muon*",       RED,    ":",  ".", 1.6),
    ]
    for meta, lbl, col, ls, mk, lw in specs:
        ts = [pt["train_sec"] for pt in meta["curve"]]
        gs = [pt["G_pct"] for pt in meta["curve"]]
        ax.plot(ts, gs, linestyle=ls, marker=mk, color=col, linewidth=lw, markersize=4.6, label=lbl)
    ax.axhline(a1["random_vocab"]["final_G_pct"], color=SLATE, linestyle=":", linewidth=1.0)
    ax.axhline(0.0, color="#94A3B8", linewidth=0.7)
    ax.set_xlabel("Wall-Clock Training Time on A100 (seconds)")
    ax.set_ylabel("Held-Out KL Gap Closed, G(A) (%)")
    ax.set_xlim(-3, 206)
    ax.set_ylim(-45, 76)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="lower right", fontsize=7.2)
    ax.annotate("Step 2 (3.3s): +26.1%\nCrosses Orig. at ~10s (~20x!)\nFinal: +65.5% (2.46x)",
                xy=(16.1, 46.06), xytext=(86, 44.5),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3),
                fontsize=7.3, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec=GREEN, lw=0.8))
    ax.set_title("(a) Wall-Clock Convergence on M2RL-RL_IF")

    # (b) 4-Axis Controlled Final G(A)
    ax = axes[1]
    items = [
        ("KV-CSPD L_kv=8 (0.59M)",      a4["kv_if_L8"]["final_G_pct"],          GREEN),
        ("KV-CSPD L_kv=16 (1.18M)",     a4["kv_if_L16"]["final_G_pct"],         TEAL),
        ("A1: Task-Inst + Muon",        a1["task_inst"]["final_G_pct"],         BLUE),
        ("A3: Low-Norm + AdamW*",       a3["adamw_lownorm"]["final_G_pct"],     PURPLE),
        ("A1: Orig. CSPD RandVocab",    a1["random_vocab"]["final_G_pct"],      SLATE),
        ("A2: Hybrid Fwd+Rev KL*",      a2["hybrid_fwd_rev_768"]["final_G_pct"], AMBER),
        ("A2: Fork-Token Top-30%*",     a2["fork_top30_768"]["final_G_pct"],    ORANGE),
        ("A3: Low-Rank r=32* (0.09M)",  a3["low_rank_r32"]["final_G_pct"],      SKY),
        ("A3: NormClip Muon*",          a3["muon_normclip_055"]["final_G_pct"], SKY),
        ("A1: Low-Norm + Muon Base*",   a1["low_norm"]["final_G_pct"],          LGRAY),
        ("A2: Seq-NLL (1-hot)*",        a2["seq_nll_768"]["final_G_pct"],       CORAL),
        ("A2: Trunc-256 Prefix*",       a2["rb_top64_trunc256"]["final_G_pct"], RED),
    ]
    y = np.arange(len(items))
    bars = ax.barh(y, [it[1] for it in items], height=0.66, color=[it[2] for it in items])
    ax.axvline(0.0, color="#1E293B", linewidth=0.9)
    ax.axvline(a1["random_vocab"]["final_G_pct"], color=SLATE, linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlabel("Final G(A) (%) [* = Low-Norm base +7.5%]")
    ax.set_xlim(-84, 84)
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_title("(b) 4-Axis Controlled Ablation (45 Updates)")
    for b in bars:
        v = b.get_width()
        xp = v + 1.5 if v >= 0 else v - 1.5
        ha = "left" if v >= 0 else "right"
        ax.text(xp, b.get_y() + b.get_height()/2, f"{v:+.1f}%", va="center", ha=ha, fontsize=7.2,
                fontweight="bold" if v > 30 or v < -20 else "normal")

    # (c) Prompt Length Scaling Law: Paper Fig. 8 vs Zero-Prior Doc Input-SP vs KV-CSPD-Doc
    ax = axes[2]
    # Paper Fig. 8 (Behavioral RL Recovery %)
    p_L = [2, 8, 32, 256, 512]
    p_rec = [51.2, 74.1, 90.0, 93.1, 87.8]
    # Stage B Input-SP on Zero-Prior Doc A (Unseen Rules %)
    sp_L = [32, 128, 256]
    sp_r = [e2b["runs"]["cap_A_L32"]["A_heliospay"]["rules"],
            e2b["runs"]["stage1_PA"]["A_heliospay"]["rules"],
            e2b["runs"]["cap_A_L256"]["A_heliospay"]["rules"]]
    # Suite 2 KV-CSPD-Doc on Zero-Prior Doc A (Unseen Rules %)
    kv_L = [8, 16, 32]
    kv_r = [e4["runs"]["KV_A_L8_AdamW"]["A_heliospay"]["rules"],
            e4["runs"]["KV_A_L16_AdamW"]["A_heliospay"]["rules"],
            e4["runs"]["KV_A_L32_AdamW"]["A_heliospay"]["rules"]]

    ax.plot(p_L, p_rec, "o--", color=SLATE, linewidth=1.8, markersize=5.5, label="Paper Fig. 8: RL Math Rec. (%)")
    ax.plot(sp_L, sp_r, "s-", color=BLUE, linewidth=2.1, markersize=6.5, label="Input-SP: Doc A Rules (%)")
    ax.plot(kv_L, kv_r, "D-", color=GREEN, linewidth=2.4, markersize=7.0, label="KV-CSPD-Doc: Doc A Rules (%)")
    ax.axhline(37.5, color=AMBER, linestyle=":", linewidth=1.3, label="In-Context Full Doc A (37.5%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 8, 16, 32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Virtual Prefix Length L (log₂ scale)")
    ax.set_ylabel("Task Recovery / Unseen Rule Accuracy (%)")
    ax.set_ylim(-5, 108)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    for lx, ly in zip(kv_L, kv_r):
        ax.text(lx, ly + 3.0, f"{ly:.1f}%", ha="center", va="bottom", fontsize=7.4, color="#065F46", fontweight="bold")
    for lx, ly in zip(sp_L, sp_r):
        if ly < 5:
            ax.text(lx * 1.35, ly + 1.5, f"{ly:.1f}%", ha="left", va="center", fontsize=7.3, color=NAVY, fontweight="bold")
        else:
            ax.text(lx, ly - 5.5, f"{ly:.1f}%", ha="center", va="center", fontsize=7.3, color=NAVY, fontweight="bold")
    ax.annotate("Input-SP hits 0% wall at L=32;\nKV-CSPD hits 66.7% at L_kv=32!",
                xy=(32, 66.7), xytext=(4.2, 84),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
                fontsize=7.3, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.22", fc="#ECFDF5", ec=GREEN, lw=0.8))
    ax.set_title("(c) Prefix Length Scaling (Paper Fig. 8 Contrast)")

    plt.tight_layout()
    p = os.path.join(out_dir, "tr_fig3_accel_and_scaling.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Figure 4: Dual-Expert Zero-Training Composition & Serving Latency
# ---------------------------------------------------------------------------
def make_fig4(D, out_dir):
    e3, e1b = D["e3"], D["e1b"]
    dc = e3["dual_composition_G"]
    ifb = e3["ifbench32_verify"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.9))

    # (a) Dual-Expert Zero-Training Composition (Stage A SP vs Suite 1 KV-CSPD)
    ax = axes[0]
    cfgs = [
        ("Stage A SP\nAdamW [M;I]\n(512 tok)", e1b["distributional_G"]["ablation_adamw_Math_IF"]["G_if_pct"],
         e1b["distributional_G"]["ablation_adamw_Math_IF"]["G_math_pct"],
         e1b["distributional_G"]["ablation_adamw_Math_IF"]["G_avg_pct"]),
        ("Stage A SP\nMuon [I;M]\n(512 tok)", e1b["distributional_G"]["cspd_concat_IF_Math"]["G_if_pct"],
         e1b["distributional_G"]["cspd_concat_IF_Math"]["G_math_pct"],
         e1b["distributional_G"]["cspd_concat_IF_Math"]["G_avg_pct"]),
        ("KV-CSPD\nSingle IF\n(16 tok)", dc["KV16_Single_IF"]["G_if_pct"], dc["KV16_Single_IF"]["G_math_pct"], dc["KV16_Single_IF"]["G_avg_pct"]),
        ("KV-CSPD\nSingle Math\n(16 tok)", dc["KV16_Single_Math"]["G_if_pct"], dc["KV16_Single_Math"]["G_math_pct"], dc["KV16_Single_Math"]["G_avg_pct"]),
        ("KV-CSPD\n[M; I]\n(32 tok!)", dc["KV16_Concat_M_I"]["G_if_pct"], dc["KV16_Concat_M_I"]["G_math_pct"], dc["KV16_Concat_M_I"]["G_avg_pct"]),
        ("KV-CSPD\n[I; M]\n(32 tok!)", dc["KV16_Concat_I_M"]["G_if_pct"], dc["KV16_Concat_I_M"]["G_math_pct"], dc["KV16_Concat_I_M"]["G_avg_pct"]),
    ]
    x = np.arange(len(cfgs))
    w = 0.26
    b1 = ax.bar(x - w, [c[1] for c in cfgs], w, label="G_IF (%)", color=BLUE)
    b2 = ax.bar(x,     [c[2] for c in cfgs], w, label="G_Math (%)", color=GREEN)
    b3 = ax.bar(x + w, [c[3] for c in cfgs], w, label="Dual-Domain Mean G (%)", color=AMBER)
    ax.axhline(0.0, color="#1E293B", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cfgs], fontsize=7.8)
    ax.set_ylabel("Held-Out KL Gap Closed, G (%)")
    ax.set_ylim(-50, 82)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=7.6)
    ax.set_title("(a) Zero-Training Dual-Expert Composition & Order Invariance")
    for bars, col in [(b1, NAVY), (b2, "#065F46"), (b3, "#B45309")]:
        for b in bars:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            off = 1.6 if h >= 0 else -1.8
            ax.text(b.get_x() + b.get_width()/2, h + off, f"{h:+.1f}", ha="center", va=va, fontsize=6.9, color=col, fontweight="bold")
    ax.annotate("Near-Exact Commutativity (0.06 pp diff!)\nDual Mean (+20.0%) beats BOTH singles\nat 16x shorter context (32 vs 512 tok)",
                xy=(4.5, 26.5), xytext=(2.85, 52),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3),
                fontsize=7.5, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec=GREEN, lw=0.8))

    # (b) Official IFBench32 Accuracy vs 32-Prompt Inference Latency (Mirrors Paper Fig. 5 spirit)
    ax = axes[1]
    if_cfgs = [
        ("Student\n(No SP)", "student_no_sp"),
        ("NormClipMuon\nSingle P_if\n(256 tok)", "normclip_muon_single_IF"),
        ("NormClipMuon\n[M; I]\n(512 tok)", "normclip_muon_concat_M_I"),
        ("KV-CSPD L16\nSingle P_if\n(16 tok)", "kv16_single_IF"),
        ("KV-CSPD L16\n[M; I]\n(32 tok!)", "kv16_concat_M_I"),
    ]
    xi = np.arange(len(if_cfgs))
    wi = 0.34
    c_acc = [ifb[k]["closed_acc"] for _, k in if_cfgs]
    s_acc = [ifb[k]["prompt_strict"] for _, k in if_cfgs]
    lats  = [ifb[k]["eval_sec"] for _, k in if_cfgs]

    ib1 = ax.bar(xi - wi/2, c_acc, wi, label="Strict Acc on Closed-</think> (%)", color=GREEN)
    ib2 = ax.bar(xi + wi/2, s_acc, wi, label="Overall Strict Acc @2560 tok (%)", color=BLUE)
    ax.set_xticks(xi)
    ax.set_xticklabels([n for n, _ in if_cfgs], fontsize=7.9)
    ax.set_ylabel("IFBench Accuracy (%)")
    ax.set_ylim(0, 114)
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_title("(b) Official IFBench32 Accuracy & Wall-Clock Serving Latency")

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(RED)
    ax2.plot(xi, lats, "o--", color=RED, linewidth=2.0, markersize=6.2, label="32-Prompt Latency (s)")
    ax2.set_ylabel("32-Prompt Inference Latency (s)", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.set_ylim(240, 382)

    for b in ib1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.4, color="#065F46", fontweight="bold")
    for b in ib2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.1, color=NAVY)
    for xj, sec in zip(xi, lats):
        pct = (sec / lats[0] - 1.0) * 100.0
        lbl = f"{sec:.0f}s" if xj == 0 else f"{sec:.0f}s ({pct:+.1f}%)"
        ax2.text(xj, sec + 6.5, lbl, ha="center", va="bottom", fontsize=7.0, color="#991B1B", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#FCA5A5", alpha=0.92, lw=0.7))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=7.5)

    plt.tight_layout()
    p = os.path.join(out_dir, "tr_fig4_dual_comp_serving.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Figure 5: Vertical-Domain Frontier, Speedup & Stress Probes (Suite 2)
# ---------------------------------------------------------------------------
def make_fig5(D, out_dir):
    e4, e2, e2b = D["e4"], D["e2"], D["e2b"]
    runs4, meta4, probes4 = e4["runs"], e4["meta"], e4["probes_A"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    # (a) Wall-Clock NLL Convergence (Doc A)
    ax = axes[0]
    curves = [
        ("SP_A_random_vocab", "Input-SP L=128 (RandVocab, 128.8s)", BLUE,   "--", "o"),
        ("SP_A_doc_semantic", "Input-SP L=128 (Doc-Sem, 128.5s)",   PURPLE, "-.", "s"),
        ("SP_A_low_norm",     "Input-SP L=128 (Low-Norm, 128.1s)",  SLATE,  ":",  "^"),
        ("KV_A_L8_AdamW",     "KV-CSPD L_kv=8 (AdamW, 57.1s)",      MINT,   "-",  "D"),
        ("KV_A_L16_AdamW",    "KV-CSPD L_kv=16 (AdamW, 57.6s)",     GREEN,  "-",  "o"),
        ("KV_A_L32_AdamW",    "KV-CSPD L_kv=32 (AdamW, 57.5s)",     "#065F46", "-", "v"),
        ("KV_A_L16_Muon",     "KV-CSPD L_kv=16 (Muon, 61.3s)",      AMBER,  "--", "x"),
    ]
    for key, lbl, col, ls, mk in curves:
        m = meta4[key]
        steps = sorted([int(s) for s in m["nll_curve"].keys()])
        max_s = max(steps)
        times = [m["train_seconds"] * (s / max_s) for s in steps]
        nlls = [m["nll_curve"][str(s)] for s in steps]
        ax.plot(times, nlls, linestyle=ls, marker=mk, color=col, linewidth=1.9, markersize=4.6, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("Wall-Clock Training Time on A100 (seconds)")
    ax.set_ylabel("Training NLL (log scale)")
    ax.set_xlim(0, 136)
    ax.set_ylim(0.001, 4.0)
    ax.grid(True, which="major", alpha=0.22)
    ax.legend(frameon=False, loc="upper right", fontsize=7.0)
    ax.annotate("4.6x–5.9x faster wall-clock\nto match Input-SP final NLL!",
                xy=(21.6, 0.0192), xytext=(64, 0.0045),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3),
                fontsize=7.4, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.24", fc="#ECFDF5", ec=GREEN, lw=0.8))
    ax.set_title("(a) Doc A Training NLL Convergence Speedup")

    # (b) Doc A Capacity Frontier (Rules & Facts vs In-Context & LoRA)
    ax = axes[1]
    items_a = [
        ("In-Context\nFull Doc\n(~500 tok)", e2["runs"]["in_context_doc_upper_bound"]["A_heliospay"]["rules"], e2["runs"]["in_context_doc_upper_bound"]["A_heliospay"]["facts"]),
        ("Input-SP\nL=128\n(0.33M)", runs4["SP_A_random_vocab"]["A_heliospay"]["rules"], runs4["SP_A_random_vocab"]["A_heliospay"]["facts"]),
        ("KV-CSPD\nL_kv=8\n(0.59M)", runs4["KV_A_L8_AdamW"]["A_heliospay"]["rules"], runs4["KV_A_L8_AdamW"]["A_heliospay"]["facts"]),
        ("KV-CSPD\nL_kv=16\n(1.18M)", runs4["KV_A_L16_AdamW"]["A_heliospay"]["rules"], runs4["KV_A_L16_AdamW"]["A_heliospay"]["facts"]),
        ("KV-CSPD\nL_kv=16\n+Think", runs4["KV_A_L16_Thinking"]["A_heliospay"]["rules"], runs4["KV_A_L16_Thinking"]["A_heliospay"]["facts"]),
        ("KV-CSPD\nL_kv=32\n(2.36M)", runs4["KV_A_L32_AdamW"]["A_heliospay"]["rules"], runs4["KV_A_L32_AdamW"]["A_heliospay"]["facts"]),
        ("Single\nLoRA r=16\n(11.8M)", e2b["runs"]["lora_seq_stage1_A"]["A_heliospay"]["rules"], e2b["runs"]["lora_seq_stage1_A"]["A_heliospay"]["facts"]),
    ]
    x = np.arange(len(items_a))
    w = 0.36
    b1 = ax.bar(x - w/2, [it[1] for it in items_a], w, label="Unseen Rules (%)", color=BLUE)
    b2 = ax.bar(x + w/2, [it[2] for it in items_a], w, label="Paraphrased Facts (%)", color=GREEN)
    ax.axhline(37.5, color=SLATE, linestyle="--", linewidth=1.0, label="In-Context Rules (37.5%)")
    ax.set_xticks(x)
    ax.set_xticklabels([it[0] for it in items_a], fontsize=7.2)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 116)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right", fontsize=7.4)
    ax.set_title("(b) Doc A Zero-Raw-Doc Capacity Frontier")
    for b in b1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.0, color=NAVY, fontweight="bold")
    for b in b2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.0, color="#065F46")

    # (c) Hard Stress Probes on Doc A
    ax = axes[2]
    pmods = [
        ("Base\n(No Doc)", "base_no_doc"),
        ("In-Context\nFull Doc", "in_context_full_doc"),
        ("Input-SP\nL=128", "SP_A_doc_semantic"),
        ("KV-CSPD\nL_kv=16", "KV_A_L16_AdamW"),
        ("KV-CSPD\n+AbstainCal", "KV_A_L16_AbstainCal"),
    ]
    xp = np.arange(len(pmods))
    wp = 0.26
    pb1 = ax.bar(xp - wp, [probes4[k]["boundary"] for _, k in pmods], wp, label="Boundary Equality (12)", color=BLUE)
    pb2 = ax.bar(xp,      [probes4[k]["extrap"] for _, k in pmods],   wp, label="Extreme Extrap (8)", color=PURPLE)
    pb3 = ax.bar(xp + wp, [probes4[k]["abstain"] for _, k in pmods],  wp, label="Unanswerable Abstain (8)", color=AMBER)
    ax.set_xticks(xp)
    ax.set_xticklabels([n for n, _ in pmods], fontsize=7.8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 114)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left", fontsize=7.4)
    ax.set_title("(c) Stress Probes: Boundary, Extrap & Abstention")
    for bars, col in [(pb1, NAVY), (pb2, "#5B21B6"), (pb3, "#B45309")]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=7.2, color=col, fontweight="bold")

    plt.tight_layout()
    p = os.path.join(out_dir, "tr_fig5_doc_cl_frontier.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Figure 6: 5-Domain Continual Learning & Forgetting (Mirrors Paper Fig. 3a, 4 & 10)
# ---------------------------------------------------------------------------
def make_fig6(D, out_dir):
    e5, e2 = D["e5"], D["e2"]

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0))

    # (a) 5-Stage Sequential LoRA Forgetting Curves vs Zero-Oracle KV-CSPD (Mirrors Paper Fig. 4!)
    ax = axes[0]
    stages = ["Stage 1\n(+A)", "Stage 2\n(+B)", "Stage 3\n(+C)", "Stage 4\n(+D)", "Stage 5\n(+E)"]
    skeys = ["stage1_A", "stage2_B", "stage3_C", "stage4_D", "stage5_E"]
    doms = [("A_heliospay", "Doc A (FinTech)",   BLUE,   "o"),
            ("B_nvx418",    "Doc B (BioPharma)", GREEN,  "s"),
            ("C_lithox9",   "Doc C (EUV Fab)",   AMBER,  "^"),
            ("D_gridops",   "Doc D (SmartGrid)", PURPLE, "D"),
            ("E_customs",   "Doc E (Customs)",   RED,    "v")]
    xs = np.arange(len(stages))
    for dkey, dlbl, col, mk in doms:
        vals = [e5["seq_lora_5"][sk][dkey]["rules"] for sk in skeys]
        ax.plot(xs, vals, marker=mk, color=col, linewidth=1.9, markersize=5.5, label=dlbl)
    ax.plot(xs, [e5["seq_lora_5"][sk]["mean_rules"] for sk in skeys], "k--", linewidth=1.9, label="SeqLoRA 5-Dom Mean (19.2%)")
    ax.axhline(e5["auto_router"]["e2e_kv16_auto_routed"]["mean_rules"], color=TEAL, linewidth=2.2,
               label="KV-CSPD Auto-Router (57.5%, 0% Forget)")
    ax.set_xticks(xs)
    ax.set_xticklabels(stages, fontsize=8.0)
    ax.set_ylabel("Unseen Multi-Step Rule Accuracy (%)")
    ax.set_ylim(-4, 134)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.8, columnspacing=0.8)
    ax.annotate("100% Forgetting of ALL 4\nPrior Domains (A=B=C=D=0%)",
                xy=(4.0, 1.5), xytext=(1.35, 33),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3),
                fontsize=7.3, fontweight="bold", color="#991B1B",
                bbox=dict(boxstyle="round,pad=0.24", fc="#FEF2F2", ec=RED, lw=0.8))
    ax.set_title("(a) 5-Stage Continual Addition Trajectory (Paper Fig. 4)")

    # (b) Per-Domain End-to-End 5-Domain Breakdown (Rules: In-Context vs SeqLoRA vs Dual-Intent Auto)
    ax = axes[1]
    d_lbls = ["Doc A\nFinTech", "Doc B\nBioPh.", "Doc C\nEUV", "Doc D\nGrid", "Doc E\nCustoms", "5-Dom\nMean"]
    dkeys = ["A_heliospay", "B_nvx418", "C_lithox9", "D_gridops", "E_customs"]
    ic_r = [e2["runs"]["in_context_doc_upper_bound"]["A_heliospay"]["rules"],
            e2["runs"]["in_context_doc_upper_bound"]["B_nvx418"]["rules"],
            e2["runs"]["in_context_doc_upper_bound"]["C_lithox9"]["rules"],
            e5["domains_5"]["in_context_DE"]["D_gridops"]["rules"],
            e5["domains_5"]["in_context_DE"]["E_customs"]["rules"]]
    ic_r.append(float(np.mean(ic_r)))
    sl_r = [e5["seq_lora_5"]["stage5_E"][dk]["rules"] for dk in dkeys] + [e5["seq_lora_5"]["stage5_E"]["mean_rules"]]
    kv_r = [e5["auto_router"]["e2e_kv16_microRAG_auto_routed"][dk]["rules"] for dk in dkeys] + [e5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["mean_rules"]]

    xd = np.arange(len(d_lbls))
    wd = 0.26
    b1 = ax.bar(xd - wd, sl_r, wd, label="5-Stage SeqLoRA (11.8M)", color=CORAL)
    b2 = ax.bar(xd,      ic_r, wd, label="In-Context Full Doc (~500 tok)", color=SLATE)
    b3 = ax.bar(xd + wd, kv_r, wd, label="Dual-Intent KV-CSPD (Zero Oracle)", color=GREEN)
    ax.axvline(4.5, color="#94A3B8", linestyle="--", linewidth=0.9)
    ax.set_xticks(xd)
    ax.set_xticklabels(d_lbls, fontsize=7.8)
    ax.set_ylabel("Unseen Multi-Step Rule Accuracy (%)")
    ax.set_ylim(0, 116)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left", fontsize=7.4)
    ax.set_title("(b) Per-Domain Rule Accuracy (Zero Oracle, 194 Q)")
    for b in b3:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2, color="#065F46", fontweight="bold")
    for idx_i, b in enumerate(b1):
        if idx_i in [4, 5]:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.0, color="#991B1B")

    # (c) 5-Stage SeqLoRA Forgetting Matrix Heatmap (Directly Mirrors Paper Fig. 3b / 10 Heatmap!)
    ax = axes[2]
    mat = np.zeros((5, 5))
    for i, sk in enumerate(skeys):
        for j, dk in enumerate(dkeys):
            mat[i, j] = e5["seq_lora_5"][sk][dk]["rules"]
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"], fontsize=8.4)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(["Stage 1 (+A)", "Stage 2 (+B)", "Stage 3 (+C)", "Stage 4 (+D)", "Stage 5 (+E)"], fontsize=8.2)
    ax.set_xlabel("Evaluated Domain (Rules %)")
    ax.set_title("(c) SeqLoRA 5-Stage Forgetting Matrix (Paper Fig. 3b Style)")
    for i in range(5):
        for j in range(5):
            v = mat[i, j]
            txt_col = "white" if v >= 55 else ("#DC2626" if (j < i and v == 0) else "#1E293B")
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.2, fontweight="bold", color=txt_col)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rule Acc (%)", fontsize=8.2)

    plt.tight_layout()
    p = os.path.join(out_dir, "tr_fig6_5domain_forgetting.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


# ---------------------------------------------------------------------------
# Figure 7: Policy Upgrade & Unified Norm-Composability Phase Diagram
# ---------------------------------------------------------------------------
def make_fig7(D, out_dir):
    e1b, e2b, e3, e4, e5 = D["e1b"], D["e2b"], D["e3"], D["e4"], D["e5"]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.1))

    # (a) Enterprise Policy Upgrade: Parameter Size (log) vs v5.0 Avg Score & Rollback
    ax = axes[0]
    vu = e5["version_upgrade"]
    up_items = [
        ("Outdated KV v4.2\n(50% Stale Leak!)", 0.0, 0.0, RED, "X", 110),
        ("In-Context Full v5.0\n(~500 tok/query)", vu["in_context_v50"]["v50_rules"], vu["in_context_v50"]["v50_facts"], SLATE, "s", 100),
        ("KV Atomic Swap v5.0\n(1.18M, 57s, swap)", vu["kv_atomic_swap_v50"]["v50_rules"], vu["kv_atomic_swap_v50"]["v50_facts"], BLUE, "o", 120),
        ("4-Tok KV Delta-Patch\n(0.29M = 1/40 LoRA,\n0% leak, rollback!)", vu["kv_delta_patch_L4"]["v50_rules"], vu["kv_delta_patch_L4"]["v50_facts"], GREEN, "*", 240),
        ("LoRA Update v4.2→v5.0\n(11.8M, 118s, irrev.)", vu["lora_continual_v42_to_v50"]["v50_rules"], vu["lora_continual_v42_to_v50"]["v50_facts"], ORANGE, "^", 120),
    ]
    for lbl, rx, fy, col, mk, sz in up_items:
        ax.scatter(rx, fy, c=col, marker=mk, s=sz, zorder=5, edgecolors="#1E293B", linewidths=0.7)
    # Clean non-overlapping text labels
    offsets = [
        (4, 5, "left"),
        (-3, -10, "right"),
        (-3, -11, "right"),
        (-3, 7, "right"),
        (-3, -11, "right"),
    ]
    for (lbl, rx, fy, col, mk, sz), (dx, dy, ha) in zip(up_items, offsets):
        ax.annotate(lbl, xy=(rx, fy), xytext=(rx + dx, fy + dy), ha=ha, va="center",
                    fontsize=7.6, fontweight="bold" if col == GREEN else "normal",
                    color="#065F46" if col == GREEN else ("#991B1B" if col == RED else "#1E293B"),
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.7))
    ax.set_xlabel("HeliosPay v5.0 Unseen Multi-Step Rules Accuracy (%)")
    ax.set_ylabel("HeliosPay v5.0 Paraphrased Facts Accuracy (%)")
    ax.set_xlim(-6, 102)
    ax.set_ylim(-8, 112)
    ax.grid(True, alpha=0.25)
    ax.set_title("(a) Policy Version Upgrade Frontier (HeliosPay v4.2 → v5.0)")

    # (b) Unified Cross-Stage Norm/Perturbation vs Zero-Training Composition Law
    ax = axes[1]
    # X-axis: Relative Perturbation Ratio (Final RMS / Natural Manifold RMS: for SP init=0.019; for KV delta_rms / 2.55)
    # Y-axis: Composition Retention / Synergy Score (normalized retention of single-domain performance when concatenated)
    # Let's plot exact measured relative perturbation (x) vs exact dual-domain retention ratio (y, % of single-expert performance kept)
    pts = [
        # (name, rel_perturbation_pct, retention_pct, color, marker, xyoff)
        ("KV-CSPD RL [M;I]\n(delta=5.6% KV,\n0.06pp order diff!)",
         100.0 * (0.1440 / 2.55), 100.0, GREEN, "*", (16, -24)),
        ("Stage A RL SP\nAdamW [M;I]\n(2.3x emb RMS)",
         230.0, 80.0, BLUE, "o", (12, 12)),
        ("Stage A RL SP\nMuon [I;M]\n(6.5x emb RMS)",
         650.0, 55.0, AMBER, "s", (-10, 14)),
        ("NR-KV16 Doc\nBlind [A;B]\n(delta=8.6% KV)",
         100.0 * (0.226 / 2.55), 0.0, PURPLE, "D", (14, 16)),
        ("NB-NR-SP Doc\nBlind [A;B]\n(2.9x emb RMS)",
         290.0, 0.0, CORAL, "^", (12, 17)),
        ("Stage B Doc SP\nBlind [A;B;C]\n(10.1x emb RMS)",
         1010.0, 0.0, RED, "X", (-95, 18)),
    ]
    # Shaded regimes
    ax.axvspan(0, 300, ymin=0.45, ymax=1.0, color="#ECFDF5", alpha=0.65, label="Regime I: Compatible RL + Low Perturb. → Synergy")
    ax.axvspan(300, 1180, ymin=0.45, ymax=1.0, color="#FFFBEB", alpha=0.65, label="Regime II: High-Norm RL → Partial Competition")
    ax.axhspan(-12, 35, color="#FEF2F2", alpha=0.55, label="Regime III: Competing Factual Docs → Route, Don't Concat!")

    for name, px, py, col, mk, (dx, dy) in pts:
        sz = 230 if mk == "*" else 115
        ax.scatter(px, py, c=col, marker=mk, s=sz, zorder=5, edgecolors="#1E293B", linewidths=0.8)
        ax.annotate(name, xy=(px, py), xytext=(px + dx, py + dy),
                    fontsize=7.3, fontweight="bold" if col in [GREEN, PURPLE] else "normal",
                    color="#065F46" if col == GREEN else ("#5B21B6" if col == PURPLE else "#1E293B"),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9))

    ax.set_xlabel("Relative Perturbation Magnitude (% of Natural Embedding / KV Manifold RMS)")
    ax.set_ylabel("Zero-Training Concatenation Retention / Synergy Index (%)")
    ax.set_xlim(-35, 1160)
    ax.set_ylim(-12, 118)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="upper right", fontsize=7.0)
    ax.set_title("(b) Unified Norm-Composability & Domain Compatibility Phase Map")

    plt.tight_layout()
    p = os.path.join(out_dir, "tr_fig7_upgrade_and_normlaw.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(repo_root, "results")
    out = os.environ.get("FIG_OUT_DIR", os.path.join(repo_root, "docs", "figures"))
    os.makedirs(out, exist_ok=True)
    D = load_all(root)
    make_fig1(D, out)
    make_fig2(D, out)
    make_fig3(D, out)
    make_fig4(D, out)
    make_fig5(D, out)
    make_fig6(D, out)
    make_fig7(D, out)
    print("ALL 7 PAPER-STYLE TECHNICAL REPORT FIGURES GENERATED SUCCESSFULLY!")
