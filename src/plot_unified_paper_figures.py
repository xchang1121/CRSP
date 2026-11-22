"""
Generate 5 camera-ready, self-contained academic figures for the unified research paper:
  1. paper_fig1_input_sp_distillation.png
  2. paper_fig2_kv_pd_acceleration_and_cl.png
  3. paper_fig3_positional_collapse_and_crsp.png
  4. paper_fig4_multiseed_and_scaled_ifbench64.png
  5. paper_fig5_multiexpert_composition_and_laws.png

All plotted values are loaded directly from the raw JSON outputs in ../results/.
Zero rhetorical phrasing, zero internal suite/experiment IDs, zero external paper references.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = r"C:\Users\singm\.gemini\antigravity\scratch\composable-soft-prompt-distillation"
RES = os.path.join(ROOT, "results")
OUT = r"C:\Users\singm\.gemini\antigravity\brain\8aba4b5d-6a54-4e44-b641-a0c487b94d0d"

def load_json(name):
    with open(os.path.join(RES, name), "r", encoding="utf-8") as f:
        return json.load(f)

# Publication typography and palette (Okabe-Ito / academic neutral palette)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "figure.titlesize": 12.5,
    "figure.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "grid.color": "#CCCCCC",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

C_NAVY   = "#1B4F72"
C_TEAL   = "#008080"
C_GREEN  = "#2E8B57"
C_ORANGE = "#D55E00"
C_AMBER  = "#E69F00"
C_PURPLE = "#6C3483"
C_CRIMSON= "#B03A2E"
C_SLATE  = "#5D6D7E"
C_LIGHT  = "#AEB6BF"
C_BLUE   = "#2874A6"

def style_ax(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

def add_bar_labels(ax, bars, fmt="{:.1f}", dy=0.8, fontsize=7.8, color="#222222", neg_dy=-2.8):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        va = "bottom" if h >= 0 else "top"
        y = h + dy if h >= 0 else h + neg_dy
        ax.text(b.get_x() + b.get_width() / 2.0, y, fmt.format(h),
                ha="center", va=va, fontsize=fontsize, color=color, fontweight="normal")


# ==============================================================================
# FIGURE 1: Input-Space Soft Prompt Distillation & Bottlenecks
# ==============================================================================
def make_fig1():
    d1b = load_json("exp1b_results.json")
    d2b = load_json("exp2b_results.json")
    d3  = load_json("exp3_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.2))
    fig.subplots_adjust(hspace=0.38, wspace=0.28, top=0.92)
    fig.suptitle("Figure 1: Input-Space Soft Prompt Distillation — Alignment, Mechanistic Attribution, Sensitivity, and Composition Limits", y=0.975)

    # (a) IFBench Strict (N=40) & Distributional Gain G_IF (%)
    ax = axes[0, 0]
    style_ax(ax, "y")
    ev = d1b["ifbench_evals"]
    dg = d1b["distributional_G"]
    labels = [
        "Base\nStudent",
        "Single P_IF\n(Muon+PR)",
        "Concat\n[IF; Math]",
        "Concat\n[Math; IF]",
        "Concat NoPR\n[Math; IF]",
        "Concat AdamW\n[Math; IF]",
    ]
    strict_vals = [
        ev["student_no_sp"]["prompt_strict"],
        ev["single_P_if"]["prompt_strict"],
        ev["cspd_concat_IF_Math"]["prompt_strict"],
        ev["cspd_concat_Math_IF"]["prompt_strict"],
        ev["ablation_nopr_Math_IF"]["prompt_strict"],
        ev["ablation_adamw_Math_IF"]["prompt_strict"],
    ]
    g_vals = [
        dg["student_no_sp"]["G_if_pct"],
        dg["single_P_if"]["G_if_pct"],
        dg["cspd_concat_IF_Math"]["G_if_pct"],
        dg["cspd_concat_Math_IF"]["G_if_pct"],
        dg["ablation_nopr_Math_IF"]["G_if_pct"],
        dg["ablation_adamw_Math_IF"]["G_if_pct"],
    ]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w/2, strict_vals, w, label="IFBench Prompt Strict (%, N=40)", color=C_NAVY, zorder=3)
    b2 = ax.bar(x + w/2, g_vals, w, label="Distributional Alignment G_IF (%)", color=C_TEAL, zorder=3)
    teacher_strict = ev["teacher_M2RL_RL_IF"]["prompt_strict"]
    ax.axhline(teacher_strict, color=C_CRIMSON, linestyle="--", linewidth=1.2, zorder=4,
               label=f"Post-RL Teacher Strict ({teacher_strict:.1f}%)")
    ax.axhline(0, color="#333333", linewidth=0.8)
    add_bar_labels(ax, b1, fmt="{:.1f}", dy=1.2, fontsize=7.6)
    add_bar_labels(ax, b2, fmt="{:+.1f}", dy=1.2, neg_dy=-1.5, fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel("Score / Alignment Gain (%)")
    ax.set_ylim(-24, 72)
    ax.set_title("(a) Behavioral Accuracy & Distributional Alignment (L=256)")
    ax.legend(loc="upper center", frameon=True, framealpha=0.95, fontsize=7.6, ncol=2)

    # (b) Mechanistic Attention-Routing Ablation (Held-out IF KL, lower is better)
    ax = axes[0, 1]
    style_ax(ax, "x")
    m = d1b["mechanistic_IF"]["mean_kl"]
    items = [
        ("Full SP (Muon)",        m["sp_muon"],            C_NAVY),
        ("Full SP (AdamW)",       m["sp_adamw"],           C_BLUE),
        ("Attn: Response-Only",   m["attn_response_only"], C_TEAL),
        ("Attn: Input-Only",      m["attn_input_only"],    C_AMBER),
        ("Base Student (No SP)",  m["student"],            C_SLATE),
        ("Text: Neutral Prompt",  m["neutral_instruction"],C_LIGHT),
        ("Text: Check Prompt",    m["check_instruction"],  C_LIGHT),
        ("Text: Verify Prompt",   m["verify_instruction"], C_LIGHT),
        ("Attn: Blocked",         m["attn_blocked"],       C_ORANGE),
        ("Untrained SP Init",     m["sp_init"],            C_CRIMSON),
    ]
    items = items[::-1]
    y_pos = np.arange(len(items))
    vals = [it[1] for it in items]
    cols = [it[2] for it in items]
    bars = ax.barh(y_pos, vals, height=0.66, color=cols, zorder=3)
    ax.axvline(m["student"], color=C_SLATE, linestyle=":", linewidth=1.2, zorder=4)
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height()/2, f"{v:.3f}",
                va="center", ha="left", fontsize=7.8, color="#222222")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([it[0] for it in items], fontsize=8.2)
    ax.set_xlabel("Held-Out Forward KL to RL Teacher (nats, lower is better)")
    ax.set_xlim(0, 0.375)
    ax.set_title("(b) Mechanistic Attention-Routing & Text-Prompt Ablation")
    ax.text(0.205, 7.6, "Response-path recovers\n54.6% of KL gap vs 0.8%\nfor input-only path",
            fontsize=7.7, bbox=dict(boxstyle="round,pad=0.3", fc="#F4F6F7", ec="#85929E", lw=0.8))

    # (c) Multi-Axis Input-Space Sensitivity (45 updates, G_IF %)
    ax = axes[1, 0]
    style_ax(ax, "x")
    a1, a2, a3 = d3["axis1_init"], d3["axis2_loss_trunc"], d3["axis3_opt_len"]
    sens = [
        ("Init: Task-Inst (Muon)",     a1["task_inst"]["final_G_pct"],          C_NAVY),
        ("Opt: AdamW (Low-Norm)",      a3["adamw_lownorm"]["final_G_pct"],      C_BLUE),
        ("Init: Random-Vocab (Muon)",  a1["random_vocab"]["final_G_pct"],       C_TEAL),
        ("Loss: Hybrid Fwd+Rev KL",    a2["hybrid_fwd_rev_768"]["final_G_pct"], C_GREEN),
        ("Loss: Fork Top-30 KL",       a2["fork_top30_768"]["final_G_pct"],     C_GREEN),
        ("Param: Low-Rank r=32",       a3["low_rank_r32"]["final_G_pct"],       C_PURPLE),
        ("Opt: Muon NormClip 0.055",   a3["muon_normclip_055"]["final_G_pct"],  C_AMBER),
        ("Length: L=64",               a3["len_L64"]["final_G_pct"],            C_SLATE),
        ("Init: Low-Norm (Muon)",      a1["low_norm"]["final_G_pct"],           C_SLATE),
        ("Length: L=16",               a3["len_L16"]["final_G_pct"],            C_SLATE),
        ("Loss: Sequence NLL Only",    a2["seq_nll_768"]["final_G_pct"],        C_ORANGE),
        ("Trunc: 256-Token KL",        a2["rb_top64_trunc256"]["final_G_pct"],  C_CRIMSON),
    ]
    sens = sens[::-1]
    y_pos = np.arange(len(sens))
    vals = [s[1] for s in sens]
    cols = [s[2] for s in sens]
    bars = ax.barh(y_pos, vals, height=0.66, color=cols, zorder=3)
    ax.axvline(0, color="#333333", linewidth=0.9)
    for b, v in zip(bars, vals):
        if v >= 0:
            ax.text(v + 1.5, b.get_y() + b.get_height()/2, f"{v:+.1f}%", va="center", ha="left", fontsize=7.7)
        else:
            ax.text(v - 1.5, b.get_y() + b.get_height()/2, f"{v:+.1f}%", va="center", ha="right", fontsize=7.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in sens], fontsize=8.0)
    ax.set_xlabel("Held-Out Distributional Alignment Gain G_IF (%) at 45 Updates")
    ax.set_xlim(-74, 48)
    ax.set_title("(c) Input-Space Optimization Sensitivity Across 3 Axes")

    # (d) Zero-Leakage Declarative Continual Learning Failure in Input Space
    ax = axes[1, 1]
    style_ax(ax, "y")
    r2 = d2b["runs"]
    groups = ["Domain A\n(HeliosPay)", "Domain B\n(NVX-418)", "Domain C\n(Litho-X9)"]
    single_sp = [r2["stage1_PA"]["A_heliospay"]["overall"],
                 r2["single_PB"]["B_nvx418"]["overall"],
                 r2["single_PC"]["C_lithox9"]["overall"]]
    concat_ab = [r2["stage2_PA_PB"]["A_heliospay"]["overall"],
                 r2["stage2_PA_PB"]["B_nvx418"]["overall"],
                 r2["stage2_PA_PB"]["C_lithox9"]["overall"]]
    concat_abc = [r2["stage3_PA_PB_PC"]["A_heliospay"]["overall"],
                  r2["stage3_PA_PB_PC"]["B_nvx418"]["overall"],
                  r2["stage3_PA_PB_PC"]["C_lithox9"]["overall"]]
    lora_s123_peak = [r2["lora_seq_stage1_A"]["A_heliospay"]["overall"],
                      r2["lora_seq_stage2_AB"]["B_nvx418"]["overall"],
                      r2["lora_seq_stage3_ABC"]["C_lithox9"]["overall"]]
    lora_s3 = [r2["lora_seq_stage3_ABC"]["A_heliospay"]["overall"],
               r2["lora_seq_stage3_ABC"]["B_nvx418"]["overall"],
               r2["lora_seq_stage3_ABC"]["C_lithox9"]["overall"]]
    x = np.arange(3)
    w = 0.17
    b1 = ax.bar(x - 2*w, single_sp,      w, label="Input-SP Solo (L=128)", color=C_NAVY, zorder=3)
    b2 = ax.bar(x - 1*w, concat_ab,      w, label="Input-SP Concat [A;B]", color=C_AMBER, zorder=3)
    b3 = ax.bar(x + 0*w, concat_abc,     w, label="Input-SP Concat [A;B;C]", color=C_CRIMSON, zorder=3)
    b4 = ax.bar(x + 1*w, lora_s123_peak, w, label="Seq LoRA (Peak at Stage t)", color=C_LIGHT, edgecolor="#555555", lw=0.6, zorder=3)
    b5 = ax.bar(x + 2*w, lora_s3,        w, label="Seq LoRA (After Stage 3)", color=C_ORANGE, zorder=3)
    for bars in [b1, b2, b3, b4, b5]:
        add_bar_labels(ax, bars, fmt="{:.1f}", dy=1.0, fontsize=6.9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("Zero-Leakage Overall Accuracy (%)")
    ax.set_ylim(0, 88)
    ax.set_title("(d) Multi-Domain Declarative Failure: Concat Collapse vs LoRA Forgetting")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.4, ncol=1)

    path = os.path.join(OUT, "paper_fig1_input_sp_distillation.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIGURE 2: KV-Space Prompt Distillation (KV-PD) & Zero-Forgetting Continual Learning
# ==============================================================================
def make_fig2():
    d3 = load_json("exp3_results.json")
    d4 = load_json("exp4_results.json")
    d5 = load_json("exp5_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.2))
    fig.subplots_adjust(hspace=0.38, wspace=0.28, top=0.92)
    fig.suptitle("Figure 2: KV-Space Prompt Distillation (KV-PD) — Acceleration, Compositional Stability, Continual Learning, and Delta Patching", y=0.975)

    # (a) Wall-clock convergence curves: KV-PD vs Input-SP
    ax = axes[0, 0]
    style_ax(ax, "both")
    curves = [
        ("KV-PD L=8 (590k params)",         d3["axis4_kv_cspd"]["kv_if_L8"]["curve"],          C_TEAL,   "o-", 2.2),
        ("KV-PD L=16 (1.18M params)",       d3["axis4_kv_cspd"]["kv_if_L16"]["curve"],         C_NAVY,   "s-", 2.0),
        ("Input-SP Task-Inst (655k)",       d3["axis1_init"]["task_inst"]["curve"],            C_ORANGE, "^--",1.8),
        ("Input-SP AdamW Low-Norm (655k)",  d3["axis3_opt_len"]["adamw_lownorm"]["curve"],     C_AMBER,  "d--",1.7),
        ("Input-SP Random-Vocab (655k)",    d3["axis1_init"]["random_vocab"]["curve"],         C_PURPLE, "v:", 1.6),
        ("Input-SP Muon NormClip (655k)",   d3["axis3_opt_len"]["muon_normclip_055"]["curve"], C_SLATE,  "x:", 1.5),
    ]
    for label, pts, col, fmt, lw in curves:
        xs = [p["train_sec"] for p in pts]
        ys = [p["G_pct"] for p in pts]
        ax.plot(xs, ys, fmt, color=col, linewidth=lw, markersize=4.8, label=label, zorder=4)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.annotate("KV-PD L=8: +65.5%\nin 72.3s (2.71x faster)",
                xy=(72.3, 65.46), xytext=(95, 57),
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.2),
                fontsize=8.0, fontweight="bold", color=C_TEAL,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_TEAL, lw=0.8))
    ax.annotate("Best Input-SP: +34.6%\nin 196.7s",
                xy=(196.7, 34.59), xytext=(116, 43),
                arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.2),
                fontsize=8.0, color=C_ORANGE,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_ORANGE, lw=0.8))
    ax.set_xlabel("Wall-Clock Training Time on A100-80GB (seconds)")
    ax.set_ylabel("Held-Out Distributional Gain G_IF (%)")
    ax.set_xlim(-5, 212)
    ax.set_ylim(-85, 78)
    ax.set_title("(a) Wall-Clock Convergence: KV-PD vs Input-Space SP")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.5)

    # (b) Dual-Expert Composition Stability: Input-SP vs KV-PD
    ax = axes[0, 1]
    style_ax(ax, "y")
    dc = d3["dual_composition_G"]
    keys = [
        ("SP\nSolo IF",    "NormClipMuon_Single_IF"),
        ("SP\nSolo Math",  "NormClipMuon_Single_Math"),
        ("SP\n[Math; IF]", "NormClipMuon_Concat_M_I"),
        ("SP\n[IF; Math]", "NormClipMuon_Concat_I_M"),
        ("KV L=16\nSolo IF",   "KV16_Single_IF"),
        ("KV L=16\nSolo Math", "KV16_Single_Math"),
        ("KV L=16\n[Math; IF]","KV16_Concat_M_I"),
        ("KV L=16\n[IF; Math]","KV16_Concat_I_M"),
    ]
    x = np.arange(len(keys))
    w = 0.27
    g_if   = [dc[k]["G_if_pct"]   for _, k in keys]
    g_math = [dc[k]["G_math_pct"] for _, k in keys]
    g_avg  = [dc[k]["G_avg_pct"]  for _, k in keys]
    b1 = ax.bar(x - w, g_if,   w, label="G_IF (%)",   color=C_NAVY, zorder=3)
    b2 = ax.bar(x,     g_math, w, label="G_Math (%)", color=C_TEAL, zorder=3)
    b3 = ax.bar(x + w, g_avg,  w, label="G_Avg (%)",  color=C_PURPLE, zorder=3)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.axvline(3.5, color="#777777", linestyle="--", linewidth=1.0)
    for b in b3:
        h = b.get_height()
        y = h + 1.8 if h >= 0 else h - 2.2
        ax.text(b.get_x() + b.get_width()/2 + 0.04, y, f"{h:+.1f}", ha="center", va="bottom" if h >= 0 else "top",
                fontsize=7.3, fontweight="bold", color=C_PURPLE)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in keys], fontsize=7.6)
    ax.set_ylabel("Distributional Alignment Gain G (%)")
    ax.set_ylim(-64, 72)
    ax.set_title("(b) Dual-Expert Composition Stability & Order Invariance")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.8, ncol=3)
    ax.text(5.55, 50, "KV-PD Dual Order Diff:\n|19.96 - 19.84| = 0.12 pp",
            fontsize=7.7, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB", ec=C_NAVY, lw=0.8))

    # (c) 5-Domain Zero-Leakage Continual Learning & Declarative Factorization
    ax = axes[1, 0]
    style_ax(ax, "y")
    r4 = d4["runs"]
    ar = d5["auto_router"]
    sl = d5["seq_lora_5"]["stage5_E"]
    methods = [
        "Seq LoRA\n(5-Dom Final)",
        "Micro-RAG\nOnly (3-Dom)",
        "Input-SP L=128\n(Dom A Solo)",
        "KV-PD L=32\n(Dom A Solo)",
        "KV-PD L=16\nAuto-Routed\n(5-Dom)",
        "KV-PD L=16 +\nMicro-RAG\n(5-Dom)",
    ]
    rules_v = [
        sl["mean_rules"],
        r4["micro_rag_only"]["mean_rules"],
        r4["SP_A_random_vocab"]["mean_rules"],
        r4["KV_A_L32_AdamW"]["mean_rules"],
        ar["e2e_kv16_auto_routed"]["mean_rules"],
        ar["e2e_kv16_microRAG_auto_routed"]["mean_rules"],
    ]
    facts_v = [
        sl["mean_facts"],
        r4["micro_rag_only"]["mean_facts"],
        r4["SP_A_random_vocab"]["mean_facts"],
        r4["KV_A_L32_AdamW"]["mean_facts"],
        ar["e2e_kv16_auto_routed"]["mean_facts"],
        ar["e2e_kv16_microRAG_auto_routed"]["mean_facts"],
    ]
    x = np.arange(len(methods))
    w = 0.36
    b1 = ax.bar(x - w/2, rules_v, w, label="Procedural Decision Rules Acc (%)", color=C_NAVY, zorder=3)
    b2 = ax.bar(x + w/2, facts_v, w, label="Verbatim Constant Facts Acc (%)", color=C_TEAL, zorder=3)
    add_bar_labels(ax, b1, fmt="{:.1f}", dy=1.4, fontsize=7.6)
    add_bar_labels(ax, b2, fmt="{:.1f}", dy=1.4, fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7.7)
    ax.set_ylabel("Zero-Leakage Accuracy (%)")
    ax.set_ylim(0, 128)
    ax.set_title("(c) Zero-Leakage Declarative Factorization & 5-Domain Continual Learning")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=7.7)
    ax.text(4.35, 113, "5-Dom OOD Abstention: 91.7%\nZero Cross-Domain Forgetting",
            ha="center", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.25", fc="#E8F8F5", ec=C_TEAL, lw=0.8))

    # (d) Dynamic Version Upgrade (HeliosPay v4.2 -> v5.0) & 4-Token KV Delta Patch
    ax = axes[1, 1]
    style_ax(ax, "y")
    vu = d5["version_upgrade"]
    v_labels = [
        "Outdated\nKV v4.2",
        "In-Context\nManual v5.0",
        "KV Atomic\nSwap (L=16)",
        "KV Delta Patch\n(L=4, 295k, 57s)",
        "LoRA Continual\n(v4.2 -> v5.0)",
    ]
    v_keys = ["outdated_KV_v42", "in_context_v50", "kv_atomic_swap_v50", "kv_delta_patch_L4", "lora_continual_v42_to_v50"]
    v_rules = [vu[k]["v50_rules"] for k in v_keys]
    v_facts = [vu[k]["v50_facts"] for k in v_keys]
    v_stale = [vu[k]["stale_v42_rules_pct"] for k in v_keys]
    x = np.arange(len(v_labels))
    w = 0.26
    b1 = ax.bar(x - w, v_rules, w, label="v5.0 Rules Acc (%)", color=C_NAVY, zorder=3)
    b2 = ax.bar(x,     v_facts, w, label="v5.0 Facts Acc (%)", color=C_TEAL, zorder=3)
    b3 = ax.bar(x + w, v_stale, w, label="Stale v4.2 Rule Leakage (%, lower better)", color=C_CRIMSON, zorder=3)
    add_bar_labels(ax, b1, fmt="{:.1f}", dy=1.4, fontsize=7.3)
    add_bar_labels(ax, b2, fmt="{:.1f}", dy=1.4, fontsize=7.3)
    add_bar_labels(ax, b3, fmt="{:.1f}", dy=1.4, fontsize=7.3)
    ax.set_xticks(x)
    ax.set_xticklabels(v_labels, fontsize=7.8)
    ax.set_ylabel("Accuracy / Stale Leakage Rate (%)")
    ax.set_ylim(0, 118)
    ax.set_title("(d) Policy Version Upgrade (v4.2 -> v5.0) & 4-Token KV Delta Patch")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.5)

    path = os.path.join(OUT, "paper_fig2_kv_pd_acceleration_and_cl.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIGURE 3: Positional Collapse, CRSP, KV Functional Inversion & Zero-Post-RL
# ==============================================================================
def make_fig3():
    d7 = load_json("exp7_results.json")
    d8 = load_json("exp8_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.2))
    fig.subplots_adjust(hspace=0.38, wspace=0.28, top=0.92)
    fig.suptitle("Figure 3: Context-Robust Soft Prompts (CRSP) — Positional Collapse, KV Functional Inversion, and Zero-Post-RL Pathways", y=0.975)

    regimes = ["P0_Clean_Start", "P1_MidSys_Persona", "P2_MultiTurn_Mid",
               "P3_PostRAG_Doc", "P4_PostQuery_Suffix", "P5_LongDrift_1000"]
    r_short = ["P0\nClean", "P1\nMidSys", "P2\nMultiTurn", "P3\nPostRAG", "P4\nPostQuery", "P5\nLong1000"]
    rx = np.arange(len(regimes))

    # (a) 6-Regime Positional Gain G_vs_ctx (exp7)
    ax = axes[0, 0]
    style_ax(ax, "both")
    pm7 = d7["position_matrix"]["methods"]
    lines7 = [
        ("Full CRSP + L_inv (L=256, mean P1-P5: +35.2%)", "A4_Full_CR_SP_Inv",  C_NAVY,    "o-", 2.3),
        ("Multi-Slot CRSP (L=256, mean P1-P5: +35.8%)",   "A3_MultiSlot_CR_SP", C_TEAL,    "s--",1.9),
        ("Compact CRSP (L=64, mean P1-P5: +32.6%)",       "A5_CR_SP_L64",       C_GREEN,   "^-", 1.9),
        ("Fixed-P0 KV-PD (L=16, prefix-only ref)",        "KV_Distilled_L16",   C_PURPLE,  "D:", 1.8),
        ("Prefix-Robust SP (mean P1-P5: +14.4%)",         "A2_Paper_PR_SP",     C_AMBER,   "v-", 1.9),
        ("Pos0-Only SP (mean P1-P5: +8.9%)",              "A1_Pos0_Only_SP",    C_CRIMSON, "x-", 2.0),
    ]
    for lbl, key, col, fmt, lw in lines7:
        ys = [pm7[key][r]["G_vs_ctx_student_pct"] for r in regimes]
        ax.plot(rx, ys, fmt, color=col, linewidth=lw, markersize=5.2, label=lbl, zorder=4)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.annotate("PR-SP collapse\n(-12.1%)", xy=(4, -12.1), xytext=(3.15, -24),
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.1), fontsize=7.7, color="#9A6700")
    ax.annotate("Pos0-Only (-1.3%)", xy=(2, -1.25), xytext=(2.25, 9.5),
                arrowprops=dict(arrowstyle="->", color=C_CRIMSON, lw=1.1), fontsize=7.6, color=C_CRIMSON)
    ax.set_xticks(rx)
    ax.set_xticklabels(r_short, fontsize=8.2)
    ax.set_ylabel("Distributional Alignment Gain G vs Context Student (%)")
    ax.set_ylim(-30, 56)
    ax.set_title("(a) 6-Regime Positional Distributional Gain (P0 to P5)")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=7.1)

    # (b) IFBench32 Behavioral Accuracy Across Deployment Positions (exp7)
    ax = axes[0, 1]
    style_ax(ax, "y")
    ib7 = d7["ifbench32_by_position"]
    slots = ["P0 Clean", "P2 MultiTurn", "P3 Post-RAG", "P4 Post-Query"]
    stu   = [ib7["P0_Student"]["prompt_strict"], ib7["P2_Student"]["prompt_strict"],
             ib7["P3_Student"]["prompt_strict"], ib7["P2_Student"]["prompt_strict"]]
    pos0  = [ib7["P0_A1_Pos0_Only"]["prompt_strict"], ib7["P2_A1_Pos0_Only"]["prompt_strict"],
             ib7["P3_A1_Pos0_Only"]["prompt_strict"], ib7["P4_A1_Pos0_Only"]["prompt_strict"]]
    prsp  = [np.nan, ib7["P2_A2_Paper_PR"]["prompt_strict"],
             np.nan, ib7["P4_A2_Paper_PR"]["prompt_strict"]]
    crsp  = [ib7["P0_A4_Full_CR_Inv"]["prompt_strict"], ib7["P2_A4_Full_CR_Inv"]["prompt_strict"],
             ib7["P3_A4_Full_CR_Inv"]["prompt_strict"], ib7["P4_A4_Full_CR_Inv"]["prompt_strict"]]
    crsp_cl = [ib7["P0_A4_Full_CR_Inv"]["closed_acc"], ib7["P2_A4_Full_CR_Inv"]["closed_acc"],
               ib7["P3_A4_Full_CR_Inv"]["closed_acc"], ib7["P4_A4_Full_CR_Inv"]["closed_acc"]]
    x = np.arange(len(slots))
    w = 0.20
    b1 = ax.bar(x - 1.5*w, stu,  w, label="Base Student", color=C_SLATE, zorder=3)
    b2 = ax.bar(x - 0.5*w, pos0, w, label="Pos0-Only SP", color=C_CRIMSON, zorder=3)
    b3 = ax.bar(x + 0.5*w, prsp, w, label="Prefix-Robust SP (P2, P4)", color=C_AMBER, zorder=3)
    b4 = ax.bar(x + 1.5*w, crsp, w, label="Full CRSP + L_inv (L=256)", color=C_NAVY, zorder=3)
    for bars in [b1, b2, b3, b4]:
        add_bar_labels(ax, bars, fmt="{:.1f}", dy=1.0, fontsize=7.2)
    for i, cval in enumerate(crsp_cl):
        ax.text(x[i] + 1.5*w, crsp[i] + 5.2, f"({cval:.0f}% cl.)", ha="center", fontsize=6.9, color=C_NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels(slots, fontsize=8.3)
    ax.set_ylabel("IFBench32 Prompt Strict Accuracy (%, N=32)")
    ax.set_ylim(0, 64)
    ax.set_title("(b) End-to-End Behavioral Accuracy Across Insertion Slots")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.5, ncol=2)

    # (c) KV-to-CRSP Functional Inversion Ablation (exp8 Q1)
    ax = axes[1, 0]
    style_ax(ax, "both")
    pm8 = d8["position_matrix"]["methods"]
    inv_lines = [
        ("Inv-2: Pure-Logit CR Inv (L=256, drift=0.056)", "Inv2_LogitCR_L256",  C_TEAL,    "o-", 2.3),
        ("Inv-3: Hidden+Logit Inv (L=256, drift=0.065)",  "Inv3_FuncOp_L256",   C_NAVY,    "s--",1.9),
        ("Inv-3: Hidden+Logit Inv (L=64, drift=0.077)",   "Inv3_FuncOp_L64",    C_BLUE,    "^:", 1.8),
        ("Inv-1b: Data-Free CR Geo KV (drift=0.142)",     "Inv1b_CR_GeoKV_DF",  C_AMBER,   "v-", 1.8),
        ("Inv-1a: Data-Free Geo KV (drift=0.180)",        "Inv1a_GeoKV_DF",     C_CRIMSON, "x-", 1.9),
    ]
    for lbl, key, col, fmt, lw in inv_lines:
        ys = [pm8[key][r]["G_vs_ctx_student_pct"] for r in regimes]
        ax.plot(rx, ys, fmt, color=col, linewidth=lw, markersize=5.2, label=lbl, zorder=4)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(rx)
    ax.set_xticklabels(r_short, fontsize=8.2)
    ax.set_ylabel("Distributional Alignment Gain G vs Context Student (%)")
    ax.set_ylim(-15, 64)
    ax.set_title("(c) KV-to-CRSP Inversion Ablation: Pure-Logit vs Hidden & Geometric")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=7.2)

    # (d) Zero-Post-RL CRSP Pathways (TF1-TF4) on IFBench32 (exp8 Q2)
    ax = axes[1, 1]
    style_ax(ax, "y")
    ib8 = d8["ifbench32_evals"]
    tf_labels = [
        "Base\nStudent",
        "TF-KV\nGRPO (P0)",
        "TF1\nKV->Inv",
        "TF2\nDirect GRPO",
        "TF3\nSelf-Boot",
        "TF4\nSelf-Rubric",
        "Post-RL\nInv-3 (Ref)",
    ]
    p2_vals = [
        ib7["P2_Student"]["prompt_strict"],
        ib8["P0_KV_GRPO_TF_L16"]["prompt_strict"],  # P0 KV ref
        ib8["P2_TF1_CR_KVGRPO_Inv"]["prompt_strict"],
        ib8["P2_TF2_CR_Direct_GRPO"]["prompt_strict"],
        ib8["P2_TF3_CR_Anchored_SelfBoot"]["prompt_strict"],
        ib8["P2_TF4_CR_SelfRubric"]["prompt_strict"],
        ib8["P2_Inv3_FuncOp_L256"]["prompt_strict"],
    ]
    p4_vals = [
        np.nan,
        np.nan,
        ib8["P4_TF1_CR_KVGRPO_Inv"]["prompt_strict"],
        ib8["P4_TF2_CR_Direct_GRPO"]["prompt_strict"],
        ib8["P4_TF3_CR_Anchored_SelfBoot"]["prompt_strict"],
        ib8["P4_TF4_CR_SelfRubric"]["prompt_strict"],
        ib8["P4_Inv3_FuncOp_L256"]["prompt_strict"],
    ]
    x = np.arange(len(tf_labels))
    w = 0.36
    b1 = ax.bar(x - w/2, p2_vals, w, color=C_BLUE, zorder=3)
    b2 = ax.bar(x + w/2, p4_vals, w, color=C_TEAL, zorder=3)
    b1[0].set_color(C_SLATE)
    b1[1].set_color(C_PURPLE)
    b1[-1].set_color(C_NAVY)
    b2[-1].set_color(C_GREEN)
    add_bar_labels(ax, b1, fmt="{:.1f}", dy=1.0, fontsize=7.3)
    add_bar_labels(ax, b2, fmt="{:.1f}", dy=1.0, fontsize=7.3)
    ax.axhline(ib7["P2_Student"]["prompt_strict"], color=C_SLATE, linestyle=":", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(tf_labels, fontsize=7.5)
    ax.set_ylabel("IFBench32 Prompt Strict Accuracy (%, N=32)")
    ax.set_ylim(0, 64)
    ax.set_title("(d) Zero-Post-RL CRSP Construction Pathways (TF1-TF4)")
    handles = [
        mpatches.Patch(color=C_BLUE,   label="TF1-TF4 @ P2 Multi-Turn Mid"),
        mpatches.Patch(color=C_TEAL,   label="TF1-TF4 @ P4 Post-Query"),
        mpatches.Patch(color=C_SLATE,  label="Base Student (P2)"),
        mpatches.Patch(color=C_PURPLE, label="TF-KV GRPO Source (P0)"),
        mpatches.Patch(color=C_NAVY,   label="Post-RL Inv-3 Ref (P2 / P4)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95, fontsize=7.1, ncol=2)

    path = os.path.join(OUT, "paper_fig3_positional_collapse_and_crsp.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIGURE 4: Multi-Seed Stability, Scaled IFBench64 & Adversarial P3 Post-RAG
# ==============================================================================
def make_fig4():
    d9  = load_json("exp9_results.json")
    d9b = load_json("exp9b_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.4))
    fig.subplots_adjust(hspace=0.40, wspace=0.28, top=0.92)
    fig.suptitle("Figure 4: Multi-Seed Stability (IFEval96), Scaled Official IFBench64 (N=64), and Adversarial P3 Post-RAG Evaluation", y=0.975)

    # (a) 3-Seed IFEval96 Mean +/- SD + Seed Points
    ax = axes[0, 0]
    style_ax(ax, "y")
    s9  = d9["track1_ifeval96_summary"]
    s9b = d9b["ifeval96_multiseed_summary"]
    ev9b = d9b["ifeval96_evals"]
    bare_if96 = d9["track1_ifeval96_multiseed"]["Bare_Student"]["prompt_strict"]

    items = [
        ("TF3\n(P2)",          s9["P2_Stage1_TF3"]["strict_mean"],      s9["P2_Stage1_TF3"]["strict_std"],      s9["P2_Stage1_TF3"]["strict_per_seed"],      C_AMBER),
        ("TF2\n(P2)",          s9["P2_Cold_TF2_step10"]["strict_mean"], s9["P2_Cold_TF2_step10"]["strict_std"], s9["P2_Cold_TF2_step10"]["strict_per_seed"], C_ORANGE),
        ("Alg B\n(P2)",        s9["P2_Full_AlgB_step10"]["strict_mean"],s9["P2_Full_AlgB_step10"]["strict_std"],s9["P2_Full_AlgB_step10"]["strict_per_seed"],C_GREEN),
        ("Direct\nL64 (P2)",   s9b["P2_PostRL_Direct_L64"]["strict_mean"],s9b["P2_PostRL_Direct_L64"]["strict_std"],s9b["P2_PostRL_Direct_L64"]["strict_per_seed"],C_BLUE),
        ("Alg B*\n(P4)",       s9["P4_Full_AlgB_step10"]["strict_mean"],s9["P4_Full_AlgB_step10"]["strict_std"],s9["P4_Full_AlgB_step10"]["strict_per_seed"],C_CRIMSON),
        ("Direct\nL64 (P4)",   s9b["P4_PostRL_Direct_L64"]["strict_mean"],s9b["P4_PostRL_Direct_L64"]["strict_std"],s9b["P4_PostRL_Direct_L64"]["strict_per_seed"],C_NAVY),
        ("KV-Inv2\nL64 (P4)",  ev9b["P4_PostRL_KVInv2_L64"]["prompt_strict"], 0.0, [], C_TEAL),
        ("Direct\nL256 (P4)",  ev9b["P4_PostRL_Direct_L256"]["prompt_strict"],0.0, [], C_PURPLE),
    ]
    x = np.arange(len(items))
    means = [it[1] for it in items]
    stds  = [it[2] for it in items]
    cols  = [it[4] for it in items]
    bars = ax.bar(x, means, width=0.60, color=cols, alpha=0.88, zorder=3)
    for i, (lbl, m, sd, pts, col) in enumerate(items):
        if sd > 0:
            ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="#222222", elinewidth=1.2, capsize=4, zorder=5)
            for j, p in enumerate(pts):
                ax.scatter(i + (j - 1)*0.11, p, color="white", edgecolor="#111111", s=24, linewidth=0.9, zorder=6)
            top_y = max(m + sd, max(pts)) + 2.0
            ax.text(i, top_y, f"{m:.1f}\n±{sd:.1f}", ha="center", va="bottom", fontsize=7.1, fontweight="normal")
        else:
            ax.text(i, m + 1.8, f"{m:.1f}\n(1-seed)", ha="center", va="bottom", fontsize=7.0, color="#222222")
    ax.axhline(bare_if96, color=C_SLATE, linestyle="--", linewidth=1.2, zorder=4, label=f"Bare Student ({bare_if96:.1f}%)")
    ax.axvline(3.5, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([it[0] for it in items], fontsize=7.6)
    ax.set_ylabel("Held-Out IFEval96 Prompt Strict Acc (%, N=96)")
    ax.set_ylim(30, 88)
    ax.set_title("(a) 3-Seed Stability on Held-Out IFEval96 (*P4 Co-Presence Mismatch)")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.8)

    # (b) Scaled Official IFBench64 (N=64) across all pathways
    ax = axes[0, 1]
    style_ax(ax, "y")
    ib9  = d9["track1_ifbench64"]
    ib9b = d9b["ifbench64_evals"]
    b_entries = [
        ("Bare\nStudent",      ib9["Bare_Student"]),
        ("P0 KV\nL16",         ib9b["P0_KV_PostRL_IF_L16"]),
        ("Cold\nTF2",          ib9["P4_Cold_TF2"]),
        ("Stage1\nTF3",        ib9["P4_Stage1_TF3"]),
        ("Alg B\nP4",          ib9["P4_Full_AlgB"]),
        ("Direct\nL64 P2",     ib9b["P2_PostRL_Direct_L64"]),
        ("Direct\nL256 P4",    ib9b["P4_PostRL_Direct_L256"]),
        ("KV-Inv2\nL256 P2",   ib9b["P2_PostRL_KVInv2_L256"]),
        ("KV-Inv2\nL64 P4",    ib9b["P4_PostRL_KVInv2_L64"]),
    ]
    x = np.arange(len(b_entries))
    w = 0.36
    sv = [e[1]["prompt_strict"] for e in b_entries]
    cv = [e[1]["closed_acc"]    for e in b_entries]
    b1 = ax.bar(x - w/2, sv, w, label="Prompt Strict Acc (%)", color=C_NAVY, zorder=3)
    b2 = ax.bar(x + w/2, cv, w, label="Closed-Think Acc (%)", color=C_TEAL, alpha=0.88, zorder=3)
    b1[-1].set_color(C_GREEN)
    b2[-1].set_color("#145A32")
    add_bar_labels(ax, b1, fmt="{:.1f}", dy=1.2, fontsize=7.0)
    add_bar_labels(ax, b2, fmt="{:.1f}", dy=1.2, fontsize=7.0)
    ax.axhline(ib9["Bare_Student"]["prompt_strict"], color=C_SLATE, linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in b_entries], fontsize=7.5)
    ax.set_ylabel("IFBench64 Accuracy (%, N=64)")
    ax.set_ylim(0, 99)
    ax.set_title("(b) Scaled Official IFBench64 Across Pathways (Strict & Closed)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=7.8)
    ax.annotate("141.7% retention\nof P0 KV-L16", xy=(7.82, 57.2), xytext=(6.05, 88.0),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2, connectionstyle="arc3,rad=-0.22"),
                fontsize=7.6, fontweight="bold", color="#145A32",
                bbox=dict(boxstyle="round,pad=0.25", fc="#EAFAF1", ec=C_GREEN, lw=0.8))

    # (c) Adversarial P3 Post-RAG 4-Tier Monotonic Hierarchy (IFBench64)
    ax = axes[1, 0]
    style_ax(ax, "y")
    p3_entries = [
        ("Tier 1: Zero-Post-RL\nAlg B (L=64)",       ib9b["P3_ZeroPostRL_AlgB_L64"]),
        ("Tier 2: Frozen\nBare Student",             ib9b["P3_Bare_Student"]),
        ("Tier 3: Post-RL\nAlg A KV-Inv2 (L=256)",   ib9b["P3_PostRL_KVInv2_L256"]),
        ("Tier 4: Post-RL\nDirect CRSP (L=256)",     ib9b["P3_PostRL_Direct_L256"]),
    ]
    x = np.arange(len(p3_entries))
    w = 0.25
    ps = [e[1]["prompt_strict"] for e in p3_entries]
    ins = [e[1]["inst_strict"]  for e in p3_entries]
    cl = [e[1]["closed_acc"]    for e in p3_entries]
    b1 = ax.bar(x - w, ps,  w, label="Prompt Strict (%)", color=C_NAVY, zorder=3)
    b2 = ax.bar(x,     ins, w, label="Instruction Strict (%)", color=C_BLUE, zorder=3)
    b3 = ax.bar(x + w, cl,  w, label="Closed-Think Acc (%)", color=C_TEAL, zorder=3)
    add_bar_labels(ax, b1, fmt="{:.1f}", dy=1.2, fontsize=7.6)
    add_bar_labels(ax, b2, fmt="{:.1f}", dy=1.2, fontsize=7.6)
    add_bar_labels(ax, b3, fmt="{:.1f}", dy=1.2, fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in p3_entries], fontsize=8.0)
    ax.set_ylabel("Adversarial P3 Post-RAG Accuracy (%, N=64)")
    ax.set_ylim(0, 84)
    ax.set_title("(c) Adversarial P3 Post-RAG (+420-Token Distractor) Monotonic Hierarchy")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.7)
    ax.text(3.0, 74.5, "0.00 pp drift vs P4\n(+9.4 pp vs Bare)", ha="center", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.25", fc="#EBF5FB", ec=C_NAVY, lw=0.8))

    # (d) Strict Accuracy vs Closed-Think Precision Pareto Plane (IFBench64)
    ax = axes[1, 1]
    style_ax(ax, "both")
    pts = [
        ("Bare Student",           ib9["Bare_Student"],               C_SLATE,   "s", (-78, 12)),
        ("P3 Bare Student",        ib9b["P3_Bare_Student"],           C_LIGHT,   "s", (-70, -24)),
        ("P0 Source KV-L16",       ib9b["P0_KV_PostRL_IF_L16"],       C_PURPLE,  "D", (-15, -30)),
        ("Zero-RL Cold TF2",       ib9["P4_Cold_TF2"],                C_CRIMSON, "v", (8, -6)),
        ("Zero-RL Alg B (P4)",     ib9["P4_Full_AlgB"],               C_AMBER,   "^", (-92, -16)),
        ("Direct L64 (P2)",        ib9b["P2_PostRL_Direct_L64"],      C_BLUE,    "o", (12, -10)),
        ("Direct L256 (P4)",       ib9b["P4_PostRL_Direct_L256"],     C_NAVY,    "o", (-112, 16)),
        ("KV-Inv2 L256 (P2)",      ib9b["P2_PostRL_KVInv2_L256"],     C_TEAL,    "P", (14, 14)),
        ("Alg A KV-Inv2 L64 (P4)", ib9b["P4_PostRL_KVInv2_L64"],      C_GREEN,   "*", (-128, -4)),
        ("Triple Alg A KV-Inv2",   ib9b["Triple_PostRL_KVInv2_Split"],C_ORANGE,  "X", (-105, -42)),
    ]
    for lbl, rec, col, mk, offset in pts:
        sx, sy = rec["closed_acc"], rec["prompt_strict"]
        ms = 145 if mk == "*" else 68
        ax.scatter(sx, sy, color=col, marker=mk, s=ms, edgecolor="#111111", linewidth=0.7, zorder=5)
        ax.annotate(f"{lbl}\n({sy:.1f}%, {sx:.1f}%)", xy=(sx, sy), xytext=offset,
                    textcoords="offset points", fontsize=7.1, color="#222222",
                    arrowprops=dict(arrowstyle="->", color="#777777", lw=0.65))
    ax.set_xlabel("Closed-Think Precision: Strict / Closed-Think Count (%)")
    ax.set_ylabel("Overall Prompt Strict Accuracy (%)")
    ax.set_xlim(43, 89)
    ax.set_ylim(29, 58)
    ax.set_title("(d) Strict Accuracy vs Closed-Think Precision Pareto Plane (IFBench64)")

    path = os.path.join(OUT, "paper_fig4_multiseed_and_scaled_ifbench64.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIGURE 5: Multi-Expert Arbitrary-Slot Composition & Structural Laws
# ==============================================================================
def make_fig5():
    d4  = load_json("exp4_results.json")
    d5  = load_json("exp5_results.json")
    d9  = load_json("exp9_results.json")
    d9b = load_json("exp9b_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.4))
    fig.subplots_adjust(hspace=0.40, wspace=0.28, top=0.92)
    fig.suptitle("Figure 5: Multi-Expert Arbitrary-Slot Composition — Interference Suppression, Permutation Invariance, and Declarative Duality", y=0.975)

    # (a) 4-Way Cross-Expert Top-32 KL Interference Matrix
    ax = axes[0, 0]
    style_ax(ax, "y")
    k9  = d9["track2_interference_kl"]
    k9b = d9b["interference_kl"]
    packs = [
        ("Naive CRSP\n(192v)",             k9["Naive"]),
        ("Zero-RL\nCS-CR (192v)",          k9["CSCR"]),
        ("Post-RL Direct\nCS-CR (256v)",   k9b["PostRL_Direct_CSCR"]),
        ("Post-RL Alg A\nKV-Inv2 (256v)",  k9b["PostRL_KVInv2_CSCR"]),
    ]
    x = np.arange(len(packs))
    w = 0.20
    if_tr   = [p[1]["if"]["triple_interference_kl"] for p in packs]
    doc_if  = [p[1]["if"]["kl_from_bare_doc"]       for p in packs]
    math_tr = [p[1]["math"]["triple_interference_kl"] for p in packs]
    if_sig  = [p[1]["if"]["kl_from_bare_if"]        for p in packs]

    b1 = ax.bar(x - 1.5*w, if_tr,   w, label="IF Own->Triple Interference KL (lower better)", color=C_CRIMSON, zorder=3)
    b2 = ax.bar(x - 0.5*w, doc_if,  w, label="Doc->IF Cross-Leakage KL (lower better)",      color=C_AMBER,   zorder=3)
    b3 = ax.bar(x + 0.5*w, math_tr, w, label="Math Own->Triple Interference KL (lower better)",color=C_SLATE,  zorder=3)
    b4 = ax.bar(x + 1.5*w, if_sig,  w, label="IF Own-Domain Signal KL (higher better)",      color=C_TEAL,    zorder=3)
    for bars in [b1, b2, b3, b4]:
        add_bar_labels(ax, bars, fmt="{:.3f}", dy=0.004, fontsize=6.9)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in packs], fontsize=8.0)
    ax.set_ylabel("Top-32 Logit KL Divergence (nats)")
    ax.set_ylim(0, 0.295)
    ax.set_title("(a) 4-Way Cross-Expert KL Interference & Signal Separation")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.2)
    ax.annotate("2.95x lower\ninterference", xy=(3 - 1.5*w, 0.066), xytext=(2.82, 0.155),
                ha="center",
                arrowprops=dict(arrowstyle="->", color=C_CRIMSON, lw=1.15),
                fontsize=7.6, fontweight="bold", color=C_CRIMSON,
                bbox=dict(boxstyle="round,pad=0.2", fc="#FDEDEC", ec=C_CRIMSON, lw=0.7))

    # (b) Simultaneous 3-Expert Split-Slot Composition [Doc@P1 + Math@P2 + IF@P4]
    ax = axes[0, 1]
    style_ax(ax, "y")
    c9  = d9["track2_composition_evals"]
    t_labels = [
        "Triple Naive\n(192v)",
        "Triple Zero-RL\nCS-CR (192v)",
        "Triple Post-RL\nDirect (256v)",
        "Triple Post-RL\nAlg A KV-Inv2 (256v)",
    ]
    ifeval_v = [
        c9["ifeval96"]["Triple_Naive_Split"]["prompt_strict"],
        c9["ifeval96"]["Triple_CSCR_Split"]["prompt_strict"],
        d9b["ifeval96_evals"]["Triple_PostRL_Direct_Split"]["prompt_strict"],
        d9b["ifeval96_evals"]["Triple_PostRL_KVInv2_Split"]["prompt_strict"],
    ]
    ifb64_v = [
        c9["ifbench64"]["Triple_Naive_Split"]["prompt_strict"],
        c9["ifbench64"]["Triple_CSCR_Split"]["prompt_strict"],
        d9b["ifbench64_evals"]["Triple_PostRL_Direct_Split"]["prompt_strict"],
        d9b["ifbench64_evals"]["Triple_PostRL_KVInv2_Split"]["prompt_strict"],
    ]
    ifb64_cl = [
        c9["ifbench64"]["Triple_Naive_Split"]["closed_acc"],
        c9["ifbench64"]["Triple_CSCR_Split"]["closed_acc"],
        d9b["ifbench64_evals"]["Triple_PostRL_Direct_Split"]["closed_acc"],
        d9b["ifbench64_evals"]["Triple_PostRL_KVInv2_Split"]["closed_acc"],
    ]
    x = np.arange(len(t_labels))
    w = 0.25
    b1 = ax.bar(x - w, ifeval_v, w, label="IFEval96 Prompt Strict (%)",   color=C_NAVY, zorder=3)
    b2 = ax.bar(x,     ifb64_v,  w, label="IFBench64 Prompt Strict (%)",  color=C_TEAL, zorder=3)
    b3 = ax.bar(x + w, ifb64_cl, w, label="IFBench64 Closed-Think Acc (%)", color=C_GREEN, zorder=3)
    add_bar_labels(ax, b1, fmt="{:.1f}", dy=1.2, fontsize=7.5)
    add_bar_labels(ax, b2, fmt="{:.1f}", dy=1.2, fontsize=7.5)
    add_bar_labels(ax, b3, fmt="{:.1f}", dy=1.2, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(t_labels, fontsize=7.9)
    ax.set_ylabel("3-Expert Simultaneous Accuracy (%)")
    ax.set_ylim(0, 88)
    ax.set_title("(b) Simultaneous 3-Expert Split-Slot Composition [Doc@P1 + Math@P2 + IF@P4]")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.6)

    # (c) Split-Slot Isolation Synergy & Exact Permutation Invariance
    ax = axes[1, 0]
    style_ax(ax, "y")
    m9  = c9["math48"]
    m9b = d9b["math48_evals"]
    cats = [
        "Bare\nStudent",
        "Zero-RL\nSolo Math\n(P2)",
        "Zero-RL Split\n[Math@P2+\nIF@P4]",
        "Zero-RL Adj\n[IF;Math]\n@P2",
        "Zero-RL Adj\n[Math;IF]\n@P2",
        "Post-RL\nSolo Math\n(P2)",
        "Post-RL Adj\n[IF;Math]\n@P2",
        "Post-RL Adj\n[Math;IF]\n@P2",
    ]
    math_acc = [
        m9["Bare_Student"]["acc"],
        m9["Single_Math_CSCR_P2"]["acc"],
        m9["Dual_CSCR_MathP2_IFP4"]["acc"],
        m9["Dual_CSCR_Adj_IF_Math_P2"]["acc"],
        m9["Dual_CSCR_Adj_Math_IF_P2"]["acc"],
        m9b["P2_PostRL_Direct_Math_L64"]["acc"],
        m9b["Dual_PostRL_Adj_IF_Math_P2"]["acc"],
        m9b["Dual_PostRL_Adj_Math_IF_P2"]["acc"],
    ]
    ifev_adj = [
        c9["ifeval96"]["Bare_Student"]["prompt_strict"],
        np.nan,
        c9["ifeval96"]["Dual_CSCR_MathP2_IFP4"]["prompt_strict"],
        c9["ifeval96"]["Dual_CSCR_Adj_IF_Math_P2"]["prompt_strict"],
        c9["ifeval96"]["Dual_CSCR_Adj_Math_IF_P2"]["prompt_strict"],
        np.nan,
        d9b["ifeval96_evals"]["Dual_PostRL_Adj_IF_Math_P2"]["prompt_strict"],
        d9b["ifeval96_evals"]["Dual_PostRL_Adj_Math_IF_P2"]["prompt_strict"],
    ]
    x = np.arange(len(cats))
    w = 0.36
    b1 = ax.bar(x - w/2, math_acc, w, label="MATH48 Accuracy (%, N=48)", color=C_NAVY, zorder=3)
    b2 = ax.bar(x + w/2, ifev_adj, w, label="IFEval96 Prompt Strict (%, N=96)", color=C_TEAL, zorder=3)
    b1[2].set_color(C_GREEN)
    add_bar_labels(ax, b1, fmt="{:.1f}", dy=1.2, fontsize=6.9)
    add_bar_labels(ax, b2, fmt="{:.1f}", dy=1.2, fontsize=6.9)
    ax.axvline(4.5, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=6.9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 88)
    ax.set_title("(c) Split-Slot Isolation Synergy & Exact 0.00 pp Permutation Invariance")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.6)
    ax.text(6.35, 79, "Post-RL Math Perm Diff:\n|50.0 - 50.0| = 0.00 pp", ha="center", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.22", fc="#EBF5FB", ec=C_NAVY, lw=0.8))
    ax.text(3.25, 79, "Zero-RL IFEval Perm Diff:\n|71.9 - 71.9| = 0.00 pp", ha="center", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.22", fc="#E8F8F5", ec=C_TEAL, lw=0.8))

    # (d) Zero-Leakage Declarative Capacity & Recency Gradient (HeliosPay Doc36)
    ax = axes[1, 1]
    style_ax(ax, "y")
    doc9  = c9["doc36"]
    doc9b = d9b["doc36_evals"]
    d_labels = [
        "In-Context\nUpper\nBound",
        "KV-PD L=32\n(P0 Prefix)",
        "KV-PD L=16\n+ Micro-RAG",
        "CRSP Doc\nL=64\n(RMS 0.055)",
        "CRSP Doc\nL=128\n@ P1 (Far)",
        "CRSP Doc\nL=128\n@ P4 (Post-Q)",
        "CRSP Doc\nL=128\n@ P2 (Pre-Q)",
    ]
    d_rules = [
        doc9["InContext_Doc_UpperBound"]["rule_acc"],
        d4["runs"]["KV_A_L32_AdamW"]["A_heliospay"]["rules"],
        d5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["A_heliospay"]["rules"],
        doc9["Single_Doc_CSCR_P2"]["rule_acc"],
        doc9b["Single_Doc_CSCR_MuonL128_P1"]["rule_acc"],
        doc9b["Single_Doc_CSCR_MuonL128_P4"]["rule_acc"],
        doc9b["Single_Doc_CSCR_MuonL128_P2"]["rule_acc"],
    ]
    d_facts = [
        doc9["InContext_Doc_UpperBound"]["fact_acc"],
        d4["runs"]["KV_A_L32_AdamW"]["A_heliospay"]["facts"],
        d5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["A_heliospay"]["facts"],
        doc9["Single_Doc_CSCR_P2"]["fact_acc"],
        doc9b["Single_Doc_CSCR_MuonL128_P1"]["fact_acc"],
        doc9b["Single_Doc_CSCR_MuonL128_P4"]["fact_acc"],
        doc9b["Single_Doc_CSCR_MuonL128_P2"]["fact_acc"],
    ]
    x = np.arange(len(d_labels))
    w = 0.36
    b1 = ax.bar(x - w/2, d_rules, w, label="Procedural Decision Rules Acc (%)", color=C_NAVY, zorder=3)
    b2 = ax.bar(x + w/2, d_facts, w, label="Verbatim Constant Facts Acc (%)", color=C_TEAL, zorder=3)
    add_bar_labels(ax, b1, fmt="{:.1f}", dy=1.4, fontsize=7.2)
    add_bar_labels(ax, b2, fmt="{:.1f}", dy=1.4, fontsize=7.2)
    ax.axvline(2.5, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(d_labels, fontsize=7.1)
    ax.set_ylabel("HeliosPay v4.2 Zero-Leakage Accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_title("(d) Declarative Capacity Boundary & Positional Recency Gradient")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.5)
    ax.annotate("5.0x recency gradient:\nP1 (8.3%) < P4 (25.0%) < P2 (41.7%)",
                xy=(6 + w/2, 41.67), xytext=(3.15, 68),
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.2),
                fontsize=7.6, fontweight="bold", color=C_TEAL,
                bbox=dict(boxstyle="round,pad=0.25", fc="#E8F8F5", ec=C_TEAL, lw=0.8))

    path = os.path.join(OUT, "paper_fig5_multiexpert_composition_and_laws.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print("All 5 unified paper figures generated successfully.")
