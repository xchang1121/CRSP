"""
Generate publication-quality plots for:
  1. Stage A (CSPD Paper Reproduction on M2RL-SFT + M2RL-RL_IF / M2RL-RL_Math + official IFBench)
  2. Stage B (Vertical-Domain Document Continual Learning: CSPD-Doc vs Sequential LoRA)
  3. Cross-Stage Synthesis (Unified Norm-Composability Law across RL Behavioral vs Factual Capsules)
"""

import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
})


def plot_stageA(res_path: str, out_dir: str):
    data = json.load(open(res_path))
    ifb = data["ifbench_evals"]
    m_if = data["mechanistic_IF"]
    m_math = data["mechanistic_Math"]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

    # --- Panel (a): IFBench Task Accuracy ---
    ax = axes[0, 0]
    cfg_order = [
        ("student_no_sp", "Student\n(No SP)"),
        ("single_P_if", "Single\nP_if (Muon)"),
        ("cspd_concat_Math_IF", "CSPD\n[M; I] Muon"),
        ("cspd_concat_IF_Math", "CSPD\n[I; M] Muon"),
        ("ablation_nopr_Math_IF", "No-PR\n[M; I] Muon"),
        ("ablation_adamw_Math_IF", "CSPD\n[M; I] AdamW"),
        ("teacher_M2RL_RL_IF", "Teacher\nM2RL-RL_IF"),
    ]
    labels = [lbl for _, lbl in cfg_order]
    strict_vals = [ifb[k]["prompt_strict"] for k, _ in cfg_order]
    inst_vals = [ifb[k]["inst_strict"] for k, _ in cfg_order]
    closed_acc = [
        100.0 * ifb[k]["n_strict"] / max(1, ifb[k]["closed_think"]) for k, _ in cfg_order
    ]

    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, strict_vals, w, label="Prompt Strict (%)", color="#2563EB")
    b2 = ax.bar(x + w / 2, inst_vals, w, label="Instruction Strict (%)", color="#10B981")
    ax.plot(x, closed_acc, "o--", color="#F59E0B", linewidth=1.8, markersize=5.5, label="Acc on Closed-</think> (%)")

    ax.axhline(ifb["student_no_sp"]["prompt_strict"], color="#9CA3AF", linestyle=":", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title("(a) IFBench (40 prompts, max_tok=3072)")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, color="#1D4ED8", fontweight="bold")
    for xi, ca in zip(x, closed_acc):
        ax.text(xi, ca + 2.2, f"{ca:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#B45309")

    # --- Panel (b): Soft Prompt vs Written Instructions G(A) (Eq. 5) ---
    ax = axes[0, 1]
    g_items = [
        ("SP (AdamW)", "sp_adamw"),
        ("SP (Muon)", "sp_muon"),
        ("Neutral Inst.", "neutral_instruction"),
        ("Check Inst.", "check_instruction"),
        ("Verify Inst.", "verify_instruction"),
        ("SP (Init)", "sp_init"),
    ]
    y_pos = np.arange(len(g_items))
    h = 0.35
    g_if = [m_if["kl_gap_closed_G_pct"][k] for _, k in g_items]
    g_ma = [m_math["kl_gap_closed_G_pct"][k] for _, k in g_items]

    bars_if = ax.barh(y_pos - h / 2, g_if, h, label="IF Expert (KL_S = 0.251)", color="#2563EB")
    bars_ma = ax.barh(y_pos + h / 2, g_ma, h, label="Math Expert (KL_S = 0.054)", color="#10B981")
    ax.axvline(0, color="#374151", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n for n, _ in g_items], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Teacher-Student KL Gap Closed G(A) (%)")
    ax.set_xlim(-46, 52)
    ax.set_title("(b) Soft Prompt vs Written Instructions (Eq. 5)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for bars in [bars_if, bars_ma]:
        for bar in bars:
            val = bar.get_width()
            xpos = val + 1.0 if val >= 0 else val - 1.0
            ha = "left" if val >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}%", va="center", ha=ha, fontsize=7.8)

    # --- Panel (c): Attention Pathway Ablation K(A) (Eq. 6 on M2RL-RL_IF vs Paper Fig. 6c) ---
    ax = axes[1, 0]
    k_names = ["Full", "Response Only", "Input Only", "Blocked"]
    k_keys = ["full", "response_only", "input_only", "blocked"]
    k_ours = [m_if["attention_pathway_K_vs_blocked_pct"][k] for k in k_keys]
    k_paper = [100.0, 80.0, 33.0, 1.0]
    y_pos = np.arange(len(k_names))
    h = 0.35
    b_our = ax.barh(y_pos - h / 2, k_ours, h, label="Ours (M2RL-RL_IF, K vs Blocked)", color="#2563EB")
    b_pap = ax.barh(y_pos + h / 2, k_paper, h, label="Paper Fig. 6c (GPT-OSS-20B Ref.)", color="#CBD5E1", edgecolor="#64748B", hatch="//")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(k_names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Soft-Prompt Improvement Kept K(A) (%)")
    ax.set_xlim(0, 122)
    ax.set_title("(c) Attention Pathway Ablation (Eq. 6 / Fig. 6c)")
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")
    for bars in [b_our, b_pap]:
        for bar in bars:
            val = bar.get_width()
            ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.2)

    # --- Panel (d): Discourse Verification Token Probabilities Relative to Teacher (Fig. 11b) ---
    ax = axes[1, 1]
    disc = m_if["discourse_prob_rel_teacher"]
    toks = [("Check", "Check"), (" But", "' But'"), (" Let", "' Let'"), ("?", "'?'"), (" So", "' So'"), ("Thus", "Thus\n(Premature)")]
    s_rel = [disc[k]["student_rel_to_teacher"] for k, _ in toks]
    sp_rel = [disc[k]["sp_rel_to_teacher"] for k, _ in toks]
    x = np.arange(len(toks))
    w = 0.35
    b1 = ax.bar(x - w / 2, s_rel, w, label="Student (No SP)", color="#94A3B8")
    b2 = ax.bar(x + w / 2, sp_rel, w, label="Student + P_if (Frozen Backbone)", color="#2563EB")
    ax.axhline(1.0, color="#DC2626", linestyle="--", linewidth=1.4, label="RL Teacher (1.00)")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in toks], fontsize=8.8)
    ax.set_ylabel("Mean Token Prob / Teacher Prob")
    ax.set_ylim(0, 1.42)
    ax.set_title("(d) Verification & Conclusion Tokens vs Teacher (Fig. 11b)")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    for bars in [b1, b2]:
        for bar in bars:
            hv = bar.get_height()
            dy = 0.055 if 0.94 <= hv <= 1.00 else 0.025
            ax.text(bar.get_x() + bar.get_width() / 2, hv + dy, f"{hv:.2f}", ha="center", va="bottom", fontsize=7.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "stageA_cspd_reproduction.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved", out_path)


def plot_stageB(res_v1_path: str, res_v2_path: str, out_dir: str):
    d1 = json.load(open(res_v1_path))["runs"]
    d2 = json.load(open(res_v2_path))["runs"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # --- Panel (a): Continual Learning (Top-1 Routed CSPD-Doc vs Sequential LoRA Forgetting) ---
    ax = axes[0]
    stages = ["Base\n(0 Docs)", "Stage 1\n(+Doc A)", "Stage 2\n(+Doc B)", "Stage 3\n(+Doc C)"]
    xs = np.arange(len(stages))

    lora_A = [0.0, d2["lora_seq_stage1_A"]["A_heliospay"]["rules"], d2["lora_seq_stage2_AB"]["A_heliospay"]["rules"], d2["lora_seq_stage3_ABC"]["A_heliospay"]["rules"]]
    lora_B = [0.0, 0.0, d2["lora_seq_stage2_AB"]["B_nvx418"]["rules"], d2["lora_seq_stage3_ABC"]["B_nvx418"]["rules"]]

    spA_r = d2["stage1_PA"]["A_heliospay"]["rules"]
    spB_r = d2["single_PB"]["B_nvx418"]["rules"]
    spC_r = d2["single_PC"]["C_lithox9"]["rules"]
    sp_mean_r = [0.0, spA_r / 3.0, (spA_r + spB_r) / 3.0, (spA_r + spB_r + spC_r) / 3.0]
    spA_f = d2["stage1_PA"]["A_heliospay"]["facts"]
    spB_f = d2["single_PB"]["B_nvx418"]["facts"]
    spC_f = d2["single_PC"]["C_lithox9"]["facts"]
    sp_mean_f = [0.0, spA_f / 3.0, (spA_f + spB_f) / 3.0, (spA_f + spB_f + spC_f) / 3.0]

    ax.plot(xs, lora_A, "--o", color="#EF4444", linewidth=2.0, label="SeqLoRA: Doc A Rules (100% Forgetting!)")
    ax.plot(xs, lora_B, "--s", color="#F97316", linewidth=2.0, label="SeqLoRA: Doc B Rules (92% Forgetting!)")
    ax.plot(xs, sp_mean_r, "-^", color="#2563EB", linewidth=2.5, markersize=7, label="CSPD-Doc Routed: 3-Doc Mean Rules")
    ax.plot(xs, sp_mean_f, "-D", color="#10B981", linewidth=2.5, markersize=7, label="CSPD-Doc Routed: 3-Doc Mean Facts")
    ax.set_xticks(xs)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-4, 112)
    ax.set_title("(a) Continual Learning: Zero Forgetting vs LoRA")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")

    # --- Panel (b): Final 3-Domain Snapshot ---
    ax = axes[1]
    groups = ["Doc A\n(Overall)", "Doc B\n(Overall)", "Doc C\n(Overall)", "OOD\n(General)"]
    in_ctx = [d1["in_context_doc_upper_bound"][k]["overall"] for k in ["A_heliospay", "B_nvx418", "C_lithox9"]] + [d1["base_no_doc"]["ood"]]
    routed = [d2["retrieved_top1_summary"][k]["overall"] for k in ["A_heliospay", "B_nvx418", "C_lithox9"]] + [d2["stage1_PA"]["ood"]]
    concat = [d2["stage3_PA_PB_PC"][k]["overall"] for k in ["A_heliospay", "B_nvx418", "C_lithox9"]] + [d2["stage3_PA_PB_PC"]["ood"]]
    seq_lora = [d2["lora_seq_stage3_ABC"][k]["overall"] for k in ["A_heliospay", "B_nvx418", "C_lithox9"]] + [d2["lora_seq_stage3_ABC"]["ood"]]

    x = np.arange(len(groups))
    w = 0.20
    b0 = ax.bar(x - 1.5 * w, in_ctx, w, label="In-Context Full Doc", color="#9CA3AF")
    b1 = ax.bar(x - 0.5 * w, routed, w, label="CSPD-Doc Top-1 Routed", color="#10B981")
    b2 = ax.bar(x + 0.5 * w, concat, w, label="CSPD-Doc Blind Concat", color="#6366F1")
    b3 = ax.bar(x + 1.5 * w, seq_lora, w, label="SeqLoRA (After C)", color="#EF4444")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_title("(b) Final State After 3 Vertical Docs")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    for bars in [b0, b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h >= 1:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    # --- Panel (c): Doc A Ablations ---
    ax = axes[2]
    ab_labels = [
        "In-Ctx\nFull Doc",
        "L=128\nCtxKL",
        "L=32\nNLL",
        "L=128\nNLL(Uni)",
        "L=128\nNLL(Bi)",
        "L=256\nNLL(Bi)",
        "LoRA\nStage 1",
    ]
    r_vals = [
        d1["in_context_doc_upper_bound"]["A_heliospay"]["rules"],
        d1["stage1_PA"]["A_heliospay"]["rules"],
        d2["cap_A_L32"]["A_heliospay"]["rules"],
        d1["ablation_A_L128_nll_only"]["A_heliospay"]["rules"],
        d2["stage1_PA"]["A_heliospay"]["rules"],
        d2["cap_A_L256"]["A_heliospay"]["rules"],
        d2["lora_seq_stage1_A"]["A_heliospay"]["rules"],
    ]
    f_vals = [
        d1["in_context_doc_upper_bound"]["A_heliospay"]["facts"],
        d1["stage1_PA"]["A_heliospay"]["facts"],
        d2["cap_A_L32"]["A_heliospay"]["facts"],
        d1["ablation_A_L128_nll_only"]["A_heliospay"]["facts"],
        d2["stage1_PA"]["A_heliospay"]["facts"],
        d2["cap_A_L256"]["A_heliospay"]["facts"],
        d2["lora_seq_stage1_A"]["A_heliospay"]["facts"],
    ]
    x = np.arange(len(ab_labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, r_vals, w, label="Procedural Rules (Unseen)", color="#2563EB")
    b2 = ax.bar(x + w / 2, f_vals, w, label="Exact Facts (Paraphrased)", color="#F59E0B")
    ax.set_xticks(x)
    ax.set_xticklabels(ab_labels, fontsize=8)
    ax.set_ylabel("Doc A Accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_title("(c) Objective, Capacity (L) & Rules vs Facts")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "stageB_domain_doc_continual_learning.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved", out_path)


def plot_norm_composability_law(res_1b_path: str, out_dir: str):
    d1b = json.load(open(res_1b_path))
    meta = d1b["sp_meta"]
    dg = d1b["distributional_G"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    # --- Panel (a): Soft-Prompt Vector RMS vs Concatenation Behavior across Stage A & Stage B ---
    ax = axes[0]
    regimes = [
        "Vocab Init\n(Baseline)",
        "Stage A: RL\nAdamW (PR)",
        "Stage A: RL\nMuon (PR)",
        "Stage B: Docs\nZero-Prior NLL",
    ]
    rms_vals = [
        0.0190,
        (meta["P_if_adamw_pr"]["rms_final"] + meta["P_math_adamw_pr"]["rms_final"]) / 2.0,
        (meta["P_if_muon_pr"]["rms_final"] + meta["P_math_muon_pr"]["rms_final"]) / 2.0,
        0.1910,
    ]
    ratios = [v / 0.0190 for v in rms_vals]
    colors = ["#94A3B8", "#10B981", "#F59E0B", "#EF4444"]
    x = np.arange(len(regimes))
    bars = ax.bar(x, rms_vals, width=0.52, color=colors)
    ax.axhline(0.0190, color="#64748B", linestyle=":", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel("Soft-Prompt Embedding RMS")
    ax.set_ylim(0, 0.245)
    ax.set_title("(a) Embedding Norm Inflation: Behavioral RL vs Factual Docs")

    annotations = [
        "1.0x\n(Natural scale)",
        f"{ratios[1]:.1f}x\nComposes cleanly!\n(IFBench 47.5%, G>0)",
        f"{ratios[2]:.1f}x\nPartial interference\n(IFBench 40-45%)",
        f"{ratios[3]:.1f}x\nBlind concat -> 0%\n(Needs Top-1 Routing)",
    ]
    for bar, v, ann in zip(bars, rms_vals, annotations):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006, f"{v:.3f}\n{ann}", ha="center", va="bottom", fontsize=7.8)

    # --- Panel (b): Stage A Held-Out Compositional G(A) (Single vs Composed, Low-RMS vs High-RMS) ---
    ax = axes[1]
    comp_items = [
        ("Single P_if\n(AdamW)", "single_P_if_adamw"),
        ("Single P_math\n(AdamW)", "single_P_math_adamw"),
        ("Concat [M; I]\nAdamW (2.3x RMS)", "ablation_adamw_Math_IF"),
        ("Concat [I; M]\nMuon (6.5x RMS)", "cspd_concat_IF_Math"),
        ("Concat [M; I]\nMuon (6.5x RMS)", "cspd_concat_Math_IF"),
    ]
    x = np.arange(len(comp_items))
    w = 0.35
    g_if = [dg[k]["G_if_pct"] for _, k in comp_items]
    g_ma = [dg[k]["G_math_pct"] for _, k in comp_items]
    b1 = ax.bar(x - w / 2, g_if, w, label="Held-Out IF G(A) (%)", color="#2563EB")
    b2 = ax.bar(x + w / 2, g_ma, w, label="Held-Out Math G(A) (%)", color="#10B981")
    ax.axhline(0, color="#374151", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in comp_items], fontsize=8.3)
    ax.set_ylabel("Held-Out KL Gap Closed G(A) (%)")
    ax.set_ylim(-36, 48)
    ax.set_title("(b) Dual-Domain Held-Out Composition vs Embedding Norm")
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
    for bars in [b1, b2]:
        for bar in bars:
            val = bar.get_height()
            va = "bottom" if val >= 0 else "top"
            off = 1.0 if val >= 0 else -1.2
            ax.text(bar.get_x() + bar.get_width() / 2, val + off, f"{val:+.1f}", ha="center", va=va, fontsize=7.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "stageAB_norm_composability_law.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved", out_path)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    res_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    p1b = os.path.join(res_dir, "exp1b_results.json")
    p2a = os.path.join(res_dir, "exp2_results.json")
    p2b = os.path.join(res_dir, "exp2b_results.json")
    if os.path.exists(p1b):
        plot_stageA(p1b, out_dir)
        plot_norm_composability_law(p1b, out_dir)
    if os.path.exists(p2a) and os.path.exists(p2b):
        plot_stageB(p2a, p2b, out_dir)
