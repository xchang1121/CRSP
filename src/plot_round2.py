"""
Generate publication-quality plots for Round 2 Deep Exploration & Acceleration:
  1. suite2_doc_cl_breakthroughs.png  (exp4_results.json + exp2/exp2b baselines)
  2. suite3_5domain_cl_and_upgrade.png (exp5_results.json + exp2/exp2b baselines)
  3. suite1_cspd_acceleration.png     (exp3_results.json — generated once available)
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
    "axes.titlesize": 11.2,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
})


def plot_suite2(res4_path: str, res2_path: str, res2b_path: str, out_dir: str):
    d4 = json.load(open(res4_path))
    d2 = json.load(open(res2_path))
    d2b = json.load(open(res2b_path))

    runs4 = d4["runs"]
    meta4 = d4["meta"]
    probes4 = d4["probes_A"]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))

    # --- Panel (a): Doc A Architecture & Capacity Frontier (Zero Raw Doc vs In-Context & LoRA) ---
    ax = axes[0, 0]
    items_a = [
        ("In-Context\nFull Doc\n(~500 tok)", d2["runs"]["in_context_doc_upper_bound"]["A_heliospay"]["rules"], d2["runs"]["in_context_doc_upper_bound"]["A_heliospay"]["facts"], "#94A3B8"),
        ("Input-SP\nL=128\n(0.33M, 129s)", runs4["SP_A_random_vocab"]["A_heliospay"]["rules"], runs4["SP_A_random_vocab"]["A_heliospay"]["facts"], "#60A5FA"),
        ("KV-CSPD\nL_kv=8\n(0.59M, 57s)", runs4["KV_A_L8_AdamW"]["A_heliospay"]["rules"], runs4["KV_A_L8_AdamW"]["A_heliospay"]["facts"], "#34D399"),
        ("KV-CSPD\nL_kv=16\n(1.18M, 58s)", runs4["KV_A_L16_AdamW"]["A_heliospay"]["rules"], runs4["KV_A_L16_AdamW"]["A_heliospay"]["facts"], "#10B981"),
        ("KV-CSPD\nL_kv=16+Think\n(1.18M, 58s)", runs4["KV_A_L16_Thinking"]["A_heliospay"]["rules"], runs4["KV_A_L16_Thinking"]["A_heliospay"]["facts"], "#0D9488"),
        ("KV-CSPD\nL_kv=32\n(2.36M, 58s)", runs4["KV_A_L32_AdamW"]["A_heliospay"]["rules"], runs4["KV_A_L32_AdamW"]["A_heliospay"]["facts"], "#047857"),
        ("Single LoRA\nr=16\n(11.8M, 118s)", d2b["runs"]["lora_seq_stage1_A"]["A_heliospay"]["rules"], d2b["runs"]["lora_seq_stage1_A"]["A_heliospay"]["facts"], "#F97316"),
    ]
    x = np.arange(len(items_a))
    w = 0.36
    r_vals = [it[1] for it in items_a]
    f_vals = [it[2] for it in items_a]
    b1 = ax.bar(x - w / 2, r_vals, w, label="Unseen Multi-Step Rules (%)", color="#2563EB")
    b2 = ax.bar(x + w / 2, f_vals, w, label="Paraphrased Facts (%)", color="#10B981")
    ax.axhline(d2["runs"]["in_context_doc_upper_bound"]["A_heliospay"]["rules"], color="#64748B", linestyle="--", linewidth=1.1, label="In-Context Full Doc Rules (37.5%)")
    ax.set_xticks(x)
    ax.set_xticklabels([it[0] for it in items_a], fontsize=7.8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("(a) Doc A (HeliosPay v4.2): KV-CSPD vs Input-SP, In-Context & LoRA")
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    for b in b1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.4, color="#1D4ED8", fontweight="bold")
    for b in b2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.4, color="#047857")

    # --- Panel (b): Wall-Clock Training NLL Convergence Curves (KV-CSPD 5.9x Faster) ---
    ax = axes[0, 1]
    curve_specs = [
        ("SP_A_random_vocab", "Input-SP L=128 (Random Vocab, 128.8s)", "#2563EB", "--", "o"),
        ("SP_A_doc_semantic", "Input-SP L=128 (Doc-Semantic, 128.5s)", "#7C3AED", "-.", "s"),
        ("SP_A_low_norm", "Input-SP L=128 (Low-Norm, 128.1s)", "#9CA3AF", ":", "^"),
        ("KV_A_L8_AdamW", "KV-CSPD L_kv=8 (AdamW, 57.1s)", "#34D399", "-", "D"),
        ("KV_A_L16_AdamW", "KV-CSPD L_kv=16 (AdamW, 57.6s)", "#059669", "-", "o"),
        ("KV_A_L32_AdamW", "KV-CSPD L_kv=32 (AdamW, 57.5s)", "#065F46", "-", "v"),
        ("KV_A_L16_Muon", "KV-CSPD L_kv=16 (LayerwiseMuon, 61.3s)", "#F59E0B", "--", "x"),
    ]
    for key, label, col, ls, mk in curve_specs:
        m = meta4[key]
        steps = sorted([int(s) for s in m["nll_curve"].keys()])
        max_s = max(steps)
        times = [m["train_seconds"] * (s / max_s) for s in steps]
        nlls = [m["nll_curve"][str(s)] for s in steps]
        ax.plot(times, nlls, linestyle=ls, marker=mk, color=col, linewidth=2.0, markersize=5, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Wall-Clock Training Time on A100 (seconds)")
    ax.set_ylabel("Training NLL (log scale)")
    ax.set_xlim(0, 138)
    ax.set_ylim(0.001, 4.0)
    ax.grid(True, which="major", alpha=0.25)
    ax.annotate("L_kv=32 hits Input-SP final NLL (0.019)\nin ~22s (5.9x faster wall-clock);\nL_kv=16 hits it in ~28s (4.6x)!",
                xy=(21.6, 0.0192), xytext=(48, 0.0028),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=1.4),
                fontsize=8.0, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981", lw=1.0))
    ax.set_title("(b) Training NLL Convergence: Direct Layerwise KV vs Input-SP")
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")

    # --- Panel (c): Hard Stress Batteries (Boundary, Extrapolation, Abstention) ---
    ax = axes[1, 0]
    probe_models = [
        ("Base\n(No Doc)", "base_no_doc"),
        ("In-Context\nFull Doc", "in_context_full_doc"),
        ("Input-SP\n(Doc-Sem)", "SP_A_doc_semantic"),
        ("KV-CSPD\nL_kv=16", "KV_A_L16_AdamW"),
        ("KV-CSPD\n+AbstainCal", "KV_A_L16_AbstainCal"),
    ]
    xp = np.arange(len(probe_models))
    wp = 0.25
    v_bnd = [probes4[k]["boundary"] for _, k in probe_models]
    v_ext = [probes4[k]["extrap"] for _, k in probe_models]
    v_abs = [probes4[k]["abstain"] for _, k in probe_models]

    pb1 = ax.bar(xp - wp, v_bnd, wp, label="Boundary Equality (12 cases)", color="#2563EB")
    pb2 = ax.bar(xp, v_ext, wp, label="Out-of-Range Extrap (8 cases)", color="#8B5CF6")
    pb3 = ax.bar(xp + wp, v_abs, wp, label="Unanswerable Abstention (8 cases)", color="#F59E0B")
    ax.set_xticks(xp)
    ax.set_xticklabels([n for n, _ in probe_models], fontsize=8.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 114)
    ax.set_title("(c) Stress Probes on Doc A: Boundary, Extrapolation & Abstention")
    ax.legend(frameon=False, fontsize=8.0, loc="upper left")
    for bars, col in [(pb1, "#1D4ED8"), (pb2, "#6D28D9"), (pb3, "#B45309")]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=7.4, color=col, fontweight="bold")

    # --- Panel (d): 3-Domain Rules/Facts + OOD Null-Replay Calibration & Concat Diagnosis ---
    ax = axes[1, 1]
    comp_items = [
        ("Input-SP\nTop-1 Routed\n(128 tok)", d2b["runs"]["retrieved_top1_summary"]["mean_rules"], d2b["runs"]["retrieved_top1_summary"]["mean_facts"], d2b["runs"]["stage1_PA"]["ood"]),
        ("KV-CSPD\nTop-1 Routed\n(16 tok)", (runs4["KV_A_L16_AdamW"]["A_heliospay"]["rules"] + runs4["KV_B_L16_AdamW"]["B_nvx418"]["rules"] + runs4["KV_C_L16_AdamW"]["C_lithox9"]["rules"]) / 3.0,
         (runs4["KV_A_L16_AdamW"]["A_heliospay"]["facts"] + runs4["KV_B_L16_AdamW"]["B_nvx418"]["facts"] + runs4["KV_C_L16_AdamW"]["C_lithox9"]["facts"]) / 3.0,
         runs4["KV_A_L16_AdamW"]["ood"]),
        ("KV-16 + Naive\nSys MicroRAG\n(RoPE shift!)", runs4["hybrid_KV16_microRAG_top1"]["mean_rules"], runs4["hybrid_KV16_microRAG_top1"]["mean_facts"], np.nan),
        ("NB-NR-SP\nBlind [A;B]\n(rms=0.055)", runs4["nb_nr_sp_concat_AB"]["mean_rules"], runs4["nb_nr_sp_concat_AB"]["mean_facts"], runs4["nb_nr_sp_concat_AB"]["ood"]),
        ("NR-KV16\nBlind [A;B]\n(d_rms=0.23)", runs4["nr_kv_concat_AB"]["mean_rules"], runs4["nr_kv_concat_AB"]["mean_facts"], runs4["nr_kv_concat_AB"]["ood"]),
        ("NR-KV16\nBlind [A;B;C]\n(48 tok)", runs4["nr_kv_concat_ABC"]["mean_rules"], runs4["nr_kv_concat_ABC"]["mean_facts"], runs4["nr_kv_concat_ABC"]["ood"]),
    ]
    xc = np.arange(len(comp_items))
    wc = 0.25
    cr = [it[1] for it in comp_items]
    cf = [it[2] for it in comp_items]
    co = [it[3] if not np.isnan(it[3]) else 0.0 for it in comp_items]

    cb1 = ax.bar(xc - wc, cr, wc, label="3-Domain Mean Rules (%)", color="#2563EB")
    cb2 = ax.bar(xc, cf, wc, label="3-Domain Mean Facts (%)", color="#10B981")
    cb3 = ax.bar(xc + wc, co, wc, label="OOD Preservation (24 Q, %)", color="#EF4444", alpha=0.82)
    ax.set_xticks(xc)
    ax.set_xticklabels([it[0] for it in comp_items], fontsize=7.6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_title("(d) 3-Domain Summary: Routing vs Blind Concat & Null-Replay OOD Fix")
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    for idx_i, b in enumerate(cb1):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.1, color="#1D4ED8", fontweight="bold")
    for idx_i, b in enumerate(cb2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.1, color="#047857")
    for idx_i, b in enumerate(cb3):
        if not np.isnan(comp_items[idx_i][3]):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=7.1, color="#B91C1C", fontweight="bold")
        else:
            ax.text(b.get_x() + b.get_width() / 2, 2.0, "N/A", ha="center", va="bottom", fontsize=6.8, color="#6B7280")

    fig.suptitle("Suite 2: Vertical-Domain Document Continual Learning Breakthroughs (KV-CSPD-Doc, Probes & Null-Replay)", fontsize=13.2, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "suite2_doc_cl_breakthroughs.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_suite3(res5_path: str, res2_path: str, out_dir: str):
    d5 = json.load(open(res5_path))
    d2 = json.load(open(res2_path))

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.6))

    # --- Panel (a): 5-Stage Sequential LoRA Forgetting Curve vs Zero-Oracle Auto-Router ---
    ax = axes[0]
    stages = ["Stage 1\n(+Doc A)", "Stage 2\n(+Doc B)", "Stage 3\n(+Doc C)", "Stage 4\n(+Doc D)", "Stage 5\n(+Doc E)"]
    skeys = ["stage1_A", "stage2_B", "stage3_C", "stage4_D", "stage5_E"]
    doms = [("A_heliospay", "Doc A (FinTech)", "#2563EB", "o"),
            ("B_nvx418", "Doc B (BioPharma)", "#10B981", "s"),
            ("C_lithox9", "Doc C (EUV Fab)", "#F59E0B", "^"),
            ("D_gridops", "Doc D (SmartGrid)", "#8B5CF6", "D"),
            ("E_customs", "Doc E (Customs)", "#EF4444", "v")]
    xs = np.arange(len(stages))
    for dkey, dlbl, col, mk in doms:
        vals = [d5["seq_lora_5"][sk][dkey]["rules"] for sk in skeys]
        ax.plot(xs, vals, marker=mk, color=col, linewidth=2.1, markersize=6, label=f"SeqLoRA: {dlbl}")
    mean_seq = [d5["seq_lora_5"][sk]["mean_rules"] for sk in skeys]
    ax.plot(xs, mean_seq, "k--", linewidth=2.0, label="SeqLoRA 5-Dom Mean (19.2%)")

    kv_auto_mean = d5["auto_router"]["e2e_kv16_auto_routed"]["mean_rules"]
    ax.axhline(kv_auto_mean, color="#059669", linestyle="-", linewidth=2.2, alpha=0.85,
               label=f"KV-CSPD Auto-Router (57.5%, 0% Forget)")
    ax.set_xticks(xs)
    ax.set_xticklabels(stages, fontsize=8.4)
    ax.set_ylabel("Unseen Multi-Step Rule Accuracy (%)")
    ax.set_ylim(-4, 134)
    ax.grid(True, alpha=0.22)
    ax.annotate("100% Catastrophic Forgetting\nof ALL 4 Prior Domains\n(A=B=C=D=0.0% at Stage 5!)",
                xy=(4.0, 1.5), xytext=(1.48, 34),
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.4),
                fontsize=7.7, fontweight="bold", color="#991B1B",
                bbox=dict(boxstyle="round,pad=0.28", fc="#FEF2F2", ec="#EF4444", lw=1.0))
    ax.set_title("(a) 5-Stage Sequential LoRA Forgetting vs KV-CSPD")
    ax.legend(frameon=False, fontsize=6.9, loc="upper left", ncol=2, columnspacing=0.9)

    # --- Panel (b): End-to-End 5-Domain Mean (Zero Oracle, 194 Unseen Queries) ---
    ax = axes[1]
    ic_5_r = (d2["runs"]["in_context_doc_upper_bound"]["A_heliospay"]["rules"] +
              d2["runs"]["in_context_doc_upper_bound"]["B_nvx418"]["rules"] +
              d2["runs"]["in_context_doc_upper_bound"]["C_lithox9"]["rules"] +
              d5["domains_5"]["in_context_DE"]["D_gridops"]["rules"] +
              d5["domains_5"]["in_context_DE"]["E_customs"]["rules"]) / 5.0
    ic_5_f = 100.0

    m5 = [
        ("Base\n(No Doc)", 0.0, 2.0, d2["runs"]["base_no_doc"]["ood"]),
        ("5-Stage\nSeqLoRA\n(11.8M)", d5["seq_lora_5"]["stage5_E"]["mean_rules"], d5["seq_lora_5"]["stage5_E"]["mean_facts"], d5["seq_lora_5"]["stage5_E"]["ood"]),
        ("In-Context\nFull Doc\n(~500 tok)", ic_5_r, ic_5_f, np.nan),
        ("KV-CSPD\nAuto-Router\n(16 tok)", d5["auto_router"]["e2e_kv16_auto_routed"]["mean_rules"], d5["auto_router"]["e2e_kv16_auto_routed"]["mean_facts"], d5["auto_router"]["e2e_kv16_auto_routed"]["ood"]),
        ("Dual-Intent\nAuto-Router\n(16-60 tok)", d5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["mean_rules"], d5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["mean_facts"], d5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["ood"]),
    ]
    xm = np.arange(len(m5))
    wm = 0.25
    b_r = ax.bar(xm - wm, [it[1] for it in m5], wm, label="5-Domain Rules (%)", color="#2563EB")
    b_f = ax.bar(xm, [it[2] for it in m5], wm, label="5-Domain Facts (%)", color="#10B981")
    b_o = ax.bar(xm + wm, [it[3] if not np.isnan(it[3]) else 0.0 for it in m5], wm, label="OOD Acc (24 Q, %)", color="#F59E0B")
    ax.set_xticks(xm)
    ax.set_xticklabels([it[0] for it in m5], fontsize=8.0)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 134)
    ax.set_title("(b) End-to-End 5-Domain Continual Learning (ZERO Oracle)")
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", ncol=3, columnspacing=0.8)
    for b in b_r:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.3, color="#1D4ED8", fontweight="bold")
    for b in b_f:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.3, color="#047857", fontweight="bold")
    for idx_i, b in enumerate(b_o):
        if not np.isnan(m5[idx_i][3]):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.3, color="#B45309")
        else:
            ax.text(b.get_x() + b.get_width() / 2, 2.0, "N/A", ha="center", va="bottom", fontsize=6.8, color="#6B7280")
    ax.text(0.5, 0.895, "Zero-Oracle Router: 100% Rule / 98% Fact / 79.2% OOD->None",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.6, fontweight="bold",
            color="#065F46", bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec="#10B981", lw=0.9))

    # --- Panel (c): Enterprise Document Version Upgrade (HeliosPay v4.2 -> v5.0) ---
    ax = axes[2]
    vu = d5["version_upgrade"]
    v_items = [
        ("Outdated\nKV v4.2", vu["outdated_KV_v42"]["v50_rules"], vu["outdated_KV_v42"]["v50_facts"], vu["outdated_KV_v42"]["stale_v42_rules_pct"]),
        ("In-Context\nFull v5.0\n(~500 tok)", vu["in_context_v50"]["v50_rules"], vu["in_context_v50"]["v50_facts"], vu["in_context_v50"]["stale_v42_rules_pct"]),
        ("KV Atomic\nSwap v5.0\n(1.18M, 57s)", vu["kv_atomic_swap_v50"]["v50_rules"], vu["kv_atomic_swap_v50"]["v50_facts"], vu["kv_atomic_swap_v50"]["stale_v42_rules_pct"]),
        ("4-Tok KV\nDelta-Patch\n(0.29M, 57s)", vu["kv_delta_patch_L4"]["v50_rules"], vu["kv_delta_patch_L4"]["v50_facts"], vu["kv_delta_patch_L4"]["stale_v42_rules_pct"]),
        ("LoRA Update\nv4.2->v5.0\n(11.8M, irrev.)", vu["lora_continual_v42_to_v50"]["v50_rules"], vu["lora_continual_v42_to_v50"]["v50_facts"], vu["lora_continual_v42_to_v50"]["stale_v42_rules_pct"]),
    ]
    xv = np.arange(len(v_items))
    wv = 0.25
    vb1 = ax.bar(xv - wv, [it[1] for it in v_items], wv, label="v5.0 New Rules (%)", color="#2563EB")
    vb2 = ax.bar(xv, [it[2] for it in v_items], wv, label="v5.0 New Facts (%)", color="#10B981")
    vb3 = ax.bar(xv + wv, [it[3] for it in v_items], wv, label="Stale v4.2 Rule Leak (%) ↓", color="#EF4444")
    ax.set_xticks(xv)
    ax.set_xticklabels([it[0] for it in v_items], fontsize=7.9)
    ax.set_ylabel("Accuracy / Leak Rate (%)")
    ax.set_ylim(0, 134)
    ax.set_title("(c) Policy Version Upgrade (HeliosPay v4.2 -> v5.0)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for b in vb1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.3, color="#1D4ED8", fontweight="bold")
    for b in vb2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.3, color="#047857", fontweight="bold")
    for b in vb3:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.3, color="#B91C1C", fontweight="bold")
    ax.annotate("40x smaller than LoRA,\n0% stale leak, instant rollback!",
                xy=(3, 86.0), xytext=(2.05, 114),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=1.3),
                fontsize=7.6, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec="#10B981", lw=0.9))

    fig.suptitle("Suite 3: 5-Domain Zero-Oracle Continual Learning & Incremental 4-Token KV Policy Patching", fontsize=13.0, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "suite3_5domain_cl_and_upgrade.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_suite1(res3_path: str, out_dir: str):
    d3 = json.load(open(res3_path))
    a1 = d3["axis1_init"]
    a2 = d3["axis2_loss_trunc"]
    a3 = d3["axis3_opt_len"]
    a4 = d3["axis4_kv_cspd"]
    dc = d3["dual_composition_G"]
    ifb = d3["ifbench32_verify"]

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.4))

    # --- Panel (a): Wall-Clock Convergence Curves (train_sec vs Held-Out G(A) %) ---
    ax = axes[0, 0]
    curves = [
        (a4["kv_if_L8"], "KV-CSPD L_kv=8 (0.59M, 32x shorter) [72.3s]", "#059669", "-", "o", 2.5),
        (a4["kv_if_L16"], "KV-CSPD L_kv=16 (1.18M, 16x shorter) [72.5s]", "#10B981", "--", "s", 2.1),
        (a1["task_inst"], "Axis 1: Input-SP Task-Inst + Muon [196.7s]", "#2563EB", "-", "^", 2.2),
        (a3["adamw_lownorm"], "Axis 3: Input-SP Low-Norm + AdamW [197.3s]", "#7C3AED", "-.", "D", 2.0),
        (a1["random_vocab"], "Original CSPD (Random Vocab + Muon) [195.8s]", "#475569", "--", "v", 2.0),
        (a2["hybrid_fwd_rev_768"], "Axis 2: Hybrid Fwd+Rev KL (Low-Norm) [196.6s]", "#F59E0B", ":", "x", 1.8),
        (a1["low_norm"], "Axis 1: Low-Norm + Muon (Off-Manifold) [195.7s]", "#EF4444", ":", ".", 1.7),
    ]
    for meta, lbl, col, ls, mk, lw in curves:
        ts = [pt["train_sec"] for pt in meta["curve"]]
        gs = [pt["G_pct"] for pt in meta["curve"]]
        ax.plot(ts, gs, linestyle=ls, marker=mk, color=col, linewidth=lw, markersize=5.2, label=lbl)

    ax.axhline(a1["random_vocab"]["final_G_pct"], color="#475569", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axhline(0.0, color="#9CA3AF", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Wall-Clock Training Time on A100 (seconds)")
    ax.set_ylabel("Held-Out Teacher-Student KL Gap Closed G(A) (%)")
    ax.set_xlim(-3, 208)
    ax.set_ylim(-48, 76)
    ax.grid(True, alpha=0.22)
    ax.annotate("KV-CSPD (L_kv=8, 0.59M):\nHits +26.1% at step 2 (3.3s),\ncrosses Orig. CSPD final (+26.7%,\n195.8s) at ~10s (~20x faster!)\n& reaches +65.5% (2.46x higher)!",
                xy=(16.1, 46.06), xytext=(86, 42.5),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5),
                fontsize=7.9, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.32", fc="#ECFDF5", ec="#10B981", lw=1.0))
    ax.set_title("(a) Wall-Clock Convergence Speedup on M2RL-RL_IF")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    # --- Panel (b): Complete 4-Axis Controlled Ablation Summary (Final G(A) %) ---
    ax = axes[0, 1]
    bar_items = [
        ("KV-CSPD L_kv=8 (0.59M, 72s)", a4["kv_if_L8"]["final_G_pct"], "#059669"),
        ("KV-CSPD L_kv=16 (1.18M, 73s)", a4["kv_if_L16"]["final_G_pct"], "#10B981"),
        ("A1: Task-Inst + Muon (197s)", a1["task_inst"]["final_G_pct"], "#2563EB"),
        ("A3: Low-Norm + AdamW (197s)", a3["adamw_lownorm"]["final_G_pct"], "#7C3AED"),
        ("A1: Orig. CSPD RandVocab (196s)", a1["random_vocab"]["final_G_pct"], "#475569"),
        ("A2: Hybrid Fwd+Rev KL* (197s)", a2["hybrid_fwd_rev_768"]["final_G_pct"], "#F59E0B"),
        ("A2: Fork-Token Top-30%* (197s)", a2["fork_top30_768"]["final_G_pct"], "#D97706"),
        ("A3: Low-Rank r=32* (0.09M)", a3["low_rank_r32"]["final_G_pct"], "#6366F1"),
        ("A3: NormClip Muon* (rms<=0.055)", a3["muon_normclip_055"]["final_G_pct"], "#60A5FA"),
        ("A3: Short SP L=64* (165s)", a3["len_L64"]["final_G_pct"], "#93C5FD"),
        ("A1: Low-Norm + Muon Base*", a1["low_norm"]["final_G_pct"], "#94A3B8"),
        ("A2: Seq-NLL (1-hot)*", a2["seq_nll_768"]["final_G_pct"], "#F87171"),
        ("A2: Trunc-256 Prefix*", a2["rb_top64_trunc256"]["final_G_pct"], "#DC2626"),
    ]
    y_pos = np.arange(len(bar_items))
    vals = [it[1] for it in bar_items]
    cols = [it[2] for it in bar_items]
    bars = ax.barh(y_pos, vals, height=0.68, color=cols)
    ax.axvline(0.0, color="#374151", linewidth=1.0)
    ax.axvline(a1["random_vocab"]["final_G_pct"], color="#475569", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([it[0] for it in bar_items], fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlabel("Final Held-Out KL Gap Closed G(A) (%)  [* = same Low-Norm init baseline +7.45%]")
    ax.set_xlim(-74, 80)
    ax.set_title("(b) All 4 Acceleration Axes: Final G(A) Comparison (45 Updates)")
    for b in bars:
        v = b.get_width()
        xpos = v + 1.5 if v >= 0 else v - 1.5
        ha = "left" if v >= 0 else "right"
        ax.text(xpos, b.get_y() + b.get_height() / 2, f"{v:+.1f}%", va="center", ha=ha, fontsize=7.6, fontweight="bold",
                color="#065F46" if v > 50 else ("#991B1B" if v < 0 else "#1E293B"))

    # --- Panel (c): Zero-Training Dual-Expert Composition & Commutativity ---
    ax = axes[1, 0]
    comp_cfgs = [
        ("KV16 Single\nIF (16 tok)", "KV16_Single_IF"),
        ("KV16 Single\nMath (16 tok)", "KV16_Single_Math"),
        ("KV16 [M; I]\n(32 tok!)", "KV16_Concat_M_I"),
        ("KV16 [I; M]\n(32 tok!)", "KV16_Concat_I_M"),
        ("NormClipMuon\n[M; I] (512 tok)", "NormClipMuon_Concat_M_I"),
        ("NormClipMuon\n[I; M] (512 tok)", "NormClipMuon_Concat_I_M"),
    ]
    xc = np.arange(len(comp_cfgs))
    wc = 0.26
    g_if = [dc[k]["G_if_pct"] for _, k in comp_cfgs]
    g_ma = [dc[k]["G_math_pct"] for _, k in comp_cfgs]
    g_av = [dc[k]["G_avg_pct"] for _, k in comp_cfgs]

    cb1 = ax.bar(xc - wc, g_if, wc, label="G_IF (%)", color="#2563EB")
    cb2 = ax.bar(xc, g_ma, wc, label="G_Math (%)", color="#10B981")
    cb3 = ax.bar(xc + wc, g_av, wc, label="Dual-Domain Mean G (%)", color="#F59E0B")
    ax.axhline(0.0, color="#374151", linewidth=1.0)
    ax.set_xticks(xc)
    ax.set_xticklabels([n for n, _ in comp_cfgs], fontsize=8.0)
    ax.set_ylabel("Held-Out KL Gap Closed G (%)")
    ax.set_ylim(-60, 88)
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_title("(c) Zero-Training Dual-Expert Composition & Order Invariance")
    ax.legend(frameon=False, fontsize=8.0, loc="upper center", ncol=3, columnspacing=1.0)
    for bars, col in [(cb1, "#1D4ED8"), (cb2, "#047857"), (cb3, "#B45309")]:
        for b in bars:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            off = 1.6 if h >= 0 else -1.8
            ax.text(b.get_x() + b.get_width() / 2, h + off, f"{h:+.1f}", ha="center", va=va, fontsize=6.9, color=col, fontweight="bold")
    ax.annotate("Near-Exact Commutativity (0.06 pp diff!)\nBoth domains POSITIVE (+25.2% / +14.7%)\n& Dual Avg (+20.0%) beats BOTH singles!",
                xy=(2.5, 27.0), xytext=(1.95, 53.0),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=1.4),
                fontsize=7.7, fontweight="bold", color="#065F46",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981", lw=1.0))

    # --- Panel (d): Official allenai/IFBench (32 Prompts) Accuracy & Inference Speed ---
    ax = axes[1, 1]
    if_cfgs = [
        ("Student\n(No SP)", "student_no_sp"),
        ("NormClipMuon\nSingle P_if\n(256 tok)", "normclip_muon_single_IF"),
        ("NormClipMuon\n[M; I]\n(512 tok)", "normclip_muon_concat_M_I"),
        ("KV-CSPD L16\nSingle P_if\n(16 tok)", "kv16_single_IF"),
        ("KV-CSPD L16\n[M; I]\n(32 tok!)", "kv16_concat_M_I"),
    ]
    xi = np.arange(len(if_cfgs))
    wi = 0.34
    closed_acc = [ifb[k]["closed_acc"] for _, k in if_cfgs]
    strict_all = [ifb[k]["prompt_strict"] for _, k in if_cfgs]
    eval_secs = [ifb[k]["eval_sec"] for _, k in if_cfgs]

    ib1 = ax.bar(xi - wi / 2, closed_acc, wi, label="Strict Acc on Closed-</think> (%)", color="#059669")
    ib2 = ax.bar(xi + wi / 2, strict_all, wi, label="Overall Strict Acc @2560 tok (%)", color="#3B82F6")
    ax.set_xticks(xi)
    ax.set_xticklabels([n for n, _ in if_cfgs], fontsize=8.0)
    ax.set_ylabel("IFBench Accuracy (%)")
    ax.set_ylim(0, 114)
    ax.set_title("(d) Official allenai/IFBench (32 Prompts): Accuracy & Inference Latency")

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#DC2626")
    l_lat = ax2.plot(xi, eval_secs, "o--", color="#DC2626", linewidth=2.0, markersize=6.5, label="32-Prompt Inference Latency (s)")
    ax2.set_ylabel("Inference Wall-Clock Time (s)", color="#DC2626")
    ax2.tick_params(axis="y", colors="#DC2626")
    ax2.set_ylim(240, 380)

    for b in ib1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.4, color="#065F46", fontweight="bold")
    for b in ib2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.2, color="#1D4ED8")
    for xj, sec in zip(xi, eval_secs):
        ax2.text(xj, sec + 6.5, f"{sec:.0f}s", ha="center", va="bottom", fontsize=7.5, color="#B91C1C", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#FCA5A5", alpha=0.92, lw=0.8))

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.6, loc="upper left")

    fig.suptitle("Suite 1: Accelerating CSPD — 4-Axis Convergence Speedup, Direct Layerwise KV-CSPD & Official IFBench Verification", fontsize=13.0, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "suite1_cspd_acceleration.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    root = r"C:\Users\singm\.gemini\antigravity\scratch\composable-soft-prompt-distillation\results"
    out = r"C:\Users\singm\.gemini\antigravity\brain\8aba4b5d-6a54-4e44-b641-a0c487b94d0d"
    res3 = os.path.join(root, "exp3_results.json")
    if os.path.exists(res3):
        plot_suite1(res3, out)
    plot_suite2(
        os.path.join(root, "exp4_results.json"),
        os.path.join(root, "exp2_results.json"),
        os.path.join(root, "exp2b_results.json"),
        out,
    )
    plot_suite3(
        os.path.join(root, "exp5_results.json"),
        os.path.join(root, "exp2_results.json"),
        out,
    )
