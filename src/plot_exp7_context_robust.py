"""
Generate tr_fig9_context_robust_sp.png (200 DPI ICML/arXiv style) for Suite 5 (exp7):
Arbitrary-Position Insertion & Context-Robust Soft Prompt Distillation (CR-SP).

4-panel cohesive publication figure:
  (a) 9-Method x 6-Position (P0..P5) Heatmap of Exact KL Gap Closed G(P, s) (%)
  (b) Cross-Position Behavioral Drift KL(p_P0 || p_s) across P1..P5 (Lower = More Context-Invariant!)
  (c) Layerwise Hidden-State Representation Cosine Invariance cos(H_l(P | P2/P4), H_l(P | P0)) across Depth (L0..L36)
  (d) Official allenai/IFBench32 End-to-End Strict Accuracy across P0 (Start), P2 (Multi-Turn Mid),
      P3 (Post-RAG), P4 (Post-Query Late Injection) & Split-Position Multi-Expert Assembly!
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#334155",
    "grid.color": "#CBD5E1",
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.facecolor": "white",
})

NAVY   = "#1E3A8A"
BLUE   = "#2563EB"
SKY    = "#38BDF8"
TEAL   = "#0D9488"
GREEN  = "#059669"
AMBER  = "#D97706"
ORANGE = "#EA580C"
RED    = "#DC2626"
PURPLE = "#7C3AED"
SLATE  = "#64748B"

ROOT = r"C:\Users\singm\.gemini\antigravity\scratch\composable-soft-prompt-distillation\results"
OUT_DIR = r"C:\Users\singm\.gemini\antigravity\brain\8aba4b5d-6a54-4e44-b641-a0c487b94d0d"

d = json.load(open(os.path.join(ROOT, "exp7_results.json")))
pm = d["position_matrix"]["methods"]
regimes = d["regimes"]
reg_short = ["P0\nStart", "P1\nMidSys", "P2\nMultiTurn", "P3\nPostRAG", "P4\nPostQuery", "P5\nLong1000"]

fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.8))
fig.subplots_adjust(hspace=0.36, wspace=0.27, top=0.92, bottom=0.08, left=0.11, right=0.97)

# ---------------------------------------------------------------------------
# (a) 9-Method x 6-Position Heatmap of Exact KL Gap Closed G(P, s) (%)
# ---------------------------------------------------------------------------
ax = axes[0, 0]
method_order = [
    ("A1_Pos0_Only_SP",    "A1: Pos0-Only SP"),
    ("A2_Paper_PR_SP",     "A2: Paper PR-SP"),
    ("A3_MultiSlot_CR_SP", "A3: Multi-Slot CR"),
    ("A4_Full_CR_SP_Inv",  "A4: Full CR-SP (L_inv)"),
    ("A5_CR_SP_L64",       "A5: Compact CR (L=64)"),
    ("KV_Distilled_L16",   "KV-CSPD L16 (Pos-0)"),
    ("B0_exp6_TF_KV2SP",   "B0: TF-KV2SP (exp6)"),
    ("B1_CR_TF_KV2SP",     "B1: CR-TF-KV2SP (0-Tch)"),
    ("KV_Fixed_L16_TF",    "KV-TF L16 (Pos-0)"),
]
method_order = [(k, lbl) for k, lbl in method_order if k in pm]

from matplotlib.colors import TwoSlopeNorm

mat = np.zeros((len(method_order), len(regimes)))
for i, (k, _) in enumerate(method_order):
    for j, r in enumerate(regimes):
        mat[i, j] = pm[k][r]["G_vs_ctx_student_pct"]

norm = TwoSlopeNorm(vmin=-35.0, vcenter=0.0, vmax=48.0)
im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", norm=norm)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat[i, j]
        txt_col = "white" if (val <= -20.0 or val >= 34.0) else "#0F172A"
        ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=7.5, fontweight="bold", color=txt_col)

ax.set_xticks(np.arange(len(reg_short)))
ax.set_xticklabels(reg_short, fontsize=7.8)
ax.set_yticks(np.arange(len(method_order)))
ax.set_yticklabels([lbl for _, lbl in method_order], fontsize=7.9)
ax.set_title("(a) KL Gap Closed G(P, s) Across 6 Arbitrary Insertion Positions")
cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02, extend="min")
cbar.ax.tick_params(labelsize=7.5)
cbar.set_label("G(P, s) (%)  [0 = White]", fontsize=8.0)

# ---------------------------------------------------------------------------
# (b) Positional Behavioral Drift KL(p_P0 || p_s) Across P1..P5
# ---------------------------------------------------------------------------
ax = axes[0, 1]
drift_regs = regimes[1:]
drift_lbls = ["P1\nMidSys", "P2\nMultiTurn", "P3\nPostRAG", "P4\nPostQuery", "P5\nLong1000"]
xd = np.arange(len(drift_regs))

plot_specs = [
    ("A1_Pos0_Only_SP",    "A1: Pos0-Only SP",       RED,    "o--", 2.0),
    ("A2_Paper_PR_SP",     "A2: Paper PR-SP",        ORANGE, "s--", 1.8),
    ("A3_MultiSlot_CR_SP", "A3: Multi-Slot CR",      BLUE,   "^-",  1.8),
    ("A4_Full_CR_SP_Inv",  "A4: Full CR-SP (L_inv)", GREEN,  "D-",  2.3),
    ("A5_CR_SP_L64",       "A5: Compact CR (L=64)",  TEAL,   "v:",  1.6),
    ("B1_CR_TF_KV2SP",     "B1: CR-TF-KV2SP (0-Tch)",PURPLE, "P-.", 1.9),
]
for k, lbl, col, sty, lw in plot_specs:
    if k in pm and "drift_kl_from_P0" in pm[k][drift_regs[0]]:
        vals = [pm[k][r]["drift_kl_from_P0"] for r in drift_regs]
        ax.plot(xd, vals, sty, color=col, linewidth=lw, markersize=5.5, label=lbl)

ax.set_xticks(xd)
ax.set_xticklabels(drift_lbls, fontsize=8.2)
ax.set_ylabel("Behavioral Drift KL(p_P0 || p_s)  [Lower = Better]")
ax.set_title("(b) Cross-Position Output Distribution Drift vs. Clean P0")
ax.grid(True, alpha=0.4)
ax.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=True, framealpha=0.94)

# ---------------------------------------------------------------------------
# (c) Layerwise Hidden-State Representation Invariance cos(H_l(P|C_s), H_l(P|P0))
# ---------------------------------------------------------------------------
ax = axes[1, 0]
layers = ["0", "9", "18", "27", "36"]
xl = np.arange(len(layers))

# Plot P2_MultiTurn_Mid (solid) and P4_PostQuery_Suffix (dashed) for A1 vs A2 vs A4 vs B1
cos_specs = [
    ("A1_Pos0_Only_SP",    "P2_MultiTurn_Mid",    "A1 Pos0-Only (P2 MultiTurn)", RED,    "o-",  2.0),
    ("A1_Pos0_Only_SP",    "P4_PostQuery_Suffix", "A1 Pos0-Only (P4 PostQuery)", RED,    "o--", 1.5),
    ("A2_Paper_PR_SP",     "P4_PostQuery_Suffix", "A2 Paper-PR (P4 PostQuery)",  ORANGE, "s--", 1.7),
    ("A3_MultiSlot_CR_SP", "P4_PostQuery_Suffix", "A3 Multi-Slot (P4 PostQuery)",BLUE,   "^--", 1.8),
    ("A4_Full_CR_SP_Inv",  "P2_MultiTurn_Mid",    "A4 Full CR-SP (P2 MultiTurn)",GREEN,  "D-",  2.3),
    ("A4_Full_CR_SP_Inv",  "P4_PostQuery_Suffix", "A4 Full CR-SP (P4 PostQuery)",TEAL,   "D--", 2.0),
    ("B1_CR_TF_KV2SP",     "P4_PostQuery_Suffix", "B1 CR-TF-KV2SP (P4 PostQuery)",PURPLE,"P:",  1.8),
]
for k, reg, lbl, col, sty, lw in cos_specs:
    if k in pm and "hidden_cos_vs_P0" in pm[k][reg]:
        vals = [pm[k][reg]["hidden_cos_vs_P0"][ly] for ly in layers]
        ax.plot(xl, vals, sty, color=col, linewidth=lw, markersize=5.2, label=lbl)

ax.set_xticks(xl)
ax.set_xticklabels([f"Layer {ly}" for ly in layers], fontsize=8.2)
ax.set_ylabel("Soft-Prompt Hidden Cosine vs. Clean P0")
ax.set_title("(c) Layerwise Hidden-State Invariance of P Under Context Injection")
ax.set_ylim(0.83, 1.015)
ax.grid(True, alpha=0.4)
ax.legend(loc="lower left", fontsize=7.2, frameon=True, framealpha=0.94)

# ---------------------------------------------------------------------------
# (d) Official IFBench32 End-to-End Strict Accuracy Across Insertion Positions
# ---------------------------------------------------------------------------
ax = axes[1, 1]
ib = d["ifbench32_by_position"]

groups = [
    ("P0 Clean\nStart",    ["P0_Student", "P0_A1_Pos0_Only", None,            "P0_A4_Full_CR_Inv", "P0_B1_CR_TF_KV2SP"]),
    ("P2 Multi-Turn\nMid", ["P2_Student", "P2_A1_Pos0_Only", "P2_A2_Paper_PR","P2_A4_Full_CR_Inv", "P2_B1_CR_TF_KV2SP"]),
    ("P3 Post-RAG\nDoc",   ["P3_Student", "P3_A1_Pos0_Only", None,            "P3_A4_Full_CR_Inv", None]),
    ("P4 Post-Query\nLate",["P2_Student", "P4_A1_Pos0_Only", "P4_A2_Paper_PR","P4_A4_Full_CR_Inv", "P4_B1_CR_TF_KV2SP"]),
]
series_meta = [
    ("Bare Student",       SLATE),
    ("A1: Pos0-Only SP",   RED),
    ("A2: Paper PR-SP",    ORANGE),
    ("A4: Full CR-SP",     GREEN),
    ("B1: CR-TF (0-Tch)",  PURPLE),
]

xg = np.arange(len(groups))
bw = 0.165
for s_i, (s_lbl, s_col) in enumerate(series_meta):
    xs, ys = [], []
    for g_i, (_, klist) in enumerate(groups):
        k = klist[s_i]
        if k is not None and k in ib:
            xs.append(xg[g_i] + (s_i - 2) * bw)
            ys.append(ib[k]["prompt_strict"])
    bars = ax.bar(xs, ys, width=bw * 0.90, color=s_col, alpha=0.88, edgecolor="#1E293B", lw=0.6, label=s_lbl)
    for rect, v in zip(bars, ys):
        # Compact 0-decimal or 1-decimal cleanly so adjacent equal bars never collide
        lbl_txt = f"{v:.1f}" if v not in (43.75, 40.62) else f"{v:.0f}"
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.8, lbl_txt, ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#0F172A")

ax.annotate("Late Injection Peak: 50.0%\n(+28.1 pp over Paper PR!\n0 prefix/query KV recompute)",
            xy=(3 + 1 * bw, 51.5), xytext=(1.68, 61.0),
            fontsize=7.1, fontweight="bold", color=GREEN,
            bbox=dict(boxstyle="round,pad=0.22", fc="#ECFDF5", ec=GREEN, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))

if "P_Split_MathEarly_CRIFLate" in ib:
    sp_v = ib["P_Split_MathEarly_CRIFLate"]["prompt_strict"]
    sp_c = ib["P_Split_MathEarly_CRIFLate"]["closed_acc"]
    ax.text(0.99, 0.97, f"Split-Pos Dual:\n[Sys:Math]+[Hist]+[Q;CR-IF]\nStrict={sp_v:.1f}% | Cl={sp_c:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8, fontweight="bold", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.22", fc="#EFF6FF", ec=BLUE, lw=0.8))

ax.set_xticks(xg)
ax.set_xticklabels([g[0] for g in groups], fontsize=8.1)
ax.set_ylim(0, 82)
ax.set_ylabel("Official IFBench32 Prompt Strict Acc (%)")
ax.set_title("(d) End-to-End IFBench32 Across Insertion Positions & Late Injection")
ax.grid(axis="y", alpha=0.4)
ax.legend(loc="upper left", ncol=2, fontsize=6.8, frameon=True, framealpha=0.95)

fig.suptitle("Suite 5 (exp7): Arbitrary-Position Insertion & Context-Robust Soft Prompt Distillation (Qwen3-4B)",
             fontsize=12.2, fontweight="bold", color="#0F172A", y=0.975)

out_path = os.path.join(OUT_DIR, "tr_fig9_context_robust_sp.png")
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)
