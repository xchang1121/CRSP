"""
Generate tr_fig11_multiseed_and_composition.png (200 DPI ICML/arXiv style) for Suite 7 (exp9):
  Track 1: 3-Seed Algorithm B (TF3 Warm-Start -> TF2 GRPO) Convergence, Variance & Scaled IFBench64
  Track 2: Zero-Interference 3-Expert Arbitrary-Slot Composition (Doc@P1 + Math@P2 + IF@P4)
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
    "font.size": 8.8,
    "axes.titlesize": 9.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.8,
    "xtick.labelsize": 7.9,
    "ytick.labelsize": 7.9,
    "legend.fontsize": 7.1,
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
TEAL   = "#0D9488"
GREEN  = "#059669"
AMBER  = "#D97706"
ORANGE = "#EA580C"
RED    = "#DC2626"
PURPLE = "#7C3AED"
SLATE  = "#64748B"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO_ROOT, "results")
OUT_DIR = os.environ.get("FIG_OUT_DIR", os.path.join(REPO_ROOT, "docs", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

d = json.load(open(os.path.join(ROOT, "exp9_results.json")))

fig, axes = plt.subplots(2, 2, figsize=(16.6, 11.4))
fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.08, left=0.065, right=0.975)

# ---------------------------------------------------------------------------
# (a) Track 1: 3-Seed Algorithm B Convergence & Variance on Held-Out IFEval96
# ---------------------------------------------------------------------------
ax = axes[0, 0]
s96 = d["track1_ifeval96_summary"]
bare96 = d["track1_ifeval96_multiseed"]["Bare_Student"]["prompt_strict"]

bars_spec = [
    ("Bare\nStudent",         bare96, 0.0, [], SLATE),
    ("P2\nStage1 TF3",        s96["P2_Stage1_TF3"]["strict_mean"],      s96["P2_Stage1_TF3"]["strict_std"],      s96["P2_Stage1_TF3"]["strict_per_seed"],      AMBER),
    ("P2\nCold TF2",          s96["P2_Cold_TF2_step10"]["strict_mean"], s96["P2_Cold_TF2_step10"]["strict_std"], s96["P2_Cold_TF2_step10"]["strict_per_seed"], BLUE),
    ("P2\nFull AlgB",         s96["P2_Full_AlgB_step10"]["strict_mean"],s96["P2_Full_AlgB_step10"]["strict_std"],s96["P2_Full_AlgB_step10"]["strict_per_seed"],GREEN),
    ("P4\nStage1 TF3",        s96["P4_Stage1_TF3"]["strict_mean"],      s96["P4_Stage1_TF3"]["strict_std"],      s96["P4_Stage1_TF3"]["strict_per_seed"],      ORANGE),
    ("P4 Cold\n@step5",       s96["P4_Cold_TF2_step5"]["strict_mean"],  s96["P4_Cold_TF2_step5"]["strict_std"],  s96["P4_Cold_TF2_step5"]["strict_per_seed"],  "#93C5FD"),
    ("P4 AlgB\n@step5",       s96["P4_Full_AlgB_step5"]["strict_mean"], s96["P4_Full_AlgB_step5"]["strict_std"], s96["P4_Full_AlgB_step5"]["strict_per_seed"], "#6EE7B7"),
    ("P4 Cold\n@step10",      s96["P4_Cold_TF2_step10"]["strict_mean"], s96["P4_Cold_TF2_step10"]["strict_std"], s96["P4_Cold_TF2_step10"]["strict_per_seed"], BLUE),
    ("P4 AlgB\n@step10",      s96["P4_Full_AlgB_step10"]["strict_mean"],s96["P4_Full_AlgB_step10"]["strict_std"],s96["P4_Full_AlgB_step10"]["strict_per_seed"],TEAL),
]
xs = np.arange(len(bars_spec))
for i, (lbl, m, sd, pts, col) in enumerate(bars_spec):
    ax.bar(i, m, width=0.66, color=col, alpha=0.88, edgecolor="#1E293B", lw=0.6)
    if sd > 0:
        ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="#0F172A", elinewidth=1.2, capsize=3.5, zorder=4)
    for p_i, pv in enumerate(pts):
        ax.scatter(i + (p_i - 1) * 0.11, pv, color="#0F172A", s=16, alpha=0.82, zorder=5)
    ax.text(i, m + sd + 1.0, f"{m:.1f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#0F172A")

ax.axhline(bare96, color=SLATE, linestyle="--", lw=1.0, alpha=0.75)
ax.axvline(3.5, color="#94A3B8", linestyle=":", lw=1.2)
ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in bars_spec], fontsize=7.2)
ymin_a = max(40, min(b[1] for b in bars_spec) - 10)
ymax_a = min(100, max(b[1] + b[2] for b in bars_spec) + 8)
ax.set_ylim(ymin_a, ymax_a)
ax.set_ylabel("Held-Out IFEval96 Prompt Strict Acc (%)  [3-Seed Mean +/- Std]")
ax.set_title("(a) Track 1: 3-Seed Algorithm B Convergence & Variance (IFEval96, L=64)")
ax.grid(axis="y", alpha=0.4)

# ---------------------------------------------------------------------------
# (b) Track 1 + 2: Scaled IFBench64 (2x IFBench32) + Triple Split
# ---------------------------------------------------------------------------
ax = axes[0, 1]
b64 = d["track1_ifbench64"]
c64 = d["track2_composition_evals"]["ifbench64"]

b_spec = [
    ("Bare\nStudent",         b64["Bare_Student"]["prompt_strict"],     b64["Bare_Student"]["closed_acc"],     SLATE),
    ("P4\nStage1 TF3",        b64["P4_Stage1_TF3"]["prompt_strict"],    b64["P4_Stage1_TF3"]["closed_acc"],    AMBER),
    ("P4\nCold TF2",          b64["P4_Cold_TF2"]["prompt_strict"],      b64["P4_Cold_TF2"]["closed_acc"],      BLUE),
    ("P2 Full\nAlgB CS-CR",   b64["P2_Full_AlgB"]["prompt_strict"],     b64["P2_Full_AlgB"]["closed_acc"],     GREEN),
    ("P4 Full\nAlgB CS-CR",   b64["P4_Full_AlgB"]["prompt_strict"],     b64["P4_Full_AlgB"]["closed_acc"],     TEAL),
    ("Triple Split\nNaive",   c64["Triple_Naive_Split"]["prompt_strict"],c64["Triple_Naive_Split"]["closed_acc"],RED),
    ("Triple Split\nCS-CR",   c64["Triple_CSCR_Split"]["prompt_strict"], c64["Triple_CSCR_Split"]["closed_acc"], NAVY),
]
xb = np.arange(len(b_spec))
for i, (lbl, st, ca, col) in enumerate(b_spec):
    ax.bar(i, st, width=0.62, color=col, alpha=0.88, edgecolor="#1E293B", lw=0.6)
    ax.text(i, st + 0.9, f"{st:.1f}%\n(cl:{ca:.0f}%)", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#0F172A")

ax.axhline(b64["Bare_Student"]["prompt_strict"], color=SLATE, linestyle="--", lw=1.0, alpha=0.75)
ax.axvline(4.5, color="#94A3B8", linestyle=":", lw=1.2)
ax.set_xticks(xb)
ax.set_xticklabels([b[0] for b in b_spec], fontsize=7.5)
ax.set_ylim(0, max(b[1] for b in b_spec) + 15)
ax.set_ylabel("Official IFBench64 Prompt Strict Acc (%)")
ax.set_title("(b) Scaled Official IFBench64 (64 Prompts, 0 Post-RL): Single vs 3-Expert")
ax.grid(axis="y", alpha=0.4)

# ---------------------------------------------------------------------------
# (c) Track 2: Exact Cross-Expert KL Interference Matrix (Naive vs CS-CR)
# ---------------------------------------------------------------------------
ax = axes[1, 0]
kn = d["track2_interference_kl"]["Naive"]
kc = d["track2_interference_kl"]["CSCR"]

doms = ["if", "math", "doc"]
dom_lbls = ["IF Queries\n(Own -> Triple)", "Math Queries\n(Own -> Triple)", "Doc Queries\n(Own -> Triple)", "Off-Domain Leak\nDoc on IF+Math", "Off-Domain Leak\nMath on IF"]
naive_vals = [
    kn["if"]["triple_interference_kl"],
    kn["math"]["triple_interference_kl"],
    kn["doc"]["triple_interference_kl"],
    0.5 * (kn["if"]["kl_from_bare_doc"] + kn["math"]["kl_from_bare_doc"]),
    kn["if"]["kl_from_bare_math"],
]
cscr_vals = [
    kc["if"]["triple_interference_kl"],
    kc["math"]["triple_interference_kl"],
    kc["doc"]["triple_interference_kl"],
    0.5 * (kc["if"]["kl_from_bare_doc"] + kc["math"]["kl_from_bare_doc"]),
    kc["if"]["kl_from_bare_math"],
]
xc = np.arange(len(dom_lbls))
bw = 0.34
b1 = ax.bar(xc - bw / 2, naive_vals, width=bw * 0.92, color=RED, alpha=0.85, edgecolor="#1E293B", lw=0.6, label="Naive CR-SP (No Co-Presence, No L_null)")
b2 = ax.bar(xc + bw / 2, cscr_vals, width=bw * 0.92, color=TEAL, alpha=0.90, edgecolor="#1E293B", lw=0.6, label="Proposed CS-CR (Co-Presence + L_null)")
for rect, v in zip(b1, naive_vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#991B1B")
for rect, v in zip(b2, cscr_vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#115E59")

ax.axvline(2.5, color="#94A3B8", linestyle=":", lw=1.2)
ax.set_xticks(xc)
ax.set_xticklabels(dom_lbls, fontsize=7.5)
ax.set_ylim(0, max(max(naive_vals), max(cscr_vals)) * 1.28 + 0.02)
ax.set_ylabel("Top-32 Output KL Divergence (Lower = Less Interference)")
ax.set_title("(c) Track 2 Diagnostic: Cross-Expert KL Interference & Off-Domain Leakage")
ax.grid(axis="y", alpha=0.4)
ax.legend(loc="upper right", fontsize=7.2, frameon=True, framealpha=0.95)

# ---------------------------------------------------------------------------
# (d) Track 2 Grand Finale: 3-Expert Simultaneous Split-Slot Composition Across 3 Domains!
# ---------------------------------------------------------------------------
ax = axes[1, 1]
ie = d["track2_composition_evals"]["ifeval96"]
me = d["track2_composition_evals"]["math48"]
de = d["track2_composition_evals"]["doc36"]

tasks = ["IFEval96\n(Instruction-Following)", "MATH48\n(Level 3-5 Math)", "HeliosPay Doc36\n(0-Leakage Policy)"]
series = [
    ("Bare Student (0 SP)",               SLATE, [ie["Bare_Student"]["prompt_strict"],        me["Bare_Student"]["acc"],             de["Bare_Student"]["overall_acc"]]),
    ("Single CS-CR Specialist (L=64)",    GREEN, [ie["Single_IF_CSCR_P4"]["prompt_strict"],   me["Single_Math_CSCR_P2"]["acc"],      de["Single_Doc_CSCR_P1"]["overall_acc"]]),
    ("Triple-Split Naive [P1+P2+P4]",     RED,   [ie["Triple_Naive_Split"]["prompt_strict"],  me["Triple_Naive_Split"]["acc"],       de["Triple_Naive_Split"]["overall_acc"]]),
    ("Triple-Split CS-CR [P1+P2+P4]",     NAVY,  [ie["Triple_CSCR_Split"]["prompt_strict"],   me["Triple_CSCR_Split"]["acc"],        de["Triple_CSCR_Split"]["overall_acc"]]),
]
xd = np.arange(len(tasks))
bw4 = 0.19
for s_i, (s_lbl, s_col, vals) in enumerate(series):
    pos = xd + (s_i - 1.5) * bw4
    rects = ax.bar(pos, vals, width=bw4 * 0.90, color=s_col, alpha=0.89, edgecolor="#1E293B", lw=0.6, label=s_lbl)
    for rect, v in zip(rects, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#0F172A")

# Permutation invariance summary box
perm_if_a = ie["Dual_CSCR_Adj_IF_Math_P2"]["prompt_strict"]
perm_if_b = ie["Dual_CSCR_Adj_Math_IF_P2"]["prompt_strict"]
perm_m_a = me["Dual_CSCR_Adj_IF_Math_P2"]["acc"]
perm_m_b = me["Dual_CSCR_Adj_Math_IF_P2"]["acc"]
ax.text(
    0.985, 0.96,
    f"Permutation Invariance ([IF;Math] vs [Math;IF]):\n"
    f"  IFEval96: {perm_if_a:.1f}% vs {perm_if_b:.1f}% (|d|={abs(perm_if_a - perm_if_b):.1f} pp)\n"
    f"  MATH48:   {perm_m_a:.1f}% vs {perm_m_b:.1f}% (|d|={abs(perm_m_a - perm_m_b):.1f} pp)",
    transform=ax.transAxes, ha="right", va="top", fontsize=6.8, fontweight="bold", color="#0F172A",
    bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec="#475569", lw=0.8),
)

ax.set_xticks(xd)
ax.set_xticklabels(tasks, fontsize=8.1)
ax.set_ylim(0, 114)
ax.set_ylabel("Task Strict / Exact Accuracy (%)")
ax.set_title("(d) Grand Finale: 3-Expert Simultaneous Split-Slot Composition (192 Vecs Total)")
ax.grid(axis="y", alpha=0.4)
ax.legend(loc="upper left", ncol=1, fontsize=6.9, frameon=True, framealpha=0.95)

fig.suptitle("Suite 7 (exp9): 3-Seed Algorithm B Validation (Track 1) & Zero-Interference 3-Expert Composition (Track 2)",
             fontsize=12.1, fontweight="bold", color="#0F172A", y=0.975)

out_path = os.path.join(OUT_DIR, "tr_fig11_multiseed_and_composition.png")
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)
