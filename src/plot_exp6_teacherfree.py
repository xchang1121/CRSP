"""
Generate tr_fig8_teacherfree_input_sp.png (200 DPI ICML/arXiv style) for Suite 4 (exp6):
Teacher-Free Input-Space Soft Prompt Acquisition & Arbitrary-Context Assembly.
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
    "font.size": 9.2,
    "axes.titlesize": 10.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 7.8,
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

ROOT = r"C:\Users\singm\.gemini\antigravity\scratch\composable-soft-prompt-distillation\results"
OUT_DIR = r"C:\Users\singm\.gemini\antigravity\brain\8aba4b5d-6a54-4e44-b641-a0c487b94d0d"

d = json.load(open(os.path.join(ROOT, "exp6_results.json")))

fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.2))
fig.subplots_adjust(hspace=0.36, wspace=0.26, top=0.92, bottom=0.08, left=0.06, right=0.98)

# ---------------------------------------------------------------------------
# (a) Official IFBench32 (C0 Clean): All 5 Teacher-Free Input-SP Methods + KV Scaffold + Dual
# ---------------------------------------------------------------------------
ax = axes[0, 0]
c0 = d["ifbench40_C0_clean"]
labels = [
    "Student\n(No SP)", "TF-RFT\n(1-Rd NLL)", "TF-IterRFT\n(2-Rd OnPol)",
    "TF-DPO\n(54 pairs)", "TF-GRPO\n(Online RL)", "TF-KV L16\n(Scaffold)",
    "TF-KV2SP\n(Input-SP)", "Dual\n[M;IF]", "Dual\n[IF;M]"
]
keys = [
    "student_no_sp", "TF_RFT", "TF_IterRFT", "TF_Contrastive",
    "TF_GRPO", "TF_KV_L16_scaffold", "TF_KV2SP", "TF_Dual_Math_IF", "TF_Dual_IF_Math"
]
strict_vals = [c0[k]["prompt_strict"] for k in keys]
closed_vals = [c0[k]["closed_acc"] for k in keys]
x = np.arange(len(labels))
w = 0.38

colors_s = [SLATE, ORANGE, BLUE, RED, TEAL, AMBER, GREEN, PURPLE, PURPLE]
b1 = ax.bar(x - w/2, strict_vals, width=w, color=colors_s, alpha=0.90, edgecolor="#1E293B", linewidth=0.6, label="Prompt Strict Acc (%)")
b2 = ax.bar(x + w/2, closed_vals, width=w, color=colors_s, alpha=0.35, edgecolor="#1E293B", linewidth=0.6, hatch="//", label="Closed-</think> Acc (%)")

ax.axhline(50.0, color=SLATE, linestyle="--", linewidth=0.9, alpha=0.7)
for rect, v in zip(b1, strict_vals):
    ax.text(rect.get_x() + rect.get_width()/2, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#0F172A")
for rect, v in zip(b2, closed_vals):
    ax.text(rect.get_x() + rect.get_width()/2, v + 1.0, f"{v:.0f}", ha="center", va="bottom", fontsize=6.8, color="#475569")

ax.annotate("+18.8 pp over direct DPO\n& +6.3 pp over own KV!", xy=(6 - w/2, 53.0), xytext=(2.6, 79.0),
            fontsize=7.6, fontweight="bold", color=GREEN,
            bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec=GREEN, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylim(0, 94)
ax.set_ylabel("Official IFBench32 Accuracy (%)")
ax.set_title("(a) Teacher-Free Input-SP Landscape on Official IFBench32 (C0 Clean)")
ax.grid(axis="y", alpha=0.4)
ax.legend(loc="upper right", frameon=True, framealpha=0.95)

# ---------------------------------------------------------------------------
# (b) Arbitrary-Context Assembly Stress Test (C0 vs C1 Persona vs C2 History)
# ---------------------------------------------------------------------------
ax = axes[0, 1]
arb = d["ifbench40_arbitrary_ctx"]
regimes = ["C0 Clean\n(0 pre-tok)", "C1 Dynamic Persona\n(+120 pre-tok)", "C2 Multi-Turn Pre-Ctx\n(+180 pre-tok)"]
xr = np.arange(len(regimes))
w = 0.21

s_st  = [c0["student_no_sp"]["prompt_strict"], arb["C1_student_no_sp"]["prompt_strict"], arb["C2_student_no_sp"]["prompt_strict"]]
kv_st = [c0["TF_KV_L16_scaffold"]["prompt_strict"], arb["C1_TF_KV_L16"]["prompt_strict"], arb["C2_TF_KV_L16"]["prompt_strict"]]
ir_st = [c0["TF_IterRFT"]["prompt_strict"], arb["C1_best_InputSP_after_persona"]["prompt_strict"], arb["C2_best_InputSP_after_history"]["prompt_strict"]]
k2_st = [c0["TF_KV2SP"]["prompt_strict"], arb["C1_KV2SP_after_persona"]["prompt_strict"], np.nan]

ax.bar(xr - 1.5*w, s_st, width=w, color=SLATE, alpha=0.85, edgecolor="#1E293B", lw=0.6, label="Bare Student (No SP)")
ax.bar(xr - 0.5*w, kv_st, width=w, color=RED, alpha=0.85, edgecolor="#1E293B", lw=0.6, label="Fixed TF-KV L16 (Pos-0 locked)")
ax.bar(xr + 0.5*w, ir_st, width=w, color=BLUE, alpha=0.88, edgecolor="#1E293B", lw=0.6, label="[Ctx; TF-IterRFT] (Input-SP)")
ax.bar(xr[:2] + 1.5*w, k2_st[:2], width=w, color=GREEN, alpha=0.88, edgecolor="#1E293B", lw=0.6, label="[Ctx; TF-KV2SP] (Input-SP)")

# Closed-think accuracy markers overlaid
s_cl  = [c0["student_no_sp"]["closed_acc"], arb["C1_student_no_sp"]["closed_acc"], arb["C2_student_no_sp"]["closed_acc"]]
kv_cl = [c0["TF_KV_L16_scaffold"]["closed_acc"], arb["C1_TF_KV_L16"]["closed_acc"], arb["C2_TF_KV_L16"]["closed_acc"]]
ir_cl = [c0["TF_IterRFT"]["closed_acc"], arb["C1_best_InputSP_after_persona"]["closed_acc"], arb["C2_best_InputSP_after_history"]["closed_acc"]]
ax.plot(xr - 0.5*w, kv_cl, "o--", color="#991B1B", markersize=5.5, linewidth=1.3, label="Fixed KV Closed-</think> Acc")
ax.plot(xr + 0.5*w, ir_cl, "s-", color=NAVY, markersize=5.5, linewidth=1.3, label="Input-SP Closed-</think> Acc")

for i in range(3):
    ax.text(xr[i] - 1.5*w, s_st[i] + 0.9, f"{s_st[i]:.1f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold")
    ax.text(xr[i] - 0.5*w, kv_st[i] + 0.9, f"{kv_st[i]:.1f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=RED)
    ax.text(xr[i] + 0.5*w, ir_st[i] + 0.9, f"{ir_st[i]:.1f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=NAVY)
    if i < 2:
        ax.text(xr[i] + 1.5*w, k2_st[i] + 0.9, f"{k2_st[i]:.1f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=GREEN)

ax.annotate("C1 Persona: Input-SP beats Fixed KV\n+6.3 pp strict / +15.6 pp closed!", xy=(1 + 1.5*w, 46.5), xytext=(1.35, 76.5),
            fontsize=7.4, fontweight="bold", color=GREEN,
            bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec=GREEN, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
ax.annotate("C2 Drift: Simple sys-PR drops\nto 40.6% (motivates CR-SP!)", xy=(2 + 0.5*w, 40.0), xytext=(1.42, 11.0),
            fontsize=7.3, fontweight="bold", color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF7ED", ec=ORANGE, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))

ax.set_xticks(xr)
ax.set_xticklabels(regimes, fontsize=8.2)
ax.set_ylim(0, 94)
ax.set_ylabel("Official IFBench32 Accuracy (%)")
ax.set_title("(b) Arbitrary-Context Stress Test: Fixed KV vs. Input-Space SP")
ax.grid(axis="y", alpha=0.4)
ax.legend(loc="upper left", ncol=2, fontsize=6.8, frameon=True, framealpha=0.95)

# ---------------------------------------------------------------------------
# (c) Online TF-GRPO Training Trajectory Directly on Layer-0 Input-SP
# ---------------------------------------------------------------------------
ax = axes[1, 0]
curve = d["train_meta"]["TF_GRPO"]["online_curve"]
steps = [r["step"] for r in curve]
rews  = [r["mean_reward"] for r in curve]
strc  = [r["strict_pct"] for r in curve]
rmss  = [r["rms"] for r in curve]

ax2 = ax.twinx()
ax2.spines["top"].set_visible(False)
l1, = ax.plot(steps, strc, "o-", color=TEAL, linewidth=2.0, markersize=5.5, label="Online Batch Strict Pass (%)")
l2, = ax2.plot(steps, rews, "s--", color=BLUE, linewidth=1.6, markersize=4.8, label="Dense Verifiable Reward")
ax.fill_between(steps, 0, strc, color=TEAL, alpha=0.12)

ax.annotate("Peak Step 9:\n75.0% strict\n100% closed", xy=(9, 75.5), xytext=(9.6, 72.0),
            fontsize=7.4, fontweight="bold", color=TEAL,
            bbox=dict(boxstyle="round,pad=0.22", fc="#F0FDFA", ec=TEAL, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9))
ax.text(0.03, 0.94, "Zero collapse: RMS 0.0200 -> 0.0206\nIterRFT R2 bootstrap: 4/14 (28.6%) newly solved",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#94A3B8", lw=0.7))

ax.set_xlabel("Online GRPO Optimization Step (24 on-policy rollouts/step, 0 Teacher)")
ax.set_ylabel("Online Batch Strict Pass Rate (%)", color=TEAL)
ax2.set_ylabel("Mean Verifiable Reward", color=BLUE)
ax.set_xticks(steps)
ax.set_ylim(0, 96)
ax2.set_ylim(-0.15, 0.98)
ax.set_title("(c) Direct Online On-Policy GRPO on Layer-0 Input-SP (Zero Teacher)")
ax.grid(True, alpha=0.35)
ax.legend(handles=[l1, l2], loc="lower right", frameon=True, framealpha=0.92)

# ---------------------------------------------------------------------------
# (d) Post-Hoc Alignment vs UNSEEN M2RL-RL_IF: Spontaneous Discourse Token Shifts
# ---------------------------------------------------------------------------
ax = axes[1, 1]
ph = d["posthoc_unseen_RL_alignment"]
toks = ["Check", " But", " Let", "?", "Thus"]
xt = np.arange(len(toks))
w = 0.25

st_rel   = [ph["student_tok_rel"][t] for t in toks]
grpo_rel = [ph["models"]["TF_GRPO"]["tok_rel"][t] for t in toks]
iter_rel = [ph["models"]["TF_IterRFT"]["tok_rel"][t] for t in toks]

ax.axhline(1.0, color=GREEN, linestyle="--", linewidth=1.2, label="Unseen RL Teacher Target (Ratio = 1.00)")
ax.bar(xt - w, st_rel, width=w, color=SLATE, alpha=0.85, edgecolor="#1E293B", lw=0.6, label="Bare Student (KL=0.194)")
ax.bar(xt, grpo_rel, width=w, color=TEAL, alpha=0.90, edgecolor="#1E293B", lw=0.6, label="TF-GRPO (0 Teacher; G=-33.4%)")
ax.bar(xt + w, iter_rel, width=w, color=BLUE, alpha=0.85, edgecolor="#1E293B", lw=0.6, label="TF-IterRFT (0 Teacher; G=-15.3%)")

for i in range(len(toks)):
    ax.text(xt[i] - w, st_rel[i] + 0.04, f"{st_rel[i]:.2f}", ha="center", va="bottom", fontsize=7.0, color="#334155")
    ax.text(xt[i], grpo_rel[i] + 0.04, f"{grpo_rel[i]:.2f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color=TEAL)

ax.annotate("TF-GRPO spontaneously\nboosts Check/But/? &\nsuppresses premature Thus\n(WITHOUT seeing RL Teacher!)",
            xy=(4, 1.76), xytext=(1.35, 1.62), fontsize=7.5, fontweight="bold", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.28", fc="#EFF6FF", ec=BLUE, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))

ax.set_xticks(xt)
ax.set_xticklabels(["'Check'", "' But'", "' Let'", "'?'", "'Thus'\n(premature)"], fontsize=8.3)
ax.set_ylim(0, 2.38)
ax.set_ylabel("Token Prob. Ratio (Model / Unseen RL Teacher)")
ax.set_title("(d) Post-Hoc Check vs. Unseen M2RL-RL_IF: Spontaneous Policy Shift")
ax.grid(axis="y", alpha=0.4)
ax.legend(loc="upper left", fontsize=7.3, frameon=True, framealpha=0.92)

fig.suptitle("Suite 4 (exp6): Teacher-Free Input-Space Soft Prompt Acquisition & Arbitrary-Context Assembly (Qwen3-4B)",
             fontsize=12.2, fontweight="bold", color="#0F172A", y=0.975)

out_path = os.path.join(OUT_DIR, "tr_fig8_teacherfree_input_sp.png")
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)
