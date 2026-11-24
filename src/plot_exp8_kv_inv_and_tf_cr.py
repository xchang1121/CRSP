"""
Generate tr_fig10_kv_inv_and_tf_cr.png (200 DPI ICML/arXiv style) for Suite 6 (exp8):
  Q1: KV-Prefix Inversion (Geometric Pre-Image vs Functional Operator Inversion) -> CR-SP
  Q2: Direct Teacher-Free Context-Robust Soft Prompt Constructions (4 Methods, 0 Post-RL!)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8.8,
    "axes.titlesize": 9.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.8,
    "xtick.labelsize": 7.9,
    "ytick.labelsize": 7.8,
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

d = json.load(open(os.path.join(ROOT, "exp8_results.json")))
e7 = json.load(open(os.path.join(ROOT, "exp7_results.json")))
pm = d["position_matrix"]["methods"]
regimes = d["regimes"]
reg_short = ["P0\nStart", "P1\nMidSys", "P2\nMultiTurn", "P3\nPostRAG", "P4\nPostQuery", "P5\nLong1000"]

fig, axes = plt.subplots(2, 2, figsize=(16.4, 11.2))
fig.subplots_adjust(hspace=0.34, wspace=0.42, top=0.92, bottom=0.07, left=0.065, right=0.955)

# ---------------------------------------------------------------------------
# (a) Layerwise Off-Manifold Geometric KV Inversion Error Profile (Layers 0..35)
# ---------------------------------------------------------------------------
ax = axes[0, 0]
m1a = d["q1_kv_inversion_meta"]["Inv1a_GeoKV_DataFree"]
m1b = d["q1_kv_inversion_meta"]["Inv1b_CR_GeoKV_DataFree"]
layers = np.arange(len(m1a["layer_err_k"]))

ax.plot(layers, m1a["layer_err_k"], "-", color=RED, lw=2.0, label=f"Inv1a GeoKV: Key RelErr (Mean={m1a['mean_rel_err_K']:.3f})")
ax.plot(layers, m1a["layer_err_v"], "--", color=ORANGE, lw=2.0, label=f"Inv1a GeoKV: Val RelErr (Mean={m1a['mean_rel_err_V']:.3f})")
ax.plot(layers, m1b["layer_err_k"], ":", color=BLUE, lw=1.7, label=f"Inv1b CR-GeoKV: Key RelErr (Mean={m1b['mean_rel_err_K']:.3f})")
ax.plot(layers, m1b["layer_err_v"], "-.", color=TEAL, lw=1.7, label=f"Inv1b CR-GeoKV: Val RelErr (Mean={m1b['mean_rel_err_V']:.3f})")

sig_amp = 0.1382 / 2.55
ax.axhline(sig_amp, color="#0F172A", linestyle="--", lw=1.4, alpha=0.9, label=f"RL Signal Amp. ||dK*||/||K0|| ~ {sig_amp:.3f}")
ax.axhspan(0.0, sig_amp, color="#10B981", alpha=0.11)
ax.annotate(
    "L0 Val Ellipsoid Barrier\n(96.3% RelErr vs 5.4% Signal)",
    xy=(0.2, 0.9626), xytext=(5.5, 0.85),
    fontsize=7.2, fontweight="bold", color="#9A3412",
    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF7ED", ec=ORANGE, lw=0.8),
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1),
)
ax.annotate(
    "All 36 layers > RL signal (5.4%)\nProj. Energy 9.02% = 31x Signal (0.29%)",
    xy=(23.0, 0.072), xytext=(11.0, 0.50),
    fontsize=7.2, fontweight="bold", color="#0F172A",
    bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#475569", lw=0.8),
    arrowprops=dict(arrowstyle="->", color="#475569", lw=1.0),
)

ax.set_xlabel("Frozen Transformer Layer Index (l = 0 .. 35)")
ax.set_ylabel("Relative KV Error  ||Phi_l(P) - K*_l|| / ||K*_l||")
ax.set_title("(a) Q1 Diagnostic: Geometric Pre-Image Barrier Across Depth")
ax.set_xlim(0, 35)
ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.4)
ax.legend(loc="upper right", fontsize=6.7, frameon=True, framealpha=0.95)

# ---------------------------------------------------------------------------
# (b) 11-Method x 6-Position (P0..P5) KL Gap Closed G(P, s) Diverging Heatmap
# ---------------------------------------------------------------------------
ax = axes[0, 1]
method_order = [
    ("KV_Distilled_L16",         "Q1 KV_Distilled (Pos-0)"),
    ("Inv1a_GeoKV_DF",           "Q1 Inv1a: GeoKV (L16)"),
    ("Inv1b_CR_GeoKV_DF",        "Q1 Inv1b: CR-GeoKV"),
    ("Inv2_LogitCR_L256",        "Q1 Inv2: Logit CR-Inv*"),
    ("Inv3_FuncOp_L256",         "Q1 Inv3: FuncOp L256"),
    ("Inv3_FuncOp_L64",          "Q1 Inv3: FuncOp L64"),
    ("KV_GRPO_TF_L16",           "Q2 KV-GRPO (0-Tch)"),
    ("TF1_CR_KVGRPO_Inv",        "Q2 TF1: KVGRPO->Inv3"),
    ("TF2_CR_Direct_GRPO",       "Q2 TF2: Direct GRPO"),
    ("TF3_CR_Anchored_SelfBoot", "Q2 TF3: AnchSelfBoot"),
    ("TF4_CR_SelfRubric",        "Q2 TF4: SelfRubric"),
]
method_order = [(k, lbl) for k, lbl in method_order if k in pm]

mat = np.zeros((len(method_order), len(regimes)))
for i, (k, _) in enumerate(method_order):
    for j, r in enumerate(regimes):
        mat[i, j] = pm[k][r]["G_vs_ctx_student_pct"]

norm = TwoSlopeNorm(vmin=-30.0, vcenter=0.0, vmax=48.0)
im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", norm=norm)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat[i, j]
        txt_col = "white" if (val <= -18.0 or val >= 34.0) else "#0F172A"
        ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=7.0, fontweight="bold", color=txt_col)

ax.axhline(5.5, color="#0F172A", linewidth=1.8)
ax.set_xticks(np.arange(len(reg_short)))
ax.set_xticklabels(reg_short, fontsize=7.6)
ax.set_yticks(np.arange(len(method_order)))
ax.set_yticklabels([lbl for _, lbl in method_order], fontsize=7.5)
ax.set_title("(b) 11-Method x 6-Position KL Gap Closed G(P, s) Matrix (Q1 vs Q2)")
cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.025, extend="both")
cbar.ax.tick_params(labelsize=7.3)
cbar.set_label("G(P, s) (%)  [0 = White]", fontsize=7.8)

# Helper to plot dynamically-centered grouped bars cleanly
def plot_centered_bars(ax, groups, s_meta, bw=0.135):
    seen = set()
    xg = np.arange(len(groups))
    bar_coords = {}
    for g_i, g in enumerate(groups):
        present = [(s_i, g[s_i + 1]) for s_i in range(len(s_meta)) if g[s_i + 1] is not None]
        k = len(present)
        for pos_i, (s_i, val) in enumerate(present):
            s_lbl, s_col = s_meta[s_i]
            x_pos = xg[g_i] + (pos_i - (k - 1) / 2.0) * bw
            lbl = s_lbl if s_i not in seen else None
            seen.add(s_i)
            rect = ax.bar([x_pos], [val], width=bw * 0.88, color=s_col, alpha=0.89, edgecolor="#1E293B", lw=0.55, label=lbl)[0]
            ax.text(x_pos, val + 0.85, f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#0F172A")
            bar_coords[(g_i, s_i)] = (x_pos, val)
    ax.set_xticks(xg)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.1)
    return bar_coords

# ---------------------------------------------------------------------------
# (c) Q1 End-to-End Official IFBench32: KV_Distilled -> Inv1a vs Inv3 (L256 & L64)
# ---------------------------------------------------------------------------
ax = axes[1, 0]
ib = d["ifbench32_evals"]
e7ib = e7["ifbench32_by_position"]

q1_groups = [
    ("P0 Clean\nStart",     e7ib["P0_Student"]["prompt_strict"], ib["P0_KV_Distilled_L16"]["prompt_strict"], ib["P0_Inv1a_GeoKV_DF"]["prompt_strict"], ib["P0_Inv3_FuncOp_L256"]["prompt_strict"], None, e7ib["P0_A4_Full_CR_Inv"]["prompt_strict"]),
    ("P2 Multi-Turn\nMid",  e7ib["P2_Student"]["prompt_strict"], None, None, ib["P2_Inv3_FuncOp_L256"]["prompt_strict"], ib["P2_Inv3_FuncOp_L64"]["prompt_strict"], e7ib["P2_A4_Full_CR_Inv"]["prompt_strict"]),
    ("P3 Post-RAG\nDoc",    e7ib["P3_Student"]["prompt_strict"], None, None, ib["P3_Inv3_FuncOp_L256"]["prompt_strict"], None, e7ib["P3_A4_Full_CR_Inv"]["prompt_strict"]),
    ("P4 Post-Query\nLate", e7ib["P2_Student"]["prompt_strict"], None, None, ib["P4_Inv3_FuncOp_L256"]["prompt_strict"], ib["P4_Inv3_FuncOp_L64"]["prompt_strict"], e7ib["P4_A4_Full_CR_Inv"]["prompt_strict"]),
]
s_meta_q1 = [
    ("Bare Student",               SLATE),
    ("KV_Distilled L16 (Pos-0)",   AMBER),
    ("Inv1a GeoKV (Data-Free)",    RED),
    ("Inv3 FuncOp L256 (0-Tch)",   GREEN),
    ("Inv3 FuncOp L64 (4x Short)", TEAL),
    ("Oracle Direct-Tch CR-SP",    NAVY),
]
bc1 = plot_centered_bars(ax, q1_groups, s_meta_q1, bw=0.145)

x_p4_inv3, y_p4_inv3 = bc1[(3, 3)]
ax.annotate(
    "Matches Direct-Tch Oracle\n(50.0%, closed_acc=84.2%)!",
    xy=(x_p4_inv3, y_p4_inv3 + 3.2), xytext=(2.15, 59.5),
    fontsize=7.0, fontweight="bold", color="#065F46",
    bbox=dict(boxstyle="round,pad=0.28", fc="#ECFDF5", ec=GREEN, lw=0.8),
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1),
)

ax.set_ylim(0, 72)
ax.set_ylabel("Official IFBench32 Prompt Strict Acc (%)")
ax.set_title("(c) Q1 End-to-End IFBench32: Inverting KV-Prefix into CR-SP")
ax.grid(axis="y", alpha=0.4)
ax.legend(loc="upper left", ncol=2, fontsize=6.7, frameon=True, framealpha=0.95)

# ---------------------------------------------------------------------------
# (d) Q2 End-to-End Official IFBench32: 4 Zero-Post-RL Teacher-Free CR Methods!
# ---------------------------------------------------------------------------
ax = axes[1, 1]
q2_groups = [
    ("P0 Clean\nStart",     e7ib["P0_Student"]["prompt_strict"], ib["P0_KV_GRPO_TF_L16"]["prompt_strict"], ib["P0_TF1_CR_KVGRPO_Inv"]["prompt_strict"], None, None, ib["P0_TF4_CR_SelfRubric"]["prompt_strict"]),
    ("P2 Multi-Turn\nMid",  e7ib["P2_Student"]["prompt_strict"], None, ib["P2_TF1_CR_KVGRPO_Inv"]["prompt_strict"], ib["P2_TF2_CR_Direct_GRPO"]["prompt_strict"], ib["P2_TF3_CR_Anchored_SelfBoot"]["prompt_strict"], ib["P2_TF4_CR_SelfRubric"]["prompt_strict"]),
    ("P4 Post-Query\nLate", e7ib["P2_Student"]["prompt_strict"], None, ib["P4_TF1_CR_KVGRPO_Inv"]["prompt_strict"], ib["P4_TF2_CR_Direct_GRPO"]["prompt_strict"], ib["P4_TF3_CR_Anchored_SelfBoot"]["prompt_strict"], ib["P4_TF4_CR_SelfRubric"]["prompt_strict"]),
]
s_meta_q2 = [
    ("Bare Student",                  SLATE),
    ("KV-GRPO L16 (0-Tch Pos-0)",     AMBER),
    ("TF1: KV-GRPO -> Inv3 CR-SP",    BLUE),
    ("TF2: Direct MultiSlot GRPO",    TEAL),
    ("TF3: KL-Anch SelfBoot CR",      ORANGE),
    ("TF4: SelfRubric -> Inv3 CR",    PURPLE),
]
bc2 = plot_centered_bars(ax, q2_groups, s_meta_q2, bw=0.145)

ax.axhline(e7ib["P2_A2_Paper_PR"]["prompt_strict"], color=RED, linestyle=":", lw=1.2, alpha=0.85, label="Direct-Tch Paper PR (P2: 40.6%)")
ax.axhline(e7ib["P4_A2_Paper_PR"]["prompt_strict"], color="#991B1B", linestyle="--", lw=1.2, alpha=0.80, label="Direct-Tch Paper PR (P4: 21.9%)")

x_p4_tf2, y_p4_tf2 = bc2[(2, 3)]
ax.annotate(
    "0-Post-RL TF2 (L64): 50.0% at P2 & P4\n(Beats Oracle P2 & 2.29x Paper PR P4!)",
    xy=(x_p4_tf2, y_p4_tf2 + 3.2), xytext=(0.82, 57.5),
    fontsize=7.0, fontweight="bold", color="#115E59",
    bbox=dict(boxstyle="round,pad=0.28", fc="#F0FDFA", ec=TEAL, lw=0.8),
    arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.1),
)

ax.set_ylim(0, 76)
ax.set_ylabel("Official IFBench32 Prompt Strict Acc (%)")
ax.set_title("(d) Q2 End-to-End IFBench32: 100% Zero-Post-RL Teacher-Free CR-SP")
ax.grid(axis="y", alpha=0.4)
ax.legend(loc="upper left", ncol=2, fontsize=6.5, frameon=True, framealpha=0.95)

fig.suptitle("Suite 6 (exp8): KV-Prefix Inversion to CR-SP (Q1) & Direct Teacher-Free CR-SP Construction (Q2)",
             fontsize=12.2, fontweight="bold", color="#0F172A", y=0.975)

out_path = os.path.join(OUT_DIR, "tr_fig10_kv_inv_and_tf_cr.png")
plt.savefig(out_path, dpi=200)
plt.close()
print("Saved:", out_path)
