"""
Generate two publication-grade 200 DPI ICML/arXiv figures for Suite 7 (exp9 + exp9b):
  1) tr_fig11_multiseed_and_postrl_scaled.png:
       (a) Track 1: 3-Seed IFEval96 Mean +/- Std (Zero-Post-RL vs Post-RL Direct & Alg A KV-Inv2)
       (b) Scaled IFBench64 (64 Prompts): 3-Way Head-to-Head + P0 KV Ceiling
       (c) Adversarial P3 Post-RAG (420-tok Distractor) & Closed-Think Precision Frontier
       (d) Zero-Leakage Declarative Capacity & Recency Law on HeliosPay v4.2 (Doc36)
  2) tr_fig12_composition_and_interference.png:
       (a) 4-Way Exact Cross-Expert KL Interference & Off-Domain Leakage Diagnostic
       (b) 3-Expert Simultaneous Split-Slot Composition ([Doc@P1 + Math@P2 + IF@P4]) on IFEval96 & IFBench64
       (c) Architectural Law: Split-Slot Synergy (+16.7 pp) vs Adjacent Crowding & 0.00 pp Permutation Invariance
       (d) Grand Pareto Frontier: Strict Acc vs Closed-Think Precision across All CR-SP Pathways (IFBench64)
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
    "font.size": 8.6,
    "axes.titlesize": 9.6,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.6,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.0,
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
SKY    = "#0284C7"
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

d9 = json.load(open(os.path.join(ROOT, "exp9_results.json")))
d9b = json.load(open(os.path.join(ROOT, "exp9b_results.json")))


# ===========================================================================
# FIGURE 11: Multi-Seed Convergence, Scaled IFBench64, P3-RAG & Doc36 Law
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(17.2, 11.8))
fig.subplots_adjust(hspace=0.38, wspace=0.24, top=0.92, bottom=0.09, left=0.06, right=0.98)

# --- (a) 3-Seed Held-Out IFEval96 (Zero-Post-RL vs Post-RL Direct & Alg A) ---
ax = axes[0, 0]
s96 = d9["track1_ifeval96_summary"]
s96b = d9b["ifeval96_multiseed_summary"]
bare96 = d9["track1_ifeval96_multiseed"]["Bare_Student"]["prompt_strict"]
ie9b = d9b["ifeval96_evals"]

spec_a = [
    ("Bare\nStudent",         bare96, 0.0, [], 81.0, SLATE),
    ("P2 S1\nTF3 (0-RL)",     s96["P2_Stage1_TF3"]["strict_mean"],      s96["P2_Stage1_TF3"]["strict_std"],      s96["P2_Stage1_TF3"]["strict_per_seed"],      s96["P2_Stage1_TF3"]["closed_acc_mean"],      AMBER),
    ("P2 Cold\nTF2 (0-RL)",   s96["P2_Cold_TF2_step10"]["strict_mean"], s96["P2_Cold_TF2_step10"]["strict_std"], s96["P2_Cold_TF2_step10"]["strict_per_seed"], s96["P2_Cold_TF2_step10"]["closed_acc_mean"], "#60A5FA"),
    ("P2 AlgB\n(0-RL)",       s96["P2_Full_AlgB_step10"]["strict_mean"],s96["P2_Full_AlgB_step10"]["strict_std"],s96["P2_Full_AlgB_step10"]["strict_per_seed"],s96["P2_Full_AlgB_step10"]["closed_acc_mean"],GREEN),
    ("P2 PR\nDir L64",        s96b["P2_PostRL_Direct_L64"]["strict_mean"], s96b["P2_PostRL_Direct_L64"]["strict_std"], s96b["P2_PostRL_Direct_L64"]["strict_per_seed"], s96b["P2_PostRL_Direct_L64"]["closed_acc_mean"], BLUE),
    ("P4 S1\nTF3 (0-RL)",     s96["P4_Stage1_TF3"]["strict_mean"],      s96["P4_Stage1_TF3"]["strict_std"],      s96["P4_Stage1_TF3"]["strict_per_seed"],      s96["P4_Stage1_TF3"]["closed_acc_mean"],      ORANGE),
    ("P4 AlgB\n(0-RL)",       s96["P4_Full_AlgB_step10"]["strict_mean"],s96["P4_Full_AlgB_step10"]["strict_std"],s96["P4_Full_AlgB_step10"]["strict_per_seed"],s96["P4_Full_AlgB_step10"]["closed_acc_mean"],RED),
    ("P4 PR\nDir L64",        s96b["P4_PostRL_Direct_L64"]["strict_mean"], s96b["P4_PostRL_Direct_L64"]["strict_std"], s96b["P4_PostRL_Direct_L64"]["strict_per_seed"], s96b["P4_PostRL_Direct_L64"]["closed_acc_mean"], NAVY),
    ("P4 PR\nDir L256",       ie9b["P4_PostRL_Direct_L256"]["prompt_strict"], 0.0, [], ie9b["P4_PostRL_Direct_L256"]["closed_acc"], PURPLE),
    ("P4 AlgA\nInv2 L64",     ie9b["P4_PostRL_KVInv2_L64"]["prompt_strict"],  0.0, [], ie9b["P4_PostRL_KVInv2_L64"]["closed_acc"],  TEAL),
]
xa = np.arange(len(spec_a))
for i, (lbl, m, sd, pts, ca, col) in enumerate(spec_a):
    ax.bar(i, m, width=0.68, color=col, alpha=0.88, edgecolor="#1E293B", lw=0.6)
    if sd > 0:
        ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="#0F172A", elinewidth=1.2, capsize=3.2, zorder=4)
    for pi, pv in enumerate(pts):
        ax.scatter(i + (pi - 1) * 0.11, pv, color="#0F172A", s=13, alpha=0.82, zorder=5)
    top_y = max(m + sd, max(pts) if pts else m)
    ax.text(i, top_y + 0.9, f"{m:.1f}\n({ca:.0f}%)", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color="#0F172A")

ax.axhline(bare96, color=SLATE, linestyle="--", lw=1.0, alpha=0.75)
ax.axvline(4.5, color="#94A3B8", linestyle=":", lw=1.2)
ax.text(2.0, 91.5, "P2 Multi-Turn Mid\n(AlgB std 4.1x lower)", ha="center", va="top", fontsize=7.0, fontweight="bold", color=GREEN,
        bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec=GREEN, lw=0.7))
ax.text(7.2, 91.5, "P4 Post-Query Suffix\n(Post-RL 3.6x lower std & 92.1% cl.)", ha="center", va="top", fontsize=7.0, fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.25", fc="#EFF6FF", ec=NAVY, lw=0.7))
ax.set_xticks(xa)
ax.set_xticklabels([b[0] for b in spec_a], fontsize=6.9)
ax.set_ylim(35, 95)
ax.set_ylabel("Held-Out IFEval96 Strict Acc (%)  [Label: Strict (Closed%)]")
ax.set_title("(a) 3-Seed IFEval96 (288 Eval Pairs/Method): Zero-Post-RL vs Post-RL CS-CR")
ax.grid(axis="y", alpha=0.38)

# --- (b) Scaled IFBench64 (64 Stratified Prompts) ---
ax = axes[0, 1]
b9 = d9["track1_ifbench64"]
b9b = d9b["ifbench64_evals"]

spec_b = [
    ("Bare\nStudent",         b9["Bare_Student"]["prompt_strict"],         b9["Bare_Student"]["closed_acc"],         SLATE),
    ("P4 Cold\nTF2 (0-RL)",   b9["P4_Cold_TF2"]["prompt_strict"],          b9["P4_Cold_TF2"]["closed_acc"],          "#60A5FA"),
    ("P4 S1\nTF3 (0-RL)",     b9["P4_Stage1_TF3"]["prompt_strict"],        b9["P4_Stage1_TF3"]["closed_acc"],        AMBER),
    ("P4 AlgB\n(0-RL)",       b9["P4_Full_AlgB"]["prompt_strict"],         b9["P4_Full_AlgB"]["closed_acc"],         GREEN),
    ("P0 KV\nL16 (Src)",      b9b["P0_KV_PostRL_IF_L16"]["prompt_strict"], b9b["P0_KV_PostRL_IF_L16"]["closed_acc"], "#94A3B8"),
    ("P2 PR\nDir L64",        b9b["P2_PostRL_Direct_L64"]["prompt_strict"],b9b["P2_PostRL_Direct_L64"]["closed_acc"],BLUE),
    ("P4 PR\nDir L256",       b9b["P4_PostRL_Direct_L256"]["prompt_strict"],b9b["P4_PostRL_Direct_L256"]["closed_acc"],NAVY),
    ("P2 AlgA\nInv2 L256",    b9b["P2_PostRL_KVInv2_L256"]["prompt_strict"],b9b["P2_PostRL_KVInv2_L256"]["closed_acc"],PURPLE),
    ("P4 AlgA\nInv2 L256",    b9b["P4_PostRL_KVInv2_L256"]["prompt_strict"],b9b["P4_PostRL_KVInv2_L256"]["closed_acc"],"#A78BFA"),
    ("P4 AlgA\nInv2 L64*",    b9b["P4_PostRL_KVInv2_L64"]["prompt_strict"], b9b["P4_PostRL_KVInv2_L64"]["closed_acc"], TEAL),
]
xb = np.arange(len(spec_b))
for i, (lbl, st, ca, col) in enumerate(spec_b):
    lw = 1.6 if "L64*" in lbl else 0.6
    ec = "#0F172A" if "L64*" in lbl else "#1E293B"
    ax.bar(i, st, width=0.66, color=col, alpha=0.90, edgecolor=ec, lw=lw)
    ax.text(i, st + 0.9, f"{st:.1f}%\n(cl:{ca:.0f}%)", ha="center", va="bottom", fontsize=6.3, fontweight="bold",
            color="#0F766E" if "L64*" in lbl else "#0F172A")

ax.axhline(b9["Bare_Student"]["prompt_strict"], color=SLATE, linestyle="--", lw=1.0, alpha=0.75)
ax.axvline(3.5, color="#94A3B8", linestyle=":", lw=1.2)
ax.annotate("BREAKTHROUGH: 53.1% (79.1% closed)\n141.7% of Source P0 KV | +7.8 pp vs Bare",
            xy=(8.68, 51.8), xytext=(3.85, 62.8),
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4, connectionstyle="arc3,rad=-0.10"),
            fontsize=7.3, fontweight="bold", color="#115E59",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F0FDFA", ec=TEAL, lw=0.9))
ax.set_xticks(xb)
ax.set_xticklabels([b[0] for b in spec_b], fontsize=6.9)
ax.set_ylim(0, 72)
ax.set_ylabel("Official IFBench64 Strict Acc (%)  [2x IFBench32]")
ax.set_title("(b) Scaled IFBench64 (64 Prompts): Zero-Post-RL vs Post-RL Pathways")
ax.grid(axis="y", alpha=0.38)

# --- (c) Adversarial P3 Post-RAG (420-tok Distractor) Staircase & Closed-Think Gain ---
ax = axes[1, 0]
p3_spec = [
    ("Zero-Post-RL\nAlgB (L=64)",     b9b["P3_ZeroPostRL_AlgB_L64"]["prompt_strict"], b9b["P3_ZeroPostRL_AlgB_L64"]["closed_acc"], ORANGE),
    ("Bare Student\n(P3 Post-RAG)",   b9b["P3_Bare_Student"]["prompt_strict"],        b9b["P3_Bare_Student"]["closed_acc"],        SLATE),
    ("Alg A KV-Inv2\n(L=256, 0-Tch)", b9b["P3_PostRL_KVInv2_L256"]["prompt_strict"],  b9b["P3_PostRL_KVInv2_L256"]["closed_acc"],  TEAL),
    ("Post-RL Direct\nCS-CR (L=256)", b9b["P3_PostRL_Direct_L256"]["prompt_strict"],  b9b["P3_PostRL_Direct_L256"]["closed_acc"],  NAVY),
]
xc = np.arange(len(p3_spec))
w = 0.34
s_vals = [x[1] for x in p3_spec]
c_vals = [x[2] for x in p3_spec]
cols_c = [x[3] for x in p3_spec]
r1 = ax.bar(xc - w/2, s_vals, width=w*0.92, color=cols_c, alpha=0.90, edgecolor="#1E293B", lw=0.7, label="Prompt Strict Acc (%)")
r2 = ax.bar(xc + w/2, c_vals, width=w*0.92, color=cols_c, alpha=0.42, edgecolor="#1E293B", lw=0.7, hatch="//", label="Closed-Think Precision (%)")
for rect, v in zip(r1, s_vals):
    ax.text(rect.get_x() + rect.get_width()/2, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#0F172A")
for rect, v in zip(r2, c_vals):
    ax.text(rect.get_x() + rect.get_width()/2, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#334155")

ax.axhline(b9["Bare_Student"]["prompt_strict"], color=SLATE, linestyle="--", lw=1.0, alpha=0.75, label="Clean Multi-Turn Bare (45.3%)")
ax.text(0.02, 0.96, "P3 Post-RAG = Persona + 420-tok Distractor RAG + [SP] + Query\nDirect L=256 recovers +9.4 pp over RAG Bare & beats Clean Bare!",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.1, fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec=NAVY, lw=0.75))
ax.set_xticks(xc)
ax.set_xticklabels([x[0] for x in p3_spec], fontsize=7.9)
ax.set_ylim(0, 90)
ax.set_ylabel("Official IFBench64 Accuracy (%) under P3 Post-RAG")
ax.set_title("(c) Adversarial P3 Post-RAG Stress Test: Monotonic 4-Tier Hierarchy")
ax.grid(axis="y", alpha=0.38)
ax.legend(loc="upper right", fontsize=6.9, frameon=True, framealpha=0.95)

# --- (d) Zero-Leakage Declarative Capacity & Recency Law (HeliosPay v4.2 Doc36) ---
ax = axes[1, 1]
de9 = d9["track2_composition_evals"]["doc36"]
de9b = d9b["doc36_evals"]

doc_spec = [
    ("Bare\nStudent",              0.0,   0.0,   SLATE),
    ("Input SP L64\nRMS<=0.055\nNo-EOS (P1-P4)", 0.0, 0.0, RED),
    ("Muon L128\nEOS @ P1\n(Far Sys)",    de9b["Single_Doc_CSCR_MuonL128_P1"]["overall_acc"], de9b["Single_Doc_CSCR_MuonL128_P1"]["fact_acc"], AMBER),
    ("Muon L128\nEOS @ P4\n(Post-Q)",     de9b["Single_Doc_CSCR_MuonL128_P4"]["overall_acc"], de9b["Single_Doc_CSCR_MuonL128_P4"]["fact_acc"], ORANGE),
    ("Muon L128\nEOS @ P2\n(Pre-Q)",      de9b["Single_Doc_CSCR_MuonL128_P2"]["overall_acc"], de9b["Single_Doc_CSCR_MuonL128_P2"]["fact_acc"], TEAL),
    ("KV-CSPD\nL=32 (Exp4)\n[Ref Ceiling]", 61.11, 50.0, BLUE),
    ("In-Context\nRaw Doc\n(Upper)",       de9["InContext_Doc_UpperBound"]["overall_acc"], de9["InContext_Doc_UpperBound"]["fact_acc"], NAVY),
]
xd = np.arange(len(doc_spec))
w2 = 0.36
ov_v = [x[1] for x in doc_spec]
fa_v = [x[2] for x in doc_spec]
cd = [x[3] for x in doc_spec]
b_ov = ax.bar(xd - w2/2, ov_v, width=w2*0.92, color=cd, alpha=0.90, edgecolor="#1E293B", lw=0.65, label="Overall Strict Acc (Rules+Facts)")
b_fa = ax.bar(xd + w2/2, fa_v, width=w2*0.92, color=cd, alpha=0.45, edgecolor="#1E293B", lw=0.65, hatch="//", label="Exact Verbatim Fact Recall (%)")
for rect, v in zip(b_ov, ov_v):
    ax.text(rect.get_x() + rect.get_width()/2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#0F172A")
for rect, v in zip(b_fa, fa_v):
    ax.text(rect.get_x() + rect.get_width()/2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#334155")

ax.axvline(4.5, color="#94A3B8", linestyle=":", lw=1.2)
ax.text(0.02, 0.96, "Declarative Capacity & Recency Law (0% Pretrain Leakage):\n- L=64 RMS<=0.055 underfits (0%)\n- Muon L128+EOS: P1(8.3%) < P4(25.0%) < P2(41.7% facts, 5x!)\n- Exact multi-step rules still require 36-layer KV-CSPD",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.8, fontweight="bold", color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec=AMBER, lw=0.8))
ax.set_xticks(xd)
ax.set_xticklabels([x[0] for x in doc_spec], fontsize=6.8)
ax.set_ylim(0, 118)
ax.set_ylabel("HeliosPay v4.2 Doc36 Accuracy (%)")
ax.set_title("(d) Zero-Leakage Vertical Domain Doc (HeliosPay v4.2): Input-SP vs KV-CSPD Law")
ax.grid(axis="y", alpha=0.38)
ax.legend(loc="upper right", fontsize=6.8, frameon=True, framealpha=0.95)

fig.suptitle("Suite 7 (exp9 + exp9b): 3-Seed Validation, Scaled Official IFBench64, Adversarial P3-RAG & Declarative Capacity Law",
             fontsize=12.0, fontweight="bold", color="#0F172A", y=0.975)

out11 = os.path.join(OUT_DIR, "tr_fig11_multiseed_and_postrl_scaled.png")
plt.savefig(out11, dpi=200)
plt.close()
print("Saved:", out11)


# ===========================================================================
# FIGURE 12: Cross-Expert KL Interference, 3-Expert Composition, Split vs Adj & Pareto
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(17.2, 11.8))
fig.subplots_adjust(hspace=0.38, wspace=0.25, top=0.92, bottom=0.09, left=0.06, right=0.98)

# --- (a) 4-Way Exact Cross-Expert KL Interference & Off-Domain Leakage ---
ax = axes[0, 0]
kn = d9["track2_interference_kl"]["Naive"]
kc = d9["track2_interference_kl"]["CSCR"]
kd = d9b["interference_kl"]["PostRL_Direct_CSCR"]
ki = d9b["interference_kl"]["PostRL_KVInv2_CSCR"]

metrics_lbl = [
    "IF Own -> Triple\nInterference KL\n(Lower Better)",
    "Math Own -> Triple\nInterference KL\n(Lower Better)",
    "Doc Off-Dom Leak\nonto IF Queries\n(Lower Better)",
    "IF On-Domain\nSignal (bare->if)\n(Higher Better)",
]
groups = [
    ("Naive CR-SP (exp9)",               RED,   [kn["if"]["triple_interference_kl"], kn["math"]["triple_interference_kl"], kn["if"]["kl_from_bare_doc"], kn["if"]["kl_from_bare_if"]]),
    ("Zero-Post-RL CS-CR (exp9)",        AMBER, [kc["if"]["triple_interference_kl"], kc["math"]["triple_interference_kl"], kc["if"]["kl_from_bare_doc"], kc["if"]["kl_from_bare_if"]]),
    ("Post-RL Direct CS-CR (exp9b*)",    BLUE,  [kd["if"]["triple_interference_kl"], kd["math"]["triple_interference_kl"], kd["if"]["kl_from_bare_doc"], kd["if"]["kl_from_bare_if"]]),
    ("Alg A KV-Inv2 CS-CR (exp9b*)",     TEAL,  [ki["if"]["triple_interference_kl"], ki["math"]["triple_interference_kl"], ki["if"]["kl_from_bare_doc"], ki["if"]["kl_from_bare_if"]]),
]
xm = np.arange(len(metrics_lbl))
bw4 = 0.19
for gi, (glbl, gcol, gvals) in enumerate(groups):
    pos = xm + (gi - 1.5) * bw4
    rects = ax.bar(pos, gvals, width=bw4*0.90, color=gcol, alpha=0.89, edgecolor="#1E293B", lw=0.6, label=glbl)
    for rect, v in zip(rects, gvals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=6.2, fontweight="bold", color="#0F172A")

ax.axvline(2.5, color="#94A3B8", linestyle=":", lw=1.3)
ax.annotate("2.95x Interference Cut!\n(0.157 -> 0.053)", xy=(0 + 1.5*bw4, 0.072), xytext=(0.45, 0.215),
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3, connectionstyle="arc3,rad=0.18"),
            fontsize=7.2, fontweight="bold", color="#115E59",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F0FDFA", ec=TEAL, lw=0.8))
ax.annotate("3.0x Stronger\nOn-Domain Signal!", xy=(3 + 1.5*bw4 - 0.03, 0.252), xytext=(2.42, 0.292),
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2, connectionstyle="arc3,rad=-0.10"),
            fontsize=7.0, fontweight="bold", color="#115E59")
ax.set_xticks(xm)
ax.set_xticklabels(metrics_lbl, fontsize=7.5)
ax.set_ylim(0, 0.345)
ax.set_ylabel("Top-32 Output KL Divergence")
ax.set_title("(a) 4-Way Cross-Expert KL Matrix (*exp9b mounts 256 vecs incl. Doc L=128 RMS=0.14)")
ax.grid(axis="y", alpha=0.38)
ax.legend(loc="upper center", ncol=2, fontsize=6.7, frameon=True, framealpha=0.95)

# --- (b) 3-Expert Simultaneous Split-Slot Composition ([Doc@P1 + Math@P2 + IF@P4]) ---
ax = axes[0, 1]
ie9 = d9["track2_composition_evals"]["ifeval96"]
be9 = d9["track2_composition_evals"]["ifbench64"]
ie9b = d9b["ifeval96_evals"]
be9b = d9b["ifbench64_evals"]

comp_tasks = ["IFEval96\n(Strict Acc %)", "IFEval96\n(Closed Precision %)", "Official IFBench64\n(Strict Acc %)", "Official IFBench64\n(Closed Precision %)"]
comp_series = [
    ("Triple Naive [192v] (exp9)",          RED,   [ie9["Triple_Naive_Split"]["prompt_strict"],     ie9["Triple_Naive_Split"]["closed_acc"],     be9["Triple_Naive_Split"]["prompt_strict"],     be9["Triple_Naive_Split"]["closed_acc"]]),
    ("Triple 0-RL CS-CR [192v] (exp9)",     AMBER, [ie9["Triple_CSCR_Split"]["prompt_strict"],      ie9["Triple_CSCR_Split"]["closed_acc"],      be9["Triple_CSCR_Split"]["prompt_strict"],      be9["Triple_CSCR_Split"]["closed_acc"]]),
    ("Triple PR-Direct [256v] (exp9b)",     BLUE,  [ie9b["Triple_PostRL_Direct_Split"]["prompt_strict"], ie9b["Triple_PostRL_Direct_Split"]["closed_acc"], be9b["Triple_PostRL_Direct_Split"]["prompt_strict"], be9b["Triple_PostRL_Direct_Split"]["closed_acc"]]),
    ("Triple AlgA KVInv2 [256v] (exp9b)",   TEAL,  [ie9b["Triple_PostRL_KVInv2_Split"]["prompt_strict"], ie9b["Triple_PostRL_KVInv2_Split"]["closed_acc"], be9b["Triple_PostRL_KVInv2_Split"]["prompt_strict"], be9b["Triple_PostRL_KVInv2_Split"]["closed_acc"]]),
]
xct = np.arange(len(comp_tasks))
for si, (slbl, scol, svals) in enumerate(comp_series):
    pos = xct + (si - 1.5) * bw4
    rects = ax.bar(pos, svals, width=bw4*0.90, color=scol, alpha=0.90, edgecolor="#1E293B", lw=0.6, label=slbl)
    for rect, v in zip(rects, svals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#0F172A")

ax.axvline(1.5, color="#94A3B8", linestyle=":", lw=1.3)
ax.text(0.985, 0.965, "Triple AlgA KVInv2 sweeps Triple Direct:\n  IFEval96: +8.3 pp | IFBench64: +7.8 pp\n  IFBench64 Closed: 73.0% (+23.0 pp vs Naive!)\n  0-RL CS-CR holds 98.5% solo retention on IFEval96",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#475569", lw=0.8))
ax.set_xticks(xct)
ax.set_xticklabels(comp_tasks, fontsize=7.6)
ax.set_ylim(0, 108)
ax.set_ylabel("Accuracy (%) with 3 Specialists Simultaneously Mounted")
ax.set_title("(b) Grand 3-Expert Split-Slot Composition ([Doc@P1 + Math@P2 + IF@P4])")
ax.grid(axis="y", alpha=0.38)
ax.legend(loc="upper left", ncol=1, fontsize=6.6, frameon=True, framealpha=0.95)

# --- (c) Architectural Law: Split-Slot Isolation vs Adjacent Concatenation & Permutation Invariance ---
ax = axes[1, 0]
me9 = d9["track2_composition_evals"]["math48"]
me9b = d9b["math48_evals"]

arch_spec = [
    ("MATH48\nBare",              me9["Bare_Student"]["acc"],             SLATE),
    ("MATH48\nSingle @P2",        me9["Single_Math_CSCR_P2"]["acc"],      AMBER),
    ("MATH48\nSplit\n[M@P2+IF@P4]", me9["Dual_CSCR_MathP2_IFP4"]["acc"],  GREEN),
    ("MATH48\nAdj\n[IF;M]@P2",    me9["Dual_CSCR_Adj_IF_Math_P2"]["acc"], RED),
    ("MATH48\nAdj\n[M;IF]@P2",    me9["Dual_CSCR_Adj_Math_IF_P2"]["acc"], "#F87171"),
    ("MATH48 PR\nAdj [IF;M]",     me9b["Dual_PostRL_Adj_IF_Math_P2"]["acc"], BLUE),
    ("MATH48 PR\nAdj [M;IF]",     me9b["Dual_PostRL_Adj_Math_IF_P2"]["acc"], "#60A5FA"),
    ("IFEval96\nAdj [IF;M]",      ie9["Dual_CSCR_Adj_IF_Math_P2"]["prompt_strict"], TEAL),
    ("IFEval96\nAdj [M;IF]",      ie9["Dual_CSCR_Adj_Math_IF_P2"]["prompt_strict"], "#2DD4BF"),
]
xar = np.arange(len(arch_spec))
for i, (lbl, v, col) in enumerate(arch_spec):
    ax.bar(i, v, width=0.66, color=col, alpha=0.89, edgecolor="#1E293B", lw=0.65)
    ax.text(i, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#0F172A")

ax.axvline(4.5, color="#94A3B8", linestyle=":", lw=1.2)
ax.axvline(6.5, color="#94A3B8", linestyle=":", lw=1.2)
ax.annotate("+16.7 pp Split-Slot\nSynergy vs Adjacent!", xy=(2.26, 70.0), xytext=(2.85, 81.5),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4, connectionstyle="arc3,rad=0.10"),
            fontsize=7.1, fontweight="bold", color="#065F46",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec=GREEN, lw=0.8))
ax.text(5.5, 63.0, "|d|=0.00 pp!\n(Sym L_null:\n2.5x less drop)", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.22", fc="#EFF6FF", ec=BLUE, lw=0.75))
ax.text(7.5, 80.5, "|d|=0.00 pp!\n(Exact Perm.\nInvariance)", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#115E59",
        bbox=dict(boxstyle="round,pad=0.22", fc="#F0FDFA", ec=TEAL, lw=0.75))
ax.set_xticks(xar)
ax.set_xticklabels([x[0] for x in arch_spec], fontsize=6.8)
ax.set_ylim(0, 96)
ax.set_ylabel("Strict / Exact Accuracy (%)")
ax.set_title("(c) Split-Slot Isolation Law (+16.7 pp Synergy) & Exact 0.00 pp Permutation Invariance")
ax.grid(axis="y", alpha=0.38)

# --- (d) Grand Pareto Frontier: Official IFBench64 Strict vs Closed-Think Precision ---
ax = axes[1, 1]
pts_p = [
    ("Bare Student (MultiTurn)",     b9["Bare_Student"]["closed_acc"],           b9["Bare_Student"]["prompt_strict"],           SLATE,  "s", 85,  (-65, 11),  True),
    ("Bare Student (P3-RAG)",        b9b["P3_Bare_Student"]["closed_acc"],       b9b["P3_Bare_Student"]["prompt_strict"],       SLATE,  "D", 70,  (8, -14),   False),
    ("0-RL Cold TF2 (L64)",          b9["P4_Cold_TF2"]["closed_acc"],            b9["P4_Cold_TF2"]["prompt_strict"],            RED,    "o", 85,  (9, -3),    False),
    ("0-RL Stage1 TF3 (L64)",        b9["P4_Stage1_TF3"]["closed_acc"],          b9["P4_Stage1_TF3"]["prompt_strict"],          AMBER,  "o", 90,  (-95, -12), True),
    ("0-RL Full AlgB (L64)",         b9["P4_Full_AlgB"]["closed_acc"],           b9["P4_Full_AlgB"]["prompt_strict"],           ORANGE, "o", 90,  (-108, -16),True),
    ("Source P0 KV_L16",             b9b["P0_KV_PostRL_IF_L16"]["closed_acc"],   b9b["P0_KV_PostRL_IF_L16"]["prompt_strict"],   "#475569","^", 105, (8, -6),   False),
    ("PR Direct L64 (P2)",           b9b["P2_PostRL_Direct_L64"]["closed_acc"],  b9b["P2_PostRL_Direct_L64"]["prompt_strict"],  BLUE,   "P", 100, (9, -3),    False),
    ("PR Direct L256 (P3-RAG)",      b9b["P3_PostRL_Direct_L256"]["closed_acc"], b9b["P3_PostRL_Direct_L256"]["prompt_strict"], SKY,    "D", 95,  (-108, 10), True),
    ("PR Direct L256 (P4)",          b9b["P4_PostRL_Direct_L256"]["closed_acc"], b9b["P4_PostRL_Direct_L256"]["prompt_strict"], NAVY,   "P", 115, (8, 5),     False),
    ("AlgA KVInv2 L256 (P2)",        b9b["P2_PostRL_KVInv2_L256"]["closed_acc"], b9b["P2_PostRL_KVInv2_L256"]["prompt_strict"], PURPLE, "v", 95,  (-108, 4),  True),
    ("Triple AlgA KVInv2 [256v]",    b9b["Triple_PostRL_KVInv2_Split"]["closed_acc"], b9b["Triple_PostRL_KVInv2_Split"]["prompt_strict"], "#10B981", "X", 115, (-55, -28), True),
    ("AlgA KVInv2 L64 (P4) *",       b9b["P4_PostRL_KVInv2_L64"]["closed_acc"],  b9b["P4_PostRL_KVInv2_L64"]["prompt_strict"],  TEAL,   "*", 240, (-128, -2), False),
]
for name, x_cl, y_st, col, mk, sz, (dx, dy), use_arr in pts_p:
    ax.scatter(x_cl, y_st, c=col, marker=mk, s=sz, edgecolors="#0F172A", linewidths=0.8, zorder=5)
    ap = dict(arrowstyle="-", color="#94A3B8", lw=0.7) if use_arr else None
    ax.annotate(name, (x_cl, y_st), xytext=(dx, dy), textcoords="offset points",
                arrowprops=ap,
                fontsize=6.7, fontweight="bold" if "*" in name or "L256 (P4)" in name else "normal",
                color="#0F766E" if "*" in name else "#0F172A")

# Draw Pareto frontier connecting dominant points
pareto_x = [b9["P4_Stage1_TF3"]["closed_acc"], b9b["P3_PostRL_Direct_L256"]["closed_acc"], b9b["P4_PostRL_Direct_L256"]["closed_acc"], b9b["P4_PostRL_KVInv2_L64"]["closed_acc"]]
pareto_y = [b9["P4_Stage1_TF3"]["prompt_strict"], b9b["P3_PostRL_Direct_L256"]["prompt_strict"], b9b["P4_PostRL_Direct_L256"]["prompt_strict"], b9b["P4_PostRL_KVInv2_L64"]["prompt_strict"]]
ax.plot(pareto_x, pareto_y, linestyle="--", color=TEAL, lw=1.3, alpha=0.8, zorder=3, label="Empirical Pareto Frontier")

ax.axvline(b9["Bare_Student"]["closed_acc"], color=SLATE, linestyle=":", lw=0.9, alpha=0.7)
ax.axhline(b9["Bare_Student"]["prompt_strict"], color=SLATE, linestyle=":", lw=0.9, alpha=0.7)
ax.set_xlim(43, 86.5)
ax.set_ylim(29, 57.5)
ax.set_xlabel("Closed-Think Precision (% Correct among Completed <think> Chains)")
ax.set_ylabel("Official IFBench64 Strict Accuracy (%)")
ax.set_title("(d) Unified Pareto Frontier on Official IFBench64: Alg A KV-Inv2 L64 Dominates")
ax.grid(True, alpha=0.35)
ax.legend(loc="lower right", fontsize=7.2, frameon=True, framealpha=0.95)

fig.suptitle("Suite 7 (exp9 + exp9b): Zero-Interference Multi-Expert Composition, Split-Slot Architecture & Grand Pareto Frontier",
             fontsize=12.0, fontweight="bold", color="#0F172A", y=0.975)

out12 = os.path.join(OUT_DIR, "tr_fig12_composition_and_interference.png")
plt.savefig(out12, dpi=200)
plt.close()
print("Saved:", out12)
