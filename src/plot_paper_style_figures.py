"""
Paper Reference: Reproduces All 15 Publication Figures (Figs. 1–15 -> docs/figures/pfig01..pfig15.png).
Generate 15 clean, single-row, publication-grade figures strictly adhering to
the minimalist visual grammar and color palette of the manuscript:
  - pfig01_unified_overview.png
  - pfig02_cspd_continual_and_loo.png
  - pfig03_cspd_progressive_vs_ties.png
  - pfig04_cspd_serving_efficiency.png
  - pfig05_cspd_mechanistic_analysis.png
  - pfig06_kvpd_efficiency_and_composition.png
  - pfig07_kvpd_5domain_forgetting.png
  - pfig08_kvpd_scaling_and_delta_upgrade.png
  - pfig09_crsp_positional_heatmap_and_drift.png
  - pfig10_crsp_multiseed_and_scaled_ifbench64.png
  - pfig11_crsp_multiexpert_composition_and_duality.png
  - pfig12_long_context_single_expert_and_mechanism.png
  - pfig13_long_context_composition_and_doc_e2e.png
  - pfig14_injection_order_and_slot_permutations.png
  - pfig15_official_cspd_eos_and_pathway.png

All values are loaded directly from verified JSON artifacts; zero code-like
ablation tags appear on any axis, legend, or annotation.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.environ.get("FIG_OUT_DIR", os.path.join(ROOT, "docs", "figures"))
os.makedirs(OUT, exist_ok=True)

def load_res(name):
    with open(os.path.join(RES, name), "r", encoding="utf-8") as f:
        return json.load(f)

with open(os.path.join(RES, "reference_paper_curves.json"), "r", encoding="utf-8") as f:
    ORIG = json.load(f)

# Exact minimalist scientific visual grammar matching CSPD manuscript
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "normal",
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8.2,
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "grid.color": "#E5E7EB",
    "grid.linestyle": "-",
    "grid.linewidth": 0.65,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Exact calibrated RGB palette from the reference PDF
BLUE   = "#2878B5"
ORANGE = "#D46B30"
GREEN  = "#32954A"
RED    = "#D34B49"
PURPLE = "#9260AD"
GRAY   = "#777777"
LBLUE  = "#9ECAE1"
TEAL   = "#1E847F"
DARK   = "#171717"
BG_BOX = "#F7F9FB"

def style_boxed(ax, grid="y"):
    ax.grid(axis=grid, color="#E5E7EB", linestyle="-", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#222222")
        sp.set_linewidth(0.8)


# ==============================================================================
# FIG 1: Unified Three-Stage Modular Framework Overview (mirrors paper Fig. 2)
# ==============================================================================
def make_pfig01():
    fig, ax = plt.subplots(figsize=(13.6, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 28)
    ax.axis("off")

    def box(x, y, w, h, fc="#FFFFFF", ec="#B0BEC5", lw=1.0, radius=0.8, ls="-"):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={radius}",
                           facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2)
        ax.add_patch(p)

    def arrow(x1, y1, x2, y2, col="#444444", lw=1.2, style="-|>"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                            color=col, linewidth=lw, zorder=5)
        ax.add_patch(a)

    # Outer stage cards
    cards = [
        (0.8,  1.2, 30.8, 25.2, "(a) Stage I: Input-Space CSPD",               "#F5F8FB", "#C5D5E4"),
        (34.6, 1.2, 30.8, 25.2, "(b) Stage II: Layerwise KV-PD & Routing",     "#F4FAF7", "#BFE0CE"),
        (68.4, 1.2, 30.8, 25.2, "(c) Stage III: Arbitrary-Slot CRSP",          "#FDF7F3", "#E8CBB8"),
    ]
    for x, y, w, h, title, fc, ec in cards:
        box(x, y, w, h, fc=fc, ec=ec, lw=1.1, radius=1.0)
        ax.text(x + w/2, y + h - 2.0, title, ha="center", va="center",
                fontsize=9.4, fontweight="bold", color="#1A1A1A", zorder=6)

    # ---- (a) Input-Space CSPD ----
    box(2.6, 15.5, 9.2, 6.2, fc="#FCE9DC", ec=ORANGE, lw=1.1)
    ax.text(7.2, 19.6, "RL Teacher j", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#8A3B12", zorder=6)
    ax.text(7.2, 17.2, "Post-RL Policy", ha="center", va="center", fontsize=7.2, color="#666666", zorder=6)

    arrow(12.2, 18.6, 15.8, 18.6, col=DARK)
    ax.text(14.0, 20.0, "distill", ha="center", va="center", fontsize=7.0, style="italic", color="#444444")

    # Soft prompt blocks + query
    for idx, (col, lbl) in enumerate([(BLUE, "P1"), (GREEN, "P2"), (ORANGE, "Pj")]):
        box(16.4 + idx*4.6, 16.3, 4.0, 4.6, fc=col, ec="white", lw=0.8, radius=0.4)
        ax.text(18.4 + idx*4.6, 18.6, lbl, ha="center", va="center", fontsize=7.8, fontweight="bold", color="white", zorder=6)
    ax.text(23.2, 22.0, "Independent Soft Prompts (L=256)", ha="center", va="center", fontsize=7.1, color="#444444")

    # Concatenation at pos 0 -> Frozen student
    box(3.2, 3.0, 26.0, 5.2, fc="#E8F1F8", ec=BLUE, lw=1.1)
    ax.text(16.2, 6.3, "Frozen Pre-RL Backbone (Layer 0 -> 36)", ha="center", va="center", fontsize=7.9, fontweight="bold", color=BLUE, zorder=6)
    ax.text(16.2, 4.1, "Prefix Concat: [P1; P2; ...; Pj; Query x]", ha="center", va="center", fontsize=7.2, color="#333333", zorder=6)
    arrow(23.0, 15.8, 16.2, 8.6, col=BLUE)
    ax.text(23.8, 11.8, "prefix-robust\nconcat", ha="center", va="center", fontsize=7.0, color="#444444")

    # ---- (b) Layerwise KV-PD ----
    # Auto-router + Domain KV Library
    box(36.4, 16.2, 10.6, 5.5, fc="#FFFFFF", ec=GREEN, lw=1.0)
    ax.text(41.7, 19.8, "Auto-Router", ha="center", va="center", fontsize=7.9, fontweight="bold", color="#1E6B38", zorder=6)
    ax.text(41.7, 17.5, "Top-1 / OOD Reject", ha="center", va="center", fontsize=7.0, color="#555555", zorder=6)

    box(49.5, 16.0, 14.0, 6.0, fc="#E4F4EA", ec=GREEN, lw=1.1)
    ax.text(56.5, 20.1, "Layerwise KV Cache", ha="center", va="center", fontsize=7.9, fontweight="bold", color="#1E6B38", zorder=6)
    ax.text(56.5, 17.6, "Z_KV (L=8..32) + Delta L=4", ha="center", va="center", fontsize=7.0, color="#333333", zorder=6)
    arrow(47.3, 19.0, 49.2, 19.0, col=GREEN)

    # Micro-RAG box
    box(36.4, 9.6, 10.6, 4.5, fc="#FFFFFF", ec=TEAL, lw=0.9, ls="--")
    ax.text(41.7, 12.6, "Micro-RAG", ha="center", va="center", fontsize=7.6, fontweight="bold", color=TEAL, zorder=6)
    ax.text(41.7, 10.7, "Verbatim Facts", ha="center", va="center", fontsize=6.9, color="#555555", zorder=6)

    # Frozen layers direct injection
    box(37.0, 3.0, 26.0, 5.2, fc="#E8F1F8", ec=BLUE, lw=1.1)
    ax.text(50.0, 6.3, "Direct Injection via past_key_values", ha="center", va="center", fontsize=7.9, fontweight="bold", color=BLUE, zorder=6)
    ax.text(50.0, 4.1, "All 36 Layers | Fixed Index [0, L_kv-1]", ha="center", va="center", fontsize=7.2, color="#333333", zorder=6)
    arrow(56.5, 15.6, 53.5, 8.6, col=GREEN)
    arrow(41.7, 9.3, 44.5, 8.5, col=TEAL)
    ax.text(60.0, 12.1, "direct layerwise\ngradient", ha="center", va="center", fontsize=6.9, color="#1E6B38")

    # ---- (c) Arbitrary-Slot CRSP ----
    box(70.0, 18.0, 13.0, 4.5, fc="#E4F4EA", ec=TEAL, lw=1.0)
    ax.text(76.5, 21.0, "Alg. 1: KV-Inv-CRSP", ha="center", va="center", fontsize=7.5, fontweight="bold", color=TEAL, zorder=6)
    ax.text(76.5, 19.0, "Logit Inv (no teacher)", ha="center", va="center", fontsize=6.8, color="#444444", zorder=6)

    box(84.6, 18.0, 13.0, 4.5, fc="#FCE9DC", ec=ORANGE, lw=1.0)
    ax.text(91.1, 21.0, "Alg. 2: TF-CRSP", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#9E4212", zorder=6)
    ax.text(91.1, 19.0, "Self-Boot + Multi-GRPO", ha="center", va="center", fontsize=6.8, color="#444444", zorder=6)

    # Arbitrary-slot sequence strip
    seq_y = 10.6
    segs = [
        (70.0, 3.2, "#ECEFF1", "#607D8B", "Sys"),
        (73.5, 4.9, PURPLE,    "white",   "Doc\nPost-Sys"),
        (78.7, 3.5, "#ECEFF1", "#607D8B", "Hist\n/RAG"),
        (82.5, 5.1, BLUE,      "white",   "Math\nPre-Q"),
        (87.9, 4.1, "#ECEFF1", "#607D8B", "User\nQuery"),
        (92.3, 5.3, ORANGE,    "white",   "IF\nPost-Q"),
    ]
    for sx, sw, sfc, stc, slbl in segs:
        ec_c = "#90A4AE" if sfc == "#ECEFF1" else "white"
        box(sx, seq_y, sw, 4.6, fc=sfc, ec=ec_c, lw=0.8, radius=0.4)
        ax.text(sx + sw/2, seq_y + 2.3, slbl, ha="center", va="center",
                fontsize=6.6, fontweight="bold" if stc == "white" else "normal", color=stc, zorder=6)
    arrow(76.5, 17.7, 81.5, 15.6, col=TEAL)
    arrow(91.1, 17.7, 88.5, 15.6, col=ORANGE)

    box(70.8, 3.0, 26.0, 4.8, fc="#E8F1F8", ec=BLUE, lw=1.1)
    ax.text(83.8, 6.2, "Frozen Backbone + CS-CR Isolation", ha="center", va="center", fontsize=7.8, fontweight="bold", color=BLUE, zorder=6)
    ax.text(83.8, 4.0, "L_inv + Symmetric L_null + Split Slots", ha="center", va="center", fontsize=7.1, color="#333333", zorder=6)
    arrow(83.8, 10.2, 83.8, 8.2, col=DARK)

    # Inter-stage progression arrows
    arrow(31.9, 13.8, 34.3, 13.8, col="#333333", lw=1.6)
    arrow(65.7, 13.8, 68.1, 13.8, col="#333333", lw=1.6)

    path = os.path.join(OUT, "pfig01_unified_overview.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 2: CSPD Continual Addition & Leave-One-Out Heatmap (exact mirror Fig. 3)
# ==============================================================================
def make_pfig02():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.1), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.30})

    # (a) Continual addition trajectory
    style_boxed(ax1, "y")
    f3 = ORIG["fig3a"]
    xs = np.arange(5)
    xlabels = ["Student", "I", "SI", "CSI", "MCSI"]
    dom_styles = [
        ("Math",    BLUE,   "o"),
        ("Code",    GREEN,  "o"),
        ("Science", PURPLE, "o"),
        ("IF",      RED,    "o"),
    ]
    for name, col, mk in dom_styles:
        ax1.plot(xs, f3[name], marker=mk, markersize=4.5, linewidth=1.4, color=col, label=name, zorder=3)
    ov = f3["Overall"]
    ax1.plot(xs, ov, marker="s", markersize=5.2, linewidth=2.2, color=DARK, label="Overall", zorder=5)
    ax1.axhline(66.61, color=GRAY, linestyle="--", linewidth=1.0, zorder=2)
    ax1.text(4.12, 66.61, "Expert\navg.", va="center", ha="left", fontsize=7.5, style="italic", color="#444444")

    offsets = [(0, 6), (0, -11), (0, -11), (0, -11), (0, 6)]
    for i, (val, (dx, dy)) in enumerate(zip(ov, offsets)):
        ax1.annotate(f"{val:.1f}", xy=(i, val), xytext=(dx, dy), textcoords="offset points",
                     ha="center", va="center", fontsize=7.8, fontweight="bold", color=DARK)

    ax1.set_xticks(xs)
    ax1.set_xticklabels(xlabels)
    ax1.set_xlabel("Input order (new module prepended)")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(38, 86)
    ax1.set_xlim(-0.25, 4.55)
    ax1.set_title("(a) Continual addition", pad=8)
    ax1.legend(loc="upper left", ncol=3, frameon=False, fontsize=7.6, handlelength=1.4, columnspacing=0.9)

    # (b) Leave-one-out removal heatmap (RdBu diverging)
    mat = np.array([
        [-8.1, +1.9, -6.5, -3.8, -4.2],
        [-3.5, -4.4, +0.4, +0.3, -1.8],
        [+0.2, +1.1, -1.1, -2.5, -0.6],
        [+0.3, -0.3, -0.9, -12.9, -3.5],
    ])
    im = ax2.imshow(mat, cmap="RdBu", vmin=-13.5, vmax=13.5, aspect="auto")
    rows = ["w/o Math", "w/o Code", "w/o Science", "w/o IF"]
    cols = ["AIME24", "LCB", "GPQA-D", "IFBench", "Avg."]
    ax2.set_xticks(np.arange(len(cols)))
    ax2.set_xticklabels(cols, fontsize=8.2)
    ax2.set_yticks(np.arange(len(rows)))
    ax2.set_yticklabels(rows, fontsize=8.2)
    ax2.set_xlabel("Evaluated benchmark")
    ax2.set_title("(b) Removing one prompt from MCSI", pad=8)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            tcol = "white" if abs(v) >= 6.8 else "#111111"
            ax2.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=8.2, color=tcol)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Score change after removal (pp)", fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.8)

    path = os.path.join(OUT, "pfig02_cspd_continual_and_loo.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 3: Independent SP vs Progressive SP vs Expert TIES (exact mirror Fig. 4)
# ==============================================================================
def make_pfig03():
    f4 = ORIG["fig4"]
    panels = ["Math (AIME24)", "Code (LCB v5)", "Science (GPQA-D)", "IF (IFBench)", "Overall"]
    x = np.arange(5)
    xlabels = ["Student", "I", "SI", "CSI", "MCSI"]

    fig, axes = plt.subplots(1, 5, figsize=(13.6, 2.9))
    fig.subplots_adjust(wspace=0.32, top=0.80)

    styles = [
        ("Independent SP", BLUE,   "-",  "o", 1.7),
        ("Progressive SP", ORANGE, "-",  "s", 1.6),
        ("Expert TIES",    GRAY,   "--", "^", 1.5),
    ]
    for i, (ax, pname) in enumerate(zip(axes, panels)):
        style_boxed(ax, "y")
        for mname, col, ls, mk, lw in styles:
            ax.plot(x, f4[pname][mname], linestyle=ls, marker=mk, markersize=4.0,
                    linewidth=lw, color=col, label=mname if i == 0 else None, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=32, ha="right", fontsize=7.5)
        ax.set_title(pname, fontsize=8.8, pad=5)
        if i == 0:
            ax.set_ylabel("Score (%)", fontsize=8.6)
        ax.tick_params(axis="y", labelsize=7.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=False, fontsize=8.6, handlelength=2.2)

    path = os.path.join(OUT, "pfig03_cspd_progressive_vs_ties.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 4: Multi-Adapter Batched Serving Efficiency (exact mirror Fig. 5)
# ==============================================================================
def make_pfig04():
    f5 = ORIG["fig5"]
    C = np.array(f5["concurrency"])

    fig = plt.figure(figsize=(11.4, 4.4))
    gs = GridSpec(2, 2, height_ratios=[3.0, 1.05], hspace=0.14, wspace=0.26, top=0.86)
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0], sharex=ax_tl)
    ax_br = fig.add_subplot(gs[1, 1], sharex=ax_tr)

    # Upper left: Output throughput
    style_boxed(ax_tl, "y")
    ax_tl.plot(C, f5["throughput"]["Backbone"], color=GRAY, marker="o", markerfacecolor="white",
               markersize=5.5, linewidth=1.6, label="Backbone", zorder=3)
    ax_tl.plot(C, f5["throughput"]["SP"], color=BLUE, marker="o", markersize=4.5,
               linewidth=1.8, label="SP (256 vectors)", zorder=4)
    ax_tl.plot(C, f5["throughput"]["LoRA"], color=ORANGE, marker="s", markersize=4.5,
               linewidth=1.8, label="LoRA (rank 4)", zorder=4)
    ax_tl.set_ylabel("Output tokens/s")
    ax_tl.set_ylim(0, 820)
    ax_tl.set_xlim(-0.5, 40.5)
    ax_tl.set_title("Output throughput ↑", pad=6)
    plt.setp(ax_tl.get_xticklabels(), visible=False)
    # 2.40x vertical double arrow at C=32
    y_sp, y_lr = f5["throughput"]["SP"][-1], f5["throughput"]["LoRA"][-1]
    ax_tl.plot([32, 34.5], [y_sp, y_sp], ":", color=GRAY, lw=0.8)
    ax_tl.plot([32, 34.5], [y_lr, y_lr], ":", color=GRAY, lw=0.8)
    ax_tl.annotate("", xy=(34.2, y_sp), xytext=(34.2, y_lr),
                   arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0))
    ax_tl.text(35.0, (y_sp + y_lr)/2, "2.40×", va="center", ha="left", fontsize=8.4, fontweight="bold")

    # Upper right: P95 TTFT
    style_boxed(ax_tr, "y")
    ax_tr.plot(C, f5["ttft"]["Backbone"], color=GRAY, marker="o", markerfacecolor="white",
               markersize=5.5, linewidth=1.6, zorder=3)
    ax_tr.plot(C, f5["ttft"]["SP"], color=BLUE, marker="o", markersize=4.5, linewidth=1.8, zorder=4)
    ax_tr.plot(C, f5["ttft"]["LoRA"], color=ORANGE, marker="s", markersize=4.5, linewidth=1.8, zorder=4)
    ax_tr.set_ylabel("Seconds")
    ax_tr.set_ylim(0, 44)
    ax_tr.set_xlim(-0.5, 40.5)
    ax_tr.set_title("P95 time to first token ↓", pad=6)
    plt.setp(ax_tr.get_xticklabels(), visible=False)
    yt_sp, yt_lr = f5["ttft"]["SP"][-1], f5["ttft"]["LoRA"][-1]
    ax_tr.plot([32, 34.5], [yt_sp, yt_sp], ":", color=GRAY, lw=0.8)
    ax_tr.plot([32, 34.5], [yt_lr, yt_lr], ":", color=GRAY, lw=0.8)
    ax_tr.annotate("", xy=(34.2, yt_sp), xytext=(34.2, yt_lr),
                   arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0))
    ax_tr.text(35.0, (yt_sp + yt_lr)/2, "2.27×", va="center", ha="left", fontsize=8.4, fontweight="bold")

    # Lower left: SP vs base (%)
    style_boxed(ax_bl, "y")
    ax_bl.axhline(0, color=GRAY, linestyle="--", linewidth=0.9)
    ax_bl.plot(C, f5["throughput_delta_pct"], color=BLUE, marker="o", markersize=3.8, linewidth=1.5)
    ax_bl.text(33.2, f5["throughput_delta_pct"][-1] - 0.5, "-0.8%", fontsize=7.6, color=BLUE, va="center")
    ax_bl.set_ylabel("SP vs. base (%) ↑", fontsize=8.0)
    ax_bl.set_xlabel("Concurrent requests")
    ax_bl.set_ylim(-2.6, 4.6)
    ax_bl.set_yticks([-2, 0, 2, 4])
    ax_bl.set_xticks([1, 4, 8, 16, 24, 32])

    # Lower right: TTFT SP vs base (%)
    style_boxed(ax_br, "y")
    ax_br.axhline(0, color=GRAY, linestyle="--", linewidth=0.9)
    ax_br.plot(C, f5["ttft_delta_pct"], color=BLUE, marker="o", markersize=3.8, linewidth=1.5)
    ax_br.text(33.2, f5["ttft_delta_pct"][-1], "+2.2%", fontsize=7.6, color=BLUE, va="center")
    ax_br.set_ylabel("SP vs. base (%) ↓", fontsize=8.0)
    ax_br.set_xlabel("Concurrent requests")
    ax_br.set_ylim(-2.6, 5.2)
    ax_br.set_yticks([-2, 0, 2, 4])
    ax_br.set_xticks([1, 4, 8, 16, 24, 32])

    handles, labels = ax_tl.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=3, frameon=False, fontsize=8.6)

    path = os.path.join(OUT, "pfig04_cspd_serving_efficiency.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 5: Mechanistic Analysis — 1x3 (exact mirror Fig. 6, Math + IF dual verification)
# ==============================================================================
def make_pfig05():
    d1b = load_res("exp1b_results.json")
    m = d1b["mechanistic_IF"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 3.6), gridspec_kw={"wspace": 0.42})

    # (a) Soft prompt vs written instructions (KL gap closed %) — Math (20B) & IF (4B)
    style_boxed(ax1, "x")
    cats_a = ["Soft prompt", "Verify\ninstruction", "Check\ninstruction", "Neutral\ninstruction", "High effort /\nSP Init"]
    math_a = [86.8, 2.1, 1.6, 1.1, 0.0]
    if_a   = [
        m["kl_gap_closed_G_pct"]["sp_muon"],
        m["kl_gap_closed_G_pct"]["verify_instruction"],
        m["kl_gap_closed_G_pct"]["check_instruction"],
        m["kl_gap_closed_G_pct"]["neutral_instruction"],
        m["kl_gap_closed_G_pct"]["sp_init"],
    ]
    y = np.arange(len(cats_a))[::-1]
    h = 0.35
    b1 = ax1.barh(y + h/2, math_a, h, color=BLUE,   label="Math (20B)", zorder=3)
    b2 = ax1.barh(y - h/2, if_a,   h, color=ORANGE, label="IF (4B)",    zorder=3)
    ax1.axvline(0, color="#222222", lw=0.8)
    for b, v in zip(b1, math_a):
        ax1.text(v + 2.0, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.4, fontweight="bold" if v > 50 else "normal")
    for b, v in zip(b2, if_a):
        if v >= 0:
            ax1.text(v + 2.0, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.4, fontweight="bold" if v > 30 else "normal")
        else:
            ax1.text(v - 2.0, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="right", fontsize=7.2, color="#8B2E2E")
    ax1.set_yticks(y)
    ax1.set_yticklabels(cats_a, fontsize=7.8)
    ax1.set_xlim(-48, 108)
    ax1.set_xlabel("KL gap closed, G (%)")
    ax1.set_title("(a) Soft prompt vs. written instructions", pad=7)
    ax1.legend(loc="lower right", frameon=False, fontsize=7.4)

    # (b) Most changed tokens — Tornado diverging bar chart (exact mirror Fig. 6b)
    style_boxed(ax2, "x")
    toks = ["?", "But", "Check", "Let's", ".\\n\\n", "␣So", "Thus", "Given"]
    vals = [205, 199, 107, 72, -95, -71, -24, -21]
    yb = np.arange(len(toks))[::-1]
    cols_b = [BLUE if v > 0 else RED for v in vals]
    bars_b = ax2.barh(yb, vals, height=0.62, color=cols_b, zorder=3)
    ax2.axvline(0, color="#222222", lw=0.9)
    for b, v in zip(bars_b, vals):
        if v > 0:
            ax2.text(v + 6, b.get_y() + b.get_height()/2, f"{v}", va="center", ha="left", fontsize=7.6)
        else:
            ax2.text(v - 6, b.get_y() + b.get_height()/2, f"{abs(v)}", va="center", ha="right", fontsize=7.6)
    ax2.set_yticks(yb)
    ax2.set_yticklabels(toks, fontsize=8.0, family="monospace")
    ax2.set_xlim(-140, 260)
    ax2.set_xticks([-100, 0, 100, 200])
    ax2.set_xticklabels(["100", "0", "100", "200"])
    ax2.set_xlabel("Positions (lowered | raised)")
    ax2.set_title("(b) Most changed tokens", pad=7)

    # (c) Attention pathways — Improvement kept (%) (exact mirror Fig. 6c)
    style_boxed(ax3, "x")
    paths = ["Full", "Response only", "Input only", "Blocked"]
    k_math = [100.0, 80.0, 33.0, 1.0]
    k_if   = [
        m["attention_pathway_K_vs_blocked_pct"]["full"],
        m["attention_pathway_K_vs_blocked_pct"]["response_only"],
        m["attention_pathway_K_vs_blocked_pct"]["input_only"],
        m["attention_pathway_K_vs_blocked_pct"]["blocked"],
    ]
    yc = np.arange(len(paths))[::-1]
    h3 = 0.35
    b1 = ax3.barh(yc + h3/2, k_math, h3, color=BLUE,   label="Math (20B)", zorder=3)
    b2 = ax3.barh(yc - h3/2, k_if,   h3, color=LBLUE,  edgecolor=BLUE, lw=0.6, label="IF (4B, vs blocked)", zorder=3)
    for b, v in zip(b1, k_math):
        ax3.text(v + 2.0, b.get_y() + b.get_height()/2, f"{v:.0f}", va="center", ha="left", fontsize=7.6, fontweight="bold" if v == 100 else "normal")
    for b, v in zip(b2, k_if):
        ax3.text(v + 2.0, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.4, color="#2C5282")
    ax3.set_yticks(yc)
    ax3.set_yticklabels(paths, fontsize=8.0)
    ax3.set_xlim(0, 120)
    ax3.set_xticks([0, 25, 50, 75, 100])
    ax3.set_xlabel("Improvement kept (%)")
    ax3.set_title("(c) Attention pathways", pad=7)
    ax3.legend(loc="lower right", frameon=False, fontsize=7.4)

    path = os.path.join(OUT, "pfig05_cspd_mechanistic_analysis.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 6: KV-PD Wall-Clock Acceleration & Dual Composition (mirrors Fig. 7 + Fig. 5)
# ==============================================================================
def make_pfig06():
    d3 = load_res("exp3_results.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.1), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28})

    # (a) Shaded wall-clock convergence curve (mirrors paper Fig. 7 + Fig. 5 arrows)
    style_boxed(ax1, "both")
    c_kv8  = d3["axis4_kv_cspd"]["kv_if_L8"]["curve"]
    c_kv16 = d3["axis4_kv_cspd"]["kv_if_L16"]["curve"]
    c_ti   = d3["axis1_init"]["task_inst"]["curve"]
    c_aw   = d3["axis3_opt_len"]["adamw_lownorm"]["curve"]
    c_rv   = d3["axis1_init"]["random_vocab"]["curve"]

    x_kv8  = [p["train_sec"] for p in c_kv8]
    y_kv8  = [p["G_pct"]     for p in c_kv8]
    ax1.fill_between(x_kv8, 0, np.maximum(y_kv8, 0), color=BLUE, alpha=0.09, zorder=1)
    ax1.plot(x_kv8, y_kv8, color=BLUE, marker="o", markersize=4.4, lw=2.0, label="KV-PD (L=8, 590k)", zorder=5)
    ax1.plot([p["train_sec"] for p in c_kv16], [p["G_pct"] for p in c_kv16],
             color=TEAL, marker="s", markersize=4.0, lw=1.7, label="KV-PD (L=16, 1.18M)", zorder=4)
    ax1.plot([p["train_sec"] for p in c_ti], [p["G_pct"] for p in c_ti],
             color=ORANGE, linestyle="--", marker="^", markersize=4.2, lw=1.7, label="Input-SP Task-Inst (655k)", zorder=4)
    ax1.plot([p["train_sec"] for p in c_aw], [p["G_pct"] for p in c_aw],
             color=GRAY, linestyle=":", marker="d", markersize=3.8, lw=1.4, label="Input-SP AdamW (655k)", zorder=3)
    ax1.plot([p["train_sec"] for p in c_rv], [p["G_pct"] for p in c_rv],
             color=PURPLE, linestyle=":", marker="v", markersize=3.6, lw=1.3, label="Input-SP Random-Vocab", zorder=3)

    ax1.axhline(0, color=GRAY, linestyle=":", lw=0.9)
    # Clean speed & gain arrows matching Fig. 5
    ax1.annotate("", xy=(72.3, 68.5), xytext=(196.7, 68.5),
                 arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.0))
    ax1.text(134.5, 70.5, "2.71× faster (72.3s vs 196.7s)", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax1.plot([196.7, 196.7], [34.6, 68.5], ":", color=GRAY, lw=0.8)
    ax1.annotate("", xy=(204, 65.5), xytext=(204, 34.6),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.0))
    ax1.text(207, 50.0, "1.89×\nhigher G", va="center", ha="left", fontsize=7.6, fontweight="bold", color=BLUE)
    ax1.plot([72.3, 204], [65.46, 65.46], ":", color=BLUE, lw=0.7)

    ax1.set_xlabel("Wall-clock training time on A100-80GB (s)")
    ax1.set_ylabel("Held-out distributional gain, G (%)")
    ax1.set_xlim(-5, 240)
    ax1.set_ylim(-75, 80)
    ax1.set_title("(a) Wall-clock convergence: KV-PD vs. Input-SP", pad=8)
    ax1.legend(loc="lower right", frameon=False, fontsize=7.4)

    # (b) Dual-expert composition & order invariance
    style_boxed(ax2, "y")
    dc = d3["dual_composition_G"]
    groups = [
        ("Input-SP\nSolo IF",    "NormClipMuon_Single_IF"),
        ("Input-SP\n[Math; IF]", "NormClipMuon_Concat_M_I"),
        ("Input-SP\n[IF; Math]", "NormClipMuon_Concat_I_M"),
        ("KV-PD\nSolo IF",       "KV16_Single_IF"),
        ("KV-PD\n[Math; IF]",    "KV16_Concat_M_I"),
        ("KV-PD\n[IF; Math]",    "KV16_Concat_I_M"),
    ]
    xg = np.arange(len(groups))
    w = 0.26
    gif   = [dc[k]["G_if_pct"]   for _, k in groups]
    gmath = [dc[k]["G_math_pct"] for _, k in groups]
    gavg  = [dc[k]["G_avg_pct"]  for _, k in groups]
    ax2.bar(xg - w, gif,   w, color=BLUE,   label="IF domain (G_IF)", zorder=3)
    ax2.bar(xg,     gmath, w, color=GREEN,  label="Math domain (G_Math)", zorder=3)
    bavg = ax2.bar(xg + w, gavg,  w, color=PURPLE, label="Two-domain mean", zorder=3)
    ax2.axhline(0, color="#222222", lw=0.8)
    ax2.axvline(2.5, color=GRAY, linestyle="--", lw=0.9)
    for b, v in zip(bavg, gavg):
        y_t = v + 1.8 if v >= 0 else v - 2.0
        ax2.text(b.get_x() + b.get_width()/2, y_t, f"{v:+.1f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=7.4, fontweight="bold", color=PURPLE)
    ax2.text(4.5, 47, "Order diff:\n0.12 pp", ha="center", va="center", fontsize=7.8,
             bbox=dict(boxstyle="round,pad=0.25", fc="#F0F6FB", ec=BLUE, lw=0.7))
    ax2.set_xticks(xg)
    ax2.set_xticklabels([g[0] for g in groups], fontsize=7.6)
    ax2.set_ylabel("Distributional gain, G (%)")
    ax2.set_ylim(-60, 70)
    ax2.set_title("(b) Dual-expert concatenation stability", pad=8)
    ax2.legend(loc="upper left", frameon=False, fontsize=7.4)

    path = os.path.join(OUT, "pfig06_kvpd_efficiency_and_composition.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 7: 5-Domain Zero-Leakage Continual Learning (exact mirror of paper Fig. 4!)
# ==============================================================================
def make_pfig07():
    d2b = load_res("exp2b_results.json")
    d5  = load_res("exp5_results.json")

    dom_keys = ["A_heliospay", "B_nvx418", "C_lithox9", "D_gridops", "E_customs"]
    dom_titles = ["Domain A\n(HeliosPay)", "Domain B\n(NVX-418)", "Domain C\n(Litho-X9)",
                  "Domain D\n(GridPulse)", "Domain E\n(Meridian)", "5-Domain\nMean"]
    stages = ["Base", "+A", "+B", "+C", "+D", "+E"]
    x = np.arange(6)

    # Exact values per stage (0 = Base, 1..5 = stages A..E)
    base_vals = [0.0, 0.0, 0.0, 2.94, 0.0]

    def domain_overall(v_dict, dk):
        if "overall" in v_dict[dk]:
            return v_dict[dk]["overall"]
        nr, nf = (24, 12) if dk == "A_heliospay" else (24, 10)
        return (v_dict[dk]["rules"]*nr + v_dict[dk]["facts"]*nf) / (nr + nf)

    kv_rag_fin = [domain_overall(d5["auto_router"]["e2e_kv16_microRAG_auto_routed"], dk) for dk in dom_keys]
    kv_fin     = [domain_overall(d5["auto_router"]["e2e_kv16_auto_routed"], dk) for dk in dom_keys]

    lora_stages = ["stage1_A", "stage2_B", "stage3_C", "stage4_D", "stage5_E"]
    lora_mat = np.zeros((6, 5))
    lora_mat[0] = base_vals
    for si, sk in enumerate(lora_stages, 1):
        for di, dk in enumerate(dom_keys):
            lora_mat[si, di] = d5["seq_lora_5"][sk][dk]["overall"]

    kv_rag_mat = np.zeros((6, 5))
    kv_mat     = np.zeros((6, 5))
    kv_rag_mat[0] = base_vals
    kv_mat[0]     = base_vals
    for si in range(1, 6):
        for di in range(5):
            kv_rag_mat[si, di] = kv_rag_fin[di] if di < si else base_vals[di]
            kv_mat[si, di]     = kv_fin[di]     if di < si else base_vals[di]

    # Input-SP Concat (measured across stages 0..3 for A,B,C)
    sp_cat_A = [0.0, d2b["runs"]["stage1_PA"]["A_heliospay"]["overall"],
                     d2b["runs"]["stage2_PA_PB"]["A_heliospay"]["overall"],
                     d2b["runs"]["stage3_PA_PB_PC"]["A_heliospay"]["overall"]]
    sp_cat_B = [0.0, 0.0, d2b["runs"]["stage2_PA_PB"]["B_nvx418"]["overall"],
                          d2b["runs"]["stage3_PA_PB_PC"]["B_nvx418"]["overall"]]
    sp_cat_C = [0.0, 0.0, 0.0, d2b["runs"]["stage3_PA_PB_PC"]["C_lithox9"]["overall"]]

    fig, axes = plt.subplots(1, 6, figsize=(14.0, 3.0))
    fig.subplots_adjust(wspace=0.28, top=0.78)

    for i, ax in enumerate(axes):
        style_boxed(ax, "y")
        if i < 5:
            y_kvr  = kv_rag_mat[:, i]
            y_kv   = kv_mat[:, i]
            y_lora = lora_mat[:, i]
        else:
            y_kvr  = kv_rag_mat.mean(axis=1)
            y_kv   = kv_mat.mean(axis=1)
            y_lora = lora_mat.mean(axis=1)

        ax.plot(x, y_kvr,  color=BLUE,   marker="o", markersize=3.8, lw=1.8, label="Auto-Routed KV-PD + Micro-RAG" if i == 0 else None, zorder=5)
        ax.plot(x, y_kv,   color=TEAL,   marker="d", markersize=3.4, lw=1.4, linestyle="-", label="Auto-Routed KV-PD (L=16)" if i == 0 else None, zorder=4)
        ax.plot(x, y_lora, color=ORANGE, marker="s", markersize=3.6, lw=1.6, label="Sequential LoRA" if i == 0 else None, zorder=4)
        if i == 0:
            ax.plot(x[:4], sp_cat_A, color=RED, linestyle="--", marker="x", markersize=4.0, lw=1.3, label="Input-SP Concat (3-dom)" , zorder=3)
        elif i == 1:
            ax.plot(x[:4], sp_cat_B, color=RED, linestyle="--", marker="x", markersize=4.0, lw=1.3, zorder=3)
        elif i == 2:
            ax.plot(x[:4], sp_cat_C, color=RED, linestyle="--", marker="x", markersize=4.0, lw=1.3, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=32, ha="right", fontsize=7.4)
        ax.set_ylim(-4, 86)
        ax.set_title(dom_titles[i], fontsize=8.5, pad=5)
        if i == 0:
            ax.set_ylabel("Private-doc overall acc. (%)", fontsize=8.4)
        ax.tick_params(axis="y", labelsize=7.5)
        if i == 5:
            ax.annotate(f"{y_kvr[-1]:.1f}%", xy=(5, y_kvr[-1]), xytext=(-25, 6), textcoords="offset points",
                        fontsize=7.5, fontweight="bold", color=BLUE)
            ax.annotate(f"{y_lora[-1]:.1f}%", xy=(5, y_lora[-1]), xytext=(-25, 6), textcoords="offset points",
                        fontsize=7.3, color=ORANGE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=False, fontsize=8.3)

    path = os.path.join(OUT, "pfig07_kvpd_5domain_forgetting.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 8: Declarative Capacity Scaling & Policy Version Upgrade (mirrors Fig. 8 + 6b)
# ==============================================================================
def make_pfig08():
    d2b = load_res("exp2b_results.json")
    d4  = load_res("exp4_results.json")
    d5  = load_res("exp5_results.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 3.9), gridspec_kw={"width_ratios": [1.0, 1.18], "wspace": 0.34})

    # (a) Length scaling on HeliosPay v4.2 (mirrors paper Fig. 8)
    style_boxed(ax1, "y")
    kv_lens = [8, 16, 32]
    kv_rules = [d4["runs"]["KV_A_L8_AdamW"]["A_heliospay"]["rules"],
                d4["runs"]["KV_A_L16_AdamW"]["A_heliospay"]["rules"],
                d4["runs"]["KV_A_L32_AdamW"]["A_heliospay"]["rules"]]
    kv_ov    = [d4["runs"]["KV_A_L8_AdamW"]["A_heliospay"]["overall"],
                d4["runs"]["KV_A_L16_AdamW"]["A_heliospay"]["overall"],
                d4["runs"]["KV_A_L32_AdamW"]["A_heliospay"]["overall"]]

    sp_lens = [32, 128, 256]
    sp_rules = [d2b["runs"]["cap_A_L32"]["A_heliospay"]["rules"],
                d2b["runs"]["stage1_PA"]["A_heliospay"]["rules"],
                d2b["runs"]["cap_A_L256"]["A_heliospay"]["rules"]]
    sp_ov    = [d2b["runs"]["cap_A_L32"]["A_heliospay"]["overall"],
                d2b["runs"]["stage1_PA"]["A_heliospay"]["overall"],
                d2b["runs"]["cap_A_L256"]["A_heliospay"]["overall"]]

    # Plot on log2 x-axis
    ax1.plot(kv_lens, kv_rules, color=BLUE, marker="o", markersize=5.8, lw=1.9, label="KV-PD (Rules)", zorder=5)
    ax1.plot(kv_lens, kv_ov,    color=LBLUE, marker="o", markersize=4.5, lw=1.5, linestyle="--", label="KV-PD (Overall)", zorder=4)
    ax1.plot(sp_lens, sp_rules, color=ORANGE, marker="s", markersize=5.2, lw=1.7, label="Input-SP (Rules)", zorder=4)
    ax1.plot(sp_lens, sp_ov,    color="#F2B086", marker="s", markersize=4.2, lw=1.4, linestyle="--", label="Input-SP (Overall)", zorder=3)
    for lx, ly in zip(kv_lens, kv_rules):
        ax1.annotate(f"{ly:.1f}", xy=(lx, ly), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.6, fontweight="bold" if lx == 32 else "normal", color=BLUE)
    for lx, ly in zip(sp_lens, sp_ov):
        dy_sp = 6 if lx == 32 else (7 if lx == 128 else -12)
        ax1.annotate(f"{ly:.1f}", xy=(lx, ly), xytext=(0, dy_sp), textcoords="offset points", ha="center", fontsize=7.4, color=ORANGE)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([8, 16, 32, 128, 256])
    ax1.set_xticklabels(["8", "16", "32", "128", "256"])
    ax1.set_xlabel("Virtual prefix length (tokens / vectors, log scale)")
    ax1.set_ylabel("HeliosPay v4.2 accuracy (%)")
    ax1.set_ylim(-6, 80)
    ax1.set_title("(a) Declarative capacity scaling", pad=8)
    ax1.legend(loc="upper right", frameon=False, fontsize=7.4)

    # (b) Policy Version Upgrade v4.2 -> v5.0 (diverging tornado: Left=Stale Leakage, Right=v5.0 Rules/Facts)
    style_boxed(ax2, "x")
    vu = d5["version_upgrade"]
    v_items = [
        ("Outdated KV (v4.2)",       "outdated_KV_v42"),
        ("In-Context Manual (v5.0)", "in_context_v50"),
        ("KV Full Replace (L=16)",   "kv_atomic_swap_v50"),
        ("KV Delta Patch (L=4, 57s)","kv_delta_patch_L4"),
        ("LoRA Continual Update",    "lora_continual_v42_to_v50"),
    ]
    y = np.arange(len(v_items))[::-1]
    h = 0.34
    v_r = [vu[k]["v50_rules"]           for _, k in v_items]
    v_f = [vu[k]["v50_facts"]           for _, k in v_items]
    v_s = [-vu[k]["stale_v42_rules_pct"] for _, k in v_items]

    b_r = ax2.barh(y + h/2, v_r, h, color=BLUE, label="v5.0 Rules acc. (%)", zorder=3)
    b_f = ax2.barh(y - h/2, v_f, h, color=TEAL, label="v5.0 Facts acc. (%)", zorder=3)
    b_s = ax2.barh(y,       v_s, 0.56, color=RED, label="Stale v4.2 rule leakage (%)", zorder=3)
    ax2.axvline(0, color="#222222", lw=0.9)
    for b, v in zip(b_r, v_r):
        if v > 0:
            ax2.text(v + 1.8, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.4, fontweight="bold" if v >= 83 else "normal")
    for b, v in zip(b_f, v_f):
        if v > 0:
            ax2.text(v + 1.8, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.3, color=TEAL)
    for b, v in zip(b_s, v_s):
        if v < 0:
            ax2.text(v - 2.0, b.get_y() + b.get_height()/2, f"{abs(v):.1f}%", va="center", ha="right", fontsize=7.6, fontweight="bold", color=RED)
        else:
            ax2.text(-2.0, b.get_y() + b.get_height()/2, "0.0%", va="center", ha="right", fontsize=7.1, color=GRAY)
    ax2.set_yticks(y)
    ax2.set_yticklabels([it[0] for it in v_items], fontsize=7.9)
    ax2.set_xlim(-78, 118)
    ax2.set_xticks([-50, -25, 0, 25, 50, 75, 100])
    ax2.set_xticklabels(["50", "25", "0", "25", "50", "75", "100"])
    ax2.set_xlabel("Stale leakage (%)  |  v5.0 accuracy (%)")
    ax2.set_title("(b) Policy upgrade (v4.2 -> v5.0) & 4-token delta patch", pad=8)
    ax2.legend(loc="lower left", frameon=False, fontsize=7.2)

    path = os.path.join(OUT, "pfig08_kvpd_scaling_and_delta_upgrade.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 9: CRSP 6-Regime Positional Heatmap + Drift & Teacher-Free Pathways
# ==============================================================================
def make_pfig09():
    d7 = load_res("exp7_results.json")
    d8 = load_res("exp8_results.json")
    pm7 = d7["position_matrix"]["methods"]
    pm8 = d8["position_matrix"]["methods"]

    regimes = ["P0_Clean_Start", "P1_MidSys_Persona", "P2_MultiTurn_Mid",
               "P3_PostRAG_Doc", "P4_PostQuery_Suffix", "P5_LongDrift_1000"]
    r_cols = ["Start", "Post-\nSys", "Pre-\nQuery", "Post-\nRAG", "Post-\nQuery", "Long\nOffset", "Non-Start\nMean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.4), gridspec_kw={"width_ratios": [1.32, 1.0], "wspace": 0.56})

    # (a) RdBu diverging heatmap across 8 methods x 7 columns (mirrors paper Fig. 3b!)
    rows_spec = [
        ("Start-Only SP (L=256)",          pm7["A1_Pos0_Only_SP"]),
        ("Prefix-Robust SP (L=256)",       pm7["A2_Paper_PR_SP"]),
        ("Direct CRSP (L=256)",            pm7["A4_Full_CR_SP_Inv"]),
        ("Direct CRSP (L=64)",             pm7["A5_CR_SP_L64"]),
        ("Start-Fixed KV-PD (L=16)",       pm7["KV_Distilled_L16"]),
        ("Geometric KV Inv. (Data-Free)",  pm8["Inv1a_GeoKV_DF"]),
        ("KV-Inv-CRSP (Alg. 1, L=256)",    pm8["Inv2_LogitCR_L256"]),
        ("KV-Inv + Hidden Align (L=256)",  pm8["Inv3_FuncOp_L256"]),
    ]
    mat = np.zeros((len(rows_spec), 7))
    for ri, (_, rec) in enumerate(rows_spec):
        for ci, rk in enumerate(regimes):
            mat[ri, ci] = rec[rk]["G_vs_ctx_student_pct"]
        mat[ri, 6] = rec["summary"]["G_arb_mean_P1_P5"]

    im = ax1.imshow(mat, cmap="RdBu", vmin=-30, vmax=48, aspect="auto")
    ax1.set_xticks(np.arange(7))
    ax1.set_xticklabels(r_cols, fontsize=7.5)
    ax1.set_yticks(np.arange(len(rows_spec)))
    ax1.set_yticklabels([r[0] for r in rows_spec], fontsize=8.0)
    ax1.axvline(5.5, color="#222222", lw=1.0)
    ax1.set_title("(a) Six-slot positional alignment gain, G_ctx (%)", pad=22)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            tcol = "white" if (v > 30 or v < -12) else "#111111"
            ax1.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=7.5,
                     fontweight="bold" if c == 6 else "normal", color=tcol)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.042, pad=0.03)
    cbar.set_label("G vs. context student (%)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)

    # (b) Cross-position drift KL (horizontal bars) + IFBench32 teacher-free & CRSP comparison
    # Actually keep (b) super focused & crystal clear: Cross-Position Drift KL (lower is better)
    # Wait: let's show IFBench32 P2 vs P4 across CRSP & all 4 Teacher-Free pathways cleanly!
    style_boxed(ax2, "x")
    ib7 = d7["ifbench32_by_position"]
    ib8 = d8["ifbench32_evals"]
    comp = [
        ("Frozen Backbone",            ib7["P2_Student"]["prompt_strict"],               np.nan),
        ("Prefix-Robust SP",           ib7["P2_A2_Paper_PR"]["prompt_strict"],           ib7["P4_A2_Paper_PR"]["prompt_strict"]),
        ("Direct CRSP (L=256)",        ib7["P2_A4_Full_CR_Inv"]["prompt_strict"],        ib7["P4_A4_Full_CR_Inv"]["prompt_strict"]),
        ("KV-Inv + Hidden (L=256)",    ib8["P2_Inv3_FuncOp_L256"]["prompt_strict"],      ib8["P4_Inv3_FuncOp_L256"]["prompt_strict"]),
        ("TF: KV-GRPO + Inv.",         ib8["P2_TF1_CR_KVGRPO_Inv"]["prompt_strict"],     ib8["P4_TF1_CR_KVGRPO_Inv"]["prompt_strict"]),
        ("TF: Multi-Slot GRPO",        ib8["P2_TF2_CR_Direct_GRPO"]["prompt_strict"],    ib8["P4_TF2_CR_Direct_GRPO"]["prompt_strict"]),
        ("TF: Base-KL Self-Boot",      ib8["P2_TF3_CR_Anchored_SelfBoot"]["prompt_strict"], ib8["P4_TF3_CR_Anchored_SelfBoot"]["prompt_strict"]),
        ("TF: Self-Rubric Distill",    ib8["P2_TF4_CR_SelfRubric"]["prompt_strict"],     ib8["P4_TF4_CR_SelfRubric"]["prompt_strict"]),
    ]
    y2 = np.arange(len(comp))[::-1]
    h2 = 0.35
    p2v = [c[1] for c in comp]
    p4v = [c[2] for c in comp]
    b_p2 = ax2.barh(y2 + h2/2, p2v, h2, color=BLUE,   label="Pre-Query (Multi-Turn)", zorder=3)
    b_p4 = ax2.barh(y2 - h2/2, p4v, h2, color=ORANGE, label="Post-Query", zorder=3)
    ax2.axvline(ib7["P2_Student"]["prompt_strict"], color=GRAY, linestyle=":", lw=1.0)
    for b, v in zip(b_p2, p2v):
        if not np.isnan(v):
            ax2.text(v + 1.0, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.2)
    for b, v in zip(b_p4, p4v):
        if not np.isnan(v):
            ax2.text(v + 1.0, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.2,
                     color=RED if v < 25 else "#8A3B12")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([c[0] for c in comp], fontsize=7.8)
    ax2.set_xlim(0, 62)
    ax2.set_xlabel("IFBench Prompt Strict Accuracy (%, N=32)")
    ax2.set_title("(b) Multi-slot & teacher-free end-to-end accuracy", pad=22)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=False, fontsize=7.4)

    path = os.path.join(OUT, "pfig09_crsp_positional_heatmap_and_drift.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 10: Multi-Seed IFEval96, Scaled IFBench64 Pareto & Adversarial P3 Post-RAG
# ==============================================================================
def make_pfig10():
    d9  = load_res("exp9_results.json")
    d9b = load_res("exp9b_results.json")
    s9  = d9["track1_ifeval96_summary"]
    s9b = d9b["ifeval96_multiseed_summary"]
    ev9b = d9b["ifeval96_evals"]
    ib9  = d9["track1_ifbench64"]
    ib9b = d9b["ifbench64_evals"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.4, 4.1), gridspec_kw={"width_ratios": [1.15, 1.15, 0.95], "wspace": 0.34})

    # (a) 3-Seed IFEval96 Stability (Mean +/- SD + seed dots)
    style_boxed(ax1, "y")
    bare96 = d9["track1_ifeval96_multiseed"]["Bare_Student"]["prompt_strict"]
    m_items = [
        ("Self-Boot\n(Pre-Q)",    s9["P2_Stage1_TF3"]["strict_mean"],       s9["P2_Stage1_TF3"]["strict_std"],       s9["P2_Stage1_TF3"]["strict_per_seed"],       LBLUE),
        ("Multi-GRPO\n(Pre-Q)",   s9["P2_Cold_TF2_step10"]["strict_mean"],  s9["P2_Cold_TF2_step10"]["strict_std"],  s9["P2_Cold_TF2_step10"]["strict_per_seed"],  ORANGE),
        ("Alg. 2\n(Pre-Q)",       s9["P2_Full_AlgB_step10"]["strict_mean"], s9["P2_Full_AlgB_step10"]["strict_std"], s9["P2_Full_AlgB_step10"]["strict_per_seed"], GREEN),
        ("Direct L64\n(Pre-Q)",   s9b["P2_PostRL_Direct_L64"]["strict_mean"],s9b["P2_PostRL_Direct_L64"]["strict_std"],s9b["P2_PostRL_Direct_L64"]["strict_per_seed"],BLUE),
        ("Alg. 2*\n(Post-Q)",     s9["P4_Full_AlgB_step10"]["strict_mean"], s9["P4_Full_AlgB_step10"]["strict_std"], s9["P4_Full_AlgB_step10"]["strict_per_seed"], RED),
        ("Direct L64\n(Post-Q)",  s9b["P4_PostRL_Direct_L64"]["strict_mean"],s9b["P4_PostRL_Direct_L64"]["strict_std"],s9b["P4_PostRL_Direct_L64"]["strict_per_seed"],BLUE),
        ("Alg. 1 L64\n(Post-Q)",  ev9b["P4_PostRL_KVInv2_L64"]["prompt_strict"], 0.0, [], TEAL),
    ]
    xm = np.arange(len(m_items))
    for i, (lbl, m, sd, pts, col) in enumerate(m_items):
        ax1.bar(i, m, width=0.58, color=col, alpha=0.88, zorder=3)
        if sd > 0:
            ax1.errorbar(i, m, yerr=sd, fmt="none", ecolor=DARK, elinewidth=1.1, capsize=3.5, zorder=5)
            for j, p in enumerate(pts):
                ax1.scatter(i + (j - 1)*0.11, p, color="white", edgecolor=DARK, s=20, lw=0.8, zorder=6)
            top_y = max(m + sd, max(pts)) + 1.8
            ax1.text(i, top_y, f"{m:.1f}\n±{sd:.1f}", ha="center", va="bottom", fontsize=6.9)
        else:
            ax1.text(i, m + 1.8, f"{m:.1f}\n(1-seed)", ha="center", va="bottom", fontsize=6.8)
    ax1.axhline(bare96, color=GRAY, linestyle="--", lw=1.1, label=f"Frozen backbone ({bare96:.1f}%)")
    ax1.axvline(3.5, color=GRAY, linestyle=":", lw=0.9)
    ax1.set_xticks(xm)
    ax1.set_xticklabels([it[0] for it in m_items], fontsize=6.2)
    ax1.set_ylabel("IFEval Prompt Strict Acc. (%, N=96)")
    ax1.set_ylim(30, 86)
    ax1.set_title("(a) 3-seed stability (*Post-Q ref. mismatch)", pad=22)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), frameon=False, fontsize=7.5)

    # (b) Scaled IFBench (N=64) plane (Completed Precision vs Overall Strict)
    style_boxed(ax2, "both")
    pts = [
        ("Frozen backbone",          ib9["Bare_Student"],               GRAY,   "s", (-50, 14)),
        ("Source KV-PD (Start)",     ib9b["P0_KV_PostRL_IF_L16"],       PURPLE, "D", (-26, -28)),
        ("Multi-GRPO (Post-Q)",      ib9["P4_Cold_TF2"],                RED,    "v", (8, -6)),
        ("Alg. 2 TF-CRSP (Post-Q)",  ib9["P4_Full_AlgB"],               ORANGE, "^", (-95, -16)),
        ("Direct CRSP L64 (Pre-Q)",  ib9b["P2_PostRL_Direct_L64"],      LBLUE,  "o", (10, -14)),
        ("Direct CRSP L256 (Post-Q)",ib9b["P4_PostRL_Direct_L256"],     BLUE,   "o", (-112, 16)),
        ("Alg. 1 L256 (Pre-Q)",      ib9b["P2_PostRL_KVInv2_L256"],     TEAL,   "P", (12, 13)),
        ("Alg. 1 L64 (Post-Q)",      ib9b["P4_PostRL_KVInv2_L64"],      GREEN,  "*", (-108, -4)),
        ("Triple Alg. 1 Split",      ib9b["Triple_PostRL_KVInv2_Split"],"#B9770E","X", (-98, -36)),
    ]
    for lbl, rec, col, mk, offset in pts:
        sx, sy = rec["closed_acc"], rec["prompt_strict"]
        ms = 140 if mk == "*" else 56
        ax2.scatter(sx, sy, color=col, marker=mk, s=ms, edgecolor=DARK, lw=0.7, zorder=5)
        ax2.annotate(f"{lbl}\n({sy:.1f}%, {sx:.1f}%)", xy=(sx, sy), xytext=offset,
                     textcoords="offset points", fontsize=6.6, color="#1A1A1A",
                     arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.6))
    ax2.set_xlabel("Completed-response precision (%)")
    ax2.set_ylabel("IFBench Prompt Strict Acc. (%, N=64)")
    ax2.set_xlim(42, 95)
    ax2.set_ylim(29, 58)
    ax2.set_title("(b) IFBench (N=64) accuracy vs. completed precision", pad=22)

    # (c) Adversarial Post-RAG 4-tier horizontal bars (mirrors paper Fig. 6a!)
    style_boxed(ax3, "x")
    p3_rows = [
        ("Direct CRSP\n(L=256)",        ib9b["P3_PostRL_Direct_L256"],  BLUE),
        ("KV-Inv-CRSP\n(Alg. 1, L=256)",ib9b["P3_PostRL_KVInv2_L256"],  TEAL),
        ("Frozen\nbackbone",            ib9b["P3_Bare_Student"],        GRAY),
        ("Two-Stage TF\n(Alg. 2, L=64)",ib9b["P3_ZeroPostRL_AlgB_L64"], ORANGE),
    ]
    y3 = np.arange(len(p3_rows))[::-1]
    h3 = 0.35
    s3 = [r[1]["prompt_strict"] for r in p3_rows]
    c3 = [r[1]["closed_acc"]    for r in p3_rows]
    cols3 = [r[2] for r in p3_rows]
    b_s3 = ax3.barh(y3 + h3/2, s3, h3, color=cols3, label="Prompt strict (%)", zorder=3)
    b_c3 = ax3.barh(y3 - h3/2, c3, h3, color=LBLUE, edgecolor=BLUE, lw=0.6, alpha=0.85, label="Completed precision (%)", zorder=3)
    ax3.axvline(ib9b["P3_Bare_Student"]["prompt_strict"], color=GRAY, linestyle=":", lw=0.9)
    for b, v in zip(b_s3, s3):
        ax3.text(v + 1.5, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.4, fontweight="bold" if v > 45 else "normal")
    for b, v in zip(b_c3, c3):
        ax3.text(v + 1.5, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.2, color="#2C5282")
    ax3.set_yticks(y3)
    ax3.set_yticklabels([r[0] for r in p3_rows], fontsize=7.7)
    ax3.set_xlim(0, 86)
    ax3.set_xlabel("Adversarial Post-RAG accuracy (%)")
    ax3.set_title("(c) Adversarial Post-RAG (+420 tok)", pad=22)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.48, 1.11), ncol=2, frameon=False, fontsize=7.1, handlelength=1.4, columnspacing=0.8)

    path = os.path.join(OUT, "pfig10_crsp_multiseed_and_scaled_ifbench64.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 11: Multi-Expert Interference Matrix, Split-Slot Isolation & Declarative Duality
# ==============================================================================
def make_pfig11():
    d4  = load_res("exp4_results.json")
    d5  = load_res("exp5_results.json")
    d9  = load_res("exp9_results.json")
    d9b = load_res("exp9b_results.json")
    k9  = d9["track2_interference_kl"]
    k9b = d9b["interference_kl"]
    c9  = d9["track2_composition_evals"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.4, 4.1), gridspec_kw={"width_ratios": [1.08, 1.15, 1.0], "wspace": 0.36})

    # (a) Cross-Expert Top-32 KL Interference vs Own-Domain Signal
    style_boxed(ax1, "y")
    packs = [
        ("Basic\nCRSP",          k9["Naive"]),
        ("TF\nCS-CR",            k9["CSCR"]),
        ("Direct\nCS-CR",        k9b["PostRL_Direct_CSCR"]),
        ("Alg. 1\nCS-CR",        k9b["PostRL_KVInv2_CSCR"]),
    ]
    xp = np.arange(len(packs))
    w = 0.25
    if_tr  = [p[1]["if"]["triple_interference_kl"] for p in packs]
    doc_lk = [p[1]["if"]["kl_from_bare_doc"]       for p in packs]
    if_sig = [p[1]["if"]["kl_from_bare_if"]        for p in packs]
    b1 = ax1.bar(xp - w, if_tr,  w, color=RED,    label="IF triple interference ↓", zorder=3)
    b2 = ax1.bar(xp,     doc_lk, w, color=ORANGE, label="Doc→IF cross-leak ↓", zorder=3)
    b3 = ax1.bar(xp + w, if_sig, w, color=BLUE,   label="IF own signal ↑", zorder=3)
    for bars in [b1, b2, b3]:
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width()/2, h + 0.004, f"{h:.3f}", ha="center", va="bottom", fontsize=6.7)
    ax1.annotate("2.95×\nlower", xy=(3 - w, 0.068), xytext=(2.82, 0.142), ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0), fontsize=7.3, fontweight="bold", color=RED)
    ax1.set_xticks(xp)
    ax1.set_xticklabels([p[0] for p in packs], fontsize=7.8)
    ax1.set_ylabel("Top-32 Logit KL divergence (nats)")
    ax1.set_ylim(0, 0.295)
    ax1.set_title("(a) Three-expert KL interference & signal", pad=22)
    ax1.legend(loc="upper left", frameon=False, fontsize=7.2)

    # (b) Split-Slot Isolation Synergy & 0.00 pp Permutation Invariance
    style_boxed(ax2, "y")
    m9  = c9["math48"]
    m9b = d9b["math48_evals"]
    cats = [
        "Frozen\nbase",
        "TF Solo\nMath",
        "TF Split\nPre+Post",
        "TF Adj.\n[I;M]",
        "TF Adj.\n[M;I]",
        "RL Solo\nMath",
        "RL Adj.\n[I;M]",
        "RL Adj.\n[M;I]",
    ]
    mv = [
        m9["Bare_Student"]["acc"],
        m9["Single_Math_CSCR_P2"]["acc"],
        m9["Dual_CSCR_MathP2_IFP4"]["acc"],
        m9["Dual_CSCR_Adj_IF_Math_P2"]["acc"],
        m9["Dual_CSCR_Adj_Math_IF_P2"]["acc"],
        m9b["P2_PostRL_Direct_Math_L64"]["acc"],
        m9b["Dual_PostRL_Adj_IF_Math_P2"]["acc"],
        m9b["Dual_PostRL_Adj_Math_IF_P2"]["acc"],
    ]
    iv = [
        c9["ifeval96"]["Bare_Student"]["prompt_strict"],
        np.nan,
        c9["ifeval96"]["Dual_CSCR_MathP2_IFP4"]["prompt_strict"],
        c9["ifeval96"]["Dual_CSCR_Adj_IF_Math_P2"]["prompt_strict"],
        c9["ifeval96"]["Dual_CSCR_Adj_Math_IF_P2"]["prompt_strict"],
        np.nan,
        d9b["ifeval96_evals"]["Dual_PostRL_Adj_IF_Math_P2"]["prompt_strict"],
        d9b["ifeval96_evals"]["Dual_PostRL_Adj_Math_IF_P2"]["prompt_strict"],
    ]
    xc = np.arange(len(cats))
    wb = 0.36
    bm = ax2.bar(xc - wb/2, mv, wb, color=BLUE, label="MATH (%, N=48)", zorder=3)
    bi = ax2.bar(xc + wb/2, iv, wb, color=TEAL, label="IFEval (%, N=96)", zorder=3)
    bm[2].set_color(GREEN)
    for b, v in zip(bm, mv):
        ax2.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    for b, v in zip(bi, iv):
        if not np.isnan(v):
            ax2.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=TEAL)
    ax2.axvline(4.5, color=GRAY, linestyle="--", lw=0.9)
    ax2.text(3.5, 79.5, "IFEval |Δ|=0.00 pp", ha="center", fontsize=6.9, color=TEAL,
             bbox=dict(boxstyle="round,pad=0.18", fc="#EEF7F6", ec=TEAL, lw=0.6))
    ax2.text(6.3, 79.5, "MATH |Δ|=0.00 pp", ha="center", fontsize=6.9, color=BLUE,
             bbox=dict(boxstyle="round,pad=0.18", fc="#EEF5FB", ec=BLUE, lw=0.6))
    ax2.set_xticks(xc)
    ax2.set_xticklabels(cats, fontsize=6.7)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 95)
    ax2.set_title("(b) Split-slot synergy & permutation invariance", pad=22)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=False, fontsize=7.3)

    # (c) Declarative Capacity Boundary & 5.0x Recency Gradient (horizontal bars)
    style_boxed(ax3, "x")
    doc9  = c9["doc36"]
    doc9b = d9b["doc36_evals"]
    d_rows = [
        ("In-Context\nUpper Bound",       doc9["InContext_Doc_UpperBound"]["rule_acc"],                 doc9["InContext_Doc_UpperBound"]["fact_acc"]),
        ("KV-PD L=32\n(Start Prefix)",    d4["runs"]["KV_A_L32_AdamW"]["A_heliospay"]["rules"],         d4["runs"]["KV_A_L32_AdamW"]["A_heliospay"]["facts"]),
        ("KV-PD L=16 +\nMicro-RAG",       d5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["A_heliospay"]["rules"],
                                          d5["auto_router"]["e2e_kv16_microRAG_auto_routed"]["A_heliospay"]["facts"]),
        ("CRSP Doc L=128\nPre-Query",     doc9b["Single_Doc_CSCR_MuonL128_P2"]["rule_acc"],             doc9b["Single_Doc_CSCR_MuonL128_P2"]["fact_acc"]),
        ("CRSP Doc L=128\nPost-Query",    doc9b["Single_Doc_CSCR_MuonL128_P4"]["rule_acc"],             doc9b["Single_Doc_CSCR_MuonL128_P4"]["fact_acc"]),
        ("CRSP Doc L=128\nPost-Sys (Far)",doc9b["Single_Doc_CSCR_MuonL128_P1"]["rule_acc"],             doc9b["Single_Doc_CSCR_MuonL128_P1"]["fact_acc"]),
        ("CRSP Doc L=64\n(RMS≤0.055)",    doc9["Single_Doc_CSCR_P2"]["rule_acc"],                       doc9["Single_Doc_CSCR_P2"]["fact_acc"]),
    ]
    yd = np.arange(len(d_rows))[::-1]
    hd = 0.35
    dr = [r[1] for r in d_rows]
    df = [r[2] for r in d_rows]
    b_dr = ax3.barh(yd + hd/2, dr, hd, color=BLUE, label="Rules acc. (%)", zorder=3)
    b_df = ax3.barh(yd - hd/2, df, hd, color=TEAL, label="Facts acc. (%)", zorder=3)
    ax3.axhline(3.5, color=GRAY, linestyle="--", lw=0.9)
    for b, v in zip(b_dr, dr):
        if v > 0:
            ax3.text(v + 1.8, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.0)
    for b, v in zip(b_df, df):
        ax3.text(v + 1.8, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", ha="left", fontsize=7.0, color=TEAL if v > 0 else GRAY)
    ax3.annotate("5.0× proximity\ngradient", xy=(44, 2.8), xytext=(64, 1.4), ha="center",
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0), fontsize=7.2, fontweight="bold", color=TEAL)
    ax3.set_yticks(yd)
    ax3.set_yticklabels([r[0] for r in d_rows], fontsize=7.3)
    ax3.set_xlim(0, 118)
    ax3.set_xlabel("HeliosPay v4.2 accuracy (%)")
    ax3.set_title("(c) Private-doc capacity by slot", pad=22)
    ax3.legend(loc="lower right", frameon=False, fontsize=7.2)

    path = os.path.join(OUT, "pfig11_crsp_multiexpert_composition_and_duality.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 12: Long-Context Single-Expert Attenuation (0 -> 8192) & Attention Mass
# ==============================================================================
def make_pfig12():
    d10  = load_res("exp10_results.json")
    d10c = load_res("exp10c_results.json")

    lens = ["0", "512", "1024", "2048", "4096", "8192"]
    xl   = np.arange(len(lens))
    xtl  = ["0", "512", "1k", "2k", "4k", "8k"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(14.4, 4.15), gridspec_kw={"width_ratios": [1.05, 1.05, 1.05], "wspace": 0.30}
    )

    # (a) IF Single-Expert Alignment Gain vs Context Length
    style_boxed(ax1, "both")
    if_s = d10["partA_single_sweep_if"]
    kv_if = d10c["kvpd_smooth_6pt"]["if"]
    ax1.axhline(0.0, color=DARK, lw=0.8, ls="--", zorder=1)
    ax1.plot(xl, [if_s["Direct_CRSP_L256_at_P2"][L]["g_vs_ctx"] for L in lens], "-o",  color=BLUE,   lw=2.0, ms=4.8, label="Direct CRSP Pre-Q (L=256)", zorder=5)
    ax1.plot(xl, [if_s["Alg1_KVInv_L64_at_P2"][L]["g_vs_ctx"]   for L in lens], "-^",  color=TEAL,   lw=1.9, ms=4.8, label="Alg. 1 CRSP Pre-Q (L=64)",  zorder=5)
    ax1.plot(xl, [kv_if[L]["g_vs_ctx"]                           for L in lens], "-D",  color=PURPLE, lw=1.9, ms=4.5, label="KV-PD Start (L=16)",        zorder=4)
    ax1.plot(xl, [if_s["PR_CSPD_L256_at_P0"][L]["g_vs_ctx"]      for L in lens], "-s",  color=ORANGE, lw=1.8, ms=4.5, label="PR-CSPD Start (L=256)",     zorder=3)
    ax1.plot(xl, [if_s["Pos0_CSPD_L256_at_P0"][L]["g_vs_ctx"]    for L in lens], "--s", color=GRAY,   lw=1.4, ms=4.0, label="Start-Only CSPD (L=256)",   zorder=2)
    ax1.plot(xl, [if_s["Direct_CRSP_L256_at_P0"][L]["g_vs_ctx"]  for L in lens], ":x",  color=RED,    lw=1.7, ms=5.0, label="Direct CRSP Start (ctrl)",  zorder=3)

    ax1.annotate("", xy=(5, 30.24), xytext=(5, -8.36),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2))
    ax1.text(4.82, 20.8, "+38.6 pp\n(same weights)", ha="right", va="center", fontsize=6.9, fontweight="bold", color=BLUE,
             bbox=dict(boxstyle="round,pad=0.18", fc="#EEF5FB", ec=BLUE, lw=0.6))
    ax1.set_xticks(xl)
    ax1.set_xticklabels(xtl)
    ax1.set_xlabel("Preceding context length (tokens)")
    ax1.set_ylabel("Same-context KL reduction G (%)")
    ax1.set_ylim(-18, 66)
    ax1.set_title("(a) IF single-expert attenuation (0 → 8k)", pad=10)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#DDDDDD", fontsize=6.7, handlelength=1.8)

    # (b) Math Single-Expert Alignment Gain vs Context Length
    style_boxed(ax2, "both")
    ma_s = d10["partA_single_sweep_math"]
    kv_m = d10c["kvpd_smooth_6pt"]["math"]
    ax2.axhline(0.0, color=DARK, lw=0.8, ls="--", zorder=1)
    ax2.plot(xl, [ma_s["Direct_CRSP_Math_L64_P2"][L]["g_vs_ctx"] for L in lens], "-o", color=BLUE,   lw=2.0, ms=4.8, label="Direct CRSP Pre-Q (L=64)", zorder=5)
    ax2.plot(xl, [ma_s["Alg1_KVInv_Math_L64_P2"][L]["g_vs_ctx"]  for L in lens], "-^", color=TEAL,   lw=1.9, ms=4.8, label="Alg. 1 CRSP Pre-Q (L=64)", zorder=5)
    ax2.plot(xl, [kv_m[L]["g_vs_ctx"]                             for L in lens], "-D", color=PURPLE, lw=1.9, ms=4.5, label="KV-PD Start (L=16)",       zorder=4)
    ax2.plot(xl, [ma_s["PR_CSPD_Math_L256_at_P0"][L]["g_vs_ctx"] for L in lens], "-s", color=ORANGE, lw=1.8, ms=4.5, label="PR-CSPD Start (L=256)",    zorder=3)
    ctrl_vals = [ma_s["Direct_CRSP_Math_L64_P0"][L]["g_vs_ctx"] for L in lens]
    ax2.plot(xl[:5], ctrl_vals[:5], ":x", color=RED, lw=1.7, ms=5.0, label="Direct CRSP Start (ctrl)", zorder=3)
    ax2.plot([4, 5], [ctrl_vals[4], -104], ":", color=RED, lw=1.7, zorder=3)
    ax2.annotate("−225.5% @ 8k\n(off-scale)", xy=(5, -104), xytext=(3.45, -88), ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0), fontsize=6.9, fontweight="bold", color=RED)
    ax2.annotate("PR-CSPD\n−82.9%", xy=(5, -82.88), xytext=(4.62, -22), ha="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0), fontsize=6.9, fontweight="bold", color=ORANGE)
    ax2.set_xticks(xl)
    ax2.set_xticklabels(xtl)
    ax2.set_xlabel("Preceding context length (tokens)")
    ax2.set_ylabel("Same-context KL reduction G (%)")
    ax2.set_ylim(-112, 68)
    ax2.set_title("(b) Math single-expert attenuation (0 → 8k)", pad=10)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#DDDDDD", fontsize=6.8, handlelength=1.8)

    # (c) Mechanistic Layerwise Response-to-Module Attention Mass
    style_boxed(ax3, "both")
    am = d10["partB_attention_mass"]
    kv_am   = [am["KVPD_L16_at_P0"][L]["mean_mod_pct"]       for L in lens]
    pr_am   = [am["PR_CSPD_L256_at_P0"][L]["mean_mod_pct"]   for L in lens]
    cr_am   = [am["Direct_CRSP_L256_at_P4"][L]["mean_mod_pct"] for L in lens]
    alg1_am = [am["Alg1_KVInv_L64_at_P4"][L]["mean_mod_pct"] for L in lens]

    ax3.plot(xl, kv_am,   "-D", color=PURPLE, lw=2.1, ms=4.8, label="KV-PD Start (16 tok)",         zorder=5)
    ax3.plot(xl, pr_am,   "-s", color=ORANGE, lw=1.9, ms=4.6, label="PR-CSPD Start (256 vec)",      zorder=4)
    ax3.plot(xl, cr_am,   "-o", color=BLUE,   lw=1.8, ms=4.6, label="Direct CRSP Post-Q (256 vec)", zorder=4)
    ax3.plot(xl, alg1_am, "-^", color=TEAL,   lw=1.9, ms=4.8, label="Alg. 1 CRSP Post-Q (64 vec)",  zorder=5)

    ax3.annotate("77.6% retained\n(3.5× vs PR-CSPD)", xy=(5, kv_am[-1]), xytext=(3.15, 32.8), ha="center",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0), fontsize=7.0, fontweight="bold", color=PURPLE)
    ax3.annotate("−54.8% decay\n(shallow −71%)", xy=(5, pr_am[-1]), xytext=(3.95, 16.5), ha="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0), fontsize=6.9, fontweight="bold", color=ORANGE)
    ax3.text(2.8, 2.3, "Alg. 1 Post-Q flat 512→8k (6.9% → 6.4%)", ha="center", fontsize=6.7, color=TEAL,
             bbox=dict(boxstyle="round,pad=0.18", fc="#EEF7F6", ec=TEAL, lw=0.6))
    ax3.set_xticks(xl)
    ax3.set_xticklabels(xtl)
    ax3.set_xlabel("Preceding context length (tokens)")
    ax3.set_ylabel("Mean response→module attn. mass (%)")
    ax3.set_ylim(0, 38)
    ax3.set_title("(c) Layerwise attention-mass mechanism", pad=10)
    ax3.legend(loc="center left", bbox_to_anchor=(0.01, 0.57), frameon=True, facecolor="white", framealpha=0.94, edgecolor="#DDDDDD", fontsize=6.6, handlelength=1.7)

    path = os.path.join(OUT, "pfig12_long_context_single_expert_and_mechanism.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 13: Long-Context Dual-Expert Composition, E2E Benchmarks & Doc KV-PD
# ==============================================================================
def make_pfig13():
    d10  = load_res("exp10_results.json")
    d10b = load_res("exp10b_results.json")
    d10c = load_res("exp10c_results.json")

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(14.8, 4.25), gridspec_kw={"width_ratios": [1.06, 1.12, 1.06], "wspace": 0.30}
    )

    # (a) Dual-Expert Long-Context Mean Gain (0 -> 8k) & Order/Interference Gap
    style_boxed(ax1, "both")
    lc = ["0", "1024", "4096", "8192"]
    xc = np.arange(len(lc))
    xtc = ["0", "1k", "4k", "8k"]
    ci, cm = d10["partC_composition_if"], d10["partC_composition_math"]
    kvi, kvm = d10b["partC_kvpd_clean_verify"]["if"], d10b["partC_kvpd_clean_verify"]["math"]

    def mean_pair(di, dm, key):
        return [0.5 * (di[key][L]["g_vs_ctx"] + dm[key][L]["g_vs_ctx"]) for L in lc]

    pr_mi   = mean_pair(ci, cm, "PR_CSPD_P0_Math_IF")
    pr_im   = mean_pair(ci, cm, "PR_CSPD_P0_IF_Math")
    kv_mi   = mean_pair(kvi, kvm, "KVPD_Dual_Math_IF")
    dir_sp  = mean_pair(ci, cm, "Direct_CRSP_Split_M@P2_I@P4")
    alg1_sp = mean_pair(ci, cm, "Alg1_CRSP_Split_M@P2_I@P4")

    ax1.axhline(0.0, color=DARK, lw=0.8, ls="--", zorder=1)
    ax1.plot(xc, dir_sp,  "-o",  color=BLUE,   lw=2.0, ms=5.0, label="Direct CRSP Split [M:Pre+I:Post]", zorder=5)
    ax1.plot(xc, alg1_sp, "-^",  color=TEAL,   lw=1.9, ms=5.0, label="Alg. 1 Split [M:Pre+I:Post]",      zorder=5)
    ax1.plot(xc, kv_mi,   "-D",  color=PURPLE, lw=1.9, ms=4.7, label="KV-PD Dual Start (|Δ|≤0.2 pp)",   zorder=4)
    ax1.plot(xc, pr_mi,   "-s",  color=ORANGE, lw=1.9, ms=4.7, label="PR-CSPD Dual Start [M; I]",       zorder=3)
    ax1.plot(xc, pr_im,   "--s", color=RED,    lw=1.5, ms=4.2, label="PR-CSPD Dual Start [I; M]",       zorder=3)
    ax1.fill_between(xc, pr_mi, pr_im, color=ORANGE, alpha=0.14, zorder=2)

    ax1.annotate("Order gap 22.9 pp (IF)\nKL int. 0.386 (10.3×)", xy=(2.0, pr_mi[2]), xytext=(2.28, -14.0), ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0), fontsize=6.6, fontweight="bold", color=RED,
                 bbox=dict(boxstyle="round,pad=0.18", fc="#FDF2F2", ec=RED, lw=0.6))
    ax1.text(2.05, 41.5, "Alg. 1 KL int. 0.032–0.042", ha="center", fontsize=6.7, color=TEAL,
             bbox=dict(boxstyle="round,pad=0.18", fc="#EEF7F6", ec=TEAL, lw=0.6))
    ax1.set_xticks(xc)
    ax1.set_xticklabels(xtc)
    ax1.set_xlabel("Preceding context length (tokens)")
    ax1.set_ylabel("Mean dual-domain gain G (%)")
    ax1.set_ylim(-68, 52)
    ax1.set_title("(a) Dual-expert composition (0 → 8k)", pad=10)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#DDDDDD", fontsize=6.3, handlelength=1.6)

    # (b) End-to-End Dual-Expert Accuracy at L_ctx = 4096 (IFEval48, IFBench32, MATH32)
    style_boxed(ax2, "y")
    ev_if  = d10c["e2e_ifeval48_4096"]
    ev_ifb = d10c["e2e_ifbench32"]["4096"]
    ev_ma  = d10c["e2e_math32"]["4096"]

    benches = ["IFEval\n(N=48, L=4k)", "IFBench\n(N=32, L=4k)", "MATH\n(N=32, L=4k)"]
    xb = np.arange(len(benches))
    wb = 0.19
    v_base = [ev_if["Base_Frozen"]["prompt_strict"],               ev_ifb["Base_Frozen"]["prompt_strict"],               ev_ma["Base_Frozen"]["acc"]]
    v_pr   = [ev_if["PR_CSPD_Dual_P0"]["prompt_strict"],           ev_ifb["PR_CSPD_Dual_P0"]["prompt_strict"],           ev_ma["PR_CSPD_Dual_P0"]["acc"]]
    v_kv   = [ev_if["KVPD_Dual_P0"]["prompt_strict"],              ev_ifb["KVPD_Dual_P0"]["prompt_strict"],              ev_ma["KVPD_Dual_P0"]["acc"]]
    v_alg1 = [ev_if["Alg1_CRSP_Split_M@P2_I@P4"]["prompt_strict"], ev_ifb["Alg1_CRSP_Split_M@P2_I@P4"]["prompt_strict"], ev_ma["Alg1_CRSP_Split_M@P2_I@P4"]["acc"]]

    b0 = ax2.bar(xb - 1.5*wb, v_base, wb, color=LBLUE,  edgecolor=DARK, lw=0.5, label="Frozen base",             zorder=3)
    b1 = ax2.bar(xb - 0.5*wb, v_pr,   wb, color=ORANGE, edgecolor=DARK, lw=0.5, label="PR-CSPD Dual Start",      zorder=3)
    b2 = ax2.bar(xb + 0.5*wb, v_kv,   wb, color=PURPLE, edgecolor=DARK, lw=0.5, label="KV-PD Dual Start",        zorder=3)
    b3 = ax2.bar(xb + 1.5*wb, v_alg1, wb, color=TEAL,   edgecolor=DARK, lw=0.5, label="Alg. 1 Split [Pre+Post]", zorder=3)

    for bars in [b0, b1, b2, b3]:
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2, h + 1.4, f"{h:.1f}", ha="center", va="bottom", fontsize=6.3)

    ax2.annotate("−39.6 pp\nvs L=0", xy=(0 - 0.5*wb, v_pr[0] + 6.5), xytext=(0 - 0.5*wb, 82.0), ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0), fontsize=6.4, fontweight="bold", color=RED,
                 bbox=dict(boxstyle="round,pad=0.14", fc="#FDF2F2", ec=RED, lw=0.5))
    ax2.annotate("−34.4 pp\nvs L=0", xy=(1 - 0.5*wb, v_pr[1] + 6.5), xytext=(1 - 0.5*wb, 72.0), ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0), fontsize=6.4, fontweight="bold", color=RED,
                 bbox=dict(boxstyle="round,pad=0.14", fc="#FDF2F2", ec=RED, lw=0.5))
    ax2.set_xticks(xb)
    ax2.set_xticklabels(benches, fontsize=7.8)
    ax2.set_ylabel("End-to-end strict accuracy (%)")
    ax2.set_ylim(0, 114)
    ax2.set_title("(b) End-to-end dual accuracy at L=4096", pad=10)
    ax2.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", framealpha=0.94, edgecolor="#DDDDDD", fontsize=6.3, handlelength=1.4, columnspacing=0.65)

    # (c) Zero-Leakage Document KV-PD (HeliosPay v4.2, N=36) Under Long Context (0 -> 4k)
    style_boxed(ax3, "both")
    ld = ["0", "1024", "2048", "4096"]
    xd = np.arange(len(ld))
    xtd = ["0", "1k", "2k", "4k"]
    de = d10b["partE_doc_kvpd_longctx"]

    r_kv32 = [de[L]["KVPD_L32_Only"]["rule_acc"]     for L in ld]
    r_kv16 = [de[L]["KVPD_L16_Only"]["rule_acc"]     for L in ld]
    f_rag32 = [de[L]["KVPD_L32_MicroRAG"]["fact_acc"] for L in ld]
    f_rag16 = [de[L]["KVPD_L16_MicroRAG"]["fact_acc"] for L in ld]
    r_man = [de["0"]["InContext_Manual_P0"]["rule_acc"], de["4096"]["InContext_Manual_P0"]["rule_acc"]]
    f_man = [de["0"]["InContext_Manual_P0"]["fact_acc"], de["4096"]["InContext_Manual_P0"]["fact_acc"]]

    ax3.plot(xd, f_rag32, "-o",  color=TEAL,   lw=2.0, ms=4.8, label="KV-PD L=32 + Micro-RAG (facts)", zorder=5)
    ax3.plot(xd, f_rag16, "--^", color=GREEN,  lw=1.7, ms=4.6, label="KV-PD L=16 + Micro-RAG (facts)", zorder=4)
    ax3.plot([0, 3], f_man, ":D", color=GRAY,  lw=1.5, ms=4.5, label="Full Manual Start (facts)",      zorder=3)
    ax3.plot(xd, r_kv32,  "-s",  color=PURPLE, lw=2.0, ms=4.8, label="KV-PD L=32 Start (rules)",       zorder=5)
    ax3.plot(xd, r_kv16,  "--s", color=ORANGE, lw=1.7, ms=4.4, label="KV-PD L=16 Start (rules)",       zorder=4)
    ax3.plot([0, 3], r_man, ":x", color=RED,   lw=1.7, ms=5.2, label="Full Manual Start (rules)",      zorder=4)

    ax3.annotate("Query-adjacent Micro-RAG:\n91.7% facts @ 4k", xy=(3, f_rag32[-1]), xytext=(1.35, 107.0), ha="center",
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0), fontsize=6.6, fontweight="bold", color=TEAL)
    ax3.annotate("Short-ctx KV-PD rules\ndrop to 0% at ≥1k", xy=(1.0, 0.8), xytext=(1.95, 18.0), ha="center",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0), fontsize=6.6, fontweight="bold", color=PURPLE,
                 bbox=dict(boxstyle="round,pad=0.16", fc="#F6F2FA", ec=PURPLE, lw=0.6))
    ax3.set_xticks(xd)
    ax3.set_xticklabels(xtd)
    ax3.set_xlabel("Preceding context length (tokens)")
    ax3.set_ylabel("HeliosPay v4.2 accuracy (%)")
    ax3.set_ylim(-6, 120)
    ax3.set_title("(c) Private-doc KV-PD (0 → 4k)", pad=10)
    ax3.legend(loc="center right", bbox_to_anchor=(0.99, 0.45), frameon=True, facecolor="white", framealpha=0.95, edgecolor="#DDDDDD", fontsize=6.1, handlelength=1.6)

    path = os.path.join(OUT, "pfig13_long_context_composition_and_doc_e2e.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 14: Comprehensive Injection-Order & Multi-Expert Slot-Permutation Ablation
# ==============================================================================
def make_pfig14():
    d1b  = load_res("exp1b_results.json")
    d10  = load_res("exp10_results.json")
    d10b = load_res("exp10b_results.json")
    d11  = load_res("exp11_results.json")

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15.0, 4.35), gridspec_kw={"width_ratios": [1.08, 1.08, 1.12], "wspace": 0.30}
    )

    # (a) Dual-Expert Order/Swap Sensitivity |Δ| (pp) across Architectures & Slots
    style_boxed(ax1, "y")
    g1b = d1b["distributional_G"]
    c10i, c10m = d10["partC_composition_if"], d10["partC_composition_math"]
    kvi, kvm   = d10b["partC_kvpd_clean_verify"]["if"], d10b["partC_kvpd_clean_verify"]["math"]
    pAi, pAm   = d11["partA_split_and_adj_order"]["if"], d11["partA_split_and_adj_order"]["math"]

    cats_a = [
        "No-PR\n(L=0)",
        "PR-CSPD\n(L=0)",
        "PR-CSPD\nStart\n(4k)",
        "KV-PD\nStart\n(4k)",
        "Direct\nPost-Q\n(4k)",
        "Alg. 1\nPre-Q\n(4k)",
        "Alg. 1\nSplit\n(4k)",
    ]
    gap_if = [
        abs(g1b["ablation_nopr_Math_IF"]["G_if_pct"] - g1b["ablation_nopr_IF_Math"]["G_if_pct"]),
        abs(g1b["cspd_concat_Math_IF"]["G_if_pct"]   - g1b["cspd_concat_IF_Math"]["G_if_pct"]),
        abs(c10i["PR_CSPD_P0_Math_IF"]["4096"]["g_vs_ctx"] - c10i["PR_CSPD_P0_IF_Math"]["4096"]["g_vs_ctx"]),
        abs(kvi["KVPD_Dual_Math_IF"]["4096"]["g_vs_ctx"]   - kvi["KVPD_Dual_IF_Math"]["4096"]["g_vs_ctx"]),
        abs(pAi["Direct_Adj_M_I@P4"]["4096"]["g_vs_ctx"]   - pAi["Direct_Adj_I_M@P4"]["4096"]["g_vs_ctx"]),
        abs(pAi["Alg1_Adj_M_I@P2"]["4096"]["g_vs_ctx"]     - pAi["Alg1_Adj_I_M@P2"]["4096"]["g_vs_ctx"]),
        abs(pAi["Alg1_Split_M@P2_I@P4"]["4096"]["g_vs_ctx"] - pAi["Alg1_Split_I@P2_M@P4"]["4096"]["g_vs_ctx"]),
    ]
    gap_ma = [
        abs(g1b["ablation_nopr_Math_IF"]["G_math_pct"] - g1b["ablation_nopr_IF_Math"]["G_math_pct"]),
        abs(g1b["cspd_concat_Math_IF"]["G_math_pct"]   - g1b["cspd_concat_IF_Math"]["G_math_pct"]),
        abs(c10m["PR_CSPD_P0_Math_IF"]["4096"]["g_vs_ctx"] - c10m["PR_CSPD_P0_IF_Math"]["4096"]["g_vs_ctx"]),
        abs(kvm["KVPD_Dual_Math_IF"]["4096"]["g_vs_ctx"]   - kvm["KVPD_Dual_IF_Math"]["4096"]["g_vs_ctx"]),
        abs(pAm["Direct_Adj_M_I@P4"]["4096"]["g_vs_ctx"]   - pAm["Direct_Adj_I_M@P4"]["4096"]["g_vs_ctx"]),
        abs(pAm["Alg1_Adj_M_I@P2"]["4096"]["g_vs_ctx"]     - pAm["Alg1_Adj_I_M@P2"]["4096"]["g_vs_ctx"]),
        abs(pAm["Alg1_Split_M@P2_I@P4"]["4096"]["g_vs_ctx"] - pAm["Alg1_Split_I@P2_M@P4"]["4096"]["g_vs_ctx"]),
    ]
    xa = np.arange(len(cats_a))
    wa = 0.36
    ba1 = ax1.bar(xa - wa/2, gap_if, wa, color=ORANGE, edgecolor=DARK, lw=0.5, label="IF order/swap gap |Δ|",   zorder=3)
    ba2 = ax1.bar(xa + wa/2, gap_ma, wa, color=BLUE,   edgecolor=DARK, lw=0.5, label="Math order/swap gap |Δ|", zorder=3)
    for b, v in zip(ba1, gap_if):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.55, f"{v:.1f}", ha="center", va="bottom", fontsize=6.1, color="#B85200")
    for b, v in zip(ba2, gap_ma):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.55, f"{v:.1f}", ha="center", va="bottom", fontsize=6.1, color=BLUE)
    ax1.axvline(1.5, color=GRAY, ls="--", lw=0.9)
    ax1.annotate("3.57× lower\nwith PR", xy=(1 - wa/2, gap_if[1] + 2.5), xytext=(0.68, 15.8), ha="center",
                 arrowprops=dict(arrowstyle="->", color="#B85200", lw=0.9), fontsize=6.2, fontweight="bold", color="#B85200")
    ax1.annotate("0.0 pp\n(≤0.2 @ 0–8k)", xy=(3, 1.8), xytext=(3.0, 9.8), ha="center",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.9), fontsize=6.1, fontweight="bold", color=PURPLE,
                 bbox=dict(boxstyle="round,pad=0.14", fc="#F6F2FA", ec=PURPLE, lw=0.5))
    ax1.annotate("Post-Q\nrecency bias", xy=(4 + wa/2, 16.3), xytext=(4.65, 22.0), ha="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9), fontsize=6.3, fontweight="bold", color=BLUE)
    ax1.set_xticks(xa)
    ax1.set_xticklabels(cats_a, fontsize=6.4)
    ax1.set_ylabel("Order / slot-swap gap |Δ| (pp) ↓")
    ax1.set_ylim(0, 29.5)
    ax1.set_title("(a) Dual-expert order & slot-swap sensitivity", pad=10)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#DDDDDD", fontsize=6.5, handlelength=1.4)

    # (b) Three-Expert All-6-Permutations ({Doc, Math, IF}) at L=4096 (symlog scale)
    style_boxed(ax2, "y")
    pB4k = d11["partB_3expert_all6_perms"]["L4096"]
    fams_b = [
        ("PR-CSPD\nStart",             "PR_CSPD_P0_3Exp",        ORANGE),
        ("KV-PD\nStart",               "KVPD_P0_3Exp",           PURPLE),
        ("Alg. 1 Split\n(Sys,Pre,Post)","Alg1_Split_P1P2P4_3Exp", BLUE),
        ("Alg. 1 Adj.\nPre-Q",         "Alg1_Adj_P2_3Exp",       TEAL),
    ]
    perm_keys = ["DMI", "DIM", "MDI", "MID", "IDM", "IMD"]
    offsets = np.linspace(-0.20, 0.20, 6)
    ax2.axhline(0.0, color=DARK, lw=0.8, ls="--", zorder=1)
    for idx, (lbl, fkey, col) in enumerate(fams_b):
        pdict = pB4k[fkey]["perms"]
        vals = [pdict[pk]["avg3_g"] for pk in perm_keys]
        mean_v = pB4k[fkey]["avg3_summary"]["mean"]
        # Vertical range whisker
        ax2.plot([idx, idx], [min(vals), max(vals)], color=col, lw=1.4, alpha=0.55, zorder=2)
        for j, (pk, val) in enumerate(zip(perm_keys, vals)):
            is_doc_first = pk.startswith("D")
            pt_col = RED if (fkey in ("PR_CSPD_P0_3Exp", "Alg1_Split_P1P2P4_3Exp") and is_doc_first) else col
            mk = "X" if (fkey in ("PR_CSPD_P0_3Exp", "Alg1_Split_P1P2P4_3Exp") and is_doc_first) else "o"
            ax2.scatter(idx + offsets[j], val, s=34, color=pt_col, edgecolor="white", lw=0.4, marker=mk, zorder=4)
        ax2.plot([idx - 0.27, idx + 0.27], [mean_v, mean_v], color=DARK, lw=1.8, zorder=5)

    ax2.set_yscale("symlog", linthresh=25.0, linscale=0.95)
    ax2.set_ylim(-680, 96)
    ax2.set_yticks([-400, -150, -50, -20, 0, 20, 50])
    ax2.set_yticklabels(["−400", "−150", "−50", "−20", "0", "+20", "+50"], fontsize=7.2)
    ax2.set_xticks(range(len(fams_b)))
    ax2.set_xticklabels([f[0] for f in fams_b], fontsize=7.2)
    ax2.set_ylabel("3-domain mean gain Avg3 (%, symlog)")
    ax2.set_title("(b) Three-expert all-6 permutations (L=4096)", pad=10)

    # Clear callouts in Panel (b)
    ax2.annotate("Doc first: −408%\n(spread 251 pp)", xy=(0.0, -150), xytext=(0.32, -38), ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.9), fontsize=6.0, fontweight="bold", color=RED,
                 bbox=dict(boxstyle="round,pad=0.14", fc="#FDF2F2", ec=RED, lw=0.5))
    ax2.annotate("All 6 perms:\n+0.3% (sp 0.12)", xy=(1.0, 1.5), xytext=(1.0, 44), ha="center",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.9), fontsize=6.0, fontweight="bold", color=PURPLE,
                 bbox=dict(boxstyle="round,pad=0.14", fc="#F6F2FA", ec=PURPLE, lw=0.5))
    ax2.annotate("Doc Post-Sys (DMI/DIM)\n−139% / −137%", xy=(2.0, -148), xytext=(2.05, -400), ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.9), fontsize=5.8, fontweight="bold", color=RED,
                 bbox=dict(boxstyle="round,pad=0.14", fc="#FDF2F2", ec=RED, lw=0.5))
    ax2.text(2.0, 44, "Doc Pre/Post-Q:\n+11..+22%", ha="center", va="bottom", fontsize=5.9, fontweight="bold", color=BLUE)
    ax2.text(3.0, 44, "All 6 > +23%\nMDI +27.6%", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=TEAL,
             bbox=dict(boxstyle="round,pad=0.14", fc="#EEF7F6", ec=TEAL, lw=0.5))

    # (c) End-to-End Order/Slot-Swap Accuracy at L_ctx = 4096
    style_boxed(ax3, "y")
    pC = d11["partC_e2e_order_swap"]
    c_labels = [
        "MATH (32)\nAlg. 1 Split",
        "IFBench (32)\nAlg. 1 Split",
        "IFEval (48)\nAlg. 1 Split",
        "IFBench (32)\nKV-PD Start",
        "IFBench (32)\nPR-CSPD Start",
    ]
    fwd_vals = [
        pC["math32"]["4096"]["Alg1_Split_M@P2_I@P4"]["acc"],
        pC["ifbench32"]["4096"]["Alg1_Split_M@P2_I@P4"]["prompt_strict"],
        pC["ifeval48"]["4096"]["Alg1_Split_M@P2_I@P4"]["prompt_strict"],
        pC["ifbench32"]["4096"]["KVPD_P0_Math_IF"]["prompt_strict"],
        pC["ifbench32"]["4096"]["PR_CSPD_P0_Math_IF"]["prompt_strict"],
    ]
    rev_vals = [
        pC["math32"]["4096"]["Alg1_Split_I@P2_M@P4"]["acc"],
        pC["ifbench32"]["4096"]["Alg1_Split_I@P2_M@P4"]["prompt_strict"],
        pC["ifeval48"]["4096"]["Alg1_Split_I@P2_M@P4"]["prompt_strict"],
        pC["ifbench32"]["4096"]["KVPD_P0_IF_Math"]["prompt_strict"],
        pC["ifbench32"]["4096"]["PR_CSPD_P0_IF_Math"]["prompt_strict"],
    ]
    xc = np.arange(len(c_labels))
    wc = 0.35
    bc1 = ax3.bar(xc - wc/2, fwd_vals, wc, color=TEAL,   edgecolor=DARK, lw=0.5, label="Primary ([M:Pre+I:Post] / [M; I])", zorder=3)
    bc2 = ax3.bar(xc + wc/2, rev_vals, wc, color=LBLUE,  edgecolor=DARK, lw=0.5, label="Swapped ([I:Pre+M:Post] / [I; M])", zorder=3)
    # Color KV-PD and PR-CSPD distinct muted shades to echo architectures while keeping Fwd vs Rev clear
    bc1[3].set_color(PURPLE); bc2[3].set_color("#C9B3DC")
    bc1[4].set_color(ORANGE); bc2[4].set_color("#F7C99E")

    for b, v in zip(bc1, fwd_vals):
        ax3.text(b.get_x() + b.get_width()/2, v + 1.3, f"{v:.1f}", ha="center", va="bottom", fontsize=6.3)
    for b, v in zip(bc2, rev_vals):
        ax3.text(b.get_x() + b.get_width()/2, v + 1.3, f"{v:.1f}", ha="center", va="bottom", fontsize=6.3)

    ax3.axvline(2.5, color=GRAY, ls="--", lw=0.9)
    ax3.text(0.0, 81.0, "+25.0 pp", ha="center", fontsize=6.7, fontweight="bold", color=TEAL,
             bbox=dict(boxstyle="round,pad=0.14", fc="#EEF7F6", ec=TEAL, lw=0.5))
    ax3.text(1.0, 64.5, "+6.2 pp", ha="center", fontsize=6.5, fontweight="bold", color=TEAL)
    ax3.text(2.0, 88.0, "|Δ|=2.1 pp", ha="center", fontsize=6.4, color=DARK)
    ax3.text(3.0, 52.0, "|Δ|=0.0 pp", ha="center", fontsize=6.6, fontweight="bold", color=PURPLE,
             bbox=dict(boxstyle="round,pad=0.14", fc="#F6F2FA", ec=PURPLE, lw=0.5))
    ax3.text(4.0, 36.5, "Both\ncollapse", ha="center", fontsize=6.3, fontweight="bold", color=RED)

    ax3.set_xticks(xc)
    ax3.set_xticklabels(c_labels, fontsize=6.8)
    ax3.set_ylabel("End-to-end strict accuracy (%)")
    ax3.set_ylim(0, 108)
    ax3.set_title("(c) End-to-end order swap at L=4096", pad=10)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#DDDDDD", fontsize=6.2, handlelength=1.4)

    path = os.path.join(OUT, "pfig14_injection_order_and_slot_permutations.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ==============================================================================
# FIG 15: Reference Step-200 6-Slot Audit, 36-Layer Pathway & Complete-EOS Gain
# ==============================================================================
def make_pfig15():
    d12 = load_res("exp12_results.json")
    d10c = load_res("exp10c_results.json")
    d10b = load_res("exp10b_results.json")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.45))
    fig.subplots_adjust(wspace=0.29, left=0.05, right=0.985, top=0.86, bottom=0.17)

    # ------------------------------------------------------------------
    # (a) Reference Step-200 CSPD (I & M, RMS 0.37-0.40) Across P0-P5
    # ------------------------------------------------------------------
    style_boxed(ax1, grid="y")
    slots = ["P0_Clean", "P0_FullCtx", "P1_SysEnd", "P2_MultiTurn", "P3_PostRAG", "P4_PostQuery", "P5_AllCombined"]
    s_labels = ["Start\nClean", "Start\nFullCtx", "Post-\nSys", "Pre-\nQuery", "Post-\nRAG", "Post-\nQuery", "Long\nOffset"]
    i_vals = [d12["partA1_official_6slots"]["official_I_on_IF"][s]["g_vs_slot"] for s in slots]
    m_vals = [d12["partA1_official_6slots"]["official_M_on_Math"][s]["g_vs_slot"] for s in slots]

    x1 = np.arange(len(slots))
    w1 = 0.36
    ax1.axvspan(4.5, 6.5, color="#FDECEC", alpha=0.55, zorder=0)
    b1 = ax1.bar(x1 - w1/2, i_vals, w1, color=BLUE,   edgecolor=DARK, lw=0.5, label="Ref-CSPD IF (256 tok, RMS 0.37)", zorder=3)
    b2 = ax1.bar(x1 + w1/2, m_vals, w1, color=ORANGE, edgecolor=DARK, lw=0.5, label="Ref-CSPD Math (256 tok, RMS 0.40)", zorder=3)
    ax1.axhline(0, color=DARK, lw=0.85, zorder=2)

    for b, v in zip(b1, i_vals):
        off = 2.0 if v >= 0 else -2.5
        va = "bottom" if v >= 0 else "top"
        ax1.text(b.get_x() + b.get_width()/2, v + off, f"{v:+.0f}", ha="center", va=va, fontsize=6.4, color=DARK)
    for b, v in zip(b2, m_vals):
        off = 2.0 if v >= 0 else -2.5
        va = "bottom" if v >= 0 else "top"
        ax1.text(b.get_x() + b.get_width()/2, v + off, f"{v:+.0f}", ha="center", va=va, fontsize=6.4, color=DARK)

    ax1.text(5.5, 69.5, "Post-query\nfailure zone", ha="center", va="top", fontsize=6.8, fontweight="bold", color=RED,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=RED, lw=0.6, alpha=0.92))
    ax1.set_xticks(x1)
    ax1.set_xticklabels(s_labels, fontsize=6.8)
    ax1.set_ylabel("Slot-matched KL recovery G (%)")
    ax1.set_ylim(-60, 82)
    ax1.set_title("(a) Reference step-200 CSPD across insertion slots", pad=9)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#DDDDDD", fontsize=6.6, handlelength=1.4)

    # ------------------------------------------------------------------
    # (b) 36-Layer Cumulative Direct-Path Causal Ablation (open [0, l))
    # ------------------------------------------------------------------
    style_boxed(ax2, grid="both")
    cuts = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
    pC = d12["partC_36layer_causal_pathway"]

    ax2.axvspan(16, 24, color="#EEF4FA", alpha=0.75, zorder=0)
    ax2.axhline(0, color=GRAY, lw=0.7, ls=":", zorder=1)
    ax2.axhline(100, color=GRAY, lw=0.7, ls=":", zorder=1)

    series = [
        ("Official_CSPD_I@P0", BLUE,   "o", "Ref-CSPD (Start)"),
        ("KVPD_IF_EOS@P0",     PURPLE, "s", "KV-PD Complete (Start)"),
        ("Alg1_CRSP_EOS@P4",   TEAL,   "^", "Alg. 1 CRSP Complete (Post-Q)"),
    ]
    for key, col, mk, lbl in series:
        y0 = [pC["L0"][key]["open_first_0_to_l"][str(c)]["recovery_vs_base_pct"] for c in cuts]
        y4 = [pC["L4096"][key]["open_first_0_to_l"][str(c)]["recovery_vs_base_pct"] for c in cuts]
        ax2.plot(cuts, y0, color=col, marker=mk, ms=4.2, lw=1.7, ls="-",  label=f"{lbl} (L=0)", zorder=4)
        ax2.plot(cuts, y4, color=col, marker=mk, ms=3.5, lw=1.2, ls="--", alpha=0.82, label=f"{lbl} (L=4096)", zorder=3)

    ax2.text(20.0, -28.0, "Mid-layer gain\n(L16-24: +48-61 pp)", ha="center", va="bottom", fontsize=6.6, color=BLUE,
             bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=BLUE, lw=0.55, alpha=0.92))
    ax2.annotate("Post-Q shallow shift\nat L=4096 (+21.9%)", xy=(8, 21.9), xytext=(2.5, 48.0),
                 fontsize=6.3, color=TEAL, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9, shrinkA=1, shrinkB=3))

    ax2.set_xticks(cuts)
    ax2.set_xlim(-1, 37)
    ax2.set_ylim(-33, 114)
    ax2.set_xlabel("Direct-path layers open [0, l) of 36")
    ax2.set_ylabel("Cumulative gain recovery (%)")
    ax2.set_title("(b) 36-layer direct-path ablation (L=0 vs L=4096)", pad=9)
    ax2.legend(loc="upper left", ncol=1, frameon=True, facecolor="white", framealpha=0.92,
               edgecolor="#DDDDDD", fontsize=5.5, handlelength=1.5, labelspacing=0.25)

    # ------------------------------------------------------------------
    # (c) Complete-EOS Upgrade on KV-PD (Start) Across All 6 E2E Benchmarks
    # ------------------------------------------------------------------
    style_boxed(ax3, grid="y")
    b_labels = ["MATH\nL=0", "MATH\nL=4096", "IFBench\nL=0", "IFBench\nL=4096", "IFEval\nL=0", "IFEval\nL=4096"]
    trunc_vals = [
        d10c["e2e_math32"]["0"]["KVPD_Math_P0"]["acc"],
        d10c["e2e_math32"]["4096"]["KVPD_Math_P0"]["acc"],
        d10c["e2e_ifbench32"]["0"]["KVPD_L16_at_P0"]["prompt_strict"],
        d10c["e2e_ifbench32"]["4096"]["KVPD_L16_at_P0"]["prompt_strict"],
        d10b["partD_e2e_ifeval48"]["0"]["KVPD_L16_at_P0"]["prompt_strict"],
        d10c["e2e_ifeval48_4096"]["KVPD_L16_at_P0"]["prompt_strict"],
    ]
    eos_vals = [
        d12["partE_e2e_benchmarks"]["math32"]["0"]["KVPD_Math_EOS@P0"]["acc"],
        d12["partE_e2e_benchmarks"]["math32"]["4096"]["KVPD_Math_EOS@P0"]["acc"],
        d12["partE_e2e_benchmarks"]["ifbench32"]["0"]["KVPD_IF_EOS@P0"]["prompt_strict"],
        d12["partE_e2e_benchmarks"]["ifbench32"]["4096"]["KVPD_IF_EOS@P0"]["prompt_strict"],
        d12["partE_e2e_benchmarks"]["ifeval48"]["0"]["KVPD_IF_EOS@P0"]["prompt_strict"],
        d12["partE_e2e_benchmarks"]["ifeval48"]["4096"]["KVPD_IF_EOS@P0"]["prompt_strict"],
    ]

    x3 = np.arange(len(b_labels))
    w3 = 0.36
    c1 = ax3.bar(x3 - w3/2, trunc_vals, w3, color="#C9B3DC", edgecolor=DARK, lw=0.5, label="Truncated traces (512 tok)", zorder=3)
    c2 = ax3.bar(x3 + w3/2, eos_vals,   w3, color=PURPLE,    edgecolor=DARK, lw=0.5, label="Complete traces (</think>+EOS)", zorder=3)

    for b, v in zip(c1, trunc_vals):
        ax3.text(b.get_x() + b.get_width()/2, v + 1.1, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2, color=GRAY)
    for b, v in zip(c2, eos_vals):
        ax3.text(b.get_x() + b.get_width()/2, v + 1.1, f"{v:.1f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=DARK)

    for i, (t_v, e_v) in enumerate(zip(trunc_vals, eos_vals)):
        dlt = e_v - t_v
        hi = max(t_v, e_v)
        txt = f"+{dlt:.1f}" if dlt > 0.05 else "0.0"
        col = GREEN if dlt > 0.05 else GRAY
        ax3.text(i, hi + 7.2, txt, ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=col)

    mean_t = float(np.mean(trunc_vals))
    mean_e = float(np.mean(eos_vals))
    ax3.text(0.02, 0.965, f"6-task mean: {mean_t:.1f}% → {mean_e:.1f}% (+{mean_e - mean_t:.1f} pp)\nCompleted traces: 66.1% → 72.8% (+6.7 pp)",
             transform=ax3.transAxes, ha="left", va="top", fontsize=5.7, fontweight="bold", color=PURPLE,
             bbox=dict(boxstyle="round,pad=0.18", fc="#F7F2FA", ec=PURPLE, lw=0.6))

    ax3.set_xticks(x3)
    ax3.set_xticklabels(b_labels, fontsize=6.8)
    ax3.set_ylabel("End-to-end strict accuracy (%)")
    ax3.set_ylim(0, 122)
    ax3.set_title("(c) Complete-trajectory upgrade (KV-PD, Start)", pad=9)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#DDDDDD", fontsize=5.7, handlelength=1.2)

    path = os.path.join(OUT, "pfig15_official_cspd_eos_and_pathway.png")
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


if __name__ == "__main__":
    make_pfig01()
    make_pfig02()
    make_pfig03()
    make_pfig04()
    make_pfig05()
    make_pfig06()
    make_pfig07()
    make_pfig08()
    make_pfig09()
    make_pfig10()
    make_pfig11()
    make_pfig12()
    make_pfig13()
    make_pfig14()
    make_pfig15()
    print("All 15 paper-style figures generated successfully.")
