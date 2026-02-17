#!/usr/bin/env python3
"""
Publication-quality spectral dimension plot for FEG Kuratowski simulation.
Focuses on the physically meaningful early transient (t ≤ 30).
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

# ── Style ───────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "legend.framealpha": 0.85, "legend.edgecolor": "0.7",
    "figure.dpi": 200,
})

BG         = "#0d1117"
GRID       = "#1c2333"
VAC_G      = "#58a6ff"
VAC_L      = "#79c0ff"
DEF_G      = "#f97583"
DEF_L      = "#ffa198"
GREEN      = "#56d364"
GOLD       = "#e3b341"
GREY       = "#8b949e"

# ── Read ────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
data = {"step":[], "P_vac":[], "P_def":[], "P_loc_vac":[], "P_loc_def":[],
        "dS_vac":[], "dS_def":[], "dS_loc_vac":[], "dS_loc_def":[]}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        for k in data:
            data[k].append(float(row[k]) if k != "step" else int(row[k]))

t  = np.array(data["step"])

# ── Cut to early transient (t ≤ 30) where P(t) > noise floor ──
mask = t <= 30
t_e  = t[mask]

# ── Figure ──────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 9), facecolor=BG,
    gridspec_kw={"height_ratios": [3, 2], "hspace": 0.28}
)

for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", which="both")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.6)

# ═══ Panel 1: d_S(t) – early transient ═══════════════
ds_vg = np.array(data["dS_vac"])[mask]
ds_dg = np.array(data["dS_def"])[mask]
ds_vl = np.array(data["dS_loc_vac"])[mask]
ds_dl = np.array(data["dS_loc_def"])[mask]

# Clip for visual clarity
clip = lambda a: np.clip(a, -0.5, 15)

ax1.plot(t_e, clip(ds_vg), color=VAC_G, lw=2.2, ls="-",  marker="o", ms=5,
         label=r"$d_S$ vacuum (global)", zorder=4)
ax1.plot(t_e, clip(ds_dg), color=DEF_G, lw=2.2, ls="-",  marker="s", ms=5,
         label=r"$d_S$ defect (global)", zorder=4)
ax1.plot(t_e, clip(ds_vl), color=VAC_L, lw=1.6, ls="--", marker="o", ms=3.5,
         label=r"$d_S$ vacuum (core)", alpha=0.8, zorder=3)
ax1.plot(t_e, clip(ds_dl), color=DEF_L, lw=1.6, ls="--", marker="s", ms=3.5,
         label=r"$d_S$ defect (core)", alpha=0.8, zorder=3)

# Reference bands
ax1.axhspan(3.8, 4.2, color=GREEN, alpha=0.08, zorder=0)
ax1.axhline(y=4, color=GREEN, ls=":", lw=1.2, alpha=0.7, label=r"$d_S=4$ (UV)")
ax1.axhspan(2.8, 3.2, color=GOLD, alpha=0.08, zorder=0)
ax1.axhline(y=3, color=GOLD, ls=":", lw=1.2, alpha=0.7, label=r"$d_S=3$ (IR target)")
ax1.axhline(y=2, color=GREY, ls=":", lw=1.0, alpha=0.4, label=r"$d_S=2$ (CDT)")

ax1.set_xlabel(r"Diffusion time $t$", fontsize=12)
ax1.set_ylabel(r"Spectral dimension $d_S(t)$", fontsize=13)
ax1.set_title(
    r"$\mathbf{Spectral\ Dimension\ Flow}$ — FEG Kuratowski Simulation"
    "\n"
    r"$N\!=\!50{,}000$,  $M\!=\!10$,  Pure Integer Causal Prism ($K_{2,N}$) Defect",
    fontsize=13, pad=14
)
ax1.set_xlim(0, 32)
ax1.set_ylim(-0.5, 15)
ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.yaxis.set_major_locator(MultipleLocator(2))
ax1.legend(loc="upper right", fontsize=8.5, ncol=2,
           facecolor=BG, labelcolor="white")

# Annotate the UV→IR flow
ax1.annotate(
    r"UV $\to$ IR flow", xy=(6, 7.5), fontsize=10, color="white",
    fontstyle="italic", alpha=0.6,
    arrowprops=dict(arrowstyle="->", color="white", alpha=0.4),
    xytext=(12, 13)
)

# ═══ Panel 2: P(t) log-scale ═════════════════════════
p_vg = np.array(data["P_vac"])[mask]
p_dg = np.array(data["P_def"])[mask]
p_vl = np.array(data["P_loc_vac"])[mask]
p_dl = np.array(data["P_loc_def"])[mask]

# Replace exact zeros for log scale
floor = 1e-5
p_vl = np.where(p_vl > 0, p_vl, floor)
p_dl = np.where(p_dl > 0, p_dl, floor)

ax2.semilogy(t_e, p_vg, color=VAC_G, lw=2.2, marker="o", ms=4.5,
             label=r"$P(t)$ vacuum (global)", zorder=4)
ax2.semilogy(t_e, p_dg, color=DEF_G, lw=2.2, marker="s", ms=4.5,
             label=r"$P(t)$ defect (global)", zorder=4)
ax2.semilogy(t_e, p_vl, color=VAC_L, lw=1.5, ls="--", marker="o", ms=3,
             label=r"$P(t)$ vacuum (core)", alpha=0.8, zorder=3)
ax2.semilogy(t_e, p_dl, color=DEF_L, lw=1.5, ls="--", marker="s", ms=3,
             label=r"$P(t)$ defect (core)", alpha=0.8, zorder=3)

ax2.set_xlabel(r"Diffusion time $t$", fontsize=12)
ax2.set_ylabel(r"Return probability $P(t)$", fontsize=13)
ax2.set_title(
    r"Return Probability — Vacuum vs Causal Prism Defect Core",
    fontsize=12, pad=8
)
ax2.set_xlim(0, 32)
ax2.xaxis.set_major_locator(MultipleLocator(5))
ax2.legend(loc="upper right", fontsize=8.5, ncol=2,
           facecolor=BG, labelcolor="white")

# ── Info box ────────────────────────────────────────
info = (
    r"$\bullet$ Phase 2: Pure integer Kuratowski contraction"  "\n"
    r"$\bullet$ $\sim\!1{,}120$ Causal Prisms ($K_{2,N}$) per realisation"  "\n"
    r"$\bullet$ Phase 3: MC walkers ($W\!=\!5000$)"  "\n"
    r"$\bullet$ Total runtime: 1 min 39 s"
)
fig.text(0.03, 0.01, info, fontsize=7.5, color=GREY, va="bottom",
         family="monospace")

# ── Save ────────────────────────────────────────────
out = csv_path.replace(".csv", "") + "_spectral_dimension.png"
fig.savefig(out, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
print(f"Saved: {out}")
plt.close()
