#!/usr/bin/env python3
# Usage: python data/scripts/make_composite_figure.py
"""
make_composite_figure.py — Dense 2×2 panel figure for CausalWorlds 2026.

Auto-detects the latest M snapshot from the data directory.
Output: paper/figures/fig_composite.pdf

Panels:
  (a) Spectral dimension flow (vacuum + defect, σ ≤ 15)
  (b) Mass spectrum (log-scale belly histogram + exponential fit)
  (c) Mass bars (Gen 1/2/3 + Anti-1, CPT arrow)
  (d) Cumulative mass fraction (visible/dark split)
"""

import re
import pathlib
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "ensemble_10M_final"
OUT = REPO_ROOT / "data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Auto-detect latest M snapshot ────────────────────────────────────────────
results_files = sorted(glob(str(DATA / "results_M*.csv")))
if not results_files:
    raise FileNotFoundError(f"No results_M*.csv found in {DATA}")
latest_M = max(int(re.search(r'M(\d+)', f).group(1)) for f in results_files)
pad = f"{latest_M:02d}"
print(f"[*] Using M={latest_M} snapshot from {DATA}")

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA / f"results_M{pad}.csv", comment="#")
ms = pd.read_csv(DATA / f"mass_spectrum_M{pad}.csv", comment="#")
ts = pd.read_csv(DATA / f"topology_summary_M{pad}.csv", comment="#")
td = dict(zip(ts["key"], ts["value"]))

sigma = df["step"].values
N_int = ms["intermediates_N"].values
freq  = ms["frequency"].values

m_gen1 = float(td["avg_mass_gen1"])
m_gen2 = float(td["avg_mass_gen2"])
m_gen3 = float(td["avg_mass_gen3"])
m_anti1 = float(df["Mass_Anti1"].iloc[0])

# ── Colourblind-safe palette ─────────────────────────────────────────────────
C_VAC  = "#4C72B0"
C_DEF  = "#C44E52"
C_GEN1 = "#55A868"
C_GEN2 = "#8172B2"
C_GEN3 = "#CCB974"
C_ANTI = "#64B5CD"
C_VIS  = "#E8A838"
C_DARK = "#2f2f2f"

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "lines.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.12,
    "grid.linewidth": 0.3,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ═════════════════════════════════════════════════════════════════════════════
# Figure
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0))
(ax_a, ax_b), (ax_c, ax_d) = axes

# ── (a) Spectral dimension flow ──────────────────────────────────────────────
mask = sigma <= 15
s = sigma[mask]

y_vac = df["dS_vac"].values[mask]
e_vac = df["dS_vac_std"].values[mask]
y_def = df["dS_def"].values[mask]
e_def = df["dS_def_std"].values[mask]

ax_a.plot(s, y_vac, color=C_VAC, lw=1.0, label="Vacuum", zorder=5)
ax_a.fill_between(s, y_vac - e_vac, y_vac + e_vac, color=C_VAC, alpha=0.15, zorder=2)
ax_a.plot(s, y_def, color=C_DEF, lw=1.0, label="Defect", zorder=5)
ax_a.fill_between(s, y_def - e_def, y_def + e_def, color=C_DEF, alpha=0.15, zorder=2)

ax_a.axhline(4.0, color="grey", ls="--", lw=0.5, alpha=0.6, zorder=1)
ax_a.axhline(2.0, color="grey", ls=":", lw=0.4, alpha=0.5, zorder=1)

ax_a.set_xlabel(r"Diffusion time $\sigma$")
ax_a.set_ylabel(r"$d_S(\sigma)$")
ax_a.set_xlim(1, 15)
ax_a.set_ylim(0, 10)
ax_a.legend(loc="upper right", frameon=True, framealpha=0.9,
            edgecolor="0.85", handlelength=1.2)

# Knockout annotation — data-driven
ds_uv_str = f"{y_vac[0]:.2f}"
ds_ir_peak = f"{y_vac.max():.2f}"
ax_a.text(0.03, 0.97, rf"$d_S\!=\!{ds_uv_str} \to {ds_ir_peak}$",
          transform=ax_a.transAxes, fontsize=6, fontweight="bold",
          va="top", ha="left", color="0.15",
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))

# Panel label
ax_a.text(0.03, 0.03, rf"$\mathbf{{(a)}}\ M\!={latest_M}$", transform=ax_a.transAxes,
          fontsize=7, va="bottom", ha="left")

# ── (b) Mass spectrum (log scale) ───────────────────────────────────────────
# Phase-coherence: colour by expected opacity Q ~ 1/sqrt(N)
# No hard N=5 boundary — continuous gradient from bright to dark
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
opacity = 1.0 / np.sqrt(N_int.astype(float))
opacity_norm = opacity / opacity.max()
bar_colors = [plt.cm.copper_r(q) for q in opacity_norm]

ax_b.bar(N_int, freq, width=0.8, color=bar_colors,
         edgecolor="white", lw=0.3, zorder=3)
ax_b.set_yscale("log")

# Exponential tail fit (n >= 7)
tail_mask = N_int >= 7
n_tail = N_int[tail_mask].astype(float)
f_tail = freq[tail_mask].astype(float)
valid = f_tail > 0
if np.sum(valid) >= 3:
    log_f = np.log(f_tail[valid])
    slope, intercept = np.polyfit(n_tail[valid], log_f, 1)
    n_fit = np.linspace(7, N_int.max(), 100)
    ax_b.plot(n_fit, np.exp(intercept + slope * n_fit),
              color="red", ls="--", lw=0.8,
              label=rf"$\sim e^{{{slope:.2f}n}}$", zorder=5)

ax_b.set_xlabel(r"Belly size $n$")
ax_b.set_ylabel("Frequency")
ax_b.set_xlim(2, 31)
ax_b.legend(loc="upper right", frameon=True, framealpha=0.9,
            edgecolor="0.85", handlelength=1.2)

# Knockout annotation — data-driven
total_prisms = int(float(td["total_prisms"]))
ax_b.text(0.03, 0.97, f"${total_prisms:,}$ prisms".replace(",", "{,}"),
          transform=ax_b.transAxes, fontsize=6, fontweight="bold",
          va="top", ha="left", color="0.15",
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))

ax_b.text(0.03, 0.03, rf"$\mathbf{{(b)}}\ M\!={latest_M}$", transform=ax_b.transAxes,
          fontsize=7, va="bottom", ha="left")

# ── (c) Mass bars + CPT ─────────────────────────────────────────────────────
labels = ["Gen 1", "Gen 2", "Gen 3", "Anti-1"]
masses = [m_gen1, m_gen2, m_gen3, m_anti1]
colors = [C_GEN1, C_GEN2, C_GEN3, C_ANTI]

x_pos = np.arange(len(labels))
bars = ax_c.bar(x_pos, masses, width=0.65, color=colors, edgecolor="white", lw=0.4, zorder=3)

# CPT deviation arrow between Gen1 and Anti-1
cpt_dev = abs(m_gen1 - m_anti1) / m_gen1 * 100
mid_y = max(m_gen1, m_anti1) + 0.25
ax_c.annotate("", xy=(3, mid_y), xytext=(0, mid_y),
              arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.7))
ax_c.text(1.5, mid_y + 0.25, rf"$\Delta m/m = {cpt_dev:.1f}\%$",
          ha="center", fontsize=5.5, color="0.3")

ax_c.set_ylabel("Topological mass")
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(labels, fontsize=6)
ax_c.set_ylim(0, 9.5)

# Knockout annotation
ax_c.text(0.97, 0.97, rf"$\Delta m/m = {cpt_dev:.1f}\%$",
          transform=ax_c.transAxes, fontsize=6, fontweight="bold",
          va="top", ha="right", color="0.15",
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))

ax_c.text(0.03, 0.03, rf"$\mathbf{{(c)}}\ M\!={latest_M}$", transform=ax_c.transAxes,
          fontsize=7, va="bottom", ha="left")

# ── (d) Cumulative mass fraction ────────────────────────────────────────────
mass_contribution = N_int * freq
total_mass = mass_contribution.sum()
cumulative = np.cumsum(mass_contribution) / total_mass

ax_d.plot(N_int, cumulative, "ko-", markersize=2, lw=1.0, zorder=5)

# Phase-coherence: read Ω and α directly from topology_summary (zero free parameters)
if "omega_ratio" in td:
    ratio_dm = float(td["omega_ratio"])
    alpha_em = float(td.get("alpha_em", 0))
    vis_total = float(td.get("visible_mass_total", 0))
    grav_total = float(td.get("grav_mass_total", 0))
    vis_frac = vis_total / grav_total if grav_total > 0 else 0
    dark_frac = 1.0 - vis_frac
else:
    # Fallback for old data (N>5 boundary)
    vis_mass = mass_contribution[N_int <= 5].sum()
    dark_mass = mass_contribution[N_int > 5].sum()
    vis_frac = vis_mass / total_mass
    dark_frac = dark_mass / total_mass
    ratio_dm = dark_mass / vis_mass
    alpha_em = 0.0

# Shade cumulative curve with gradient (no hard boundary)
ax_d.fill_between(N_int, 0, cumulative, color=C_VIS, alpha=0.10, zorder=2)

ax_d.axhline(vis_frac, color=C_VIS, ls=":", lw=0.6, alpha=0.7)
ax_d.text(20, vis_frac + 0.02, f"Vis: {vis_frac:.1%}", fontsize=5, color=C_VIS)
ax_d.text(20, vis_frac + 0.09, f"Dark: {dark_frac:.1%}", fontsize=5, color="0.4")

ax_d.set_xlabel(r"Belly size $n$")
ax_d.set_ylabel("Cumulative mass fraction")
ax_d.set_xlim(3, N_int.max())
ax_d.set_ylim(0, 1.05)

# Knockout annotation — phase-coherence Ω (zero free parameters)
omega_label = rf"$\Omega_d/\Omega_v = {ratio_dm:.2f}$"
if alpha_em > 0:
    omega_label += rf"$\quad \alpha = {alpha_em:.4f}$"
ax_d.text(0.97, 0.50, omega_label,
          transform=ax_d.transAxes, fontsize=6, fontweight="bold",
          va="center", ha="right", color="0.15",
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))

ax_d.text(0.03, 0.03, rf"$\mathbf{{(d)}}\ M\!={latest_M}$", transform=ax_d.transAxes,
          fontsize=7, va="bottom", ha="left")

# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.5)
out_path = OUT / "fig_composite.pdf"
fig.savefig(out_path)
fig.savefig(OUT / "fig_composite.png")
plt.close(fig)
print(f"[+] {out_path}")
print(f"[+] {OUT / 'fig_composite.png'}")
