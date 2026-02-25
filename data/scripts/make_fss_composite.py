#!/usr/bin/env python3
# Usage: python data/scripts/make_fss_composite.py
"""
make_fss_composite.py — FSS 2×2 panel figure for CausalWorlds 2026.

Panels:
  (a) Q_topo vs N^{-1/4}  — hero plot, linear extrapolation
  (b) Mass hierarchy vs N^{-1/4} — flat = topological invariants
  (c) Omega_energy vs N^{-1/4} — convergence to Planck 2018
  (d) Spectral dimension flow (vacuum + defect, σ ≤ 15)

Output: paper/figures/fig_fss_composite.pdf
"""

import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FSS_JSON = REPO_ROOT / "data" / "fss_scaling" / "fss_comprehensive_results.json"
CSV_FILE = REPO_ROOT / "data" / "ensemble_10M_final" / "results_M20.csv"
OUT = REPO_ROOT / "data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
with open(FSS_JSON) as f:
    fss = json.load(f)

dp = fss["data_points"]
fits = fss["fits"]
derived = fss["derived"]

N_vals = np.array([d["N"] for d in dp], dtype=float)
x_vals = N_vals ** (-0.25)  # N^{-1/4}

Q_vals = np.array([d["Q_topo"] for d in dp])
Om_vals = np.array([d["Omega_energy"] for d in dp])
m1_vals = np.array([d["mass_gen1"] for d in dp])
m2_vals = np.array([d["mass_gen2"] for d in dp])
m3_vals = np.array([d["mass_gen3"] for d in dp])

df = pd.read_csv(CSV_FILE, comment="#")

# ── Colourblind-safe palette ─────────────────────────────────────────────────
C_VAC  = "#4C72B0"
C_DEF  = "#C44E52"
C_GEN1 = "#55A868"
C_GEN2 = "#8172B2"
C_GEN3 = "#CCB974"
C_FIT  = "#2f2f2f"
C_PLANCK = "#E8A838"

# ── Style (matches make_composite_figure.py) ─────────────────────────────────
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

x_fit = np.linspace(0, x_vals.max() * 1.05, 100)

# ── (a) Q_topo vs N^{-1/4} ──────────────────────────────────────────────────
Q_fit = fits["Q_topo"]
Q_inf = Q_fit["O_inf"]
Q_a = Q_fit["a"]
R2_Q = Q_fit["R_sq"]

ax_a.plot(x_vals, Q_vals, "o", color=C_VAC, markersize=3.5, zorder=5)
ax_a.plot(x_fit, Q_inf + Q_a * x_fit, "--", color=C_FIT, lw=0.8, zorder=3)
ax_a.axhline(Q_inf, color="grey", ls=":", lw=0.4, alpha=0.6, zorder=1)

# Mark the continuum limit
ax_a.plot(0, Q_inf, "s", color=C_DEF, markersize=4, zorder=6, clip_on=False)

ax_a.set_xlabel(r"$N^{-1/4}$")
ax_a.set_ylabel(r"$\mathcal{Q}_{\mathrm{topo}}$")
ax_a.set_xlim(-0.003, x_vals.max() * 1.08)
ax_a.set_ylim(0.13, 0.30)

ax_a.text(0.97, 0.97,
          rf"$\mathcal{{Q}}_\infty = {Q_inf:.3f}$" + "\n" +
          rf"$R^2 = {R2_Q:.4f}$",
          transform=ax_a.transAxes, fontsize=5.5, va="top", ha="right",
          color="0.15",
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))

ax_a.text(0.03, 0.03, r"$\mathbf{(a)}$", transform=ax_a.transAxes,
          fontsize=7, va="bottom", ha="left")

# ── (b) Mass hierarchy across N ─────────────────────────────────────────────
ax_b.plot(x_vals, m1_vals, "o-", color=C_GEN1, markersize=3, lw=0.8,
          label="Gen 1", zorder=5)
ax_b.plot(x_vals, m2_vals, "s-", color=C_GEN2, markersize=3, lw=0.8,
          label="Gen 2", zorder=5)
ax_b.plot(x_vals, m3_vals, "^-", color=C_GEN3, markersize=3, lw=0.8,
          label="Gen 3", zorder=5)

# Horizontal fit lines (extrapolated values)
for fit_key, color in [("mass_gen1", C_GEN1), ("mass_gen2", C_GEN2),
                        ("mass_gen3", C_GEN3)]:
    ax_b.axhline(fits[fit_key]["O_inf"], color=color, ls=":", lw=0.4, alpha=0.6)

ax_b.set_xlabel(r"$N^{-1/4}$")
ax_b.set_ylabel("Topological mass")
ax_b.set_xlim(-0.003, x_vals.max() * 1.08)
ax_b.set_ylim(3.5, 8.5)
ax_b.legend(loc="upper right", frameon=True, framealpha=0.9,
            edgecolor="0.85", handlelength=1.2)

ax_b.text(0.03, 0.97,
          r"Flat: $R^2 \leq 0.79$",
          transform=ax_b.transAxes, fontsize=5.5, va="top", ha="left",
          color="0.15",
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))

ax_b.text(0.03, 0.03, r"$\mathbf{(b)}$", transform=ax_b.transAxes,
          fontsize=7, va="bottom", ha="left")

# ── (c) Omega_energy convergence ─────────────────────────────────────────────
Om_fit = fits["Omega_energy"]
Om_inf_direct = Om_fit["O_inf"]
Om_a = Om_fit["a"]
R2_Om = Om_fit["R_sq"]
Om_inf_derived = derived["Omega_inf_from_Q"]

ax_c.plot(x_vals, Om_vals, "o", color=C_VAC, markersize=3.5, zorder=5)
ax_c.plot(x_fit, Om_inf_direct + Om_a * x_fit, "--", color=C_FIT, lw=0.8,
          zorder=3, label=rf"Direct: $\Omega_\infty={Om_inf_direct:.2f}$")

# Derived from Q_topo extrapolation (more precise)
ax_c.axhline(Om_inf_derived, color=C_DEF, ls="-.", lw=0.6, alpha=0.8,
             zorder=2, label=rf"From $\mathcal{{Q}}_\infty$: ${Om_inf_derived:.2f}$")

# Planck 2018 reference
PLANCK_OMEGA = 5.36
ax_c.axhline(PLANCK_OMEGA, color=C_PLANCK, ls="--", lw=0.7, alpha=0.8,
             zorder=2, label=f"Planck 2018: {PLANCK_OMEGA}")

ax_c.set_xlabel(r"$N^{-1/4}$")
ax_c.set_ylabel(r"$\Omega_{\mathrm{energy}}$")
ax_c.set_xlim(-0.003, x_vals.max() * 1.08)
ax_c.set_ylim(2.0, 6.5)
ax_c.legend(loc="upper right", frameon=True, framealpha=0.9,
            edgecolor="0.85", handlelength=1.2, fontsize=5)

ax_c.text(0.03, 0.03, r"$\mathbf{(c)}$", transform=ax_c.transAxes,
          fontsize=7, va="bottom", ha="left")

# ── (d) Spectral dimension flow ──────────────────────────────────────────────
sigma = df["step"].values
mask = sigma <= 15
s = sigma[mask]

y_vac = df["dS_vac"].values[mask]
e_vac = df["dS_vac_std"].values[mask]
y_def = df["dS_def"].values[mask]
e_def = df["dS_def_std"].values[mask]

ax_d.plot(s, y_vac, color=C_VAC, lw=1.0, label="Vacuum", zorder=5)
ax_d.fill_between(s, y_vac - e_vac, y_vac + e_vac, color=C_VAC,
                   alpha=0.15, zorder=2)
ax_d.plot(s, y_def, color=C_DEF, lw=1.0, label="Defect", zorder=5)
ax_d.fill_between(s, y_def - e_def, y_def + e_def, color=C_DEF,
                   alpha=0.15, zorder=2)

ax_d.axhline(4.0, color="grey", ls="--", lw=0.5, alpha=0.6, zorder=1)
ax_d.axhline(2.0, color="grey", ls=":", lw=0.4, alpha=0.5, zorder=1)

ax_d.set_xlabel(r"Diffusion time $\sigma$")
ax_d.set_ylabel(r"$d_S(\sigma)$")
ax_d.set_xlim(1, 15)
ax_d.set_ylim(0, 10)
ax_d.legend(loc="upper right", frameon=True, framealpha=0.9,
            edgecolor="0.85", handlelength=1.2)

ds_uv_str = f"{y_vac[0]:.2f}"
ds_ir_peak = f"{y_vac.max():.2f}"
ax_d.text(0.03, 0.97, rf"$d_S\!=\!{ds_uv_str} \to {ds_ir_peak}$",
          transform=ax_d.transAxes, fontsize=6, fontweight="bold",
          va="top", ha="left", color="0.15",
          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))

ax_d.text(0.03, 0.03, rf"$\mathbf{{(d)}}\ N\!=\!10^7,\,M\!=\!20$",
          transform=ax_d.transAxes, fontsize=7, va="bottom", ha="left")

# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.5)
out_path = OUT / "fig_fss_composite.pdf"
fig.savefig(out_path)
fig.savefig(OUT / "fig_fss_composite.png")
plt.close(fig)
print(f"[+] {out_path}")
print(f"[+] {OUT / 'fig_fss_composite.png'}")
