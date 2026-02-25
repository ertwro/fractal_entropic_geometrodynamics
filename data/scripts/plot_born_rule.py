#!/usr/bin/env python3
# Usage: python data/scripts/plot_born_rule.py [path/to/born_rule.csv]
"""
plot_born_rule.py — Born rule convergence from M6 path counting.

Reads born_rule.csv (produced by --measure-decoherence) and generates
a two-panel figure: (a) predicted vs observed scatter, (b) bin-by-bin
comparison showing decoherence protection.
"""

import re
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Locate input ────────────────────────────────────────────────────────────
REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

if len(sys.argv) > 1:
    csv_path = pathlib.Path(sys.argv[1])
else:
    candidates = sorted(REPO.glob("data/ensemble_*/born_rule*.csv"))
    if not candidates:
        sys.exit("No born_rule*.csv found. Run with --measure-decoherence first.")
    csv_path = candidates[-1]

print(f"[*] Reading {csv_path}")

# ── Parse metadata from comment lines ───────────────────────────────────────
chi_sq = None
n_det = None
N_events = None
with open(csv_path) as f:
    for line in f:
        if not line.startswith("#"):
            break
        m = re.search(r"chi_sq=([\d.eE+-]+)", line)
        if m:
            chi_sq = float(m.group(1))
        m = re.search(r"n_detector_nodes=(\d+)", line)
        if m:
            n_det = int(m.group(1))
        m = re.search(r"N:\s*(\d+)", line)
        if m:
            N_events = int(m.group(1))

df = pd.read_csv(csv_path, comment="#")
if df.empty:
    sys.exit("born_rule.csv has no data rows. Check simulation output.")

pred = df["predicted_freq"].values
obs = df["observed_freq"].values
n_bins = len(pred)

# ── Two-panel figure ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

scale = 1e5
pred_s = pred * scale
obs_s = obs * scale

# ── Panel (a): Scatter plot — predicted vs observed ─────────────────────────
hi = max(pred_s.max(), obs_s.max()) * 1.15
ax1.plot([0, hi], [0, hi], ls="--", color="0.55", lw=1.2, label="Perfect Born rule")
ax1.scatter(pred_s, obs_s, s=80, edgecolors="black", linewidths=0.6,
            facecolors="#4285F4", zorder=3, label="Binned detector data")

parts = []
if chi_sq is not None:
    parts.append(f"$\\chi^2 = {chi_sq:.4f}$")
if N_events is not None:
    parts.append(f"$N = {N_events:,}$")
if n_det is not None:
    parts.append(f"$N_{{\\mathrm{{det}}}} = {n_det:,}$")
note = "\n".join(parts)
if note:
    ax1.text(0.04, 0.96, note, transform=ax1.transAxes, fontsize=9.5,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))

ax1.set_xlabel(r"Predicted $|\psi|^2$" f"  ($\\times 10^{{5}}$)", fontsize=11)
ax1.set_ylabel(f"Observed intensity  ($\\times 10^{{5}}$)", fontsize=11)
ax1.set_title("(a)  Born Rule Scatter", fontsize=12)
ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax1.set_xlim(0, hi)
ax1.set_ylim(0, hi)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.25)

# ── Panel (b): Bin-by-bin comparison ────────────────────────────────────────
x = np.arange(n_bins)
w = 0.35

# Normalize both to sum to 1 for shape comparison
pred_norm = pred / pred.sum() if pred.sum() > 0 else pred
obs_norm = obs / obs.sum() if obs.sum() > 0 else obs

bars1 = ax2.bar(x - w/2, pred_norm, w, color="#E57373", edgecolor="black",
                linewidth=0.5, label=r"Predicted $|\psi|^2$", alpha=0.85)
bars2 = ax2.bar(x + w/2, obs_norm, w, color="#4285F4", edgecolor="black",
                linewidth=0.5, label="Observed walkers", alpha=0.85)

# Uniform reference
ax2.axhline(1.0 / n_bins, ls=":", color="0.4", lw=1, label="Uniform ($1/n_{\\mathrm{bins}}$)")

ax2.set_xlabel("Bin index (sorted by predicted intensity)", fontsize=11)
ax2.set_ylabel("Normalized frequency", fontsize=11)
ax2.set_title("(b)  Decoherence Protection", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([str(i+1) for i in x])
ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax2.grid(True, axis="y", alpha=0.25)

fig.suptitle("Born Rule from Causal Path Topology (M6)", fontsize=14, y=1.02)
fig.tight_layout()

outfile = OUT / "fig_born_rule.pdf"
fig.savefig(outfile, dpi=300, bbox_inches="tight")
print(f"[*] Saved {outfile}")

outpng = OUT / "fig_born_rule.png"
fig.savefig(outpng, dpi=200, bbox_inches="tight")
print(f"[*] Saved {outpng}")
plt.close()
