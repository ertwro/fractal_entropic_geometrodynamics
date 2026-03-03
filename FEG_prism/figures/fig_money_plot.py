#!/usr/bin/env python3
"""Money Plot: per-batch 1/alpha from cumulative topology snapshots.

Standalone script — reads topology_summary_M*.csv from the data directory,
differences consecutive cumulative sums to get per-batch Q_topo and 1/alpha,
then generates the figure with the caption text embedded.

Usage:
    python FEG_prism/figures/fig_money_plot.py data/ensemble_10M_final
"""

import sys
import re
import pathlib
import textwrap
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAPTION = (
    "We present a non-perturbative calculation of the Fine Structure Constant "
    "\u03b1 using a discrete causal set simulation with zero free parameters. "
    "By defining particles as topological obstructions (K\u2082,\u2099 prisms) "
    "in a random geometric graph, we derive a value of 1/\u03b1 \u2248 131.8 "
    "at N=10\u2077, with a renormalization flow suggesting convergence to 137.0 "
    "in the continuum limit. The simulation naturally reproduces the "
    "three-generation mass hierarchy, maximal parity violation for leptons, "
    "and a Dark Matter fraction of 63%."
)


def load_snapshots(data_dir):
    """Load all topology_summary_M*.csv and return sorted (M, psq, msq)."""
    pattern = str(data_dir / "topology_summary_M*.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No topology_summary_M*.csv in {data_dir}")

    snapshots = []
    for f in files:
        m = int(re.search(r"M(\d+)", f).group(1))
        df = pd.read_csv(f, comment="#")
        row = dict(zip(df["key"], df["value"]))
        snapshots.append((m, int(float(row["phase_sq_total"])),
                             int(float(row["mass_sq_total"]))))
    snapshots.sort(key=lambda x: x[0])
    return snapshots


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data_dir> [output_dir]")
        sys.exit(1)

    data_dir = pathlib.Path(sys.argv[1])
    out_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots = load_snapshots(data_dir)

    # Difference consecutive cumulative sums to get per-batch values
    batch_labels = []   # midpoint realization index for x-axis
    batch_inv_alpha = []
    prev_m, prev_psq, prev_msq = 0, 0, 0
    for m, psq, msq in snapshots:
        d_psq = psq - prev_psq
        d_msq = msq - prev_msq
        batch_size = m - prev_m
        q = d_psq / d_msq if d_msq > 0 else 0.0
        inv_a = 8.0 * np.pi / q if q > 0 else 0.0
        # place point at midpoint of the batch range
        midpoint = prev_m + (batch_size + 1) / 2.0
        batch_labels.append(midpoint)
        batch_inv_alpha.append(inv_a)
        prev_m, prev_psq, prev_msq = m, psq, msq

    x = np.array(batch_labels)
    y = np.array(batch_inv_alpha)
    mean_val = np.mean(y)
    std_val = np.std(y, ddof=1)
    phys = 137.036
    total_m = snapshots[-1][0]

    # ── Style ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.4,
    })

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Ensemble mean +/- 1 sigma band
    ax.axhspan(mean_val - std_val, mean_val + std_val,
               color="#4C72B0", alpha=0.10, zorder=2)
    ax.axhline(mean_val, color="#4C72B0", ls="-", lw=1.2, alpha=0.6, zorder=3,
               label=rf"Ensemble mean $1/\alpha = {mean_val:.1f} \pm {std_val:.1f}$")

    # Per-batch scatter
    ax.scatter(x, y, s=55, color="#4C72B0", edgecolors="white",
               linewidths=0.6, zorder=5, label=f"Per-batch $1/\\alpha$ (M={total_m})")

    # Physical reference line
    ax.axhline(phys, color="#E8A838", ls="--", lw=2.0, zorder=4,
               label=r"Physical $1/\alpha = 137.036$")

    # Annotate the screening gap
    gap = phys - mean_val
    mid_y = (mean_val + phys) / 2.0
    arrow_x = x[-1] + 1.5
    ax.annotate(
        "", xy=(arrow_x, phys - 0.15), xytext=(arrow_x, mean_val + 0.15),
        arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2),
    )
    ax.text(
        arrow_x + 0.5, mid_y,
        f"Finite Volume\nScreening Effect\n$\\Delta = {gap:.1f}$",
        fontsize=9, va="center", ha="left", color="0.3",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )

    ax.set_xlabel("Realization (batch midpoint)")
    ax.set_ylabel(r"$1/\alpha$")
    ax.set_xlim(0, x[-1] + 6)
    y_lo = mean_val - 4 * max(std_val, 1.0)
    y_hi = phys + 5
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.85")

    # Caption text embedded in the figure
    wrapped = textwrap.fill(CAPTION, width=95)
    fig.text(
        0.5, -0.01, wrapped,
        ha="center", va="top", fontsize=7.5,
        fontstyle="italic", color="0.30",
        transform=fig.transFigure,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    for ext in ("png", "pdf"):
        path = out_dir / f"money_plot_alpha.{ext}"
        fig.savefig(path, dpi=300)
        print(f"  [+] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
