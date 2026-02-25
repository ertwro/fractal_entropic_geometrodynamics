"""Born rule scatter + decoherence bins."""

import re
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import savefig


def born_rule(data, out):
    """Two panels: (a) predicted vs observed scatter, (b) bin-by-bin comparison."""
    br = data.born_rule
    if br is None:
        print("  [!] fig_born_rule: born_rule.csv not found, skipping.")
        return
    if br.empty:
        print("  [!] fig_born_rule: born_rule.csv has no data rows, skipping.")
        return

    # Try to extract metadata from comment lines
    chi_sq = None
    n_det = None
    N_events = None
    csv_path = data._csv_path("born_rule")
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

    pred = br["predicted_freq"].values
    obs = br["observed_freq"].values
    n_bins = len(pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    scale = 1e5
    pred_s = pred * scale
    obs_s = obs * scale

    # ── Panel (a): Scatter ──
    hi = max(pred_s.max(), obs_s.max()) * 1.15
    ax1.plot([0, hi], [0, hi], ls="--", color="0.55", lw=1.2,
             label="Perfect Born rule")
    ax1.scatter(pred_s, obs_s, s=80, edgecolors="black", linewidths=0.6,
                facecolors="#4285F4", zorder=3, label="Binned detector data")

    parts = []
    if chi_sq is not None:
        parts.append(rf"$\chi^2 = {chi_sq:.4f}$")
    if N_events is not None:
        parts.append(rf"$N = {N_events:,}$")
    if n_det is not None:
        parts.append(rf"$N_{{\mathrm{{det}}}} = {n_det:,}$")
    note = "\n".join(parts)
    if note:
        ax1.text(0.04, 0.96, note, transform=ax1.transAxes, fontsize=9.5,
                 va="top", ha="left", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7",
                           alpha=0.9))

    ax1.set_xlabel(r"Predicted $|\psi|^2$" f"  ($\\times 10^{{5}}$)", fontsize=11)
    ax1.set_ylabel(f"Observed intensity  ($\\times 10^{{5}}$)", fontsize=11)
    ax1.set_title("(a)  Born Rule Scatter", fontsize=12)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax1.set_xlim(0, hi)
    ax1.set_ylim(0, hi)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.25)

    # ── Panel (b): Bin-by-bin ──
    x = np.arange(n_bins)
    w = 0.35

    pred_norm = pred / pred.sum() if pred.sum() > 0 else pred
    obs_norm = obs / obs.sum() if obs.sum() > 0 else obs

    ax2.bar(x - w / 2, pred_norm, w, color="#E57373", edgecolor="black",
            linewidth=0.5, label=r"Predicted $|\psi|^2$", alpha=0.85)
    ax2.bar(x + w / 2, obs_norm, w, color="#4285F4", edgecolor="black",
            linewidth=0.5, label="Observed walkers", alpha=0.85)

    ax2.axhline(1.0 / n_bins, ls=":", color="0.4", lw=1,
                label=r"Uniform ($1/n_{\mathrm{bins}}$)")

    ax2.set_xlabel("Bin index (sorted by predicted intensity)", fontsize=11)
    ax2.set_ylabel("Normalized frequency", fontsize=11)
    ax2.set_title("(b)  Decoherence Protection", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i + 1) for i in x])
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Born Rule from Causal Path Topology (M6)", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_born_rule", out)
