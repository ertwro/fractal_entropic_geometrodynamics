"""Figure 7: Causal flux & running coupling."""

import numpy as np
import matplotlib.pyplot as plt

from .style import C, savefig

SIGMA_MAX = 30


def fig7(data, out):
    """Two panels: raw flux (semilogy) + running coupling ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    sigma = data.sigma
    df = data.results
    mask = sigma <= SIGMA_MAX
    s = sigma[mask]
    attr = df["Flux_Attr"].values[mask]
    repu = df["Flux_Repu"].values[mask]
    attr_std = df["Flux_Attr_std"].values[mask]
    repu_std = df["Flux_Repu_std"].values[mask]

    valid_a = attr > 0
    valid_r = repu > 0
    valid_both = valid_a & valid_r

    # ── Panel 1: Raw flux ──
    ax1.semilogy(s[valid_a], attr[valid_a], "s-", color=C["flux_a"],
                 markersize=3.5,
                 label=r"Attraction (Gen1 $\to$ Anti-1)", zorder=5)
    ax1.semilogy(s[valid_r], repu[valid_r], "o-", color=C["flux_r"],
                 markersize=3.5,
                 label=r"Repulsion (Gen1 $\to$ Gen1)", zorder=5)

    if np.any(valid_a & (attr_std > 0)):
        ax1.fill_between(s[valid_a],
                         np.maximum(attr[valid_a] - attr_std[valid_a], 1e-12),
                         attr[valid_a] + attr_std[valid_a],
                         color=C["flux_a"], alpha=0.1, zorder=2)
    if np.any(valid_r & (repu_std > 0)):
        ax1.fill_between(s[valid_r],
                         np.maximum(repu[valid_r] - repu_std[valid_r], 1e-12),
                         repu[valid_r] + repu_std[valid_r],
                         color=C["flux_r"], alpha=0.1, zorder=2)

    ax1.set_xlabel(r"Diffusion time $\sigma$")
    ax1.set_ylabel("Transmission probability")
    ax1.set_title("Causal Flux: Attraction vs Repulsion")
    ax1.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax1.set_xlim(1, SIGMA_MAX)

    # ── Panel 2: Running coupling ratio ──
    ratio = np.full_like(s, np.nan, dtype=float)
    ratio[valid_both] = attr[valid_both] / repu[valid_both]

    valid_ratio = ~np.isnan(ratio) & (ratio > 0)
    inv_ratio = np.full_like(ratio, np.nan)
    inv_ratio[valid_ratio] = 1.0 / ratio[valid_ratio]

    ax2.plot(s[valid_ratio], inv_ratio[valid_ratio], "o-", color="#8B0000",
             markersize=4, lw=1.5,
             label=r"$F_{\mathrm{repu}} / F_{\mathrm{attr}}$", zorder=5)

    ax2.axhline(137.036, color="gold", ls="--", lw=1.5,
                label=r"$1/\alpha_{\mathrm{SM}} = 137.036$", zorder=3)

    ax2.set_xlabel(r"Diffusion time $\sigma$")
    ax2.set_ylabel("Flux ratio (running coupling)")
    ax2.set_title("Running Coupling Constant")
    ax2.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax2.set_xlim(1, SIGMA_MAX)

    if valid_both[0]:
        r0 = repu[0] / attr[0]
        ax2.text(0.03, 0.05,
                 rf"$\sigma = 1$: ratio $= {r0:.1f}$",
                 transform=ax2.transAxes, fontsize=9, color="0.3",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85",
                           alpha=0.9))

    fig.suptitle("Emergent Electromagnetism", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig7_causal_flux", out)
