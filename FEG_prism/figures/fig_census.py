"""Figure 6: Generation census & combinatorial prediction."""

import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb as ncr

from .style import C, savefig


def _stirling2(n, k):
    """Stirling number of the second kind."""
    if k == 0:
        return 1 if n == 0 else 0
    if k == 1 or k == n:
        return 1
    s = 0
    for j in range(k + 1):
        s += (-1) ** (k - j) * ncr(k, j, exact=True) * j ** n
    return s // math.factorial(k)


def fig6(data, out):
    """Two panels: gen population bars (log) + combinatorial P(g|n)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    P_gen1, P_gen2, P_gen3 = data.N_gen1, data.N_gen2, data.N_gen3
    P_total = P_gen1 + P_gen2 + P_gen3

    # ── Panel 1: Generation populations ──
    gen_labels = ["Gen 1\n($g=1$)", "Gen 2\n($g=2$)", "Gen 3\n($g=3$)", "Anti-1"]
    gen_counts = [P_gen1, P_gen2, P_gen3, data.N_anti1]
    gen_colors = [C["gen1"], C["gen2"], C["gen3"], C["anti1"]]

    bars = ax1.bar(gen_labels, gen_counts, color=gen_colors,
                   edgecolor="white", lw=0.6, zorder=3)
    ax1.set_ylabel("Prism count")
    ax1.set_title("Generation Census (Prism Counts)")
    ax1.set_yscale("log")

    for bar, count in zip(bars, gen_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                 f"{count:,}", ha="center", fontsize=8, color="0.3")

    ax1.text(0.02, 0.97,
             f"Total prisms (g=1+2+3): {P_total:,}\n"
             f"Gen1: {P_gen1/P_total:.1%}  "
             f"Gen2: {P_gen2/P_total:.1%}  "
             f"Gen3: {P_gen3/P_total:.1%}",
             transform=ax1.transAxes, fontsize=8, color="0.4",
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    # ── Panel 2: Combinatorial prediction ──
    n_vals = np.arange(3, 16)
    frac_g1 = np.zeros_like(n_vals, dtype=float)
    frac_g2 = np.zeros_like(n_vals, dtype=float)
    frac_g3 = np.zeros_like(n_vals, dtype=float)

    for i, n in enumerate(n_vals):
        total = 3 ** n
        w1 = 3
        w2 = 3 * _stirling2(n, 2) * 2
        w3 = 1 * _stirling2(n, 3) * 6
        frac_g1[i] = w1 / total
        frac_g2[i] = w2 / total
        frac_g3[i] = w3 / total

    ax2.plot(n_vals, frac_g1, "o-", color=C["gen1"],
             label="$g=1$ (predicted)", markersize=5)
    ax2.plot(n_vals, frac_g2, "s-", color=C["gen2"],
             label="$g=2$ (predicted)", markersize=5)
    ax2.plot(n_vals, frac_g3, "^-", color=C["gen3"],
             label="$g=3$ (predicted)", markersize=5)

    # Observed fractions
    ax2.axhline(P_gen1 / P_total, color=C["gen1"], ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(P_gen2 / P_total, color=C["gen2"], ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(P_gen3 / P_total, color=C["gen3"], ls="--", lw=0.8, alpha=0.5)

    ax2.text(14.5, P_gen1 / P_total, "observed", fontsize=7,
             color=C["gen1"], va="bottom", ha="right")
    ax2.text(14.5, P_gen2 / P_total, "observed", fontsize=7,
             color=C["gen2"], va="bottom", ha="right")
    ax2.text(14.5, P_gen3 / P_total, "observed", fontsize=7,
             color=C["gen3"], va="top", ha="right")

    ax2.set_xlabel(r"Belly size $n$")
    ax2.set_ylabel("Fraction of prisms in generation")
    ax2.set_title(r"Combinatorial Prediction: $P(g|n)$")
    ax2.legend(loc="center right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax2.set_xlim(3, 15)
    ax2.set_ylim(0, 1)

    fig.suptitle(rf"Generation Structure — $N = {data.N_total/1e6:.0f}$M events",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig6_generation_census", out)
