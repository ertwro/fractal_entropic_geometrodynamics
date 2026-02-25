"""Figure 4: Mass spectrum (belly size distribution)."""

import numpy as np
import matplotlib.pyplot as plt

from .style import C, savefig


def fig4(data, out):
    """Two panels: linear belly histogram + log-scale exponential tail fit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ms = data.mass_spectrum
    N_int = ms["intermediates_N"].values
    freq = ms["frequency"].values

    # Left: linear scale with dark/visible boundary
    vis_mask = N_int <= 5
    dark_mask = N_int > 5

    ax1.bar(N_int[vis_mask], freq[vis_mask], width=0.8,
            color=C["vis"], edgecolor="white", lw=0.4,
            label=r"Visible ($n \leq 5$)", zorder=3)
    ax1.bar(N_int[dark_mask], freq[dark_mask], width=0.8,
            color=C["dark"], edgecolor="white", lw=0.4,
            label=r"Dark ($n > 5$)", alpha=0.7, zorder=3)

    ax1.axvline(5.5, color="red", ls="--", lw=1.0, alpha=0.7, zorder=4)
    ax1.text(5.7, max(freq) * 0.9, "visible | dark", fontsize=8, color="red",
             rotation=90, va="top")

    ax1.set_xlabel(r"Belly size $n$ (intermediate nodes)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(r"Mass Spectrum of Causal Prisms $K_{2,n}$")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.85")

    peak_idx = np.argmax(freq)
    ax1.annotate(f"Peak: $n = {N_int[peak_idx]}$\n$f = {freq[peak_idx]:,}$",
                 xy=(N_int[peak_idx], freq[peak_idx]),
                 xytext=(N_int[peak_idx] + 5, freq[peak_idx] * 0.85),
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
                 fontsize=9, color="0.2")

    # Right: log scale with exponential tail
    ax2.bar(N_int, freq, width=0.8, color="#5B7C99", edgecolor="white",
            lw=0.4, zorder=3)
    ax2.set_yscale("log")
    ax2.axvline(5.5, color="red", ls="--", lw=1.0, alpha=0.7, zorder=4)
    ax2.set_xlabel(r"Belly size $n$")
    ax2.set_ylabel("Frequency (log scale)")
    ax2.set_title(r"Exponential Tail (large-$n$ dark sector)")

    tail_mask = N_int >= 7
    if np.sum(tail_mask) >= 3:
        n_tail = N_int[tail_mask].astype(float)
        f_tail = freq[tail_mask].astype(float)
        valid = f_tail > 0
        if np.sum(valid) >= 3:
            log_f = np.log(f_tail[valid])
            slope, intercept = np.polyfit(n_tail[valid], log_f, 1)
            n_fit = np.linspace(7, N_int.max(), 100)
            ax2.plot(n_fit, np.exp(intercept + slope * n_fit),
                     color="red", ls="--", lw=1.2,
                     label=rf"$\sim e^{{{slope:.2f}n}}$", zorder=5)
            ax2.legend(loc="upper right", frameon=True, framealpha=0.9,
                       edgecolor="0.85")

    fig.suptitle(f"Topological Mass Spectrum — {freq.sum():,} Prisms, "
                 rf"$n_{{\max}} = {N_int.max()}$", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig4_mass_spectrum", out)
