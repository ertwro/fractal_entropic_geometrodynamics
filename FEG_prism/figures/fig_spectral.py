"""Figures 1–3: Spectral dimension flow, generation-resolved, return probability."""

import numpy as np
import matplotlib.pyplot as plt

from .style import C, savefig

SIGMA_MAX = 30


def fig1(data, out):
    """Fig 1 — Spectral dimension flow: vacuum + defect."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sigma = data.sigma
    df = data.results
    mask = sigma <= SIGMA_MAX
    s = sigma[mask]

    y_vac = df["dS_vac"].values[mask]
    e_vac = df["dS_vac_std"].values[mask]
    ax.plot(s, y_vac, color=C["vac"], label="Vacuum", zorder=5)
    ax.fill_between(s, y_vac - e_vac, y_vac + e_vac,
                    color=C["vac"], alpha=0.15, zorder=2)

    y_def = df["dS_def"].values[mask]
    e_def = df["dS_def_std"].values[mask]
    ax.plot(s, y_def, color=C["def"], label="Defect core", zorder=5)
    ax.fill_between(s, y_def - e_def, y_def + e_def,
                    color=C["def"], alpha=0.15, zorder=2)

    ax.axhline(4.0, color="black", ls="--", lw=1.5, alpha=0.8,
               label=r"$d_S = 4$ (Minkowski)", zorder=4)
    ax.axhline(2.0, color="black", ls="--", lw=1.5, alpha=0.8,
               label=r"$d_S = 2$ (UV limit)", zorder=4)

    cross_idx = np.argmax(y_vac >= 4.0)
    if cross_idx > 0:
        tc = s[cross_idx]
        ax.annotate(rf"$d_S = 4$ at $\sigma \approx {tc}$",
                    xy=(tc, 4.0), xytext=(tc + 5, 3.0),
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
                    fontsize=9, color="0.3")

    ax.annotate(f"$d_S = {y_vac[0]:.2f}$\n(UV)",
                xy=(s[0], y_vac[0]), xytext=(3, 1.4),
                arrowprops=dict(arrowstyle="->", color=C["vac"], lw=0.8),
                fontsize=9, color=C["vac"])

    peak_idx = np.argmax(y_def)
    ax.annotate(f"$d_S = {y_def[peak_idx]:.2f}$\n(defect trapping)",
                xy=(s[peak_idx], y_def[peak_idx]),
                xytext=(s[peak_idx] + 5, y_def[peak_idx] + 1.5),
                arrowprops=dict(arrowstyle="->", color=C["def"], lw=0.8),
                fontsize=9, color=C["def"])

    ax.set_xlabel(r"Diffusion time $\sigma$ (random walk steps)")
    ax.set_ylabel(r"Spectral dimension $d_S(\sigma)$")
    ax.set_title(r"Spectral Dimension Flow: UV $\to$ IR")
    ax.set_xlim(1, SIGMA_MAX)
    ax.set_ylim(0, 10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.85")

    ax.text(0.02, 0.97,
            f"$N = {data.N_total/1e6:.0f}$M, {data.M_label}, seed 42",
            transform=ax.transAxes, fontsize=8, color="0.5",
            va="top", ha="left")

    fig.tight_layout()
    savefig(fig, "fig1_spectral_dimension_flow", out)


def fig2(data, out):
    """Fig 2 — Generation-resolved spectral dimensions."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sigma = data.sigma
    df = data.results
    mask = sigma <= 15
    s = sigma[mask]

    for col, ecol, color, label in [
        ("dS_Gen1", "dS_Gen1_std", C["gen1"], "Gen 1 ($g=1$)"),
        ("dS_Gen2", "dS_Gen2_std", C["gen2"], "Gen 2 ($g=2$)"),
        ("dS_Gen3", "dS_Gen3_std", C["gen3"], "Gen 3 ($g=3$)"),
        ("dS_Anti1", "dS_Anti1_std", C["anti1"], "Anti-1"),
    ]:
        y = df[col].values[mask]
        e = df[ecol].values[mask]
        ax.plot(s, y, color=color, label=label, zorder=5)
        ax.fill_between(s, y - e, y + e, color=color, alpha=0.12, zorder=2)

    y_vac = df["dS_vac"].values[mask]
    ax.plot(s, y_vac, color="grey", ls="--", lw=1.0,
            label="Vacuum (reference)", zorder=3)
    ax.axhline(4.0, color="grey", ls=":", lw=0.5, alpha=0.5, zorder=1)

    ax.set_xlabel(r"Diffusion time $\sigma$")
    ax.set_ylabel(r"Spectral dimension $d_S(\sigma)$")
    ax.set_title("Generation-Resolved Spectral Dimensions")
    ax.set_xlim(1, 15)
    ax.set_ylim(1.5, 9)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.85")

    ax.text(0.02, 0.97,
            "All generations exceed vacuum $d_S$\n"
            "= enhanced return probability = mass",
            transform=ax.transAxes, fontsize=8, color="0.4",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    fig.tight_layout()
    savefig(fig, "fig2_generation_spectral", out)


def fig3(data, out):
    """Fig 3 — Return probability & mass extraction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    sigma = data.sigma
    df = data.results
    mask = sigma <= SIGMA_MAX
    s = sigma[mask]

    # Left: log-log P(sigma)
    for col, color, label in [
        ("P_vac",  C["vac"],  "Vacuum"),
        ("P_Gen1", C["gen1"], "Gen 1"),
        ("P_Gen2", C["gen2"], "Gen 2"),
        ("P_Gen3", C["gen3"], "Gen 3"),
        ("P_Anti1", C["anti1"], "Anti-1"),
    ]:
        y = df[col].values[mask]
        valid = y > 0
        ax1.loglog(s[valid], y[valid], color=color, label=label, zorder=5)

    ax1.set_xlabel(r"Diffusion time $\sigma$")
    ax1.set_ylabel(r"Return probability $P(\sigma)$")
    ax1.set_title("Return Probability Decay")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9,
               edgecolor="0.85", fontsize=8)

    # Right: P_Gen / P_vac ratio
    for col, color, label in [
        ("P_Gen1", C["gen1"], "Gen 1"),
        ("P_Gen2", C["gen2"], "Gen 2"),
        ("P_Gen3", C["gen3"], "Gen 3"),
        ("P_Anti1", C["anti1"], "Anti-1"),
    ]:
        y_gen = df[col].values[mask]
        y_vac = df["P_vac"].values[mask]
        valid = (y_gen > 0) & (y_vac > 0)
        ratio = y_gen[valid] / y_vac[valid]
        ax2.plot(s[valid], ratio, color=color, label=label, zorder=5)

    ax2.axhline(1.0, color="grey", ls="--", lw=0.7, alpha=0.7, zorder=1)
    ax2.set_xlabel(r"Diffusion time $\sigma$")
    ax2.set_ylabel(r"$P_{\mathrm{gen}} / P_{\mathrm{vac}}$")
    ax2.set_title("Trapping Enhancement (mass proxy)")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9,
               edgecolor="0.85", fontsize=8)
    ax2.set_xlim(1, SIGMA_MAX)

    ax2.text(0.97, 0.97,
             "Separation = mass hierarchy\n"
             "Gen3 traps walkers longest",
             transform=ax2.transAxes, fontsize=8, color="0.4",
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    fig.suptitle("Return Probability Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig3_return_probability", out)
