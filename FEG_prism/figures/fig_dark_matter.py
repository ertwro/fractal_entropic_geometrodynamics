"""Figure 5: Dark matter analysis (3-panel)."""

import numpy as np
import matplotlib.pyplot as plt

from .style import C, savefig


def fig5(data, out):
    """Three panels: CLT opacity, cumulative mass fraction, mass-weighted histogram."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    ms = data.mass_spectrum
    N_int = ms["intermediates_N"].values
    freq = ms["frequency"].values

    # ── Panel 1: EM Opacity Q(n) = 1/sqrt(n) ──
    n_range = np.arange(3, 31)
    Q_clt = 1.0 / np.sqrt(n_range)

    ax1.plot(n_range, Q_clt, "k-", lw=1.5,
             label=r"CLT: $Q \sim 1/\sqrt{n}$", zorder=5)
    ax1.axhline(0.2, color="red", ls="--", lw=0.8, alpha=0.7,
                label="Visibility threshold")
    ax1.axvline(5.5, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax1.fill_between([3, 5.5], 0, 1.1, color=C["vis"], alpha=0.08, zorder=1)
    ax1.fill_between([5.5, 30], 0, 1.1, color=C["dark"], alpha=0.08, zorder=1)
    ax1.text(4.0, 0.05, "visible", fontsize=9, color=C["vis"],
             ha="center", fontweight="bold")
    ax1.text(15, 0.05, "dark", fontsize=9, color="0.4",
             ha="center", fontweight="bold")

    ax1.set_xlabel(r"Belly size $n$")
    ax1.set_ylabel(r"EM opacity $Q(n) = |\Phi|/n$")
    ax1.set_title("CLT Phase Cancellation")
    ax1.set_xlim(3, 30)
    ax1.set_ylim(0, 0.7)
    ax1.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")

    # ── Panel 2: Cumulative Mass Fraction ──
    mass_contribution = N_int * freq
    total_mass = mass_contribution.sum()
    cumulative = np.cumsum(mass_contribution) / total_mass

    ax2.plot(N_int, cumulative, "ko-", markersize=4, lw=1.5, zorder=5)
    ax2.axvline(5.5, color="red", ls="--", lw=1.0, alpha=0.7)

    vis_mass = mass_contribution[N_int <= 5].sum()
    dark_mass = mass_contribution[N_int > 5].sum()
    vis_frac = vis_mass / total_mass
    dark_frac = dark_mass / total_mass
    ratio_dm = dark_mass / vis_mass

    ax2.axhline(vis_frac, color=C["vis"], ls=":", lw=0.8, alpha=0.7)
    ax2.text(15, vis_frac + 0.02, f"Visible: {vis_frac:.1%}",
             fontsize=9, color=C["vis"])
    ax2.text(15, vis_frac + 0.10,
             f"Dark: {dark_frac:.1%}\n"
             rf"$\Omega_{{\mathrm{{dark}}}}/\Omega_{{\mathrm{{vis}}}} = {ratio_dm:.2f}$",
             fontsize=9, color="0.3")

    ax2.set_xlabel(r"Belly size $n$")
    ax2.set_ylabel("Cumulative mass fraction")
    ax2.set_title("Dark-to-Visible Mass Ratio")
    ax2.set_xlim(3, N_int.max())
    ax2.set_ylim(0, 1.05)

    # ── Panel 3: Mass-weighted histogram ──
    ax3.bar(N_int[N_int <= 5], mass_contribution[N_int <= 5], width=0.8,
            color=C["vis"], edgecolor="white", lw=0.4, label="Visible", zorder=3)
    ax3.bar(N_int[N_int > 5], mass_contribution[N_int > 5], width=0.8,
            color=C["dark"], edgecolor="white", lw=0.4, label="Dark",
            alpha=0.7, zorder=3)
    ax3.set_xlabel(r"Belly size $n$")
    ax3.set_ylabel(r"Mass contribution ($n \times$ frequency)")
    ax3.set_title("Mass-Weighted Spectrum")
    ax3.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")

    ax3.text(0.97, 0.70,
             rf"Simulation: $\Omega_d/\Omega_v = {ratio_dm:.2f}$" "\n"
             r"Observed:   $\Omega_d/\Omega_v \approx 5.4$",
             transform=ax3.transAxes, fontsize=9, color="0.3",
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    fig.suptitle(r"Dark Matter as Phase-Cancelled Large-$n$ Prisms",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig5_dark_matter", out)
