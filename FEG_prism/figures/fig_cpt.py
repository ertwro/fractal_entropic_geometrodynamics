"""Figure 8: CPT symmetry test."""

import numpy as np
import matplotlib.pyplot as plt

from .style import C, savefig


def fig8(data, out):
    """Two panels: mass bars (matter vs anti) + dS matter/antimatter comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    m_gen1, m_gen2, m_gen3 = data.m_gen1, data.m_gen2, data.m_gen3
    m_anti1 = data.m_anti1

    # ── Panel 1: Mass comparison ──
    masses = [m_gen1, m_gen2, m_gen3]
    mass_labels = ["Gen 1", "Gen 2", "Gen 3"]
    mass_colors = [C["gen1"], C["gen2"], C["gen3"]]

    x = np.arange(3)
    width = 0.35
    ax1.bar(x - width / 2, masses, width, label="Matter",
            color=mass_colors, edgecolor="white", lw=0.6, zorder=3)

    ax1.bar(x[0] + width / 2, m_anti1, width, label="Antimatter (Gen 1)",
            color=C["anti1"], edgecolor="white", lw=0.6, zorder=3)

    cpt_dev = abs(m_gen1 - m_anti1) / m_gen1 * 100
    ax1.text(0, max(m_gen1, m_anti1) + 0.3,
             rf"$\Delta m / m = {cpt_dev:.1f}\%$",
             ha="center", fontsize=9, color="0.3")

    ax1.set_ylabel("Topological mass (Planck units)")
    ax1.set_title("Mass Spectrum & CPT Test")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mass_labels)
    ax1.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax1.set_ylim(0, 10)

    ax1.text(0.97, 0.97,
             rf"$m_1 : m_2 : m_3 = 1 : {m_gen2/m_gen1:.2f} : {m_gen3/m_gen1:.2f}$"
             "\n"
             r"$m_e : m_\mu : m_\tau = 1 : 206.8 : 3477$"
             "\n"
             "(topological, not physical units)",
             transform=ax1.transAxes, fontsize=8, color="0.4",
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    # ── Panel 2: dS comparison (matter vs antimatter) ──
    sigma = data.sigma
    df = data.results
    mask = sigma <= 15
    s = sigma[mask]

    y_g1 = df["dS_Gen1"].values[mask]
    e_g1 = df["dS_Gen1_std"].values[mask]
    y_a1 = df["dS_Anti1"].values[mask]
    e_a1 = df["dS_Anti1_std"].values[mask]

    ax2.plot(s, y_g1, color=C["gen1"], label="Gen 1 (matter)", zorder=5)
    ax2.fill_between(s, y_g1 - e_g1, y_g1 + e_g1,
                     color=C["gen1"], alpha=0.15, zorder=2)
    ax2.plot(s, y_a1, color=C["anti1"], ls="--",
             label="Anti-1 (antimatter)", zorder=5)
    ax2.fill_between(s, y_a1 - e_a1, y_a1 + e_a1,
                     color=C["anti1"], alpha=0.15, zorder=2)

    diff = np.abs(y_g1 - y_a1)
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(s, 0, diff, color="grey", alpha=0.2, zorder=1)
    ax2_twin.set_ylabel(r"$|d_S^{\mathrm{matter}} - d_S^{\mathrm{anti}}|$",
                        color="0.5")
    ax2_twin.tick_params(axis="y", colors="0.5")
    ax2_twin.set_ylim(0, 0.15)

    ax2.set_xlabel(r"Diffusion time $\sigma$")
    ax2.set_ylabel(r"Spectral dimension $d_S$")
    ax2.set_title("CPT Symmetry: Matter vs Antimatter")
    ax2.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax2.set_xlim(1, 15)

    fig.suptitle(
        rf"CPT Test — $N = {data.N_total/1e6:.0f}$M, {data.M_label} realisations",
        fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig8_cpt_test", out)
