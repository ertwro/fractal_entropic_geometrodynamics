"""CausalWorlds 2×2 dense panel figure."""

import numpy as np
import matplotlib.pyplot as plt

from .style import C, apply_style, savefig


def composite(data, out):
    """2×2: (a) dS flow, (b) mass spectrum log, (c) mass bars, (d) cumulative."""
    apply_style(compact=True)

    ms = data.mass_spectrum
    df = data.results
    sigma = data.sigma
    td = data.topology

    N_int = ms["intermediates_N"].values
    freq = ms["frequency"].values

    m_gen1, m_gen2, m_gen3 = data.m_gen1, data.m_gen2, data.m_gen3
    m_anti1 = data.m_anti1
    latest_M = data._latest_M or "?"

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ── (a) Spectral dimension flow ──
    mask = sigma <= 15
    s = sigma[mask]

    y_vac = df["dS_vac"].values[mask]
    e_vac = df["dS_vac_std"].values[mask]
    y_def = df["dS_def"].values[mask]
    e_def = df["dS_def_std"].values[mask]

    ax_a.plot(s, y_vac, color=C["vac"], lw=1.0, label="Vacuum", zorder=5)
    ax_a.fill_between(s, y_vac - e_vac, y_vac + e_vac,
                      color=C["vac"], alpha=0.15, zorder=2)
    ax_a.plot(s, y_def, color=C["def"], lw=1.0, label="Defect", zorder=5)
    ax_a.fill_between(s, y_def - e_def, y_def + e_def,
                      color=C["def"], alpha=0.15, zorder=2)

    ax_a.axhline(4.0, color="grey", ls="--", lw=0.5, alpha=0.6, zorder=1)
    ax_a.axhline(2.0, color="grey", ls=":", lw=0.4, alpha=0.5, zorder=1)

    ax_a.set_xlabel(r"Diffusion time $\sigma$")
    ax_a.set_ylabel(r"$d_S(\sigma)$")
    ax_a.set_xlim(1, 15)
    ax_a.set_ylim(0, 10)
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.9,
                edgecolor="0.85", handlelength=1.2)

    ds_uv = f"{y_vac[0]:.2f}"
    ds_ir = f"{y_vac.max():.2f}"
    ax_a.text(0.03, 0.97, rf"$d_S\!=\!{ds_uv} \to {ds_ir}$",
              transform=ax_a.transAxes, fontsize=6, fontweight="bold",
              va="top", ha="left", color="0.15",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))
    ax_a.text(0.03, 0.03, rf"$\mathbf{{(a)}}\ M\!={latest_M}$",
              transform=ax_a.transAxes, fontsize=7, va="bottom", ha="left")

    # ── (b) Mass spectrum (log scale) ──
    opacity = 1.0 / np.sqrt(N_int.astype(float))
    opacity_norm = opacity / opacity.max()
    bar_colors = [plt.cm.copper_r(q) for q in opacity_norm]

    ax_b.bar(N_int, freq, width=0.8, color=bar_colors,
             edgecolor="white", lw=0.3, zorder=3)
    ax_b.set_yscale("log")

    tail_mask = N_int >= 7
    n_tail = N_int[tail_mask].astype(float)
    f_tail = freq[tail_mask].astype(float)
    valid = f_tail > 0
    if np.sum(valid) >= 3:
        log_f = np.log(f_tail[valid])
        slope, intercept = np.polyfit(n_tail[valid], log_f, 1)
        n_fit = np.linspace(7, N_int.max(), 100)
        ax_b.plot(n_fit, np.exp(intercept + slope * n_fit),
                  color="red", ls="--", lw=0.8,
                  label=rf"$\sim e^{{{slope:.2f}n}}$", zorder=5)

    ax_b.set_xlabel(r"Belly size $n$")
    ax_b.set_ylabel("Frequency")
    ax_b.set_xlim(2, 31)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.9,
                edgecolor="0.85", handlelength=1.2)

    total_prisms = int(float(td["total_prisms"]))
    ax_b.text(0.03, 0.97, f"${total_prisms:,}$ prisms".replace(",", "{,}"),
              transform=ax_b.transAxes, fontsize=6, fontweight="bold",
              va="top", ha="left", color="0.15",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))
    ax_b.text(0.03, 0.03, rf"$\mathbf{{(b)}}\ M\!={latest_M}$",
              transform=ax_b.transAxes, fontsize=7, va="bottom", ha="left")

    # ── (c) Mass bars + CPT ──
    labels = ["Gen 1", "Gen 2", "Gen 3", "Anti-1"]
    masses = [m_gen1, m_gen2, m_gen3, m_anti1]
    colors = [C["gen1"], C["gen2"], C["gen3"], C["anti1"]]

    x_pos = np.arange(len(labels))
    ax_c.bar(x_pos, masses, width=0.65, color=colors,
             edgecolor="white", lw=0.4, zorder=3)

    cpt_dev = abs(m_gen1 - m_anti1) / m_gen1 * 100
    mid_y = max(m_gen1, m_anti1) + 0.25
    ax_c.annotate("", xy=(3, mid_y), xytext=(0, mid_y),
                  arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.7))
    ax_c.text(1.5, mid_y + 0.25, rf"$\Delta m/m = {cpt_dev:.1f}\%$",
              ha="center", fontsize=5.5, color="0.3")

    ax_c.set_ylabel("Topological mass")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(labels, fontsize=6)
    ax_c.set_ylim(0, 9.5)

    ax_c.text(0.97, 0.97, rf"$\Delta m/m = {cpt_dev:.1f}\%$",
              transform=ax_c.transAxes, fontsize=6, fontweight="bold",
              va="top", ha="right", color="0.15",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))
    ax_c.text(0.03, 0.03, rf"$\mathbf{{(c)}}\ M\!={latest_M}$",
              transform=ax_c.transAxes, fontsize=7, va="bottom", ha="left")

    # ── (d) Cumulative mass fraction ──
    mass_contribution = N_int * freq
    total_mass = mass_contribution.sum()
    cumulative = np.cumsum(mass_contribution) / total_mass

    ax_d.plot(N_int, cumulative, "ko-", markersize=2, lw=1.0, zorder=5)

    if "omega_ratio" in td:
        ratio_dm = float(td["omega_ratio"])
        vis_total = float(td.get("visible_mass_total", 0))
        grav_total = float(td.get("grav_mass_total", 0))
        vis_frac = vis_total / grav_total if grav_total > 0 else 0
        dark_frac = 1.0 - vis_frac
        alpha_em = float(td.get("alpha_em", 0))
    else:
        vis_mass = mass_contribution[N_int <= 5].sum()
        dark_mass = mass_contribution[N_int > 5].sum()
        vis_frac = vis_mass / total_mass
        dark_frac = dark_mass / total_mass
        ratio_dm = dark_mass / vis_mass
        alpha_em = 0.0

    ax_d.fill_between(N_int, 0, cumulative, color=C["vis"], alpha=0.10, zorder=2)
    ax_d.axhline(vis_frac, color=C["vis"], ls=":", lw=0.6, alpha=0.7)
    ax_d.text(20, vis_frac + 0.02, f"Vis: {vis_frac:.1%}", fontsize=5,
              color=C["vis"])
    ax_d.text(20, vis_frac + 0.09, f"Dark: {dark_frac:.1%}", fontsize=5,
              color="0.4")

    ax_d.set_xlabel(r"Belly size $n$")
    ax_d.set_ylabel("Cumulative mass fraction")
    ax_d.set_xlim(3, N_int.max())
    ax_d.set_ylim(0, 1.05)

    omega_label = rf"$\Omega_d/\Omega_v = {ratio_dm:.2f}$"
    if alpha_em > 0:
        omega_label += rf"$\quad \alpha = {alpha_em:.4f}$"
    ax_d.text(0.97, 0.50, omega_label,
              transform=ax_d.transAxes, fontsize=6, fontweight="bold",
              va="center", ha="right", color="0.15",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))
    ax_d.text(0.03, 0.03, rf"$\mathbf{{(d)}}\ M\!={latest_M}$",
              transform=ax_d.transAxes, fontsize=7, va="bottom", ha="left")

    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.5)
    savefig(fig, "fig_composite", out)

    # Restore default style
    apply_style(compact=False)
