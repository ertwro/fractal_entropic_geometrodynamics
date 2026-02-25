"""FSS 2×2 composite figure."""

import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import C, apply_style, savefig


def fss(data, out, *, fss_json=None):
    """2×2: (a) Q_topo, (b) mass hierarchy, (c) Omega_energy, (d) dS flow.

    Parameters
    ----------
    fss_json : str or Path, optional
        Path to ``fss_comprehensive_results.json``.  Required.
    """
    if fss_json is None:
        print("  [!] fig_fss: --fss-json not provided, skipping.")
        return

    fss_path = pathlib.Path(fss_json)
    if not fss_path.exists():
        print(f"  [!] fig_fss: {fss_path} not found, skipping.")
        return

    with open(fss_path) as f:
        fss_data = json.load(f)

    apply_style(compact=True)

    dp = fss_data["data_points"]
    fits = fss_data["fits"]
    derived = fss_data["derived"]

    N_vals = np.array([d["N"] for d in dp], dtype=float)
    x_vals = N_vals ** (-0.25)

    Q_vals = np.array([d["Q_topo"] for d in dp])
    Om_vals = np.array([d["Omega_energy"] for d in dp])
    m1_vals = np.array([d["mass_gen1"] for d in dp])
    m2_vals = np.array([d["mass_gen2"] for d in dp])
    m3_vals = np.array([d["mass_gen3"] for d in dp])

    df = data.results
    sigma = data.sigma
    latest_M = data._latest_M or "?"

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    x_fit = np.linspace(0, x_vals.max() * 1.05, 100)

    # ── (a) Q_topo vs N^{-1/4} ──
    Q_fit = fits["Q_topo"]
    Q_inf = Q_fit["O_inf"]
    Q_a = Q_fit["a"]
    R2_Q = Q_fit["R_sq"]

    ax_a.plot(x_vals, Q_vals, "o", color=C["vac"], markersize=3.5, zorder=5)
    ax_a.plot(x_fit, Q_inf + Q_a * x_fit, "--", color=C["fit"], lw=0.8, zorder=3)
    ax_a.axhline(Q_inf, color="grey", ls=":", lw=0.4, alpha=0.6, zorder=1)
    ax_a.plot(0, Q_inf, "s", color=C["def"], markersize=4, zorder=6, clip_on=False)

    ax_a.set_xlabel(r"$N^{-1/4}$")
    ax_a.set_ylabel(r"$\mathcal{Q}_{\mathrm{topo}}$")
    ax_a.set_xlim(-0.003, x_vals.max() * 1.08)
    ax_a.set_ylim(0.13, 0.30)

    ax_a.text(0.97, 0.97,
              rf"$\mathcal{{Q}}_\infty = {Q_inf:.3f}$" "\n"
              rf"$R^2 = {R2_Q:.4f}$",
              transform=ax_a.transAxes, fontsize=5.5, va="top", ha="right",
              color="0.15",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))
    ax_a.text(0.03, 0.03, r"$\mathbf{(a)}$",
              transform=ax_a.transAxes, fontsize=7, va="bottom", ha="left")

    # ── (b) Mass hierarchy across N ──
    ax_b.plot(x_vals, m1_vals, "o-", color=C["gen1"], markersize=3, lw=0.8,
              label="Gen 1", zorder=5)
    ax_b.plot(x_vals, m2_vals, "s-", color=C["gen2"], markersize=3, lw=0.8,
              label="Gen 2", zorder=5)
    ax_b.plot(x_vals, m3_vals, "^-", color=C["gen3"], markersize=3, lw=0.8,
              label="Gen 3", zorder=5)

    for fit_key, color in [("mass_gen1", C["gen1"]),
                           ("mass_gen2", C["gen2"]),
                           ("mass_gen3", C["gen3"])]:
        ax_b.axhline(fits[fit_key]["O_inf"], color=color, ls=":", lw=0.4, alpha=0.6)

    ax_b.set_xlabel(r"$N^{-1/4}$")
    ax_b.set_ylabel("Topological mass")
    ax_b.set_xlim(-0.003, x_vals.max() * 1.08)
    ax_b.set_ylim(3.5, 8.5)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.9,
                edgecolor="0.85", handlelength=1.2)

    ax_b.text(0.03, 0.97, r"Flat: $R^2 \leq 0.79$",
              transform=ax_b.transAxes, fontsize=5.5, va="top", ha="left",
              color="0.15",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))
    ax_b.text(0.03, 0.03, r"$\mathbf{(b)}$",
              transform=ax_b.transAxes, fontsize=7, va="bottom", ha="left")

    # ── (c) Omega_energy convergence ──
    Om_fit = fits["Omega_energy"]
    Om_inf = Om_fit["O_inf"]
    Om_a = Om_fit["a"]
    R2_Om = Om_fit["R_sq"]
    Om_inf_Q = derived["Omega_inf_from_Q"]

    ax_c.plot(x_vals, Om_vals, "o", color=C["vac"], markersize=3.5, zorder=5)
    ax_c.plot(x_fit, Om_inf + Om_a * x_fit, "--", color=C["fit"], lw=0.8,
              zorder=3, label=rf"Direct: $\Omega_\infty={Om_inf:.2f}$")

    ax_c.axhline(Om_inf_Q, color=C["def"], ls="-.", lw=0.6, alpha=0.8,
                 zorder=2,
                 label=rf"From $\mathcal{{Q}}_\infty$: ${Om_inf_Q:.2f}$")

    PLANCK_OMEGA = 5.36
    ax_c.axhline(PLANCK_OMEGA, color=C["planck"], ls="--", lw=0.7, alpha=0.8,
                 zorder=2, label=f"Planck 2018: {PLANCK_OMEGA}")

    ax_c.set_xlabel(r"$N^{-1/4}$")
    ax_c.set_ylabel(r"$\Omega_{\mathrm{energy}}$")
    ax_c.set_xlim(-0.003, x_vals.max() * 1.08)
    ax_c.set_ylim(2.0, 6.5)
    ax_c.legend(loc="upper right", frameon=True, framealpha=0.9,
                edgecolor="0.85", handlelength=1.2, fontsize=5)
    ax_c.text(0.03, 0.03, r"$\mathbf{(c)}$",
              transform=ax_c.transAxes, fontsize=7, va="bottom", ha="left")

    # ── (d) Spectral dimension flow ──
    mask = sigma <= 15
    s = sigma[mask]

    y_vac = df["dS_vac"].values[mask]
    e_vac = df["dS_vac_std"].values[mask]
    y_def = df["dS_def"].values[mask]
    e_def = df["dS_def_std"].values[mask]

    ax_d.plot(s, y_vac, color=C["vac"], lw=1.0, label="Vacuum", zorder=5)
    ax_d.fill_between(s, y_vac - e_vac, y_vac + e_vac,
                      color=C["vac"], alpha=0.15, zorder=2)
    ax_d.plot(s, y_def, color=C["def"], lw=1.0, label="Defect", zorder=5)
    ax_d.fill_between(s, y_def - e_def, y_def + e_def,
                      color=C["def"], alpha=0.15, zorder=2)

    ax_d.axhline(4.0, color="grey", ls="--", lw=0.5, alpha=0.6, zorder=1)
    ax_d.axhline(2.0, color="grey", ls=":", lw=0.4, alpha=0.5, zorder=1)

    ax_d.set_xlabel(r"Diffusion time $\sigma$")
    ax_d.set_ylabel(r"$d_S(\sigma)$")
    ax_d.set_xlim(1, 15)
    ax_d.set_ylim(0, 10)
    ax_d.legend(loc="upper right", frameon=True, framealpha=0.9,
                edgecolor="0.85", handlelength=1.2)

    ds_uv = f"{y_vac[0]:.2f}"
    ds_ir = f"{y_vac.max():.2f}"
    ax_d.text(0.03, 0.97, rf"$d_S\!=\!{ds_uv} \to {ds_ir}$",
              transform=ax_d.transAxes, fontsize=6, fontweight="bold",
              va="top", ha="left", color="0.15",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85))
    ax_d.text(0.03, 0.03, rf"$\mathbf{{(d)}}\ N\!=\!10^7,\,M\!=\!{latest_M}$",
              transform=ax_d.transAxes, fontsize=7, va="bottom", ha="left")

    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.5)
    savefig(fig, "fig_fss_composite", out)

    # Restore default style
    apply_style(compact=False)
