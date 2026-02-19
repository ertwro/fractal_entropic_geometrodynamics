#!/usr/bin/env python3
# Usage: python data/scripts/plot_universe.py
"""
plot_universe.py — Publication-quality figures for Modulo Synthesis / FEG.

Reads the three CSV outputs from causal_set_sim and generates:
  Figure 1: Emergent Metric      (spectral dimension d_S vs diffusion time t)
  Figure 2: Inertia Quantisation  (topological mass spectrum)
  Figure 3: Causal Flux Asymmetry (attraction vs repulsion)

Dependencies: pandas, matplotlib, seaborn
Usage:        python3 plot_universe.py
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════════
# Global style — Nature/Science publication standard
# ═══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.4,
})

# Colour palette — muted, colourblind-safe
PAL = {
    "vacuum": "#4C72B0",
    "defect": "#C44E52",
    "gen1":   "#55A868",
    "gen2":   "#8172B2",
    "gen3":   "#CCB974",
    "anti1":  "#64B5CD",
    "attr":   "#C44E52",
    "repu":   "#4C72B0",
}

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "ensemble_10M"
OUT = REPO_ROOT / "data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV, skipping leading comment lines (# ...)."""
    path = DATA / name
    return pd.read_csv(path, comment="#")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
df = load_csv("results.csv")
mass = load_csv("mass_spectrum.csv")
topo = load_csv("topology_summary.csv")

# Parse topology summary into a dict for annotations
topo_dict = dict(zip(topo["key"], topo["value"]))


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — The Emergent Metric: d_S(t)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_spectral_dimension():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    t = df["step"].values

    # ── Main curves ──
    ax.plot(t, df["dS_vac"], color=PAL["vacuum"], label=r"Vacuum $d_S$",
            zorder=5)
    ax.plot(t, df["dS_def"], color=PAL["defect"], label=r"Defect $d_S$",
            zorder=5)

    # ── Generation curves (thin, secondary) ──
    for gen, col, lbl in [
        ("dS_Gen1", PAL["gen1"], "Gen1"),
        ("dS_Gen2", PAL["gen2"], "Gen2"),
        ("dS_Gen3", PAL["gen3"], "Gen3"),
        ("dS_Anti1", PAL["anti1"], "AntiGen1"),
    ]:
        ax.plot(t, df[gen], color=col, alpha=0.5, linewidth=0.9,
                label=lbl, zorder=3)

    # ── Theoretical references ──
    ax.axhline(4.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.7,
               label=r"$d_S = 4$ (continuum)", zorder=1)
    ax.axhline(2.0, color="grey", linestyle=":", linewidth=0.5, alpha=0.5,
               label=r"$d_S = 2$ (UV limit)", zorder=1)

    # ── Zoom to the physically interesting region ──
    ax.set_xlim(1, 50)
    ax.set_ylim(-0.5, 10)
    ax.set_xlabel(r"Diffusion time $t$")
    ax.set_ylabel(r"Spectral dimension $d_S(t)$")
    ax.set_title(r"\textbf{Emergent Metric} — Spectral Dimension Flow")

    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="0.8", ncol=2, fontsize=8.5)
    ax.grid(True, alpha=0.15, linewidth=0.4)

    # ── Annotation: d_S = 4 crossing ──
    # Find the first t where dS_def crosses 4
    ds_def = df["dS_def"].values
    cross_idx = np.argmax(ds_def >= 4.0)
    if cross_idx > 0:
        t_cross = t[cross_idx]
        ax.annotate(
            rf"$d_S = 4$ at $t \approx {t_cross}$",
            xy=(t_cross, 4.0), xytext=(t_cross + 6, 3.0),
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
            fontsize=9, color="0.3",
        )

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_spectral_dimension.pdf")
    fig.savefig(OUT / "fig1_spectral_dimension.png")
    print("  [+] fig1_spectral_dimension.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Inertia Quantisation: Topological Mass Spectrum
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_mass_spectrum():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    N = mass["intermediates_N"].values
    freq = mass["frequency"].values

    # ── Bar chart with colour gradient by mass ──
    norm = plt.Normalize(vmin=N.min(), vmax=N.max())
    cmap = plt.cm.viridis
    colours = [cmap(norm(n)) for n in N]

    bars = ax.bar(N, freq, width=0.75, color=colours, edgecolor="white",
                  linewidth=0.4, zorder=3)

    ax.set_xlabel(r"Topological mass $N$ (intermediate nodes)")
    ax.set_ylabel(r"Frequency (Causal Prisms)")
    ax.set_title(r"\textbf{Inertia Quantisation} — Mass Spectrum of $K_{2,N}$")

    # ── Annotate dominant peaks ──
    peak_idx = np.argmax(freq)
    ax.annotate(
        rf"$N = {N[peak_idx]}$" + "\n" + rf"$f = {freq[peak_idx]:,}$",
        xy=(N[peak_idx], freq[peak_idx]),
        xytext=(N[peak_idx] + 4, freq[peak_idx] * 0.95),
        arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
        fontsize=9, color="0.2",
    )

    # ── Total prisms annotation ──
    total = freq.sum()
    ax.text(0.97, 0.95,
            rf"$\Sigma = {total:,}$ Prisms" + "\n"
            rf"$N_{{\max}} = {N.max()}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="0.3",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8",
                      alpha=0.9))

    ax.set_xticks(N[::2])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.4)

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_mass_spectrum.pdf")
    fig.savefig(OUT / "fig2_mass_spectrum.png")
    print("  [+] fig2_mass_spectrum.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Causal Flux Asymmetry: Attraction vs Repulsion
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_causal_flux():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    t = df["step"].values
    attr = df["Flux_Attr"].values
    repu = df["Flux_Repu"].values

    # Only plot where flux is non-zero (walkers still active)
    mask_attr = attr > 0
    mask_repu = repu > 0
    mask = mask_attr | mask_repu

    t_m = t[mask]
    attr_m = attr[mask]
    repu_m = repu[mask]

    ax.semilogy(t_m, repu_m, color=PAL["repu"], marker="o", markersize=3.5,
                label=r"Repulsion (Gen1 $\to$ Gen1)", zorder=5)
    ax.semilogy(t_m[attr_m > 0], attr_m[attr_m > 0],
                color=PAL["attr"], marker="s", markersize=3.5,
                label=r"Attraction (Gen1 $\to$ AntiGen1)", zorder=5)

    # ── Ratio annotation ──
    # At t=1 the ratio is most clean
    if attr[0] > 0 and repu[0] > 0:
        ratio = repu[0] / attr[0]
        ax.text(0.03, 0.05,
                rf"$F_{{\mathrm{{repu}}}} / F_{{\mathrm{{attr}}}}(t\!=\!1)"
                rf" = {ratio:.0f}$",
                transform=ax.transAxes, fontsize=9, color="0.3",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8",
                          alpha=0.9))

    ax.set_xlabel(r"Diffusion time $t$")
    ax.set_ylabel(r"Transmission probability (log scale)")
    ax.set_title(
        r"\textbf{Causal Flux Asymmetry}"
        r" — Modus Ponens vs.\ Modus Tollens")
    ax.set_xlim(1, 50)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="0.8", fontsize=9)
    ax.grid(True, alpha=0.15, linewidth=0.4, which="both")

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_causal_flux.pdf")
    fig.savefig(OUT / "fig3_causal_flux.png")
    print("  [+] fig3_causal_flux.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures …")
    print(f"  Source: {HERE}")
    print(f"  N = {topo_dict.get('total_nodes', '?')}, "
          f"Prisms = {topo_dict.get('total_prisms', '?')}")
    print()

    fig1_spectral_dimension()
    fig2_mass_spectrum()
    fig3_causal_flux()

    print()
    print("Done. Figures saved as PDF + PNG in:")
    print(f"  {OUT}")
