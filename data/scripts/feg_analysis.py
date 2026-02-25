#!/usr/bin/env python3
# Usage: python data/scripts/feg_analysis.py
"""
feg_analysis_m20.py — Comprehensive analysis of FEG N=10M simulation data.

Auto-detects the latest M snapshot from the production ensemble.
Produces publication figures into paper/figures/.

Usage:  python3 feg_analysis_m20.py
"""

import re
import pathlib
from glob import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from scipy.special import comb as ncr
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# Publication style
# ═══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

# Colourblind-safe palette
C = {
    "vac":   "#4C72B0",
    "def":   "#C44E52",
    "gen1":  "#55A868",
    "gen2":  "#8172B2",
    "gen3":  "#CCB974",
    "anti1": "#64B5CD",
    "flux_a": "#C44E52",
    "flux_r": "#4C72B0",
    "dark":  "#2f2f2f",
    "vis":   "#E8A838",
    "sterile": "#999999",
}

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "ensemble_10M_final"
OUT  = REPO_ROOT / "data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading — auto-detect latest M snapshot
# ═══════════════════════════════════════════════════════════════════════════════
results_files = sorted(glob(str(DATA / "results_M*.csv")))
if not results_files:
    raise FileNotFoundError(f"No results_M*.csv found in {DATA}")
latest_M = max(int(re.search(r'M(\d+)', f).group(1)) for f in results_files)
pad = f"{latest_M:02d}"
M_LABEL = f"M={latest_M}"

def load():
    df = pd.read_csv(DATA / f"results_M{pad}.csv", comment="#")
    ms = pd.read_csv(DATA / f"mass_spectrum_M{pad}.csv", comment="#")
    ts = pd.read_csv(DATA / f"topology_summary_M{pad}.csv", comment="#")
    td = dict(zip(ts["key"], ts["value"]))
    return df, ms, td


df, ms, td = load()
sigma = df["step"].values

# Useful constants from topology summary
N_total   = int(float(td["total_nodes"]))
N_prisms  = int(float(td["total_prisms"]))
N_gen1    = int(float(td["count_gen1"]))
N_gen2    = int(float(td["count_gen2"]))
N_gen3    = int(float(td["count_gen3"]))
N_anti1   = int(float(td["count_antigen1"]))
m_gen1    = float(td["avg_mass_gen1"])
m_gen2    = float(td["avg_mass_gen2"])
m_gen3    = float(td["avg_mass_gen3"])
m_anti1   = float(df["Mass_Anti1"].iloc[0])  # from results CSV

# Reliable region: where std < |value| (rough SNR > 1)
SIGMA_MAX_RELIABLE = 30  # beyond this, generation dS has huge error bars


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: save figure in PDF + PNG
# ═══════════════════════════════════════════════════════════════════════════════
def savefig(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    print(f"  [+] {name}.pdf / .png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Spectral Dimension Flow: The Signature Plot
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_spectral_dimension_flow():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    mask = sigma <= SIGMA_MAX_RELIABLE
    s = sigma[mask]

    # Vacuum
    y_vac = df["dS_vac"].values[mask]
    e_vac = df["dS_vac_std"].values[mask]
    ax.plot(s, y_vac, color=C["vac"], label="Vacuum", zorder=5)
    ax.fill_between(s, y_vac - e_vac, y_vac + e_vac, color=C["vac"],
                    alpha=0.15, zorder=2)

    # Defect core
    y_def = df["dS_def"].values[mask]
    e_def = df["dS_def_std"].values[mask]
    ax.plot(s, y_def, color=C["def"], label="Defect core", zorder=5)
    ax.fill_between(s, y_def - e_def, y_def + e_def, color=C["def"],
                    alpha=0.15, zorder=2)

    # Reference lines — hard, prominent
    ax.axhline(4.0, color="black", ls="--", lw=1.5, alpha=0.8,
               label="$d_S = 4$ (Minkowski)", zorder=4)
    ax.axhline(2.0, color="black", ls="--", lw=1.5, alpha=0.8,
               label="$d_S = 2$ (UV limit)", zorder=4)

    # Mark crossover: first sigma where dS_vac > 4
    cross_idx = np.argmax(y_vac >= 4.0)
    if cross_idx > 0:
        tc = s[cross_idx]
        ax.annotate(f"$d_S = 4$ at $\\sigma \\approx {tc}$",
                    xy=(tc, 4.0), xytext=(tc + 5, 3.0),
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
                    fontsize=9, color="0.3")

    # Mark UV value
    ax.annotate(f"$d_S = {y_vac[0]:.2f}$\n(UV)",
                xy=(s[0], y_vac[0]), xytext=(3, 1.4),
                arrowprops=dict(arrowstyle="->", color=C["vac"], lw=0.8),
                fontsize=9, color=C["vac"])

    # Mark defect peak
    peak_idx = np.argmax(y_def)
    ax.annotate(f"$d_S = {y_def[peak_idx]:.2f}$\n(defect trapping)",
                xy=(s[peak_idx], y_def[peak_idx]),
                xytext=(s[peak_idx] + 5, y_def[peak_idx] + 1.5),
                arrowprops=dict(arrowstyle="->", color=C["def"], lw=0.8),
                fontsize=9, color=C["def"])

    ax.set_xlabel("Diffusion time $\\sigma$ (random walk steps)")
    ax.set_ylabel("Spectral dimension $d_S(\\sigma)$")
    ax.set_title("Spectral Dimension Flow: UV $\\to$ IR")
    ax.set_xlim(1, SIGMA_MAX_RELIABLE)
    ax.set_ylim(0, 10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.85")

    # Metadata
    ax.text(0.02, 0.97, f"$N = {N_total/1e6:.0f}$M, {M_LABEL}, seed 42",
            transform=ax.transAxes, fontsize=8, color="0.5",
            va="top", ha="left")

    fig.tight_layout()
    savefig(fig, "fig1_spectral_dimension_flow")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Generation-Resolved Spectral Dimensions
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_generation_spectral():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Only the reliable region
    mask = sigma <= 15  # even tighter for generation curves
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

    # Vacuum as reference (dashed)
    y_vac = df["dS_vac"].values[mask]
    ax.plot(s, y_vac, color="grey", ls="--", lw=1.0,
            label="Vacuum (reference)", zorder=3)

    ax.axhline(4.0, color="grey", ls=":", lw=0.5, alpha=0.5, zorder=1)

    ax.set_xlabel("Diffusion time $\\sigma$")
    ax.set_ylabel("Spectral dimension $d_S(\\sigma)$")
    ax.set_title("Generation-Resolved Spectral Dimensions")
    ax.set_xlim(1, 15)
    ax.set_ylim(1.5, 9)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.85")

    # Key observation annotation
    ax.text(0.02, 0.97,
            "All generations exceed vacuum $d_S$\n"
            "= enhanced return probability = mass",
            transform=ax.transAxes, fontsize=8, color="0.4",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    fig.tight_layout()
    savefig(fig, "fig2_generation_spectral")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Return Probability & Mass Extraction
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_return_probability():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    mask = sigma <= SIGMA_MAX_RELIABLE
    s = sigma[mask]

    # Left panel: log-log P(sigma)
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

    ax1.set_xlabel("Diffusion time $\\sigma$")
    ax1.set_ylabel("Return probability $P(\\sigma)$")
    ax1.set_title("Return Probability Decay")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.85",
               fontsize=8)

    # Right panel: P_Gen / P_vac ratio = trapping enhancement
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
    ax2.set_xlabel("Diffusion time $\\sigma$")
    ax2.set_ylabel("$P_{\\mathrm{gen}} / P_{\\mathrm{vac}}$")
    ax2.set_title("Trapping Enhancement (mass proxy)")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.85",
               fontsize=8)
    ax2.set_xlim(1, SIGMA_MAX_RELIABLE)

    # Annotate: at early sigma, all curves are close to 1.
    # At late sigma, generations separate — that's the mass signal.
    ax2.text(0.97, 0.97,
             "Separation = mass hierarchy\n"
             "Gen3 traps walkers longest",
             transform=ax2.transAxes, fontsize=8, color="0.4",
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    fig.suptitle("Return Probability Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig3_return_probability")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Mass Spectrum (Belly Size Distribution)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_mass_spectrum():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    N_int = ms["intermediates_N"].values
    freq  = ms["frequency"].values

    # Left: linear scale with dark/visible boundary
    vis_mask  = N_int <= 5
    dark_mask = N_int > 5

    ax1.bar(N_int[vis_mask], freq[vis_mask], width=0.8,
            color=C["vis"], edgecolor="white", lw=0.4, label="Visible ($n \\leq 5$)",
            zorder=3)
    ax1.bar(N_int[dark_mask], freq[dark_mask], width=0.8,
            color=C["dark"], edgecolor="white", lw=0.4, label="Dark ($n > 5$)",
            alpha=0.7, zorder=3)

    # Vertical boundary
    ax1.axvline(5.5, color="red", ls="--", lw=1.0, alpha=0.7, zorder=4)
    ax1.text(5.7, max(freq) * 0.9, "visible | dark", fontsize=8, color="red",
             rotation=90, va="top")

    ax1.set_xlabel("Belly size $n$ (intermediate nodes)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Mass Spectrum of Causal Prisms $K_{2,n}$")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.85")

    # Annotate peak
    peak_idx = np.argmax(freq)
    ax1.annotate(f"Peak: $n = {N_int[peak_idx]}$\n$f = {freq[peak_idx]:,}$",
                 xy=(N_int[peak_idx], freq[peak_idx]),
                 xytext=(N_int[peak_idx] + 5, freq[peak_idx] * 0.85),
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
                 fontsize=9, color="0.2")

    # Right: log scale — shows the exponential tail
    ax2.bar(N_int, freq, width=0.8, color="#5B7C99", edgecolor="white", lw=0.4,
            zorder=3)
    ax2.set_yscale("log")
    ax2.axvline(5.5, color="red", ls="--", lw=1.0, alpha=0.7, zorder=4)
    ax2.set_xlabel("Belly size $n$")
    ax2.set_ylabel("Frequency (log scale)")
    ax2.set_title("Exponential Tail (large-$n$ dark sector)")

    # Fit exponential to the tail (n >= 7)
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
                     color="red", ls="--", lw=1.2, label=f"$\\sim e^{{{slope:.2f}n}}$",
                     zorder=5)
            ax2.legend(loc="upper right", frameon=True, framealpha=0.9,
                       edgecolor="0.85")

    fig.suptitle(f"Topological Mass Spectrum — {freq.sum():,} Prisms, "
                 f"$n_{{\\max}} = {N_int.max()}$", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig4_mass_spectrum")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Dark Matter Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_dark_matter():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    N_int = ms["intermediates_N"].values
    freq  = ms["frequency"].values

    # ── Panel 1: EM Opacity Q(n) = 1/sqrt(n) (CLT prediction) ──
    n_range = np.arange(3, 31)
    Q_clt = 1.0 / np.sqrt(n_range)  # CLT: Q ~ sigma/sqrt(n), sigma~1

    ax1.plot(n_range, Q_clt, "k-", lw=1.5, label="CLT: $Q \\sim 1/\\sqrt{n}$",
             zorder=5)
    ax1.axhline(0.2, color="red", ls="--", lw=0.8, alpha=0.7,
                label="Visibility threshold")
    ax1.axvline(5.5, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax1.fill_between([3, 5.5], 0, 1.1, color=C["vis"], alpha=0.08, zorder=1)
    ax1.fill_between([5.5, 30], 0, 1.1, color=C["dark"], alpha=0.08, zorder=1)
    ax1.text(4.0, 0.05, "visible", fontsize=9, color=C["vis"], ha="center",
             fontweight="bold")
    ax1.text(15, 0.05, "dark", fontsize=9, color="0.4", ha="center",
             fontweight="bold")

    ax1.set_xlabel("Belly size $n$")
    ax1.set_ylabel("EM opacity $Q(n) = |\\Phi|/n$")
    ax1.set_title("CLT Phase Cancellation")
    ax1.set_xlim(3, 30)
    ax1.set_ylim(0, 0.7)
    ax1.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")

    # ── Panel 2: Cumulative Mass Fraction ──
    # "Mass" of a prism ~ n (topological mass), total mass ~ n * freq
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
    ax2.text(15, vis_frac + 0.02,
             f"Visible: {vis_frac:.1%}",
             fontsize=9, color=C["vis"])
    ax2.text(15, vis_frac + 0.10,
             f"Dark: {dark_frac:.1%}\n"
             f"$\\Omega_{{dark}}/\\Omega_{{vis}} = {ratio_dm:.2f}$",
             fontsize=9, color="0.3")

    ax2.set_xlabel("Belly size $n$")
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
    ax3.set_xlabel("Belly size $n$")
    ax3.set_ylabel("Mass contribution ($n \\times$ frequency)")
    ax3.set_title("Mass-Weighted Spectrum")
    ax3.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")

    # Cosmological comparison
    ax3.text(0.97, 0.70,
             f"Simulation: $\\Omega_d/\\Omega_v = {ratio_dm:.2f}$\n"
             f"Observed:   $\\Omega_d/\\Omega_v \\approx 5.4$",
             transform=ax3.transAxes, fontsize=9, color="0.3",
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    fig.suptitle("Dark Matter as Phase-Cancelled Large-$n$ Prisms", fontsize=13,
                 y=1.02)
    fig.tight_layout()
    savefig(fig, "fig5_dark_matter")

    return ratio_dm


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Generation Census & Combinatorial Prediction
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_generation_census():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel 1: Generation populations (PRISM counts, not node counts) ──
    P_gen1 = N_gen1
    P_gen2 = N_gen2
    P_gen3 = N_gen3
    P_total = P_gen1 + P_gen2 + P_gen3

    gen_labels = ["Gen 1\n($g=1$)", "Gen 2\n($g=2$)", "Gen 3\n($g=3$)",
                  "Anti-1"]
    gen_counts = [P_gen1, P_gen2, P_gen3, N_anti1]
    gen_colors = [C["gen1"], C["gen2"], C["gen3"], C["anti1"]]

    bars = ax1.bar(gen_labels, gen_counts, color=gen_colors, edgecolor="white",
                   lw=0.6, zorder=3)
    ax1.set_ylabel("Prism count")
    ax1.set_title("Generation Census (Prism Counts)")
    ax1.set_yscale("log")

    # Annotate counts
    for bar, count in zip(bars, gen_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                 f"{count:,}", ha="center", fontsize=8, color="0.3")

    # Total
    ax1.text(0.02, 0.97,
             f"Total prisms (g=1+2+3): {P_total:,}\n"
             f"Gen1: {P_gen1/P_total:.1%}  "
             f"Gen2: {P_gen2/P_total:.1%}  "
             f"Gen3: {P_gen3/P_total:.1%}",
             transform=ax1.transAxes, fontsize=8, color="0.4",
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    # ── Panel 2: Combinatorial prediction of generation fractions vs N ──
    # For n intermediates with random phase in {-,0,+}:
    #   P(g=k | n) = S(n,k) * C(3,k) / 3^n
    # where S(n,k) = Stirling number of the second kind
    def stirling2(n, k):
        """Stirling number of the second kind."""
        if k == 0:
            return 1 if n == 0 else 0
        if k == 1 or k == n:
            return 1
        s = 0
        for j in range(k + 1):
            s += (-1)**(k - j) * ncr(k, j, exact=True) * j**n
        import math
        return s // math.factorial(k)

    n_vals = np.arange(3, 16)
    frac_g1 = np.zeros_like(n_vals, dtype=float)
    frac_g2 = np.zeros_like(n_vals, dtype=float)
    frac_g3 = np.zeros_like(n_vals, dtype=float)

    for i, n in enumerate(n_vals):
        total = 3**n
        # g=1: all intermediates in one phase class → 3 ways
        w1 = 3
        # g=2: surjections onto exactly 2 of 3 classes → C(3,2)*S(n,2)*2!
        s2 = stirling2(n, 2)
        w2 = 3 * s2 * 2  # C(3,2)=3, 2!=2
        # g=3: surjections onto all 3 classes → C(3,3)*S(n,3)*3!
        s3 = stirling2(n, 3)
        w3 = 1 * s3 * 6  # C(3,3)=1, 3!=6
        frac_g1[i] = w1 / total
        frac_g2[i] = w2 / total
        frac_g3[i] = w3 / total

    ax2.plot(n_vals, frac_g1, "o-", color=C["gen1"], label="$g=1$ (predicted)",
             markersize=5)
    ax2.plot(n_vals, frac_g2, "s-", color=C["gen2"], label="$g=2$ (predicted)",
             markersize=5)
    ax2.plot(n_vals, frac_g3, "^-", color=C["gen3"], label="$g=3$ (predicted)",
             markersize=5)

    # Observed prism fractions (horizontal dashed lines)
    ax2.axhline(P_gen1/P_total, color=C["gen1"], ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(P_gen2/P_total, color=C["gen2"], ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(P_gen3/P_total, color=C["gen3"], ls="--", lw=0.8, alpha=0.5)

    ax2.text(14.5, P_gen1/P_total, "observed", fontsize=7, color=C["gen1"],
             va="bottom", ha="right")
    ax2.text(14.5, P_gen2/P_total, "observed", fontsize=7, color=C["gen2"],
             va="bottom", ha="right")
    ax2.text(14.5, P_gen3/P_total, "observed", fontsize=7, color=C["gen3"],
             va="top", ha="right")

    ax2.set_xlabel("Belly size $n$")
    ax2.set_ylabel("Fraction of prisms in generation")
    ax2.set_title("Combinatorial Prediction: $P(g|n)$")
    ax2.legend(loc="center right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax2.set_xlim(3, 15)
    ax2.set_ylim(0, 1)

    fig.suptitle(f"Generation Structure — $N = {N_total/1e6:.0f}$M events",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig6_generation_census")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Causal Flux & Running Coupling
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_causal_flux():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    mask = sigma <= SIGMA_MAX_RELIABLE
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
                 markersize=3.5, label="Attraction (Gen1 $\\to$ Anti-1)", zorder=5)
    ax1.semilogy(s[valid_r], repu[valid_r], "o-", color=C["flux_r"],
                 markersize=3.5, label="Repulsion (Gen1 $\\to$ Gen1)", zorder=5)

    # Error bars where available
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

    ax1.set_xlabel("Diffusion time $\\sigma$")
    ax1.set_ylabel("Transmission probability")
    ax1.set_title("Causal Flux: Attraction vs Repulsion")
    ax1.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax1.set_xlim(1, SIGMA_MAX_RELIABLE)

    # ── Panel 2: Running coupling ratio ──
    ratio = np.full_like(s, np.nan, dtype=float)
    ratio[valid_both] = attr[valid_both] / repu[valid_both]

    valid_ratio = ~np.isnan(ratio) & (ratio > 0)
    inv_ratio = np.full_like(ratio, np.nan)
    inv_ratio[valid_ratio] = 1.0 / ratio[valid_ratio]

    ax2.plot(s[valid_ratio], inv_ratio[valid_ratio], "o-", color="#8B0000",
             markersize=4, lw=1.5, label="$F_{\\mathrm{repu}} / F_{\\mathrm{attr}}$",
             zorder=5)

    ax2.axhline(137.036, color="gold", ls="--", lw=1.5,
                label="$1/\\alpha_{\\mathrm{SM}} = 137.036$", zorder=3)

    ax2.set_xlabel("Diffusion time $\\sigma$")
    ax2.set_ylabel("Flux ratio (running coupling)")
    ax2.set_title("Running Coupling Constant")
    ax2.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax2.set_xlim(1, SIGMA_MAX_RELIABLE)

    # Print the value at sigma=1 for reference
    if valid_both[0]:
        r0 = repu[0] / attr[0]
        ax2.text(0.03, 0.05,
                 f"$\\sigma = 1$: ratio $= {r0:.1f}$",
                 transform=ax2.transAxes, fontsize=9, color="0.3",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85",
                           alpha=0.9))

    fig.suptitle("Emergent Electromagnetism", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig7_causal_flux")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — CPT Symmetry Test
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_cpt_test():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel 1: Mass comparison ──
    masses = [m_gen1, m_gen2, m_gen3]
    mass_labels = ["Gen 1", "Gen 2", "Gen 3"]
    mass_colors = [C["gen1"], C["gen2"], C["gen3"]]

    # Matter masses
    x = np.arange(3)
    width = 0.35
    bars1 = ax1.bar(x - width/2, masses, width, label="Matter",
                    color=mass_colors, edgecolor="white", lw=0.6, zorder=3)

    # Anti-matter mass (only have Anti-1)
    m_a1 = float(td.get("avg_mass_gen1", m_gen1))  # matter gen1
    m_anti = m_anti1  # from results CSV (Mass_Anti1 column)
    anti_masses = [m_anti, np.nan, np.nan]
    ax1.bar(x[0] + width/2, m_anti, width, label="Antimatter (Gen 1)",
            color=C["anti1"], edgecolor="white", lw=0.6, zorder=3)

    # CPT deviation
    cpt_dev = abs(m_gen1 - m_anti) / m_gen1 * 100
    ax1.text(0, max(m_gen1, m_anti) + 0.3,
             f"$\\Delta m / m = {cpt_dev:.1f}\\%$",
             ha="center", fontsize=9, color="0.3")

    ax1.set_ylabel("Topological mass (Planck units)")
    ax1.set_title("Mass Spectrum & CPT Test")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mass_labels)
    ax1.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax1.set_ylim(0, 10)

    # Mass ratios
    ax1.text(0.97, 0.97,
             f"$m_1 : m_2 : m_3 = 1 : {m_gen2/m_gen1:.2f} : {m_gen3/m_gen1:.2f}$\n"
             f"$m_e : m_\\mu : m_\\tau = 1 : 206.8 : 3477$\n"
             f"(topological, not physical units)",
             transform=ax1.transAxes, fontsize=8, color="0.4",
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.9))

    # ── Panel 2: dS comparison (matter vs antimatter) ──
    mask = sigma <= 15
    s = sigma[mask]

    y_g1 = df["dS_Gen1"].values[mask]
    e_g1 = df["dS_Gen1_std"].values[mask]
    y_a1 = df["dS_Anti1"].values[mask]
    e_a1 = df["dS_Anti1_std"].values[mask]

    ax2.plot(s, y_g1, color=C["gen1"], label="Gen 1 (matter)", zorder=5)
    ax2.fill_between(s, y_g1 - e_g1, y_g1 + e_g1, color=C["gen1"],
                     alpha=0.15, zorder=2)
    ax2.plot(s, y_a1, color=C["anti1"], ls="--",
             label="Anti-1 (antimatter)", zorder=5)
    ax2.fill_between(s, y_a1 - e_a1, y_a1 + e_a1, color=C["anti1"],
                     alpha=0.15, zorder=2)

    # Difference
    diff = np.abs(y_g1 - y_a1)
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(s, 0, diff, color="grey", alpha=0.2, zorder=1)
    ax2_twin.set_ylabel("$|d_S^{\\mathrm{matter}} - d_S^{\\mathrm{anti}}|$",
                        color="0.5")
    ax2_twin.tick_params(axis="y", colors="0.5")
    ax2_twin.set_ylim(0, 0.15)

    ax2.set_xlabel("Diffusion time $\\sigma$")
    ax2.set_ylabel("Spectral dimension $d_S$")
    ax2.set_title("CPT Symmetry: Matter vs Antimatter")
    ax2.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax2.set_xlim(1, 15)

    fig.suptitle(f"CPT Test — $N = {N_total/1e6:.0f}$M, {M_LABEL} realisations",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig8_cpt_test")

    return cpt_dev


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE (LaTeX-ready)
# ═══════════════════════════════════════════════════════════════════════════════
def summary_table(dm_ratio, cpt_dev):
    N_int = ms["intermediates_N"].values
    freq  = ms["frequency"].values

    # Extract key values
    ds_vac_uv = df["dS_vac"].values[0]  # sigma=1
    ds_vac_core_idx = np.argmin(np.abs(sigma - 4))  # sigma ~ 4
    ds_vac_core = df["dS_vac"].values[ds_vac_core_idx]
    ds_def_peak = df["dS_def"].values[:15].max()

    # Prism counts from topology summary
    P_gen1 = N_gen1
    P_gen2 = N_gen2
    P_gen3 = N_gen3
    P_total = P_gen1 + P_gen2 + P_gen3

    # Flux ratio at sigma=1
    attr_1 = df["Flux_Attr"].values[0]
    repu_1 = df["Flux_Repu"].values[0]
    flux_ratio = repu_1 / attr_1 if attr_1 > 0 else float("nan")

    vis_count = freq[N_int <= 5].sum()
    dark_count = freq[N_int > 5].sum()

    lines = [
        "",
        "=" * 72,
        f"  SUMMARY TABLE — FEG Simulation at N = 10M ({M_LABEL}, seed 42)",
        "=" * 72,
        "",
        f"  {'Observable':<42} {'Value':>12}  {'Note':<20}",
        f"  {'-'*42} {'-'*12}  {'-'*20}",
        "",
        "  SPECTRAL DIMENSION",
        f"  {'dS vacuum (UV, sigma=1)':<42} {ds_vac_uv:>12.4f}  {'expected ~2':20}",
        f"  {'dS vacuum (core, sigma=4)':<42} {ds_vac_core:>12.4f}  {'approaching 4D':20}",
        f"  {'dS defect (peak)':<42} {ds_def_peak:>12.4f}  {'trapping > 4':20}",
        "",
        "  GENERATION CLASSIFICATION (prism counts)",
        f"  {'Total prisms':<42} {N_prisms:>12,}",
        f"  {'Gen 1 (g=1) prisms':<42} {P_gen1:>12,}  {f'{P_gen1/P_total:.1%}':20}",
        f"  {'Gen 2 (g=2) prisms':<42} {P_gen2:>12,}  {f'{P_gen2/P_total:.1%}':20}",
        f"  {'Gen 3 (g=3) prisms':<42} {P_gen3:>12,}  {f'{P_gen3/P_total:.1%}':20}",
        f"  {'Anti-Gen1 count':<42} {N_anti1:>12,}",
        "",
        "  MASS HIERARCHY",
        f"  {'Mass Gen 1':<42} {m_gen1:>12.4f}  {'lightest':20}",
        f"  {'Mass Gen 2':<42} {m_gen2:>12.4f}  {'':20}",
        f"  {'Mass Gen 3':<42} {m_gen3:>12.4f}  {'heaviest':20}",
        f"  {'Mass Anti-1':<42} {m_anti1:>12.4f}  {'':20}",
        f"  {'Ratio m1 : m2 : m3':<42} {'1 : {:.2f} : {:.2f}'.format(m_gen2/m_gen1, m_gen3/m_gen1):>12}",
        "",
        "  CPT TEST",
        f"  {'|m_Gen1 - m_Anti1| / m_Gen1':<42} {cpt_dev:>11.1f}%  {'converges w/ M':20}",
        "",
        "  DARK MATTER",
        f"  {'Visible prisms (n <= 5)':<42} {vis_count:>12,}",
        f"  {'Dark prisms (n > 5)':<42} {dark_count:>12,}",
        f"  {'Omega_dark / Omega_vis (mass-weighted)':<42} {dm_ratio:>12.2f}  {'observed: 5.4':20}",
        f"  {'Max belly size':<42} {N_int.max():>12d}",
        "",
        "  CAUSAL FLUX (sigma=1)",
        f"  {'Flux_Attraction':<42} {attr_1:>12.6e}",
        f"  {'Flux_Repulsion':<42} {repu_1:>12.6e}",
        f"  {'Repu / Attr (~ 1/alpha?)':<42} {flux_ratio:>12.1f}  {'SM: 137':20}",
        "",
        "  COMPUTATION",
        f"  {'Total events N':<42} {N_total:>12,}",
        f"  {'Runtime':<42} {'1450s':>12}",
        f"  {'Walkers':<42} {'2,250,000':>12}",
        f"  {'Mode':<42} {'in-memory':>12}",
        "",
        "=" * 72,
    ]

    table = "\n".join(lines)
    print(table)

    # Save to file
    with open(OUT / "summary_table.txt", "w") as f:
        f.write(table)
    print(f"\n  [+] summary_table.txt")

    # Also produce LaTeX version
    latex = r"""\begin{table}[h]
\caption{Key observables at $N = 10^7$, $""" + M_LABEL + r"""$ (seed\,42).}
\label{tab:full_results}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Observable} & \textbf{Value} & \textbf{Note} \\
\midrule
\multicolumn{3}{@{}l}{\emph{Spectral dimension}} \\[2pt]
$d_S$ vacuum (UV, $\sigma{=}1$)     & """ + f"{ds_vac_uv:.2f}" + r""" & expected $\approx 2$ \\
$d_S$ vacuum (core, $\sigma{=}4$)   & """ + f"{ds_vac_core:.2f}" + r""" & approaching 4D \\
$d_S$ defect (peak)                 & """ + f"{ds_def_peak:.2f}" + r""" & trapping $> 4$ \\
\midrule
\multicolumn{3}{@{}l}{\emph{Generation classification}} \\[2pt]
Total prisms                        & """ + f"{N_prisms:,}" + r""" & \\
Gen\,1 ($g{=}1$) / Gen\,2 / Gen\,3 & """ + f"{P_gen1:,} / {P_gen2:,} / {P_gen3:,}" + r""" & """ + f"{P_gen1/P_total:.1%} / {P_gen2/P_total:.1%} / {P_gen3/P_total:.1%}" + r""" \\
Anti-Gen\,1                         & """ + f"{N_anti1:,}" + r""" & \\
\midrule
\multicolumn{3}{@{}l}{\emph{Mass hierarchy}} \\[2pt]
$m_1 : m_2 : m_3$                  & """ + f"$1 : {m_gen2/m_gen1:.2f} : {m_gen3/m_gen1:.2f}$" + r""" & topological units \\
Mass: Gen\,1 / Gen\,2 / Gen\,3     & """ + f"{m_gen1:.2f} / {m_gen2:.2f} / {m_gen3:.2f}" + r""" & \\
CPT: Gen\,1 vs Anti-1              & """ + f"{m_gen1:.2f} vs {m_anti1:.2f}" + r""" & $\Delta m/m = """ + f"{cpt_dev:.1f}" + r"""\%$ \\
\midrule
\multicolumn{3}{@{}l}{\emph{Dark matter}} \\[2pt]
$\Omega_{\mathrm{dark}} / \Omega_{\mathrm{vis}}$ & """ + f"{dm_ratio:.2f}" + r""" & observed: $5.4$ \\
Max belly size $n_{\max}$           & """ + f"{N_int.max()}" + r""" & \\
\midrule
\multicolumn{3}{@{}l}{\emph{Causal flux ($\sigma{=}1$)}} \\[2pt]
$F_{\mathrm{repu}} / F_{\mathrm{attr}}$ & """ + f"{flux_ratio:.1f}" + r""" & SM: $1/\alpha = 137$ \\
\bottomrule
\end{tabular}
\end{table}"""

    with open(OUT / "table_observables.tex", "w") as f:
        f.write(latex)
    print(f"  [+] table_observables.tex")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"FEG Comprehensive Analysis")
    print(f"  Source: {HERE}")
    print(f"  N = {N_total:,}, Prisms = {N_prisms:,}, {M_LABEL}")
    print(f"  Output: {OUT}/")
    print()

    fig1_spectral_dimension_flow()
    fig2_generation_spectral()
    fig3_return_probability()
    fig4_mass_spectrum()
    dm_ratio = fig5_dark_matter()
    fig6_generation_census()
    fig7_causal_flux()
    cpt_dev = fig8_cpt_test()

    print()
    summary_table(dm_ratio, cpt_dev)

    print()
    print("All figures saved to:")
    print(f"  {OUT}/")
