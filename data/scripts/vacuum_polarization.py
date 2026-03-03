#!/usr/bin/env python3
"""
vacuum_polarization.py -- Publication-quality figures for geometric vacuum polarization.

Produces 4 figures demonstrating that the i.i.d. occupancy model succeeds for
mass but fails for charge, revealing discrete vacuum polarization from
Kuratowski planarity constraints.

Figures:
    1. vp_q_running.pdf      -- Q_topo running with N (i.i.d. line + FSS fit)
    2. vp_mu_effective.pdf   -- Effective mu(n) per belly size (UV screening)
    3. vp_inv_alpha.pdf      -- 1/alpha running with FSS extrapolation
    4. vp_mass_vs_charge.pdf -- Side-by-side: mass ratios flat, Q_topo runs

Usage:  python data/scripts/vacuum_polarization.py
Output: data/figures/vp_*.pdf
"""

import math
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# Publication style (matches feg_analysis.py)
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

# Colourblind-safe palette (from feg_analysis.py)
C = {
    "vac":   "#4C72B0",
    "def":   "#C44E52",
    "gen1":  "#55A868",
    "gen2":  "#8172B2",
    "gen3":  "#CCB974",
    "anti1": "#64B5CD",
    "dark":  "#2f2f2f",
    "vis":   "#E8A838",
    "iid":   "#C44E52",    # i.i.d. prediction
    "obs":   "#4C72B0",    # observed data
    "fit":   "#2980b9",    # FSS fit line
    "target": "#f39c12",   # physical target
    "screen": "#8172B2",   # screened region
}

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]  # data/scripts -> data -> repo root
DATA_ROOT = REPO_ROOT / "data"
ENSEMBLE_DIR = DATA_ROOT / "ensemble_10M_final"
FSS_DIR = DATA_ROOT / "fss_scaling"
OUT = DATA_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_topo(path):
    """Parse key-value topology_summary CSV -> dict."""
    d = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('key'):
                continue
            parts = s.split(',', 1)
            if len(parts) == 2:
                d[parts[0].strip()] = parts[1].strip()
    return d


def load_belly_distribution():
    """Load belly distribution from mass_spectrum CSV."""
    for name in ["mass_spectrum_M20.csv", "mass_spectrum.csv"]:
        p = ENSEMBLE_DIR / name
        if p.exists():
            ns, fs = [], []
            with open(p) as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith('#') or s.startswith('intermediates'):
                        continue
                    parts = s.split(',')
                    if len(parts) >= 2:
                        try:
                            ns.append(int(parts[0]))
                            fs.append(int(parts[1]))
                        except ValueError:
                            continue
            if ns:
                return np.array(ns, dtype=float), np.array(fs, dtype=float)
    raise FileNotFoundError(f"No mass_spectrum CSV found in {ENSEMBLE_DIR}")


def load_fss_data():
    """Load Q_topo and mass data at each FSS lattice size."""
    N_VALUES = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    results = []
    for n in N_VALUES:
        path = None
        if n == 10_000_000:
            for name in ["topology_summary_M20.csv", "topology_summary.csv"]:
                p = ENSEMBLE_DIR / name
                if p.exists():
                    path = p
                    break
        if path is None:
            p = FSS_DIR / f"N_{n}" / "topology_summary.csv"
            if p.exists():
                path = p
        if path is None:
            continue
        d = _parse_topo(str(path))
        if 'phase_sq_total' in d and 'mass_sq_total' in d:
            psq = int(d['phase_sq_total'])
            msq = int(d['mass_sq_total'])
            q = psq / msq if msq > 0 else 0.0
            entry = {
                'N': n, 'Q_topo': q,
                'mass_gen1': float(d.get('avg_mass_gen1', 0)),
                'mass_gen2': float(d.get('avg_mass_gen2', 0)),
                'mass_gen3': float(d.get('avg_mass_gen3', 0)),
            }
            if entry['mass_gen1'] > 0:
                entry['ratio_m2_m1'] = entry['mass_gen2'] / entry['mass_gen1']
                entry['ratio_m3_m1'] = entry['mass_gen3'] / entry['mass_gen1']
            results.append(entry)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Derived quantities
# ═══════════════════════════════════════════════════════════════════════════════

# Measured phases
pp, p0, pm = 4340 / 13648, 250 / 13648, 9058 / 13648
mu = pp - pm
sigma2 = (pp + pm) - mu**2

# Load belly data and compute moments
N_belly, f_belly = load_belly_distribution()
f_norm = f_belly / f_belly.sum()
mean_N = float(np.sum(N_belly * f_norm))
mean_N2 = float(np.sum(N_belly**2 * f_norm))

# i.i.d. prediction
Q_PRED = mu**2 + sigma2 * mean_N / mean_N2

# Load FSS data
fss_data = load_fss_data()

# FSS fit
x_fss = np.array([d['N']**(-0.25) for d in fss_data])
q_fss = np.array([d['Q_topo'] for d in fss_data])


def fss_linear(x, q_inf, a):
    return q_inf + a * x


popt, pcov = curve_fit(fss_linear, x_fss, q_fss)
Q_INF = popt[0]
A_FSS = popt[1]
Q_INF_ERR = math.sqrt(pcov[0, 0])
INV_ALPHA_INF = 8 * math.pi / Q_INF


# ═══════════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════════

def savefig(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    print(f"  [+] {name}.pdf / .png")
    plt.close(fig)


def n_label(n):
    if n >= 1_000_000:
        return f"$N={n // 1_000_000}$M"
    return f"$N={n // 1000}$k"


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 -- Q_topo Running with N
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_q_running():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Data points
    ax.plot(x_fss, q_fss, 'ko', markersize=8, zorder=5, label='Simulation data')

    # Labels
    for d in fss_data:
        x = d['N']**(-0.25)
        ax.annotate(n_label(d['N']), (x, d['Q_topo']),
                    textcoords="offset points", xytext=(8, -12), fontsize=9)

    # FSS fit
    x_fit = np.linspace(0, max(x_fss) * 1.1, 200)
    q_fit = fss_linear(x_fit, Q_INF, A_FSS)
    ax.plot(x_fit, q_fit, '-', color=C["fit"], linewidth=2,
            label=f'FSS fit: $\\mathcal{{Q}}_\\infty = {Q_INF:.4f} \\pm {Q_INF_ERR:.4f}$')

    # Extrapolated point
    ax.plot(0, Q_INF, '^', color=C["fit"], markersize=12, zorder=6,
            label=f'$N \\to \\infty$: $1/\\alpha = {INV_ALPHA_INF:.1f}$')

    # i.i.d. prediction (horizontal)
    ax.axhline(y=Q_PRED, color=C["iid"], linestyle='--', linewidth=2,
               label=f'i.i.d. prediction: $\\mathcal{{Q}}_{{\\mathrm{{pred}}}} = {Q_PRED:.4f}$')

    # Physical target
    Q_TARGET = 8 * math.pi / 137.036
    ax.axhline(y=Q_TARGET, color=C["target"], linestyle=':', linewidth=1.5,
               label=f'$8\\pi/137 = {Q_TARGET:.4f}$')

    # Shade the screening gap at N=10M
    q_10m = fss_data[-1]['Q_topo']
    ax.annotate('', xy=(x_fss[-1] + 0.003, q_10m), xytext=(x_fss[-1] + 0.003, Q_PRED),
                arrowprops=dict(arrowstyle='<->', color=C["screen"], lw=1.5))
    mid_q = (q_10m + Q_PRED) / 2
    overshoot = (Q_PRED - q_10m) / q_10m * 100
    ax.text(x_fss[-1] + 0.005, mid_q, f'{overshoot:.1f}%\nscreening',
            fontsize=9, color=C["screen"], va='center')

    ax.set_xlabel('$N^{-1/4}$  (finite-size variable)')
    ax.set_ylabel('$\\mathcal{Q}_{\\mathrm{topo}} = \\Sigma|\\Phi|^2 / \\Sigma N^2$')
    ax.set_title('Vacuum Polarization: i.i.d. Overshoot of Topological Charge')
    ax.legend(loc='upper left', fontsize=9, frameon=True, framealpha=0.9,
              edgecolor="0.85")
    ax.set_xlim(left=-0.003)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "vp_q_running")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 -- Effective mu(n) per Belly Size
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_mu_effective():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    Q_obs = fss_data[-1]['Q_topo']  # N=10M
    mu_iid = abs(mu)

    n_range = np.arange(3, 31)
    mu_eff = np.full_like(n_range, np.nan, dtype=float)
    screened_mask = np.zeros(len(n_range), dtype=bool)

    for i, n in enumerate(n_range):
        delta = Q_obs - sigma2 / n
        if delta < 0:
            screened_mask[i] = True
        else:
            mu_eff[i] = math.sqrt(delta)

    # Shaded UV region (screened)
    n_cross = math.ceil(sigma2 / Q_obs)
    ax.axvspan(2.5, n_cross - 0.5, color=C["screen"], alpha=0.08, zorder=1,
               label=f'Total screening ($n < {n_cross}$)')

    # Screened points (plot at y=0 with markers)
    n_scr = n_range[screened_mask]
    ax.plot(n_scr, np.zeros_like(n_scr), 'x', color=C["iid"], markersize=10,
            markeredgewidth=2, zorder=5, label=f'$\\mu_{{\\mathrm{{eff}}}}$ imaginary')

    # Real mu_eff
    valid = ~screened_mask
    ax.plot(n_range[valid], mu_eff[valid], 'o-', color=C["obs"], markersize=6,
            zorder=5, label='$\\mu_{\\mathrm{eff}}(n)$')

    # i.i.d. reference line
    ax.axhline(y=mu_iid, color=C["iid"], linestyle='--', linewidth=2,
               label=f'i.i.d. $|\\mu| = {mu_iid:.4f}$')

    # Zero line
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    # Annotation: UV screening mechanism
    ax.annotate('Kuratowski constraints\nforce phase cancellation\nat small belly sizes',
                xy=(3.5, 0), xytext=(8, -0.08),
                arrowprops=dict(arrowstyle='->', color=C["screen"], lw=1.2),
                fontsize=9, color=C["screen"],
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C["screen"],
                          alpha=0.9))

    # Annotation: convergence to i.i.d.
    ax.annotate(f'Approaches i.i.d.\nat $n \\approx {n_cross + 7}$',
                xy=(n_cross + 7, mu_eff[n_cross + 7 - 3]),
                xytext=(22, 0.20),
                arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8),
                fontsize=9, color='0.4')

    ax.set_xlabel('Belly size $n$ (intermediate nodes)')
    ax.set_ylabel('Effective charge mean $\\mu_{\\mathrm{eff}}(n)$')
    ax.set_title('Vacuum Polarization: UV Charge Screening by Belly Size')
    ax.set_xlim(2.5, 31)
    ax.set_ylim(-0.15, 0.50)
    ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.9,
              edgecolor="0.85")

    fig.tight_layout()
    savefig(fig, "vp_mu_effective")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 -- 1/alpha Running
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_inv_alpha():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Compute 1/alpha at each N
    inv_alpha_arr = np.array([8 * math.pi / d['Q_topo'] for d in fss_data])

    # Data points
    ax.plot(x_fss, inv_alpha_arr, 'ko', markersize=8, zorder=5,
            label='Simulation data')

    for d in fss_data:
        x = d['N']**(-0.25)
        inv_a = 8 * math.pi / d['Q_topo']
        ax.annotate(n_label(d['N']), (x, inv_a),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)

    # FSS fit curve (from Q fit, transformed)
    x_fit = np.linspace(0, max(x_fss) * 1.1, 200)
    q_fit = fss_linear(x_fit, Q_INF, A_FSS)
    inv_alpha_fit = 8 * math.pi / q_fit
    ax.plot(x_fit, inv_alpha_fit, '-', color=C["fit"], linewidth=2,
            label=f'FSS extrapolation: $1/\\alpha_\\infty = {INV_ALPHA_INF:.1f}$')

    # Physical reference
    ax.axhline(y=137.036, color=C["target"], linestyle='--', linewidth=2,
               label='$1/\\alpha = 137.036$ (physical)')

    # Shade the RG running domain
    ax.fill_between([0, max(x_fss) * 1.1], 137.036, INV_ALPHA_INF,
                    color=C["target"], alpha=0.06, zorder=1)
    ax.text(0.002, (137.036 + INV_ALPHA_INF) / 2,
            'RG running domain\n(Planck $\\to$ lab)',
            fontsize=9, color=C["target"], va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C["target"],
                      alpha=0.8))

    # Extrapolated point
    ax.plot(0, INV_ALPHA_INF, '^', color=C["fit"], markersize=12, zorder=6)

    ax.set_xlabel('$N^{-1/4}$  (finite-size variable)')
    ax.set_ylabel('$1/\\alpha = 8\\pi / \\mathcal{Q}_{\\mathrm{topo}}$')
    ax.set_title('Running Coupling: Planck Scale to Continuum')
    ax.legend(loc='lower left', fontsize=9, frameon=True, framealpha=0.9,
              edgecolor="0.85")
    ax.set_xlim(left=-0.003)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "vp_inv_alpha")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 -- Mass vs Charge Diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_mass_vs_charge():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    N_arr = np.array([d['N'] for d in fss_data])

    # ── Left panel: Mass ratios (flat across N) ──
    has_mass = all('ratio_m2_m1' in d for d in fss_data)
    if has_mass:
        r21 = np.array([d['ratio_m2_m1'] for d in fss_data])
        r31 = np.array([d['ratio_m3_m1'] for d in fss_data])

        ax1.plot(N_arr, r21, 'o-', color=C["gen2"], markersize=7,
                 label='$m_2/m_1$ (observed)')
        ax1.plot(N_arr, r31, 's-', color=C["gen3"], markersize=7,
                 label='$m_3/m_1$ (observed)')

        # i.i.d. predictions (horizontal) -- compute inline
        def _gen_prob(n, pa, pb, pc):
            p1 = pa**n + pb**n + pc**n
            p3 = 1 - (1-pa)**n - (1-pb)**n - (1-pc)**n + p1
            return p1, 1 - p1 - p3, p3

        _masses_pred = {}
        for g in (1, 2, 3):
            num, den = 0.0, 0.0
            for j, bn in enumerate(N_belly):
                p1, p2, p3 = _gen_prob(bn, pp, p0, pm)
                pg = [p1, p2, p3][g - 1]
                num += bn * pg * f_norm[j]
                den += pg * f_norm[j]
            _masses_pred[g] = num / den if den > 0 else 0.0
        r21_pred = _masses_pred[2] / _masses_pred[1]
        r31_pred = _masses_pred[3] / _masses_pred[1]

        ax1.axhline(y=r21_pred, color=C["gen2"], linestyle='--', linewidth=1.5,
                     alpha=0.7, label=f'i.i.d. $m_2/m_1 = {r21_pred:.3f}$')
        ax1.axhline(y=r31_pred, color=C["gen3"], linestyle='--', linewidth=1.5,
                     alpha=0.7, label=f'i.i.d. $m_3/m_1 = {r31_pred:.3f}$')

    ax1.set_xscale('log')
    ax1.set_xlabel('Lattice size $N$')
    ax1.set_ylabel('Mass ratio')
    ax1.set_title('Mass Ratios: Flat (i.i.d. works)')
    ax1.legend(loc='center right', fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax1.grid(True, alpha=0.3)

    # Annotation
    ax1.text(0.05, 0.05, 'Mass = counting observable\ni.i.d. model PASSES',
             transform=ax1.transAxes, fontsize=10, color=C["gen1"],
             va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C["gen1"],
                       alpha=0.9))

    # ── Right panel: Q_topo (runs with N) ──
    Q_arr = np.array([d['Q_topo'] for d in fss_data])

    ax2.plot(N_arr, Q_arr, 'o-', color=C["obs"], markersize=7, zorder=5,
             label='$\\mathcal{Q}_{\\mathrm{topo}}$ (observed)')
    ax2.axhline(y=Q_PRED, color=C["iid"], linestyle='--', linewidth=2,
                label=f'i.i.d. prediction: {Q_PRED:.4f}')

    # Shade the gap
    ax2.fill_between(N_arr, Q_arr, Q_PRED, color=C["screen"], alpha=0.15,
                     zorder=2)

    ax2.set_xscale('log')
    ax2.set_xlabel('Lattice size $N$')
    ax2.set_ylabel('$\\mathcal{Q}_{\\mathrm{topo}}$')
    ax2.set_title('Topological Charge: Running (i.i.d. fails)')
    ax2.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9,
               edgecolor="0.85")
    ax2.grid(True, alpha=0.3)

    # Annotation
    ax2.text(0.05, 0.05, 'Charge = summation observable\ni.i.d. model FAILS',
             transform=ax2.transAxes, fontsize=10, color=C["iid"],
             va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C["iid"],
                       alpha=0.9))

    fig.suptitle('Mass vs Charge: Why i.i.d. Succeeds for One and Fails for the Other',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "vp_mass_vs_charge")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Vacuum Polarization Figures")
    print(f"  Data: {DATA_ROOT}")
    print(f"  Output: {OUT}/")
    print(f"  Q_pred (i.i.d.) = {Q_PRED:.4f}")
    print(f"  Q_inf (FSS)     = {Q_INF:.4f} +/- {Q_INF_ERR:.4f}")
    print(f"  1/alpha_inf     = {INV_ALPHA_INF:.1f}")
    print()

    fig1_q_running()
    fig2_mu_effective()
    fig3_inv_alpha()
    fig4_mass_vs_charge()

    print()
    print(f"All figures saved to: {OUT}/")
