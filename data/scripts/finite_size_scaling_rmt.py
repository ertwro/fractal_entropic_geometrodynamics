#!/usr/bin/env python3
"""
Comprehensive Finite-Size Scaling Analysis
===========================================

Physics:
    In a 4D causal diamond of temporal extent T = (24N/π)^{1/4},
    every intensive observable O(N) receives a finite-size correction
    from the boundary-to-volume ratio:

        boundary / volume ~ T³ / T⁴ = 1/T ∝ N^{-1/4}

    Therefore: O(N) = O_∞ + a · N^{-1/4} + O(N^{-1/2}).

    This script extrapolates ALL observables to the continuum limit N→∞,
    not just Q_topo. Every measurement in the paper is FSS-corrected.

Usage:
    python3 finite_size_scaling.py --run              # Run simulations
    python3 finite_size_scaling.py --analyze           # Full FSS analysis
    python3 finite_size_scaling.py --run --analyze     # Both
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ── Configuration ────────────────────────────────────────────────────────────

N_VALUES = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
M_ENSEMBLE = 10
SEED = 42

SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "FEG_prism"
BINARY      = PROJECT_DIR / "target" / "release" / "feg_prism"
FSS_DIR     = REPO_ROOT / "data" / "fss_scaling"
PROD_DIR    = REPO_ROOT / "data" / "ensemble_10M_final"

# Physical target: Q_topo = 8π/137 for α⁻¹ = 137.036
Q_TARGET = 8.0 * math.pi / 137.036  # ≈ 0.18344

# ── Plotting style ───────────────────────────────────────────────────────────

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'data': '#2c3e50',
    'fit': '#2980b9',
    'target': '#f39c12',
    'gen1': '#e74c3c',
    'gen2': '#27ae60',
    'gen3': '#8e44ad',
    'anti1': '#e67e22',
}

# ── Models ───────────────────────────────────────────────────────────────────

def fss_linear(x, o_inf, a):
    """O(x) = O_∞ + a·x where x = N^{-1/4}."""
    return o_inf + a * x

def fss_quadratic(x, o_inf, a, b):
    """O(x) = O_∞ + a·x + b·x² where x = N^{-1/4}."""
    return o_inf + a * x + b * x * x

# ── Data parsing ─────────────────────────────────────────────────────────────

def parse_topology_summary(csv_path):
    """Parse key-value topology_summary CSV → dict."""
    d = {}
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '' or line.startswith('key'):
                continue
            parts = line.split(',', 1)
            if len(parts) == 2:
                d[parts[0].strip()] = parts[1].strip()
    return d


def parse_results_csv(csv_path):
    """Parse results.csv → list of dicts (one per step)."""
    rows = []
    with open(csv_path) as f:
        lines = f.readlines()
    # Find header line (starts with 'step,')
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('step,'):
            header_idx = i
            break
    if header_idx is None:
        return rows
    header = lines[header_idx].strip().split(',')
    for line in lines[header_idx+1:]:
        vals = line.strip().split(',')
        if len(vals) == len(header):
            row = {}
            for h, v in zip(header, vals):
                try:
                    row[h] = float(v)
                except ValueError:
                    row[h] = v
            rows.append(row)
    return rows


def get_metadata(csv_path):
    """Extract M and convergence status from CSV comments."""
    m_val = '?'
    converged = False
    with open(csv_path) as f:
        for line in f:
            if line.startswith('#') and 'M:' in line:
                m_match = re.search(r'M:\s*(\d+)', line)
                if m_match:
                    m_val = m_match.group(1)
                if 'converged=true' in line:
                    converged = True
                break
    return m_val, converged


# ── Phase 1: Run simulations ────────────────────────────────────────────────

def run_simulations():
    """Run the simulator at each N value."""
    if not BINARY.exists():
        print(f"ERROR: Binary not found at {BINARY}")
        sys.exit(1)

    FSS_DIR.mkdir(parents=True, exist_ok=True)

    for n in N_VALUES:
        if n == 10_000_000 and PROD_DIR.exists():
            if (PROD_DIR / "topology_summary.csv").exists():
                print(f"\n[N={n:,}] Reusing PROD data at {PROD_DIR}")
                continue

        out_dir = FSS_DIR / f"N_{n}"
        out_dir.mkdir(parents=True, exist_ok=True)

        final_topo = out_dir / "topology_summary.csv"
        if final_topo.exists():
            d = parse_topology_summary(str(final_topo))
            if 'phase_sq_total' in d:
                print(f"\n[N={n:,}] Already completed — skipping")
                continue

        print(f"\n{'='*60}")
        print(f"[N={n:,}] Starting ensemble (M={M_ENSEMBLE}, seed={SEED})")
        print(f"{'='*60}")

        cmd = [
            str(BINARY), str(n), str(M_ENSEMBLE), str(out_dir),
            "--inmemory", "--seed", str(SEED),
            "--max-ensemble", str(M_ENSEMBLE),
            "--min-ensemble", str(min(M_ENSEMBLE, 8)),
            "--batch-size", "4",
        ]
        print(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
        if result.returncode != 0:
            print(f"  WARNING: exited with code {result.returncode}")
        else:
            print(f"  [N={n:,}] Done.")


# ── Phase 2: Comprehensive FSS analysis ─────────────────────────────────────

def collect_all_data():
    """Collect ALL observables from every completed N."""
    data = []

    for n in N_VALUES:
        candidates = [FSS_DIR / f"N_{n}" / "topology_summary.csv"]
        if n == 10_000_000:
            candidates.append(PROD_DIR / "topology_summary.csv")

        topo_path = None
        for c in candidates:
            if c.exists():
                d = parse_topology_summary(str(c))
                if 'phase_sq_total' in d:
                    topo_path = c
                    break
        if topo_path is None:
            print(f"  [N={n:,}] No topology data — skipping")
            continue

        # Also find results.csv
        results_path = topo_path.parent / "results.csv"
        results_rows = parse_results_csv(str(results_path)) if results_path.exists() else []

        m_val, converged = get_metadata(str(topo_path))
        d = parse_topology_summary(str(topo_path))

        # ── Topology observables ───────────────────────────────────
        phase_sq = int(d['phase_sq_total'])
        mass_sq  = int(d['mass_sq_total'])
        q_topo = phase_sq / mass_sq if mass_sq > 0 else 0.0

        total_prisms = int(d['total_prisms'])
        gen1 = int(d.get('count_gen1', 0))
        gen2 = int(d.get('count_gen2', 0))
        gen3 = int(d.get('count_gen3', 0))
        anti1 = int(d.get('count_antigen1', 0))
        sterile = int(d.get('count_sterile', 0))
        gen_total = gen1 + gen2 + gen3
        m_int = int(m_val) if m_val != '?' else 1

        entry = {
            'N': n,
            'T': (24.0 * n / math.pi) ** 0.25,
            'x': n ** (-0.25),  # FSS variable
            'M': m_val,
            'converged': converged,
            'source': str(topo_path),
            # Q_topo family
            'Q_topo': q_topo,
            'alpha': q_topo / (8 * math.pi),
            'inv_alpha': 8 * math.pi / q_topo if q_topo > 0 else float('inf'),
            'Omega_energy': 1.0 / q_topo - 1.0 if q_topo > 0 else float('inf'),
            'Omega_linear': float(d.get('omega_ratio', 0)),
            # Masses
            'mass_gen1': float(d.get('avg_mass_gen1', 0)),
            'mass_gen2': float(d.get('avg_mass_gen2', 0)),
            'mass_gen3': float(d.get('avg_mass_gen3', 0)),
            'mass_sterile': float(d.get('avg_mass_sterile', 0)),
            # Generation fractions (per realization, as fraction of gen1+gen2+gen3)
            'frac_gen1': gen1 / gen_total if gen_total > 0 else 0,
            'frac_gen2': gen2 / gen_total if gen_total > 0 else 0,
            'frac_gen3': gen3 / gen_total if gen_total > 0 else 0,
            # Prisms per node (per realization)
            'prisms_per_node': total_prisms / (n * m_int) if n > 0 else 0,
            # Mass ratios
            'ratio_m2_m1': float(d.get('avg_mass_gen2', 0)) / float(d.get('avg_mass_gen1', 1)),
            'ratio_m3_m1': float(d.get('avg_mass_gen3', 0)) / float(d.get('avg_mass_gen1', 1)),
        }

        # ── Spectral dimension from results.csv ───────────────────
        if len(results_rows) >= 4:
            # σ=1 (UV): step index 0 (step=1)
            entry['dS_vac_uv'] = results_rows[0].get('dS_vac', 0)
            entry['dS_def_uv'] = results_rows[0].get('dS_def', 0)
            entry['dS_vac_uv_std'] = results_rows[0].get('dS_vac_std', 0)
            # σ=4 (IR): step index 3 (step=4)
            entry['dS_vac_ir'] = results_rows[3].get('dS_vac', 0)
            entry['dS_def_ir'] = results_rows[3].get('dS_def', 0)
            # Mass from results (per-step row has Mass_Gen1 etc.)
            entry['mass_anti1_spec'] = results_rows[0].get('Mass_Anti1', 0)
            entry['cpt_delta'] = abs(entry['mass_gen1'] - entry['mass_anti1_spec'])
            entry['cpt_frac'] = entry['cpt_delta'] / entry['mass_gen1'] if entry['mass_gen1'] > 0 else 0
        else:
            entry['dS_vac_uv'] = 0
            entry['dS_def_uv'] = 0
            entry['dS_vac_uv_std'] = 0
            entry['dS_vac_ir'] = 0
            entry['dS_def_ir'] = 0
            entry['mass_anti1_spec'] = 0
            entry['cpt_delta'] = 0
            entry['cpt_frac'] = 0

        data.append(entry)

    return data


def fit_observable(x_arr, y_arr, name, p0=None):
    """Fit O = O_∞ + a·x. Return dict with results or None on failure."""
    if len(x_arr) < 2:
        return None
    if p0 is None:
        p0 = [y_arr[-1], (y_arr[0] - y_arr[-1]) / (x_arr[0] - x_arr[-1]) if x_arr[0] != x_arr[-1] else 0]
    try:
        popt, pcov = curve_fit(fss_linear, x_arr, y_arr, p0=p0, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        y_pred = fss_linear(x_arr, *popt)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'name': name,
            'O_inf': float(popt[0]),
            'O_inf_err': float(perr[0]),
            'a': float(popt[1]),
            'a_err': float(perr[1]),
            'R_sq': float(r_sq),
        }
    except Exception as e:
        print(f"  WARNING: Fit failed for {name}: {e}")
        return None


def comprehensive_analysis():
    """Full FSS analysis of all observables."""
    data = collect_all_data()
    if len(data) < 3:
        print(f"\nERROR: Need ≥3 data points, found {len(data)}")
        sys.exit(1)

    out_dir = FSS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build arrays
    N_arr = np.array([d['N'] for d in data], dtype=np.float64)
    x_arr = N_arr ** (-0.25)
    T_arr = np.array([d['T'] for d in data])

    # ═══════════════════════════════════════════════════════════════════════
    # PRINT RAW DATA TABLE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("RAW DATA: All Observables at Each N")
    print("=" * 110)
    print(f"{'N':>12s} {'T':>7s} {'x=N^-¼':>9s} {'Q_topo':>10s} {'1/α':>7s} {'Ω_E':>7s}"
          f" {'m₁':>6s} {'m₂':>6s} {'m₃':>6s} {'m_ā':>6s}"
          f" {'dS_UV':>7s} {'dS_IR':>7s} {'CPT%':>6s} {'M':>3s}")
    print("-" * 110)
    for d in data:
        print(f"{d['N']:>12,d} {d['T']:>7.1f} {d['x']:>9.5f} {d['Q_topo']:>10.6f} {d['inv_alpha']:>7.1f}"
              f" {d['Omega_energy']:>7.3f}"
              f" {d['mass_gen1']:>6.2f} {d['mass_gen2']:>6.2f} {d['mass_gen3']:>6.2f}"
              f" {d['mass_anti1_spec']:>6.2f}"
              f" {d['dS_vac_uv']:>7.4f} {d['dS_vac_ir']:>7.4f}"
              f" {d['cpt_frac']*100:>5.1f}% {d['M']:>3s}")
    print("-" * 110)

    # ═══════════════════════════════════════════════════════════════════════
    # FIT EVERY OBSERVABLE
    # ═══════════════════════════════════════════════════════════════════════
    observables = {
        'Q_topo':      ('Q_topo',         '$\\mathcal{Q}_{\\mathrm{topo}}$'),
        'mass_gen1':   ('mass_gen1',      '$\\bar{N}_{g=1}$'),
        'mass_gen2':   ('mass_gen2',      '$\\bar{N}_{g=2}$'),
        'mass_gen3':   ('mass_gen3',      '$\\bar{N}_{g=3}$'),
        'mass_sterile':('mass_sterile',   '$\\bar{N}_{\\mathrm{sterile}}$'),
        'dS_vac_uv':   ('dS_vac_uv',     '$d_S^{\\mathrm{vac}}(\\sigma{=}1)$'),
        'dS_vac_ir':   ('dS_vac_ir',     '$d_S^{\\mathrm{vac}}(\\sigma{=}4)$'),
        'dS_def_uv':   ('dS_def_uv',     '$d_S^{\\mathrm{def}}(\\sigma{=}1)$'),
        'Omega_energy': ('Omega_energy',  '$\\Omega_{\\mathrm{energy}}$'),
        'Omega_linear': ('Omega_linear',  '$\\Omega_{\\mathrm{dark}}/\\Omega_{\\mathrm{vis}}$'),
        'frac_gen1':   ('frac_gen1',      '$f_{g=1}$'),
        'frac_gen2':   ('frac_gen2',      '$f_{g=2}$'),
        'frac_gen3':   ('frac_gen3',      '$f_{g=3}$'),
        'ratio_m2_m1': ('ratio_m2_m1',   '$m_2/m_1$'),
        'ratio_m3_m1': ('ratio_m3_m1',   '$m_3/m_1$'),
        'cpt_frac':    ('cpt_frac',       'CPT $\\Delta m/m$'),
    }

    fits = {}
    print("\n" + "=" * 100)
    print("FINITE-SIZE SCALING FITS: O(N) = O_∞ + a · N^{-1/4}")
    print("=" * 100)
    print(f"{'Observable':>35s} {'O_∞':>12s} {'± σ':>10s} {'slope a':>10s} {'R²':>10s}")
    print("-" * 100)

    for key, (field, latex) in observables.items():
        y_arr = np.array([d[field] for d in data])
        result = fit_observable(x_arr, y_arr, key)
        if result:
            fits[key] = result
            print(f"{key:>35s} {result['O_inf']:>12.6f} {result['O_inf_err']:>10.6f}"
                  f" {result['a']:>10.4f} {result['R_sq']:>10.6f}")

    print("-" * 100)

    # ── Derived quantities from Q_topo fit ─────────────────────────────
    if 'Q_topo' in fits:
        q_inf = fits['Q_topo']['O_inf']
        q_err = fits['Q_topo']['O_inf_err']
        alpha_inf = q_inf / (8 * math.pi)
        inv_alpha_inf = 1.0 / alpha_inf if alpha_inf > 0 else float('inf')
        omega_inf = 1.0 / q_inf - 1.0 if q_inf > 0 else float('inf')

        print(f"\n{'── Derived from Q_topo fit ──':>35s}")
        print(f"{'α_∞ = Q_∞/(8π)':>35s} {alpha_inf:>12.8f}")
        print(f"{'1/α_∞':>35s} {inv_alpha_inf:>12.2f}   (target: 137.036)")
        print(f"{'Gap to 137':>35s} {abs(inv_alpha_inf - 137.036):>12.2f}   ({abs(inv_alpha_inf - 137.036)/137.036*100:.1f}%)")
        print(f"{'Ω_∞ = 1/Q_∞ − 1':>35s} {omega_inf:>12.4f}   (target: 4.452)")
        print(f"{'α_∞(1+Ω_∞) [= 1/(8π)]':>35s} {alpha_inf*(1+omega_inf):>12.8f}   (exact: {1/(8*math.pi):.8f})")

    # ── CPT convergence ─────────────────────────────────────────────────
    if 'cpt_frac' in fits:
        cpt_inf = fits['cpt_frac']['O_inf']
        print(f"\n{'── CPT Symmetry ──':>35s}")
        print(f"{'Δm/m at N→∞':>35s} {cpt_inf*100:>12.2f}%")
        print(f"  (Exact CPT symmetry = 0%; current trend: {'converging' if cpt_inf < 0.02 else 'open'})")

    # ═══════════════════════════════════════════════════════════════════════
    # EXTRAPOLATED CONTINUUM-LIMIT TABLE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("CONTINUUM LIMIT: Extrapolated Observables (N → ∞)")
    print("=" * 90)
    table_rows = [
        ('$\\mathcal{Q}_{\\mathrm{topo}}$',         'Q_topo',     '{:.6f}'),
        ('$\\alpha = \\mathcal{Q}/(8\\pi)$',        None,         '{:.8f}'),
        ('$1/\\alpha$',                               None,         '{:.2f}'),
        ('$\\Omega_{\\mathrm{energy}} = 1/Q - 1$',  None,         '{:.4f}'),
        ('$d_S^{\\mathrm{vac}}(\\sigma{=}1)$',      'dS_vac_uv',  '{:.4f}'),
        ('$d_S^{\\mathrm{vac}}(\\sigma{=}4)$',      'dS_vac_ir',  '{:.4f}'),
        ('$\\bar{N}_{g=1}$ (Gen 1 mass)',            'mass_gen1',  '{:.3f}'),
        ('$\\bar{N}_{g=2}$ (Gen 2 mass)',            'mass_gen2',  '{:.3f}'),
        ('$\\bar{N}_{g=3}$ (Gen 3 mass)',            'mass_gen3',  '{:.3f}'),
        ('$m_2/m_1$',                                 'ratio_m2_m1', '{:.3f}'),
        ('$m_3/m_1$',                                 'ratio_m3_m1', '{:.3f}'),
        ('CPT $\\Delta m/m$',                         'cpt_frac',   '{:.4f}'),
        ('$f_{g=1}$',                                 'frac_gen1',  '{:.4f}'),
        ('$f_{g=2}$',                                 'frac_gen2',  '{:.4f}'),
        ('$f_{g=3}$',                                 'frac_gen3',  '{:.4f}'),
    ]

    print(f"{'Observable':>45s} {'N=10M':>12s} {'N→∞':>12s} {'± σ':>10s} {'R²':>8s}")
    print("-" * 90)
    for label, key, fmt in table_rows:
        if key and key in fits:
            val_10m = [d for d in data if d['N'] == 10_000_000]
            v10 = val_10m[0][key] if val_10m else 0
            print(f"{label:>45s} {fmt.format(v10):>12s} {fmt.format(fits[key]['O_inf']):>12s}"
                  f" {fmt.format(fits[key]['O_inf_err']):>10s} {fits[key]['R_sq']:>8.4f}")
        elif key is None and 'Q_topo' in fits:
            # Derived quantities
            v10_list = [d for d in data if d['N'] == 10_000_000]
            if '1/\\alpha' in label:
                v10 = v10_list[0]['inv_alpha'] if v10_list else 0
                print(f"{label:>45s} {fmt.format(v10):>12s} {fmt.format(inv_alpha_inf):>12s}"
                      f" {'':>10s} {'':>8s}")
            elif '\\alpha = ' in label:
                v10 = v10_list[0]['alpha'] if v10_list else 0
                print(f"{label:>45s} {fmt.format(v10):>12s} {fmt.format(alpha_inf):>12s}"
                      f" {'':>10s} {'':>8s}")
            elif 'Omega' in label:
                v10 = v10_list[0]['Omega_energy'] if v10_list else 0
                print(f"{label:>45s} {fmt.format(v10):>12s} {fmt.format(omega_inf):>12s}"
                      f" {'':>10s} {'':>8s}")
    print("-" * 90)

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════════════

    x_fit = np.linspace(0, max(x_arr) * 1.1, 200)

    def n_label(n):
        if n >= 1_000_000:
            return f"$N={n//1_000_000}$M"
        return f"$N={n//1000}$k"

    # ─── Fig 1: Q_topo vs N^{-1/4} ────────────────────────────────────
    if 'Q_topo' in fits:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        Q_arr = np.array([d['Q_topo'] for d in data])
        ax.plot(x_arr, Q_arr, 'ko', markersize=8, zorder=5, label='Simulation data')

        f = fits['Q_topo']
        ax.plot(x_fit, fss_linear(x_fit, f['O_inf'], f['a']), '-',
                color=COLORS['fit'], linewidth=2,
                label=f'$\\mathcal{{Q}}_\\infty = {f["O_inf"]:.5f} \\pm {f["O_inf_err"]:.5f}$'
                      f'  ($R^2 = {f["R_sq"]:.5f}$)')
        ax.plot(0, f['O_inf'], '^', color=COLORS['fit'], markersize=12, zorder=6,
                label=f'$N \\to \\infty$: $1/\\alpha = {inv_alpha_inf:.1f}$')
        ax.axhline(y=Q_TARGET, color=COLORS['target'], linestyle='--', linewidth=2,
                    label=f'Target: $8\\pi/137 = {Q_TARGET:.5f}$')

        for d in data:
            ax.annotate(n_label(d['N']), (d['x'], d['Q_topo']),
                        textcoords="offset points", xytext=(8, -12), fontsize=9)

        ax.set_xlabel('$N^{-1/4}$  (finite-size variable)')
        ax.set_ylabel('$\\mathcal{Q}_{\\mathrm{topo}} = \\Sigma|\\Phi|^2 / \\Sigma N^2$')
        ax.set_title('Finite-Size Scaling: Topological Charge Ratio')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(left=-0.003)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(str(out_dir / f'fss_q_topo.{ext}'), dpi=300)
        plt.close(fig)

    # ─── Fig 2: 1/α vs N^{-1/4} ──────────────────────────────────────
    if 'Q_topo' in fits:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        inv_a_arr = np.array([d['inv_alpha'] for d in data])
        ax.plot(x_arr, inv_a_arr, 'ko', markersize=8, zorder=5, label='Simulation data')

        q_fit_vals = fss_linear(x_fit, fits['Q_topo']['O_inf'], fits['Q_topo']['a'])
        inv_alpha_fit = 8 * math.pi / q_fit_vals
        ax.plot(x_fit, inv_alpha_fit, '-', color=COLORS['fit'], linewidth=2,
                label=f'FSS extrapolation: $1/\\alpha_\\infty = {inv_alpha_inf:.1f}$')
        ax.axhline(y=137.036, color=COLORS['target'], linestyle='--', linewidth=2,
                    label='$1/\\alpha = 137.036$')

        for d in data:
            ax.annotate(n_label(d['N']), (d['x'], d['inv_alpha']),
                        textcoords="offset points", xytext=(8, 8), fontsize=9)

        ax.set_xlabel('$N^{-1/4}$  (finite-size variable)')
        ax.set_ylabel('$1/\\alpha = 8\\pi / \\mathcal{Q}_{\\mathrm{topo}}$')
        ax.set_title('Finite-Size Scaling: Fine Structure Constant')
        ax.legend(loc='lower left', fontsize=9)
        ax.set_xlim(left=-0.003)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(str(out_dir / f'fss_inv_alpha.{ext}'), dpi=300)
        plt.close(fig)

    # ─── Fig 3: Mass hierarchy convergence ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key, label, color in [
        ('mass_gen1', 'Gen 1', COLORS['gen1']),
        ('mass_gen2', 'Gen 2', COLORS['gen2']),
        ('mass_gen3', 'Gen 3', COLORS['gen3']),
    ]:
        y = np.array([d[key] for d in data])
        ax.plot(x_arr, y, 'o', color=color, markersize=8, zorder=5)
        if key in fits:
            f = fits[key]
            ax.plot(x_fit, fss_linear(x_fit, f['O_inf'], f['a']), '-',
                    color=color, linewidth=2,
                    label=f'{label}: $\\bar{{N}}_\\infty = {f["O_inf"]:.3f} \\pm {f["O_inf_err"]:.3f}$')
            ax.plot(0, f['O_inf'], '^', color=color, markersize=10, zorder=6)

    # Anti-1 mass for CPT
    anti1_y = np.array([d['mass_anti1_spec'] for d in data])
    ax.plot(x_arr, anti1_y, 's', color=COLORS['anti1'], markersize=7, zorder=4,
            label='Anti-1 (CPT partner)')
    if 'mass_gen1' in fits:
        ax.axhline(y=fits['mass_gen1']['O_inf'], color=COLORS['gen1'],
                    linestyle=':', alpha=0.5)

    ax.set_xlabel('$N^{-1/4}$  (finite-size variable)')
    ax.set_ylabel('Average belly size $\\bar{N}$ (topological mass)')
    ax.set_title('Finite-Size Scaling: Mass Hierarchy and CPT Convergence')
    ax.legend(loc='center left', fontsize=9)
    ax.set_xlim(left=-0.003)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(str(out_dir / f'fss_mass_hierarchy.{ext}'), dpi=300)
    plt.close(fig)

    # ─── Fig 4: Spectral dimension convergence ───────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # UV (σ=1)
    y_uv = np.array([d['dS_vac_uv'] for d in data])
    ax1.plot(x_arr, y_uv, 'ko', markersize=8, zorder=5, label='Simulation')
    if 'dS_vac_uv' in fits:
        f = fits['dS_vac_uv']
        ax1.plot(x_fit, fss_linear(x_fit, f['O_inf'], f['a']), '-',
                 color=COLORS['fit'], linewidth=2,
                 label=f'$d_{{S,\\infty}}^{{\\mathrm{{UV}}}} = {f["O_inf"]:.4f} \\pm {f["O_inf_err"]:.4f}$')
        ax1.plot(0, f['O_inf'], '^', color=COLORS['fit'], markersize=12, zorder=6)
    ax1.axhline(y=2.0, color=COLORS['target'], linestyle='--', linewidth=2, label='$d_S = 2$ (CDT prediction)')
    ax1.set_xlabel('$N^{-1/4}$')
    ax1.set_ylabel('$d_S(\\sigma{=}1)$')
    ax1.set_title('UV Spectral Dimension')
    ax1.legend(fontsize=9)
    ax1.set_xlim(left=-0.003)
    ax1.grid(True, alpha=0.3)

    # IR (σ=4)
    y_ir = np.array([d['dS_vac_ir'] for d in data])
    ax2.plot(x_arr, y_ir, 'ko', markersize=8, zorder=5, label='Simulation')
    if 'dS_vac_ir' in fits:
        f = fits['dS_vac_ir']
        ax2.plot(x_fit, fss_linear(x_fit, f['O_inf'], f['a']), '-',
                 color=COLORS['fit'], linewidth=2,
                 label=f'$d_{{S,\\infty}}^{{\\mathrm{{IR}}}} = {f["O_inf"]:.4f} \\pm {f["O_inf_err"]:.4f}$')
        ax2.plot(0, f['O_inf'], '^', color=COLORS['fit'], markersize=12, zorder=6)
    ax2.axhline(y=4.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='$d_S = 4$')
    ax2.set_xlabel('$N^{-1/4}$')
    ax2.set_ylabel('$d_S(\\sigma{=}4)$')
    ax2.set_title('IR Spectral Dimension')
    ax2.legend(fontsize=9)
    ax2.set_xlim(left=-0.003)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(str(out_dir / f'fss_spectral_dimension.{ext}'), dpi=300)
    plt.close(fig)

    # ─── Fig 5: Ω_energy vs N^{-1/4} ────────────────────────────────
    if 'Q_topo' in fits:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        omega_arr = np.array([d['Omega_energy'] for d in data])
        ax.plot(x_arr, omega_arr, 'ko', markersize=8, zorder=5, label='Simulation data')

        # Compute Ω from Q fit (nonlinear transformation)
        q_fit_vals = fss_linear(x_fit, fits['Q_topo']['O_inf'], fits['Q_topo']['a'])
        omega_from_q = 1.0 / q_fit_vals - 1.0
        ax.plot(x_fit, omega_from_q, '-', color='#c0392b', linewidth=2,
                label=f'From $\\mathcal{{Q}}$ fit: $\\Omega_\\infty = {omega_inf:.3f}$')

        omega_target = 137.036 / (8 * math.pi) - 1
        ax.axhline(y=omega_target, color=COLORS['target'], linestyle='--', linewidth=2,
                    label=f'SM target: $\\Omega = {omega_target:.3f}$')

        for d in data:
            ax.annotate(n_label(d['N']), (d['x'], d['Omega_energy']),
                        textcoords="offset points", xytext=(8, -12), fontsize=9)

        ax.set_xlabel('$N^{-1/4}$')
        ax.set_ylabel('$\\Omega_{\\mathrm{energy}} = 1/\\mathcal{Q} - 1$')
        ax.set_title('Self-Energy Dark Matter Ratio: Convergence')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(left=-0.003)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(str(out_dir / f'fss_omega_energy.{ext}'), dpi=300)
        plt.close(fig)

    # ─── Fig 6: Generation fractions ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key, label, color in [
        ('frac_gen1', '$g{=}1$', COLORS['gen1']),
        ('frac_gen2', '$g{=}2$', COLORS['gen2']),
        ('frac_gen3', '$g{=}3$', COLORS['gen3']),
    ]:
        y = np.array([d[key] for d in data])
        ax.plot(x_arr, y * 100, 'o', color=color, markersize=8, zorder=5)
        if key in fits:
            f = fits[key]
            ax.plot(x_fit, fss_linear(x_fit, f['O_inf'], f['a']) * 100, '-',
                    color=color, linewidth=2,
                    label=f'{label}: ${f["O_inf"]*100:.2f} \\pm {f["O_inf_err"]*100:.2f}$%')

    ax.set_xlabel('$N^{-1/4}$')
    ax.set_ylabel('Generation fraction (%)')
    ax.set_title('Generation Population Fractions: Convergence')
    ax.legend(fontsize=10)
    ax.set_xlim(left=-0.003)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(str(out_dir / f'fss_generation_fractions.{ext}'), dpi=300)
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE ALL RESULTS AS JSON
    # ═══════════════════════════════════════════════════════════════════════
    results = {
        'description': 'Comprehensive FSS analysis: O(N) = O_inf + a * N^{-1/4}',
        'physics': 'Boundary/volume ratio in 4D causal diamond: T^3/T^4 = 1/T ~ N^{-1/4}',
        'data_points': [{
            'N': d['N'],
            'T': d['T'],
            'M': d['M'],
            'converged': d['converged'],
            'Q_topo': d['Q_topo'],
            'inv_alpha': d['inv_alpha'],
            'Omega_energy': d['Omega_energy'],
            'mass_gen1': d['mass_gen1'],
            'mass_gen2': d['mass_gen2'],
            'mass_gen3': d['mass_gen3'],
            'mass_anti1': d['mass_anti1_spec'],
            'dS_vac_uv': d['dS_vac_uv'],
            'dS_vac_ir': d['dS_vac_ir'],
            'frac_gen1': d['frac_gen1'],
            'frac_gen2': d['frac_gen2'],
            'frac_gen3': d['frac_gen3'],
            'cpt_frac': d['cpt_frac'],
        } for d in data],
        'fits': fits,
        'derived': {},
    }
    if 'Q_topo' in fits:
        results['derived'] = {
            'alpha_inf': alpha_inf,
            'inv_alpha_inf': inv_alpha_inf,
            'Omega_inf_from_Q': omega_inf,
            'gap_to_137_percent': abs(inv_alpha_inf - 137.036) / 137.036 * 100,
        }

    json_path = out_dir / 'fss_comprehensive_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # GENERATE LATEX TABLE FOR PAPER
    # ═══════════════════════════════════════════════════════════════════════
    latex_path = out_dir / 'fss_table.tex'
    with open(latex_path, 'w') as f:
        f.write("% Auto-generated by finite_size_scaling.py\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\caption{Finite-size scaling: all observables at five lattice sizes\n")
        f.write("  and their continuum-limit extrapolation via\n")
        f.write("  $\\mathcal{O}(N) = \\mathcal{O}_\\infty + a \\cdot N^{-1/4}$.\n")
        f.write("  The scaling variable $x = N^{-1/4}$ arises from the 4D\n")
        f.write("  boundary/volume ratio $T^3/T^4 = 1/T \\propto N^{-1/4}$.}\n")
        f.write("\\label{tab:fss}\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{@{}lcccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("& $10^5$ & $5{\\times}10^5$ & $10^6$ & $5{\\times}10^6$ & $10^7$ & $N{\\to}\\infty$ \\\\\n")
        f.write("\\midrule\n")

        def row(label, key, fmt='.4f', derived_inf=None):
            vals = []
            for d in data:
                vals.append(f"${d[key]:{fmt}}$")
            if derived_inf is not None:
                vals.append(f"$\\mathbf{{{derived_inf:{fmt}}}}$")
            elif key in fits:
                vals.append(f"$\\mathbf{{{fits[key]['O_inf']:{fmt}}}}$")
            else:
                vals.append("---")
            return f"{label} & {' & '.join(vals)} \\\\\n"

        f.write(row('$\\mathcal{Q}_{\\mathrm{topo}}$', 'Q_topo', '.5f'))
        if 'Q_topo' in fits:
            f.write(f"$1/\\alpha$ & ")
            for d in data:
                f.write(f"${d['inv_alpha']:.1f}$ & ")
            f.write(f"$\\mathbf{{{inv_alpha_inf:.1f}}}$ \\\\\n")
            f.write(f"$\\Omega_{{\\mathrm{{energy}}}}$ & ")
            for d in data:
                f.write(f"${d['Omega_energy']:.3f}$ & ")
            f.write(f"$\\mathbf{{{omega_inf:.3f}}}$ \\\\\n")
        f.write("\\midrule\n")
        f.write(row('$d_S^{\\mathrm{vac}}(\\sigma{=}1)$', 'dS_vac_uv', '.4f'))
        f.write(row('$d_S^{\\mathrm{vac}}(\\sigma{=}4)$', 'dS_vac_ir', '.4f'))
        f.write("\\midrule\n")
        f.write(row('$\\bar{N}_{g=1}$', 'mass_gen1', '.3f'))
        f.write(row('$\\bar{N}_{g=2}$', 'mass_gen2', '.3f'))
        f.write(row('$\\bar{N}_{g=3}$', 'mass_gen3', '.3f'))
        f.write(row('$m_2/m_1$', 'ratio_m2_m1', '.3f'))
        f.write(row('$m_3/m_1$', 'ratio_m3_m1', '.3f'))
        f.write("\\midrule\n")

        # CPT row
        f.write("CPT $\\Delta m/m$ & ")
        for d in data:
            f.write(f"${d['cpt_frac']*100:.1f}\\%$ & ")
        if 'cpt_frac' in fits:
            f.write(f"$\\mathbf{{{fits['cpt_frac']['O_inf']*100:.1f}\\%}}$")
        f.write(" \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"  LaTeX table: {latex_path}")

    print(f"\n  Figures saved to: {out_dir}/")
    for p in sorted(out_dir.glob("fss_*.png")):
        print(f"    {p.name}")

    print("\nDone.")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive FSS Analysis')
    parser.add_argument('--run', action='store_true', help='Run simulations')
    parser.add_argument('--analyze', action='store_true', help='Full FSS analysis')
    args = parser.parse_args()

    if not args.run and not args.analyze:
        parser.print_help()
        sys.exit(0)

    if args.run:
        run_simulations()
    if args.analyze:
        comprehensive_analysis()
