#!/usr/bin/env python3
"""
GUE Pair Correlation: Random Matrix Theory Test for the Causal Graph
====================================================================

Tests the nearest-neighbor eigenvalue spacing distribution of the Hasse
diagram Laplacian against three universality classes:

  Poisson:  P(s) = exp(-s)                  — uncorrelated (integrable)
  GOE:      P(s) = (π/2) s exp(-π s²/4)    — time-reversal symmetric (β=1)
  GUE:      P(s) = (32/π²) s² exp(-4s²/π)  — broken time-reversal  (β=2)

The Montgomery-Odlyzko law (1987-1989) established that the non-trivial
zeros of the Riemann zeta function exhibit GUE pair correlation statistics.
If the Hasse diagram Laplacian of the FEG causal set also exhibits GUE
statistics, this provides numerical evidence that the causal graph sits in
the same universality class as the Riemann zeros.

Method
------
  1. Sprinkle N events into a 4D causal diamond  (matching diamond.rs)
  2. Build the Hasse diagram via transitive reduction
  3. Construct two Laplacians:
     (a) Combinatorial:  L = D - A    (real symmetric, undirected Hasse)
     (b) Hermitian:      L_H = D - Γ  (complex Hermitian, causal phase θ)
  4. Eigendecompose, unfold the spectrum, compute nearest-neighbor spacings
  5. Compare to Poisson / GOE / GUE via KS test and spacing ratio ⟨r⟩

Why two Laplacians?
-------------------
Real symmetric matrices belong to GOE (β=1) by construction.  To access the
GUE class (β=2) one must break time-reversal symmetry.  The Hasse diagram is
a DAG — it carries a natural causal arrow.  The Hermitian Laplacian encodes
this direction as a complex phase (Guo & Mohar, 2017):

    Γ[i,j] = e^{iθ}   if i → j  in Hasse  (past → future)
    Γ[j,i] = e^{-iθ}   conjugate

This is Hermitian by construction and explicitly breaks T-symmetry for θ ≠ 0.

The spacing ratio ⟨r⟩ (Atas et al., 2013) provides an unfolding-independent
cross-check:  Poisson ≈ 0.386,  GOE ≈ 0.531,  GUE ≈ 0.600.

Usage:
    python gue_correlation.py
"""
import numpy as np
from pathlib import Path
from scipy.special import erf
from scipy.stats import kstest


SCRIPT_DIR = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Causal Diamond Construction  (matches diamond.rs)
# ══════════════════════════════════════════════════════════════════════════════

def sprinkle(N, seed=42):
    """Poisson-sprinkle N events into a 4D causal diamond  |t| + r ≤ T/2.

    Returns points sorted by coordinate time (matching time_order in diamond.rs).
    Columns: (t, x, y, z).
    """
    rng = np.random.default_rng(seed)
    T = (24.0 * N / np.pi) ** 0.25
    half_T = T / 2.0

    points = np.empty((N, 4))
    count = 0
    while count < N:
        batch_size = max(N - count, 256) * 8
        batch = rng.uniform(-half_T, half_T, size=(batch_size, 4))
        t = batch[:, 0]
        r = np.sqrt(batch[:, 1] ** 2 + batch[:, 2] ** 2 + batch[:, 3] ** 2)
        mask = np.abs(t) + r <= half_T
        accept = batch[mask]
        n_accept = min(len(accept), N - count)
        points[count:count + n_accept] = accept[:n_accept]
        count += n_accept

    order = np.argsort(points[:, 0])
    return points[order]


def build_hasse(points):
    """Build the Hasse diagram (transitive reduction of the causal order).

    Returns the directed adjacency matrix H  (bool, N×N)
    where H[i,j] = True iff there is a Hasse link from i (past) to j (future).
    """
    N = len(points)
    t = points[:, 0]
    spatial = points[:, 1:]

    # Temporal separation:  dt[i,j] = t_j - t_i
    dt = t[np.newaxis, :] - t[:, np.newaxis]

    # Spatial distance squared via  ||xi - xj||² = ||xi||² + ||xj||² - 2 xi·xj
    r2 = np.sum(spatial ** 2, axis=1)
    dr2 = r2[:, np.newaxis] + r2[np.newaxis, :] - 2.0 * (spatial @ spatial.T)
    np.maximum(dr2, 0.0, out=dr2)  # clamp floating-point noise

    # Causal order:  i ≺ j  iff  dt > 0  and  dt² > dr²  (strict timelike)
    C = (dt > 1e-12) & (dt ** 2 > dr2)

    # Transitive reduction:  remove edge i→j if ∃ intermediate path of length ≥ 2.
    # Since C is the full transitive closure, C²[i,j] > 0  ⟺  such a path exists.
    C_float = C.astype(np.float32)
    C2 = C_float @ C_float
    H = C & (C2 < 0.5)

    return H


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Laplacian Construction
# ══════════════════════════════════════════════════════════════════════════════

def hasse_to_laplacians(H, theta=np.pi / 4):
    """Construct two graph Laplacians from the directed Hasse diagram.

    L_real   Combinatorial Laplacian of the undirected symmetrisation.
             Real symmetric → GOE universality class (expected).

    L_herm   Hermitian Laplacian with causal phase encoding.
             Γ[i,j] = e^{iθ}  if i→j,   e^{-iθ}  if j→i.
             Complex Hermitian → GUE universality class (if T-breaking works).
    """
    H_float = H.astype(np.float64)

    # --- Undirected (real symmetric) ---
    A_und = np.maximum(H_float, H_float.T)
    D_und = np.diag(A_und.sum(axis=1))
    L_real = D_und - A_und

    # --- Hermitian adjacency (Guo & Mohar, 2017) ---
    z = np.exp(1j * theta)
    Gamma = H_float * z + H_float.T * np.conj(z)
    D_herm = np.diag(np.abs(Gamma).sum(axis=1).real)
    L_herm = D_herm - Gamma

    return L_real, L_herm


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Spectral Unfolding & Spacing Computation
# ══════════════════════════════════════════════════════════════════════════════

def eigenvalues_sorted(L):
    """Sorted real eigenvalues of a real symmetric or Hermitian matrix."""
    evals = np.linalg.eigvalsh(L)
    return np.sort(evals.real)


def unfold_spectrum(evals, poly_degree=7, trim_fraction=0.05):
    """Unfold the eigenvalue spectrum to unit mean spacing.

    1. Trim the edges (top and bottom trim_fraction) to avoid boundary effects.
    2. Fit a polynomial  N̂(E)  to the cumulative staircase function.
    3. Map eigenvalues: ε_i = N̂(E_i).

    Returns the unfolded eigenvalues.
    """
    N = len(evals)
    lo = int(trim_fraction * N)
    hi = N - lo
    trimmed = evals[lo:hi]

    # Staircase: index  i  vs  eigenvalue  E_i
    indices = np.arange(len(trimmed), dtype=np.float64)

    # Polynomial fit  E → N̂(E)  for the smooth cumulative density
    coeffs = np.polyfit(trimmed, indices, poly_degree)
    unfolded = np.polyval(coeffs, trimmed)

    return unfolded


def nearest_neighbor_spacings(unfolded):
    """Normalised nearest-neighbor spacings (mean = 1)."""
    spacings = np.diff(unfolded)
    spacings = spacings[spacings > 1e-10]  # drop degeneracies
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings = spacings / mean_s
    return spacings


def spacing_ratios(spacings):
    """Consecutive spacing ratios  r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1}).

    Unfolding-independent diagnostic (Atas et al., 2013).
    """
    if len(spacings) < 2:
        return np.array([])
    s1 = spacings[:-1]
    s2 = spacings[1:]
    return np.minimum(s1, s2) / np.maximum(s1, s2)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Theoretical Distributions  (Wigner surmises)
# ══════════════════════════════════════════════════════════════════════════════

def pdf_poisson(s):
    return np.exp(-s)


def cdf_poisson(s):
    return 1.0 - np.exp(-s)


def pdf_goe(s):
    return (np.pi / 2) * s * np.exp(-np.pi * s ** 2 / 4)


def cdf_goe(s):
    return 1.0 - np.exp(-np.pi * s ** 2 / 4)


def pdf_gue(s):
    return (32.0 / np.pi ** 2) * s ** 2 * np.exp(-4.0 * s ** 2 / np.pi)


def cdf_gue(s):
    return erf(2.0 * s / np.sqrt(np.pi)) - \
           (4.0 * s / np.pi) * np.exp(-4.0 * s ** 2 / np.pi)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: Statistical Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_ks_tests(spacings):
    """Kolmogorov-Smirnov tests against Poisson, GOE, GUE."""
    results = {}
    for name, cdf_func in [('Poisson', cdf_poisson),
                            ('GOE', cdf_goe),
                            ('GUE', cdf_gue)]:
        D, p = kstest(spacings, cdf_func)
        results[name] = (D, p)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Configuration ─────────────────────────────────────────────────
    N = 1000             # Events per causal diamond
    M = 20               # Ensemble realisations
    SEEDS = list(range(42, 42 + M))
    POLY_DEGREE = 7      # Unfolding polynomial degree
    TRIM_FRAC = 0.05     # Trim 5% from each spectral edge
    THETA = np.pi / 4    # Causal phase for Hermitian adjacency

    print("=" * 72)
    print("GUE PAIR CORRELATION TEST FOR THE CAUSAL GRAPH LAPLACIAN")
    print("Fractal Entropic Geometrodynamics — Random Matrix Theory")
    print("=" * 72)
    print(f"\nConfiguration: N = {N},  M = {M},  θ = π/4")
    print(f"Unfolding: polynomial degree {POLY_DEGREE}, trim {100*TRIM_FRAC:.0f}%")

    # ── Ensemble loop ─────────────────────────────────────────────────
    all_spacings_real = []
    all_spacings_herm = []
    all_ratios_real = []
    all_ratios_herm = []
    graph_stats = []

    for i, seed in enumerate(SEEDS):
        print(f"\r  Realisation {i+1}/{M} (seed {seed})...", end="", flush=True)

        pts = sprinkle(N, seed=seed)
        H = build_hasse(pts)
        n_edges = int(H.sum())
        avg_degree = 2 * n_edges / N

        L_real, L_herm = hasse_to_laplacians(H, theta=THETA)

        evals_real = eigenvalues_sorted(L_real)
        evals_herm = eigenvalues_sorted(L_herm)

        # Drop trivial zero eigenvalue(s) from connected components
        threshold = 1e-6
        evals_real = evals_real[evals_real > threshold]
        evals_herm = evals_herm[evals_herm > threshold]

        if len(evals_real) > 50:
            unf = unfold_spectrum(evals_real, POLY_DEGREE, TRIM_FRAC)
            sp = nearest_neighbor_spacings(unf)
            all_spacings_real.append(sp)
            all_ratios_real.append(spacing_ratios(sp))

        if len(evals_herm) > 50:
            unf = unfold_spectrum(evals_herm, POLY_DEGREE, TRIM_FRAC)
            sp = nearest_neighbor_spacings(unf)
            all_spacings_herm.append(sp)
            all_ratios_herm.append(spacing_ratios(sp))

        graph_stats.append((n_edges, avg_degree, len(evals_real), len(evals_herm)))

    print("\r  Done." + " " * 40)

    spacings_real = np.concatenate(all_spacings_real)
    spacings_herm = np.concatenate(all_spacings_herm)
    ratios_real = np.concatenate(all_ratios_real)
    ratios_herm = np.concatenate(all_ratios_herm)

    # ── Graph Statistics ──────────────────────────────────────────────
    edges_arr = np.array([s[0] for s in graph_stats])
    degree_arr = np.array([s[1] for s in graph_stats])

    print("\n" + "─" * 72)
    print("GRAPH STATISTICS (ensemble)")
    print("─" * 72)
    print(f"  Hasse edges per graph:    {edges_arr.mean():.0f} ± {edges_arr.std():.0f}")
    print(f"  Mean undirected degree:   {degree_arr.mean():.1f} ± {degree_arr.std():.1f}")
    print(f"  Eigenvalues per graph:    ~{graph_stats[0][2]} (real), "
          f"~{graph_stats[0][3]} (Hermitian)")
    print(f"  Total pooled spacings:    {len(spacings_real)} (real), "
          f"{len(spacings_herm)} (Hermitian)")

    # ── KS Tests ──────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("KOLMOGOROV-SMIRNOV TESTS")
    print("─" * 72)

    for label, spacings in [
        ("Combinatorial Laplacian (real symmetric)", spacings_real),
        ("Hermitian Laplacian (causal phase θ=π/4)", spacings_herm),
    ]:
        print(f"\n  {label}")
        print(f"  Spacings: {len(spacings)},  mean = {spacings.mean():.4f},  "
              f"std = {spacings.std():.4f}")

        ks = run_ks_tests(spacings)
        best_name = min(ks, key=lambda k: ks[k][0])

        print(f"\n  {'Class':>10}  {'D-statistic':>12}  {'p-value':>12}  {'Verdict':>10}")
        print("  " + "─" * 50)
        for name, (D, p) in ks.items():
            flag = " ◄ BEST" if name == best_name else ""
            print(f"  {name:>10}  {D:>12.6f}  {p:>12.6e}  {flag}")

    # ── Spacing Ratio ⟨r⟩ ─────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("SPACING RATIO ⟨r⟩  (unfolding-independent diagnostic)")
    print("─" * 72)
    print(f"\n  Theoretical:   Poisson = 0.3863,  GOE = 0.5307,  GUE = 0.5996")
    print(f"\n  Real Laplacian:       ⟨r⟩ = {ratios_real.mean():.4f} ± "
          f"{ratios_real.std()/np.sqrt(len(ratios_real)):.4f}")
    print(f"  Hermitian Laplacian:  ⟨r⟩ = {ratios_herm.mean():.4f} ± "
          f"{ratios_herm.std()/np.sqrt(len(ratios_herm)):.4f}")

    for label, r_mean in [("Real", ratios_real.mean()),
                           ("Hermitian", ratios_herm.mean())]:
        distances = {
            'Poisson': abs(r_mean - 0.3863),
            'GOE': abs(r_mean - 0.5307),
            'GUE': abs(r_mean - 0.5996),
        }
        closest = min(distances, key=distances.get)
        print(f"  {label:>10} → closest to {closest}  "
              f"(Δ = {distances[closest]:.4f})")

    # ── Summary ───────────────────────────────────────────────────────
    ks_real = run_ks_tests(spacings_real)
    ks_herm = run_ks_tests(spacings_herm)
    best_real = min(ks_real, key=lambda k: ks_real[k][0])
    best_herm = min(ks_herm, key=lambda k: ks_herm[k][0])

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"""
  Causal graph:  N = {N} events × M = {M} realisations
  Causal phase:  θ = π/4

  Real Laplacian (undirected Hasse):
    Best KS fit:   {best_real}  (D = {ks_real[best_real][0]:.4f})
    Spacing ratio: ⟨r⟩ = {ratios_real.mean():.4f}

  Hermitian Laplacian (directed, causal phase):
    Best KS fit:   {best_herm}  (D = {ks_herm[best_herm][0]:.4f})
    Spacing ratio: ⟨r⟩ = {ratios_herm.mean():.4f}

  INTERPRETATION:
    The real symmetric Laplacian is expected to show GOE statistics (β = 1).
    The Hermitian Laplacian with causal phase encoding breaks T-symmetry
    and can exhibit GUE statistics (β = 2).

    GUE agreement in either Laplacian would place the causal graph in the
    same universality class as the Riemann zeta zeros (Montgomery-Odlyzko).
    This is a necessary (but not sufficient) condition for the Hilbert-Pólya
    realization within FEG.
""")

    # ── Plotting ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            "Random Matrix Theory Test — Causal Graph Laplacian\n"
            f"FEG  N = {N} × M = {M}  |  θ = π/4",
            fontsize=13, fontweight='bold',
        )

        s_plot = np.linspace(0, 4, 500)

        # (a) NNS histogram — Real Laplacian
        ax = axes[0, 0]
        ax.hist(spacings_real, bins=80, density=True, alpha=0.6,
                color='#4393c3', edgecolor='white', linewidth=0.3,
                label='Data')
        ax.plot(s_plot, pdf_poisson(s_plot), '--', color='#999999',
                linewidth=1.5, label='Poisson')
        ax.plot(s_plot, pdf_goe(s_plot), '-', color='#d6604d',
                linewidth=2, label='GOE')
        ax.plot(s_plot, pdf_gue(s_plot), '-', color='#1a9850',
                linewidth=2, label='GUE')
        ax.set_xlabel('Spacing  s')
        ax.set_ylabel('P(s)')
        ax.set_title('(a) Real Laplacian — NNS')
        ax.set_xlim(0, 4)
        ax.legend(fontsize=8)

        # (b) NNS histogram — Hermitian Laplacian
        ax = axes[0, 1]
        ax.hist(spacings_herm, bins=80, density=True, alpha=0.6,
                color='#762a83', edgecolor='white', linewidth=0.3,
                label='Data')
        ax.plot(s_plot, pdf_poisson(s_plot), '--', color='#999999',
                linewidth=1.5, label='Poisson')
        ax.plot(s_plot, pdf_goe(s_plot), '-', color='#d6604d',
                linewidth=2, label='GOE')
        ax.plot(s_plot, pdf_gue(s_plot), '-', color='#1a9850',
                linewidth=2, label='GUE')
        ax.set_xlabel('Spacing  s')
        ax.set_ylabel('P(s)')
        ax.set_title('(b) Hermitian Laplacian — NNS')
        ax.set_xlim(0, 4)
        ax.legend(fontsize=8)

        # (c) CDF — Real Laplacian
        ax = axes[0, 2]
        sorted_sp = np.sort(spacings_real)
        ecdf = np.arange(1, len(sorted_sp) + 1) / len(sorted_sp)
        ax.plot(sorted_sp, ecdf, color='#4393c3', linewidth=1, label='Data')
        ax.plot(s_plot, cdf_poisson(s_plot), '--', color='#999999',
                linewidth=1.5, label='Poisson')
        ax.plot(s_plot, cdf_goe(s_plot), '-', color='#d6604d',
                linewidth=1.5, label='GOE')
        ax.plot(s_plot, cdf_gue(s_plot), '-', color='#1a9850',
                linewidth=1.5, label='GUE')
        ax.set_xlabel('Spacing  s')
        ax.set_ylabel('CDF')
        ax.set_title('(c) Cumulative — Real L')
        ax.set_xlim(0, 4)
        ax.legend(fontsize=8)

        # (d) CDF — Hermitian Laplacian
        ax = axes[1, 0]
        sorted_sp = np.sort(spacings_herm)
        ecdf = np.arange(1, len(sorted_sp) + 1) / len(sorted_sp)
        ax.plot(sorted_sp, ecdf, color='#762a83', linewidth=1, label='Data')
        ax.plot(s_plot, cdf_poisson(s_plot), '--', color='#999999',
                linewidth=1.5, label='Poisson')
        ax.plot(s_plot, cdf_goe(s_plot), '-', color='#d6604d',
                linewidth=1.5, label='GOE')
        ax.plot(s_plot, cdf_gue(s_plot), '-', color='#1a9850',
                linewidth=1.5, label='GUE')
        ax.set_xlabel('Spacing  s')
        ax.set_ylabel('CDF')
        ax.set_title('(d) Cumulative — Herm L')
        ax.set_xlim(0, 4)
        ax.legend(fontsize=8)

        # (e) Spacing ratio — Real Laplacian
        ax = axes[1, 1]
        ax.hist(ratios_real, bins=50, density=True, alpha=0.6,
                color='#4393c3', edgecolor='white', linewidth=0.3)
        ax.axvline(0.3863, color='#999999', linestyle='--', linewidth=1.5,
                   label=f'Poisson ⟨r⟩=0.386')
        ax.axvline(0.5307, color='#d6604d', linestyle='-', linewidth=1.5,
                   label=f'GOE ⟨r⟩=0.531')
        ax.axvline(0.5996, color='#1a9850', linestyle='-', linewidth=1.5,
                   label=f'GUE ⟨r⟩=0.600')
        ax.axvline(ratios_real.mean(), color='black', linestyle=':',
                   linewidth=2, label=f'Data ⟨r⟩={ratios_real.mean():.3f}')
        ax.set_xlabel('Ratio  r')
        ax.set_ylabel('P(r)')
        ax.set_title('(e) Spacing ratio — Real L')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)

        # (f) Spacing ratio — Hermitian Laplacian
        ax = axes[1, 2]
        ax.hist(ratios_herm, bins=50, density=True, alpha=0.6,
                color='#762a83', edgecolor='white', linewidth=0.3)
        ax.axvline(0.3863, color='#999999', linestyle='--', linewidth=1.5,
                   label=f'Poisson ⟨r⟩=0.386')
        ax.axvline(0.5307, color='#d6604d', linestyle='-', linewidth=1.5,
                   label=f'GOE ⟨r⟩=0.531')
        ax.axvline(0.5996, color='#1a9850', linestyle='-', linewidth=1.5,
                   label=f'GUE ⟨r⟩=0.600')
        ax.axvline(ratios_herm.mean(), color='black', linestyle=':',
                   linewidth=2, label=f'Data ⟨r⟩={ratios_herm.mean():.3f}')
        ax.set_xlabel('Ratio  r')
        ax.set_ylabel('P(r)')
        ax.set_title('(f) Spacing ratio — Herm L')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)

        plt.tight_layout()
        out_path = SCRIPT_DIR / "gue_correlation.pdf"
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved: {out_path}")

    except ImportError:
        print("matplotlib not available — skipping figure generation.")


if __name__ == '__main__':
    main()
