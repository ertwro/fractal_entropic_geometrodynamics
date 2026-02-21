#!/usr/bin/env python3
"""
GUE Convergence Exponent from Finite-Size RMT Data
====================================================

Reads fss_rmt.csv (produced by the fss_rmt Rust binary) and fits the
finite-size convergence model:

    |r_GUE - <r>(N)| = A * N^(-gamma)

where r_GUE = 0.5996 is the GUE Wigner surmise prediction for the
mean spacing ratio.

If fss_rmt.csv contains fewer than 3 lattice sizes, falls back to a
self-contained Python computation at small N (slower but reproducible
without the Rust binary).

GUE prediction for the convergence: gamma ~ 1.

Usage:
    python spectral_form_factor.py [--fallback-sizes 200,500,1000,2000]
                                   [--fallback-M 10] [--seed 42]

Output:
    Console: gamma +/- SE, R^2
    data/figures/spectral_form_factor.pdf (if matplotlib available)
"""
import argparse
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Theoretical GUE limit for the spacing ratio
R_GUE = 0.5996
BD_WEIGHTS = (1, -9, 16, -8)


# ── Self-contained causal diamond pipeline (Python fallback) ──────────

def sprinkle(N, rng):
    """Poisson-sprinkle N events into a 4D causal diamond |t| + r <= T/2."""
    T = (24.0 * N / np.pi) ** 0.25
    half_T = T / 2.0
    points = np.empty((N, 4))
    count = 0
    while count < N:
        batch_size = max(N - count, 256) * 8
        batch = rng.uniform(-half_T, half_T, size=(batch_size, 4))
        t = batch[:, 0]
        r = np.sqrt(batch[:, 1]**2 + batch[:, 2]**2 + batch[:, 3]**2)
        accept = batch[np.abs(t) + r <= half_T]
        take = min(len(accept), N - count)
        points[count:count + take] = accept[:take]
        count += take
    return points


def build_hasse(pts):
    """Build directed Hasse diagram (transitively reduced) via brute force.
    O(N^3) -- only practical for N <= ~2000."""
    N = len(pts)
    order = np.argsort(pts[:, 0])
    pts_sorted = pts[order]
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            dt = pts_sorted[j, 0] - pts_sorted[i, 0]
            if dt <= 0:
                continue
            dx = pts_sorted[j, 1:] - pts_sorted[i, 1:]
            r = np.sqrt(np.sum(dx**2))
            if dt > r:
                is_direct = True
                for nbr in adj[i]:
                    if j in adj[nbr]:
                        is_direct = False
                        break
                if is_direct:
                    adj[i].append(j)
    return adj


def build_bd_matrix(adj, N):
    """Build the BD matrix via layered BFS."""
    B = np.zeros((N, N), dtype=np.int64)
    for src in range(N):
        visited = set([src])
        frontier = [src]
        for depth in range(4):
            next_frontier = []
            for node in frontier:
                for nbr in adj[node]:
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.append(nbr)
                        B[src, nbr] = BD_WEIGHTS[depth]
            frontier = next_frontier
            if not frontier:
                break
    return B


def effective_hamiltonian_eigenvalues(B):
    """H_eff = (B - B^T) / (2i) -> sorted real eigenvalues."""
    D = (B - B.T).astype(np.float64)
    evals = np.linalg.eigvals(D)
    h_evals = evals.imag / 2.0
    h_evals = h_evals[np.abs(h_evals) > 1e-8]
    h_evals.sort()
    return h_evals


def spacing_ratio_mean(evals):
    """Compute mean spacing ratio <r> from sorted eigenvalues."""
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-12]
    if len(spacings) < 2:
        return float('nan')
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return np.mean(ratios)


def compute_r_at_N(N, M, base_seed):
    """Compute ensemble-averaged <r> at lattice size N."""
    r_values = []
    for i in range(M):
        seed = base_seed + i * 7919 + N
        rng = np.random.default_rng(seed)
        pts = sprinkle(N, rng)
        adj = build_hasse(pts)
        B = build_bd_matrix(adj, N)
        evals = effective_hamiltonian_eigenvalues(B)
        r = spacing_ratio_mean(evals)
        if not np.isnan(r):
            r_values.append(r)
    if len(r_values) == 0:
        return float('nan'), float('nan')
    return np.mean(r_values), np.std(r_values, ddof=1) / np.sqrt(len(r_values))


# ── Fitting ───────────────────────────────────────────────────────────

def fit_gamma(N_vals, r_vals):
    """Fit |r_GUE - <r>(N)| = A * N^(-gamma) via log-log regression."""
    delta_r = np.abs(R_GUE - r_vals)
    mask = delta_r > 0
    N_fit = N_vals[mask].astype(float)
    delta_fit = delta_r[mask]

    if len(N_fit) < 3:
        return None

    ln_N = np.log(N_fit)
    ln_delta = np.log(delta_fit)
    n = len(ln_N)

    sx = np.sum(ln_N)
    sy = np.sum(ln_delta)
    sxy = np.sum(ln_N * ln_delta)
    sx2 = np.sum(ln_N**2)

    denom = n * sx2 - sx**2
    if abs(denom) < 1e-15:
        return None

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    gamma = -slope
    A = np.exp(intercept)

    y_mean = sy / n
    ss_tot = np.sum((ln_delta - y_mean)**2)
    ln_pred = intercept + slope * ln_N
    ss_res = np.sum((ln_delta - ln_pred)**2)
    R_sq = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    if n > 2:
        s2 = ss_res / (n - 2)
        se_slope = np.sqrt(s2 * n / denom) if denom > 0 else float('nan')
        gamma_err = se_slope
    else:
        gamma_err = float('nan')

    return {
        'gamma': gamma, 'gamma_err': gamma_err,
        'A': A, 'R_sq': R_sq,
        'N_fit': N_fit, 'delta_fit': delta_fit
    }


def main():
    parser = argparse.ArgumentParser(description="GUE convergence exponent fit")
    parser.add_argument("--fallback-sizes", type=str, default="200,400,800,1500",
                        help="Lattice sizes for Python fallback (default: 200,400,800,1500)")
    parser.add_argument("--fallback-M", type=int, default=10,
                        help="Realisations per size in fallback mode (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed (default: 42)")
    args = parser.parse_args()

    # ── Try reading fss_rmt.csv first ─────────────────────────────────
    csv_path = DATA_DIR / "fss_rmt.csv"
    N_vals = None
    r_vals = None

    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, comment='#')
            df = df.dropna(subset=['r_mean']).copy()
            if len(df) >= 3:
                N_vals = df['N'].values
                r_vals = df['r_mean'].values
                print(f"Read {len(df)} data points from {csv_path}")
        except Exception as e:
            print(f"  Warning: could not read {csv_path}: {e}")

    # ── Fallback: compute in Python ───────────────────────────────────
    if N_vals is None or len(N_vals) < 3:
        sizes = [int(s) for s in args.fallback_sizes.split(',')]
        M = args.fallback_M
        print(f"Fallback mode: computing <r> at N = {sizes} with M = {M}")
        print(f"  (For production results, run the Rust binary: "
              f"cargo run --release --bin fss_rmt -- "
              f"--sizes 500,1000,2000,5000 --m 10 --seed 42)")
        print()

        computed_N = []
        computed_r = []
        for N in sizes:
            print(f"  N = {N:>6} ...", end="", flush=True)
            r_mean, r_se = compute_r_at_N(N, M, args.seed)
            print(f"  <r> = {r_mean:.4f} +/- {r_se:.4f}")
            if not np.isnan(r_mean):
                computed_N.append(N)
                computed_r.append(r_mean)

        N_vals = np.array(computed_N)
        r_vals = np.array(computed_r)

    if N_vals is None or len(N_vals) < 3:
        print("  ERROR: insufficient data points for fit")
        return

    # ── Fit ────────────────────────────────────────────────────────────
    result = fit_gamma(N_vals, r_vals)
    if result is None:
        print("  ERROR: fit failed")
        return

    print(f"\nGUE Convergence Exponent Fit")
    print(f"  Model: |r_GUE - <r>(N)| = A * N^(-gamma)")
    print(f"  r_GUE = {R_GUE}")
    print()
    print(f"  Data points:")
    delta_r = np.abs(R_GUE - r_vals)
    for i in range(len(N_vals)):
        print(f"    N = {N_vals[i]:>10.0f}   <r> = {r_vals[i]:.6f}   |delta| = {delta_r[i]:.6f}")
    print()
    print(f"  Fit results:")
    print(f"    gamma = {result['gamma']:.2f} +/- {result['gamma_err']:.2f}")
    print(f"    A     = {result['A']:.4f}")
    print(f"    R^2   = {result['R_sq']:.4f}")
    print()
    print(f"  GUE prediction: gamma ~ 1")

    # ── Optional plot ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.loglog(result['N_fit'], result['delta_fit'], 'bo', markersize=6, label="Data")

        N_line = np.logspace(np.log10(result['N_fit'].min()),
                             np.log10(result['N_fit'].max()), 100)
        ax.loglog(N_line, result['A'] * N_line**(-result['gamma']),
                  'r--', linewidth=1.2,
                  label=rf"Fit: $N^{{-{result['gamma']:.2f} \pm {result['gamma_err']:.2f}}}$")

        ax.set_xlabel("$N$")
        ax.set_ylabel(r"$|r_{\rm GUE} - \langle r \rangle(N)|$")
        ax.set_title("GUE Convergence: Finite-Size Scaling of Spacing Ratio")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = FIG_DIR / "spectral_form_factor.pdf"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\n  Saved {out_path}")
    except ImportError:
        print("\n  matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
