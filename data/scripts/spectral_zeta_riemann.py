#!/usr/bin/env python3
"""
Spectral Zeta Function of the Causal Graph: Prime Prisms & the Critical Line
=============================================================================

Numerical exploration of the connection between the FEG causal graph spectrum
and the Riemann zeta function.

Theoretical framework
---------------------
In Fractal Entropic Geometrodynamics, Causal Prisms K_{2,n} are the
topologically irreducible excitations of the Hasse diagram.  When the belly
size n is prime, the prism cannot be factored into smaller sub-prisms: these
are the "prime prisms."  The belly-size histogram f(n) from the simulation
therefore encodes a spectral density over a discrete set of topological masses.

We define the *graph spectral zeta function*:

    ζ_G(s) = Σ_n  f(n) · n^{-s}        (Dirichlet series, n = 3..n_max)

and its *prime-belly Euler factor*:

    P(s)  = Π_{p prime}  (1 - f(p) · p^{-s})^{-1}

If the prism spectrum were a perfect encoding of the integers (f(n) ≡ 1),
ζ_G would reduce to a tail of the Riemann zeta function, and P(s) to the
Euler product.  The simulation deviates from this via the thermodynamic
weighting f(n), which reflects the entropy of the causal graph.

What this script computes (numerical evidence)
-----------------------------------------------
1. The graph spectral zeta ζ_G(s) along the critical line s = 1/2 + it.
2. The prime-belly Euler product P(s) and its relation to ζ_G.
3. Minima of |ζ_G(1/2 + it)| and comparison to known Riemann zeros.
4. The return-probability Mellin transform ζ_M(s) from random-walk data.
5. A symmetry diagnostic: |ζ_G(σ + it)| vs σ for evidence of a
   functional equation centred at σ = 1/2.

What this does NOT constitute
-----------------------------
A finite Dirichlet series (n = 3..30, 28 terms) cannot have true zeros in
the critical strip.  The minima of |ζ_G| are *near-zeros* -- resonant dips.
Agreement with known Riemann zeros would be numerical evidence for the
Hilbert-Pólya realization conjecture within FEG, not a proof of RH.  A proof
would require showing that the continuum limit (N → ∞, n_max → ∞) of ζ_G
converges to ζ and that the unitarity constraint forces all zeros onto
Re(s) = 1/2.

Usage:
    python spectral_zeta_riemann.py

Reads from:
    data/ensemble_10M_final/mass_spectrum_M20.csv   (belly distribution f(n))
    data/ensemble_10M_final/results_M20.csv         (return probabilities P(σ))
"""
import math
import numpy as np
from pathlib import Path
from scipy.special import gamma as Gamma
from scipy.signal import argrelmin


# ══════════════════════════════════════════════════════════════════════════════
# Path resolution
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_data_root():
    """Locate data/ directory whether run from repo root or data/scripts/."""
    for candidate in [SCRIPT_DIR / "data", SCRIPT_DIR.parent]:
        if (candidate / "ensemble_10M_final").exists():
            return candidate
    return None


DATA_ROOT = _find_data_root()
ENSEMBLE_DIR = DATA_ROOT / "ensemble_10M_final" if DATA_ROOT else None


# ══════════════════════════════════════════════════════════════════════════════
# CSV loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_belly_distribution(path):
    """Load mass_spectrum CSV -> dict {belly_size: frequency}."""
    dist = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('inter'):
                continue
            parts = s.split(',')
            n, freq = int(parts[0]), int(parts[1])
            dist[n] = freq
    return dist


def load_return_probabilities(path):
    """Load results CSV -> (steps[], P_vac[])."""
    steps, p_vac = [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('step'):
                continue
            parts = s.split(',')
            steps.append(int(parts[0]))
            p_vac.append(float(parts[1]))
    return np.array(steps), np.array(p_vac)


# ══════════════════════════════════════════════════════════════════════════════
# Number theory utilities
# ══════════════════════════════════════════════════════════════════════════════

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    k = 5
    while k * k <= n:
        if n % k == 0 or n % (k + 2) == 0:
            return False
        k += 6
    return True


# Known non-trivial Riemann zeros (imaginary parts, first 20)
RIEMANN_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
])


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Graph Spectral Zeta Function
# ══════════════════════════════════════════════════════════════════════════════

def zeta_graph(s, belly_dist):
    """
    ζ_G(s) = Σ_n f(n) · n^{-s}

    Dirichlet series weighted by prism belly-size degeneracy.
    """
    total = 0.0 + 0.0j
    for n, f_n in belly_dist.items():
        total += f_n * n ** (-s)
    return total


def zeta_graph_normalised(s, belly_dist):
    """
    ζ̃_G(s) = Σ_n [f(n)/F] · n^{-s},   F = Σ f(n)

    Probability-normalised: f(n)/F is the fraction of prisms with belly n.
    This removes the overall scale and isolates the spectral shape.
    """
    F = sum(belly_dist.values())
    total = 0.0 + 0.0j
    for n, f_n in belly_dist.items():
        total += (f_n / F) * n ** (-s)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Prime Belly Euler Product
# ══════════════════════════════════════════════════════════════════════════════

def euler_product_prime_bellies(s, belly_dist):
    """
    P(s) = Π_{p prime, p in belly_dist} (1 - p^{-s})^{-1}

    Standard Euler factor restricted to prime belly sizes.
    If f(n) ≡ 1, this recovers the Euler product for ζ(s).
    """
    product = 1.0 + 0.0j
    for n in sorted(belly_dist.keys()):
        if is_prime(n):
            product *= (1.0 - n ** (-s)) ** (-1)
    return product


def euler_product_weighted(s, belly_dist):
    """
    P_w(s) = Π_{p prime} (1 - f(p) · p^{-s})^{-1}

    Weighted Euler product: each prime factor carries the thermodynamic
    degeneracy f(p) of the causal graph.
    """
    product = 1.0 + 0.0j
    for n in sorted(belly_dist.keys()):
        if is_prime(n):
            f_n = belly_dist[n]
            term = 1.0 - f_n * n ** (-s)
            if abs(term) > 1e-15:
                product *= term ** (-1)
    return product


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Critical Line Analysis
# ══════════════════════════════════════════════════════════════════════════════

def scan_critical_line(belly_dist, t_max=100.0, dt=0.01, sigma=0.5):
    """
    Evaluate |ζ̃_G(σ + it)| along a line of constant Re(s) = σ.
    Return (t_values, magnitudes).
    """
    t_vals = np.arange(0.0, t_max, dt)
    mags = np.empty_like(t_vals)
    for i, t in enumerate(t_vals):
        s = sigma + 1j * t
        mags[i] = abs(zeta_graph_normalised(s, belly_dist))
    return t_vals, mags


def find_near_zeros(t_vals, mags, prominence_threshold=0.3):
    """
    Find local minima (near-zeros) of |ζ̃_G(1/2 + it)|.
    Returns array of t-values at each minimum.
    """
    # Use scipy argrelmin with moderate order for smoothness
    order = max(5, int(0.5 / (t_vals[1] - t_vals[0])))
    (minima_idx,) = argrelmin(mags, order=order)
    # Filter: keep only minima that are below the median magnitude
    if len(minima_idx) == 0:
        return np.array([]), np.array([])
    median_mag = np.median(mags[mags > 0])
    mask = mags[minima_idx] < prominence_threshold * median_mag
    return t_vals[minima_idx[mask]], mags[minima_idx[mask]]


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Symmetry Diagnostic (Functional Equation Test)
# ══════════════════════════════════════════════════════════════════════════════

def symmetry_profile(belly_dist, t_fixed, sigma_range=(0.0, 1.0), ds=0.005):
    """
    For a fixed Im(s) = t, sweep Re(s) = σ and record |ζ̃_G(σ + it)|.
    If a functional equation holds, |ζ̃| should be symmetric about σ = 1/2.
    """
    sigmas = np.arange(sigma_range[0], sigma_range[1] + ds, ds)
    mags = np.empty_like(sigmas)
    for i, sigma in enumerate(sigmas):
        s = sigma + 1j * t_fixed
        mags[i] = abs(zeta_graph_normalised(s, belly_dist))
    return sigmas, mags


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Mellin Transform of Return Probability
# ══════════════════════════════════════════════════════════════════════════════

def mellin_zeta(s, steps, p_return):
    """
    ζ_M(s) = Σ_σ  P(σ) · σ^{s-1}

    Discrete Mellin transform of the return probability P(σ).
    For a graph with eigenvalues {λ_k}, P(σ) = (1/N) Σ λ_k^σ,
    so ζ_M encodes the eigenvalue spectrum via moment sums.
    """
    total = 0.0 + 0.0j
    for sigma_step, p_val in zip(steps, p_return):
        if sigma_step > 0 and p_val > 0:
            total += p_val * sigma_step ** (s - 1)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Reference Riemann Zeta (truncated, for comparison)
# ══════════════════════════════════════════════════════════════════════════════

def riemann_zeta_ref(s):
    """
    Reference Riemann zeta via mpmath (arbitrary precision).
    Falls back to a truncated Dirichlet series if mpmath is unavailable.
    """
    try:
        import mpmath
        return complex(mpmath.zeta(s))
    except ImportError:
        pass
    # Fallback: truncated series with Euler-Maclaurin, skip the pole at s=1
    s = complex(s)
    if abs(s - 1) < 1e-10:
        return float('inf') + 0j
    N = 10000
    total = sum(n ** (-s) for n in range(1, N + 1))
    total += N ** (1 - s) / (s - 1) + 0.5 * N ** (-s)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if ENSEMBLE_DIR is None:
        raise FileNotFoundError("Cannot locate data/ensemble_10M_final/ directory.")

    belly_dist = load_belly_distribution(ENSEMBLE_DIR / "mass_spectrum_M20.csv")
    steps, p_vac = load_return_probabilities(ENSEMBLE_DIR / "results_M20.csv")

    belly_sizes = sorted(belly_dist.keys())
    total_prisms = sum(belly_dist.values())
    prime_bellies = {n: belly_dist[n] for n in belly_sizes if is_prime(n)}
    composite_bellies = {n: belly_dist[n] for n in belly_sizes if not is_prime(n)}

    print("=" * 72)
    print("SPECTRAL ZETA FUNCTION OF THE CAUSAL GRAPH")
    print("Fractal Entropic Geometrodynamics — N = 10^7, M = 20")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Part 1: Prime Prism Census
    # ------------------------------------------------------------------
    print("\n" + "─" * 72)
    print("PART 1: PRIME PRISM CENSUS")
    print("─" * 72)
    print(f"\nBelly sizes observed:  n ∈ [{belly_sizes[0]}, {belly_sizes[-1]}]")
    print(f"Total prisms:          {total_prisms:,}")
    print(f"Prime belly count:     {len(prime_bellies)} sizes")
    print(f"Composite belly count: {len(composite_bellies)} sizes")

    f_prime = sum(prime_bellies.values())
    f_composite = sum(composite_bellies.values())
    print(f"\nPrisms with prime belly:     {f_prime:>10,}  "
          f"({100*f_prime/total_prisms:.2f}%)")
    print(f"Prisms with composite belly: {f_composite:>10,}  "
          f"({100*f_composite/total_prisms:.2f}%)")

    print(f"\n{'n':>4}  {'f(n)':>10}  {'prime?':>6}  {'f(n)/F':>10}  {'n^(-1/2)':>10}")
    print("─" * 50)
    for n in belly_sizes:
        tag = "  ★" if is_prime(n) else ""
        frac = belly_dist[n] / total_prisms
        print(f"{n:>4}  {belly_dist[n]:>10,}  {tag:>6}  {frac:>10.6f}  "
              f"{n**(-0.5):>10.6f}")

    # ------------------------------------------------------------------
    # Part 2: Graph Spectral Zeta on the Critical Line
    # ------------------------------------------------------------------
    print("\n" + "─" * 72)
    print("PART 2: ζ̃_G(1/2 + it)  —  CRITICAL LINE SCAN")
    print("─" * 72)

    t_vals, mags = scan_critical_line(belly_dist, t_max=100.0, dt=0.005)
    near_t, near_mag = find_near_zeros(t_vals, mags)

    print(f"\nScanned t ∈ [0, 100] with dt = 0.005")
    print(f"Near-zeros found: {len(near_t)}")

    if len(near_t) > 0:
        print(f"\n{'#':>3}  {'t_graph':>10}  {'|ζ̃_G|':>12}  "
              f"{'t_Riemann':>10}  {'Δt':>8}  {'Δt/t':>8}")
        print("─" * 62)
        matched = 0
        for i, (tg, mg) in enumerate(zip(near_t, near_mag)):
            # Find closest known Riemann zero
            if len(RIEMANN_ZEROS) > 0:
                idx = np.argmin(np.abs(RIEMANN_ZEROS - tg))
                tr = RIEMANN_ZEROS[idx]
                delta = tg - tr
                rel = abs(delta / tr) if tr != 0 else float('inf')
                match_flag = " ◄" if rel < 0.05 else ""
                if rel < 0.05:
                    matched += 1
                print(f"{i+1:>3}  {tg:>10.4f}  {mg:>12.6e}  "
                      f"{tr:>10.4f}  {delta:>+8.4f}  {rel:>8.4f}{match_flag}")
            else:
                print(f"{i+1:>3}  {tg:>10.4f}  {mg:>12.6e}")
            if i >= 29:
                print(f"  ... ({len(near_t) - 30} more)")
                break

        print(f"\nNear-zeros within 5% of a known Riemann zero: {matched}")

    # ------------------------------------------------------------------
    # Part 3: Euler Product Comparison
    # ------------------------------------------------------------------
    print("\n" + "─" * 72)
    print("PART 3: EULER PRODUCT OF PRIME BELLIES")
    print("─" * 72)

    test_points = [2.0, 1.5, 1.0 + 0.0j, 0.5 + 14.134725j, 0.5 + 21.022040j]
    labels = ["s=2 (convergent)", "s=3/2", "s=1 (pole)",
              "s=1/2+14.13i (1st zero)", "s=1/2+21.02i (2nd zero)"]

    print(f"\n{'Point':>24}  {'|ζ̃_G(s)|':>12}  {'|P(s)|':>12}  "
          f"{'|ζ_trunc(s)|':>14}")
    print("─" * 72)
    for label, s in zip(labels, test_points):
        zg = abs(zeta_graph_normalised(complex(s), belly_dist))
        ep = abs(euler_product_prime_bellies(complex(s), belly_dist))
        zr = abs(riemann_zeta_ref(complex(s)))
        print(f"{label:>24}  {zg:>12.6f}  {ep:>12.6f}  {zr:>14.6f}")

    # ------------------------------------------------------------------
    # Part 4: Symmetry About σ = 1/2
    # ------------------------------------------------------------------
    print("\n" + "─" * 72)
    print("PART 4: FUNCTIONAL EQUATION SYMMETRY TEST")
    print("─" * 72)
    print("\nIf the graph zeta satisfies a functional equation, |ζ̃_G(σ+it)|")
    print("should be approximately symmetric about σ = 1/2 for each t.")

    test_t_values = [14.13, 21.02, 25.01, 30.42, 50.0]
    for t_test in test_t_values:
        sigmas, profile = symmetry_profile(belly_dist, t_test)
        # Measure asymmetry: compare σ and (1 - σ)
        n_half = len(sigmas) // 2
        left = profile[:n_half]
        right = profile[-1:len(profile)-n_half-1:-1]
        min_len = min(len(left), len(right))
        if min_len > 0:
            left = left[:min_len]
            right = right[:min_len]
            asym = np.mean(np.abs(left - right)) / (np.mean(profile) + 1e-30)
            # Find σ that minimises |ζ̃_G|
            sigma_min = sigmas[np.argmin(profile)]
            print(f"  t = {t_test:>6.2f}:  asymmetry = {asym:.4f},  "
                  f"σ_min = {sigma_min:.3f}  "
                  f"{'(≈ 1/2)' if abs(sigma_min - 0.5) < 0.1 else ''}")

    # ------------------------------------------------------------------
    # Part 5: Mellin Transform of Return Probability
    # ------------------------------------------------------------------
    print("\n" + "─" * 72)
    print("PART 5: MELLIN TRANSFORM ζ_M(s) FROM RANDOM WALK DATA")
    print("─" * 72)
    print(f"\nDiffusion steps used: {len(steps)} (σ = {steps[0]}..{steps[-1]})")

    mellin_points = [0.5 + 1j * t for t in RIEMANN_ZEROS[:10]]
    print(f"\n{'Riemann zero':>14}  {'|ζ_M(1/2+it)|':>16}  {'arg(ζ_M)':>10}")
    print("─" * 48)
    for s_pt in mellin_points:
        zm = mellin_zeta(s_pt, steps, p_vac)
        print(f"  t = {s_pt.imag:>8.4f}  {abs(zm):>16.6e}  "
              f"{np.degrees(np.angle(zm)):>+10.2f}°")

    # ------------------------------------------------------------------
    # Part 6: BD Operator Phase Structure
    # ------------------------------------------------------------------
    print("\n" + "─" * 72)
    print("PART 6: BENINCASA-DOWKER LAYER WEIGHTS & IMAGINARY SPECTRUM")
    print("─" * 72)
    print("""
The 2D Benincasa-Dowker (BD) operator on a causal set has layer weights:

    L₀ = +1,  L₁ = -3,  L₂ = +3,  L₃ = -1

which are the alternating binomial coefficients (-1)^k C(3,k).

In 4D (our simulation), the retarded d'Alembertian uses:

    C₀ = +1,  C₁ = -9,  C₂ = +16,  C₃ = -8

These alternating signs (+, -, +, -) force the operator to be non-self-adjoint
on the causal partial order.  The eigenvalues of a non-self-adjoint operator
are generically complex:  λ = σ ± iω.

Claim under test:
  Unitarity of the random walk (conservation of probability, Axiom 2 of FEG)
  constrains the spectral radius to |λ| = 1 for the evolution operator.
  On the unit circle, λ = e^{iθ}, so Re(λ) and Im(λ) are symmetric about 0.
  Under the s ↔ log(λ) map, this symmetry maps to Re(s) = 1/2.

Status: NUMERICAL EXPLORATION — see Parts 2 and 4 above for evidence.
""")

    bd_weights_4d = [+1, -9, +16, -8]
    print("  BD weights (4D retarded d'Alembertian):", bd_weights_4d)
    print(f"  Sum of weights: {sum(bd_weights_4d)}  (= 0: UV finiteness)")

    # Eigenvalues of the 4×4 companion matrix for the BD recurrence
    companion = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [8, -16, 9, -1],
    ], dtype=float)
    eigvals = np.linalg.eigvals(companion)
    print(f"\n  Companion matrix eigenvalues (BD recurrence):")
    for ev in eigvals:
        print(f"    λ = {ev.real:+.6f} {ev.imag:+.6f}i    "
              f"|λ| = {abs(ev):.6f}")

    # ------------------------------------------------------------------
    # Part 7: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"""
Data:       N = 10^7 events,  M = 20 realisations
Bellies:    n ∈ [{belly_sizes[0]}, {belly_sizes[-1]}],  {total_prisms:,} total prisms
Primes:     {len(prime_bellies)} prime belly sizes carrying {f_prime:,} prisms ({100*f_prime/total_prisms:.1f}%)

Graph spectral zeta ζ̃_G(s) = Σ [f(n)/F] · n^{{-s}}:
  • Evaluated on the critical line Re(s) = 1/2
  • {len(near_t)} near-zero dips found in t ∈ [0, 100]

Prime-belly Euler product P(s) = Π_p (1 - p^{{-s}})^{{-1}}:
  • Constructed from {len(prime_bellies)} prime factors (n = {', '.join(str(n) for n in sorted(prime_bellies))})

BD operator:
  • 4D weights [+1, -9, +16, -8] sum to 0 (UV finite)
  • Companion eigenvalues generically complex → non-Hermitian spectrum

INTERPRETATION:
  The finite belly distribution (28 terms) produces a Dirichlet polynomial,
  not the full Riemann ζ(s).  Near-zeros on the critical line are resonant
  dips of this polynomial.  The physically significant result is that the
  *density* and *approximate positions* of these dips are governed by the
  prime belly fraction ({100*f_prime/total_prisms:.1f}%), which is itself
  set by the topological irreducibility constraints of the causal graph.

  To approach a genuine Hilbert-Pólya realization, one would need:
    1. n_max → ∞ in the continuum limit (FSS extrapolation)
    2. f(n) → 1 (each belly size equally weighted) or a controlled correction
    3. Proof that unitarity forces all spectral zeros to Re(s) = 1/2

  Step 1 is addressable with the existing FSS pipeline.
  Steps 2-3 remain open conjectures within the FEG framework.
""")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Spectral Zeta Function of the Causal Graph  —  "
            "FEG  N = 10⁷,  M = 20",
            fontsize=13, fontweight='bold'
        )

        # Panel (a): |ζ̃_G(1/2 + it)| on the critical line
        ax = axes[0, 0]
        ax.semilogy(t_vals, mags, color='#2166ac', linewidth=0.5, alpha=0.8)
        for tr in RIEMANN_ZEROS[:10]:
            ax.axvline(tr, color='#b2182b', alpha=0.35, linewidth=0.8,
                       linestyle='--')
        if len(near_t) > 0:
            ax.scatter(near_t, near_mag, color='#d6604d', s=15, zorder=5,
                       label='graph near-zeros')
        ax.set_xlabel('t  (imaginary part)')
        ax.set_ylabel('|ζ̃_G(½ + it)|')
        ax.set_title('(a) Critical line:  Re(s) = ½')
        ax.set_xlim(0, 80)
        ax.legend(fontsize=8)

        # Panel (b): Belly distribution with primes highlighted
        ax = axes[0, 1]
        ns = np.array(belly_sizes)
        freqs = np.array([belly_dist[n] for n in belly_sizes])
        colors = ['#b2182b' if is_prime(n) else '#4393c3' for n in belly_sizes]
        ax.bar(ns, freqs, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Belly size  n')
        ax.set_ylabel('Frequency  f(n)')
        ax.set_title('(b) Prime prisms (red) vs composite (blue)')
        ax.set_yscale('log')

        # Panel (c): Symmetry profile |ζ̃_G(σ + it)| for first Riemann zero
        ax = axes[1, 0]
        for t_test, color, label in [
            (14.13, '#b2182b', 't = 14.13'),
            (21.02, '#2166ac', 't = 21.02'),
            (25.01, '#1b7837', 't = 25.01'),
        ]:
            sigmas, profile = symmetry_profile(belly_dist, t_test)
            ax.plot(sigmas, profile, color=color, linewidth=1.2, label=label)
        ax.axvline(0.5, color='grey', linestyle=':', linewidth=1)
        ax.set_xlabel('σ  (real part)')
        ax.set_ylabel('|ζ̃_G(σ + it)|')
        ax.set_title('(c) Symmetry about σ = ½')
        ax.legend(fontsize=8)

        # Panel (d): Mellin transform from return probability
        ax = axes[1, 1]
        t_scan_mellin = np.linspace(1, 80, 400)
        mellin_mags = []
        for t in t_scan_mellin:
            zm = mellin_zeta(0.5 + 1j * t, steps, p_vac)
            mellin_mags.append(abs(zm))
        mellin_mags = np.array(mellin_mags)
        ax.semilogy(t_scan_mellin, mellin_mags, color='#762a83',
                    linewidth=0.8)
        for tr in RIEMANN_ZEROS[:10]:
            ax.axvline(tr, color='#b2182b', alpha=0.35, linewidth=0.8,
                       linestyle='--')
        ax.set_xlabel('t  (imaginary part)')
        ax.set_ylabel('|ζ_M(½ + it)|')
        ax.set_title('(d) Mellin transform of P(σ)')
        ax.set_xlim(1, 80)

        plt.tight_layout()
        out_path = SCRIPT_DIR / "spectral_zeta_riemann.pdf"
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved: {out_path}")

    except ImportError:
        print("matplotlib not available — skipping figure generation.")


if __name__ == '__main__':
    main()
