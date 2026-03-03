#!/usr/bin/env python3
"""
Occupancy model for the mass hierarchy.

Usage: python data/scripts/occupancy_model.py
Reads from: data/ensemble_10M_final/mass_spectrum_M20.csv (belly distribution)
Outputs to: stdout (analysis results)

The mass hierarchy m1 < m2 < m3 is a coupon-collector selection effect
on the belly size distribution f(N).  Generation classification counts
how many distinct phase signs {-1, 0, +1} appear among a prism's N
intermediates, where phi(w) = sign(out_degree - in_degree).

  g(P) = number of distinct phase signs among N intermediates
  mass(gen k) = E[N | g = k]

If phase signs are i.i.d. with probabilities (p+, p0, p-), then:
  P(g=1|N) = p+^N + p0^N + p-^N
  P(g=3|N) = 1 - (1-p+)^N - (1-p0)^N - (1-p-)^N + p+^N + p0^N + p-^N
  P(g=2|N) = 1 - P(g=1|N) - P(g=3|N)

  E[N|g=k] = sum_N  N * P(g=k|N) * f(N)  /  sum_N  P(g=k|N) * f(N)

With measured phases (p+, p0, p-) = (0.318, 0.018, 0.664), the model
reproduces the observed mass ratios m2/m1 = 1.434, m3/m1 = 1.698 to
within 1.8% -- zero free parameters.
"""
import numpy as np
from scipy.optimize import minimize

# ── Observed data (N=10^7, M=20 ensemble) ──────────────────────────────────
OBSERVED_MASS = {1: 4.5547, 2: 6.5312, 3: 7.7324}

# Belly size distribution f(N) from mass_spectrum_M20.csv
BELLY_DATA = [
    (3,  823279),
    (4,  1111148),
    (5,  1287866),
    (6,  1247014),
    (7,  1042789),
    (8,  779347),
    (9,  535037),
    (10, 350644),
    (11, 219969),
    (12, 134326),
    (13, 80510),
    (14, 47567),
    (15, 27598),
    (16, 15806),
    (17, 9183),
    (18, 5167),
    (19, 2986),
    (20, 1669),
    (21, 896),
    (22, 538),
    (23, 285),
    (24, 169),
    (25, 71),
    (26, 46),
    (27, 15),
    (28, 7),
    (29, 5),
    (30, 6),
]

# Measured intermediate phase census from N=100k simulation (19 chunks aggregated)
# φ(w) = sign(out_degree - in_degree) for each intermediate node
MEASURED_PHASES = {
    '+1': 4340,   # source-like (out > in)
    '0':  250,    # balanced
    '-1': 9058,   # sink-like (in > out)
}

# Measured prism generation census from N=100k (19 chunks)
MEASURED_GEN_COUNTS = {1: 898, 2: 1073, 3: 189}

N_vals = np.array([x[0] for x in BELLY_DATA], dtype=float)
f_vals = np.array([x[1] for x in BELLY_DATA], dtype=float)
f_norm = f_vals / f_vals.sum()

total_prisms = f_vals.sum()
mean_belly = np.sum(N_vals * f_norm)
print(f"Total prisms: {total_prisms:.0f}")
print(f"Mean belly size: {mean_belly:.4f}")
print(f"Belly range: {N_vals[0]:.0f} - {N_vals[-1]:.0f}")
print()

# Measured phase fractions
phase_total = sum(MEASURED_PHASES.values())
pp_meas = MEASURED_PHASES['+1'] / phase_total
p0_meas = MEASURED_PHASES['0'] / phase_total
pm_meas = MEASURED_PHASES['-1'] / phase_total
print(f"Measured phase fractions (N=100k, 19 chunks, {phase_total} intermediates):")
print(f"  p+ = {pp_meas:.4f}  ({MEASURED_PHASES['+1']})")
print(f"  p0 = {p0_meas:.4f}  ({MEASURED_PHASES['0']})")
print(f"  p- = {pm_meas:.4f}  ({MEASURED_PHASES['-1']})")
print(f"  Sorted: ({max(pp_meas,p0_meas,pm_meas):.4f}, "
      f"{sorted([pp_meas,p0_meas,pm_meas])[1]:.4f}, "
      f"{min(pp_meas,p0_meas,pm_meas):.4f})")
print()

gen_total = sum(MEASURED_GEN_COUNTS.values())
print(f"Measured prism generation fractions (N=100k, {gen_total} prisms):")
for g in (1, 2, 3):
    print(f"  Gen{g}: {MEASURED_GEN_COUNTS[g]} ({MEASURED_GEN_COUNTS[g]/gen_total*100:.1f}%)")
print()


def gen_probabilities(N, pp, p0, pm):
    """P(g=1|N), P(g=2|N), P(g=3|N) for phase probs (pp, p0, pm)."""
    p1 = pp**N + p0**N + pm**N
    p3 = (1.0 - (1-pp)**N - (1-p0)**N - (1-pm)**N
           + pp**N + p0**N + pm**N)
    p2 = 1.0 - p1 - p3
    return p1, p2, p3


def expected_masses(pp, p0, pm):
    """Compute E[N|g=k] for k=1,2,3."""
    masses = {}
    fractions = {}
    for g in (1, 2, 3):
        num = 0.0
        den = 0.0
        for i, N in enumerate(N_vals):
            p1, p2, p3 = gen_probabilities(N, pp, p0, pm)
            pg = [p1, p2, p3][g - 1]
            num += N * pg * f_norm[i]
            den += pg * f_norm[i]
        masses[g] = num / den if den > 0 else 0.0
        fractions[g] = den
    return masses, fractions


def print_model(label, pp, p0, pm):
    """Print model predictions vs observations."""
    print(f"{'='*70}")
    print(f"  {label}")
    print(f"  Phase probabilities: p+ = {pp:.4f}, p0 = {p0:.4f}, p- = {pm:.4f}")
    print(f"  Sorted: ({max(pp,p0,pm):.4f}, {sorted([pp,p0,pm])[1]:.4f}, {min(pp,p0,pm):.4f})")
    print(f"{'='*70}")
    masses, fracs = expected_masses(pp, p0, pm)
    print(f"  {'Gen':<6} {'Predicted':<12} {'Observed':<12} {'Pred/Obs':<10} {'PrismFrac':<10}")
    print(f"  {'-'*55}")
    for g in (1, 2, 3):
        ratio = masses[g] / OBSERVED_MASS[g]
        print(f"  {g:<6} {masses[g]:<12.4f} {OBSERVED_MASS[g]:<12.4f} {ratio:<10.4f} {fracs[g]:<10.6f}")

    r21_pred = masses[2] / masses[1]
    r31_pred = masses[3] / masses[1]
    r21_obs = OBSERVED_MASS[2] / OBSERVED_MASS[1]
    r31_obs = OBSERVED_MASS[3] / OBSERVED_MASS[1]
    print()
    print(f"  Mass ratios:")
    print(f"    m2/m1:  pred = {r21_pred:.4f},  obs = {r21_obs:.4f}  (err = {abs(r21_pred-r21_obs)/r21_obs*100:.2f}%)")
    print(f"    m3/m1:  pred = {r31_pred:.4f},  obs = {r31_obs:.4f}  (err = {abs(r31_pred-r31_obs)/r31_obs*100:.2f}%)")
    print(f"    m3/m2:  pred = {masses[3]/masses[2]:.4f},  obs = {OBSERVED_MASS[3]/OBSERVED_MASS[2]:.4f}")
    print()

    # Generation fractions (prism-weighted)
    total_frac = fracs[1] + fracs[2] + fracs[3]
    print(f"  Generation fractions (prism-weighted):")
    meas_total = sum(MEASURED_GEN_COUNTS.values())
    for g in (1, 2, 3):
        pred_pct = fracs[g]/total_frac*100
        meas_pct = MEASURED_GEN_COUNTS[g]/meas_total*100
        print(f"    Gen{g}: predicted {pred_pct:5.1f}%  measured {meas_pct:5.1f}%")
    print()


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "#"*70)
print("#  ZERO-PARAMETER TEST: measured phases + measured belly distribution")
print("#"*70 + "\n")

print_model("MEASURED PHASES (no fitting)", pp_meas, p0_meas, pm_meas)

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "#"*70)
print("#  COMPARISON MODELS")
print("#"*70 + "\n")

# ── Uniform phases ────────────────────────────────────────────────────────
print_model("Uniform phases (1/3, 1/3, 1/3)", 1/3, 1/3, 1/3)

# ── Best-fit phases (mass ratio match) ───────────────────────────────────
def objective(params):
    pp, p0 = params
    pm = 1.0 - pp - p0
    if pm < 0.001 or pp < 0.001 or p0 < 0.001:
        return 1e6
    masses, _ = expected_masses(pp, p0, pm)
    r21_pred = masses[2] / masses[1]
    r31_pred = masses[3] / masses[1]
    r21_obs = OBSERVED_MASS[2] / OBSERVED_MASS[1]
    r31_obs = OBSERVED_MASS[3] / OBSERVED_MASS[1]
    return ((r21_pred - r21_obs)/r21_obs)**2 + ((r31_pred - r31_obs)/r31_obs)**2

result = minimize(objective, x0=[0.33, 0.34], method='Nelder-Mead',
                  options={'xatol': 1e-8, 'fatol': 1e-12})
pp_fit, p0_fit = result.x
pm_fit = 1.0 - pp_fit - p0_fit
print_model("Best-fit phases (mass ratio match)", pp_fit, p0_fit, pm_fit)

# ── Permutation symmetry analysis ────────────────────────────────────────
print("="*70)
print("  PERMUTATION SYMMETRY CHECK")
print("="*70)
fit_sorted = sorted([pp_fit, p0_fit, pm_fit], reverse=True)
meas_sorted = sorted([pp_meas, p0_meas, pm_meas], reverse=True)
print(f"  Best-fit sorted:  ({fit_sorted[0]:.4f}, {fit_sorted[1]:.4f}, {fit_sorted[2]:.4f})")
print(f"  Measured sorted:  ({meas_sorted[0]:.4f}, {meas_sorted[1]:.4f}, {meas_sorted[2]:.4f})")
print(f"  Differences:      ({abs(fit_sorted[0]-meas_sorted[0]):.4f}, "
      f"{abs(fit_sorted[1]-meas_sorted[1]):.4f}, "
      f"{abs(fit_sorted[2]-meas_sorted[2]):.4f})")
print()
print(f"  Physical assignment (from measurement):")
print(f"    Dominant  ({meas_sorted[0]:.4f}) = phi=-1 (sink-like, in_deg > out_deg)")
print(f"    Secondary ({meas_sorted[1]:.4f}) = phi=+1 (source-like, out_deg > in_deg)")
print(f"    Rare      ({meas_sorted[2]:.4f}) = phi= 0 (balanced)")
print()

# ── Final verdict ─────────────────────────────────────────────────────────
print("="*70)
print("  VERDICT")
print("="*70)
masses_meas, fracs_meas = expected_masses(pp_meas, p0_meas, pm_meas)
r21_m = masses_meas[2] / masses_meas[1]
r31_m = masses_meas[3] / masses_meas[1]
r21_o = OBSERVED_MASS[2] / OBSERVED_MASS[1]
r31_o = OBSERVED_MASS[3] / OBSERVED_MASS[1]
err21 = abs(r21_m - r21_o)/r21_o * 100
err31 = abs(r31_m - r31_o)/r31_o * 100
print(f"  Measured-phase model errors:")
print(f"    m2/m1: {err21:.2f}%")
print(f"    m3/m1: {err31:.2f}%")
print()
if err21 < 5 and err31 < 5:
    print("  The occupancy model with MEASURED phases reproduces the mass")
    print("  ratios to within 5%. The mass hierarchy is a combinatorial")
    print("  selection effect: f(N) x occupancy statistics. Two inputs,")
    print("  both determined by the causal set geometry. Zero free parameters.")
else:
    print(f"  The measured-phase model has {max(err21,err31):.1f}% error.")
    print(f"  This suggests phase-belly correlations or non-independence")
    print(f"  that the i.i.d. occupancy model does not capture.")
    print(f"  The fit-sorted vs measured-sorted comparison shows whether")
    print(f"  the discrepancy is in the phase fractions or the model structure.")
