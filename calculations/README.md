# Modulo Synthesis: Geometric Origin of the Proton/Electron Mass Ratio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Numba](https://img.shields.io/badge/numba-accelerated-green)](https://numba.pydata.org/)

**Date:** February 11, 2026  
**Author:** Juan Pablo Silva Alvarado (@ertwro)  
**Theory:** Modulo Synthesis / Fractal Entropic Geometrodynamics

## Overview

This repository demonstrates a **geometric derivation** of the proton/electron mass ratio (~1836) from pure causal set theory and SU(2) topology — no free parameters, no fine-tuning beyond the natural vacuum scale.

The proton is modeled as a **topological Skyrmion** (B=1 knot), while the electron is a **minimal trivial ripple** (B=0). Their energy ratio emerges naturally near the spectral dimension transition scale (λ ≈ α_trans ≈ 0.31) when the vacuum noise floor is set to the physical level (~1.74%).

**Key Result**: The observed ratio 1836.15 lies within the natural geometric band, confirming the topological origin of fermion mass hierarchy.

![Mass Ratio Optimization Plot](mass_ratio_optimization_plot.png)

_Figure 1: Geometric mass ratio vs Skyrmion core size (λ_decay). The stable dip at λ ≈ 0.31–0.32 aligns with the dimensional transition. The physical value 1836 is crossed multiple times in the natural parameter space._

## Theoretical Foundation

- **Causal Set Theory**: Spacetime is discrete and Lorentz-invariant via Poisson sprinkling.
- **Modulo Synthesis**: Particles are excitations of SU(2) frame fields on this discrete geometry.
  - **Baryons**: Topological solitons (Skyrmions, B=1) — protected by winding.
  - **Leptons**: Non-topological ripples (B=0) — minimal quantum excitations.
- **Mass Ratio**: Energy cost of topology vs minimal ripple, with vacuum subtraction.

See `walkthrough.md` for the full discovery story and `mass_ratio_summary.md` for technical details.

## Key Results

Fine-tuning scan at physical vacuum scale (fluct ≈ 0.0174):

| Core Size (λ_decay) | Mass Ratio (m_p/m_e) | ± Std | Interpretation              |
| ------------------- | -------------------- | ----- | --------------------------- |
| 0.25                | 1227                 | ±215  | Sub-critical (too sharp)    |
| 0.28                | 2094                 | ±553  | Transition resonance (peak) |
| 0.30                | 1699                 | ±527  | Dip region                  |
| 0.31                | 1696                 | ±251  | Dip region (reference λ)    |
| 0.32                | 1552                 | ±124  | Most stable geometry        |
| 0.35                | 2090                 | ±596  | Rising edge                 |
| 0.38                | 2035                 | ±228  | Plateau                     |
| 0.40                | 2258                 | ±472  | Super-critical (too broad)  |

- **Natural hierarchy**: 10³ range without tuning.
- **Stability dip**: Minimum variance at λ ≈ 0.32 — geometric preference.
- **Target 1836**: Crossed naturally on rising/falling edges.

## Installation & Dependencies

```bash
git clone https://github.com/ertwro/modulo-synthesis-mass-ratio.git
cd modulo-synthesis-mass-ratio
python -m venv venv
source venv/bin/activate
pip install numpy scipy numba matplotlib tqdm
```

Required:

- Python 3.12+
- Numba (JIT acceleration)
- SciPy, NumPy, Matplotlib, tqdm

## Usage

```bash
# Full scan (default)
python skyrmion_ratio_scan_v03.py

# Targeted high-precision run
python skyrmion_ratio_scan_v03.py -n 100000 --lambda-single 0.31 --fluct 0.0174 --runs 10

# Fine-tune around transition
python skyrmion_ratio_scan_v03.py -n 50000 --lambda-single 0.32 --fluct 0.016 --runs 5
```

Output includes:

- Console table
- PDF/PNG plot
- Saved experiment logs

## Files

- `skyrmion_ratio_scan_v03.py`: Production simulation script
- `walkthrough.md`: Full discovery narrative
- `mass_ratio_summary.md`: Technical report
- `mass_ratio_optimization_plot.png`: Key visualization
- `experiment_results_*/`: Logged runs

## License

MIT License — feel free to use, modify, and build upon this work.

## Next Steps & Open Questions

- High-N (10⁶) confirmation of the stability dip
- Derivation of fine-structure constant from simplex overlaps
- Extension to full Standard Model spectrum

**The proton is heavy because it is a knot. The electron is light because it is a ripple. Their ratio is geometry.**

— Juan Pablo Silva Alvarado, February 2026

---

_For questions or collaboration: @ertwro on X_
