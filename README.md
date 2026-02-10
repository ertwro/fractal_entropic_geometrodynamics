# Discrete Skyrmion on Causal Sets – Exploratory Code

This repository contains Python code used in the preprint:

**"Fractal Entropic Geometrodynamics: Emergent Gravity and Standard Model Synthesis from Causal Set Information Geometry"** by Juan Pablo Silva Alvarado (metric engineer)

DOI (Zenodo): [https://doi.org/10.5281/zenodo.18526209]  
arXiv: [none – not endorsed yet]

## Pedagogic Material: Modulo Synthesis

In addition to the exploratory code, this repository hosts the **"Modulo Synthesis"** pedagogic project—a comprehensive attempt to teach discrete physics from first principles, where matter emerges as a topological defect in the Causal Set.

Located in the `pedagogic_booklets/` directory:

- **Volume I: The Geometry of Spacetime** Establishes the foundations, deriving General Relativity from the thermodynamics of causal horizons and proving Chentsov's theorem for the uniqueness of the statistical metric.

- **Volume II: The Geometry of Matter** A speculative synthesis proposing geometric origins for the Standard Model generations (via Kuratowski's theorem), gauge groups ($S_3 \to SU(3)$), and constants like $\alpha$ and $\theta_C$.

- **Emma's Thought Experiments** A companion booklet of intuitive scenarios (e.g., "The Quantum Ghost", "The Tilted Loaf") designed to make these advanced discrete concepts accessible to a general audience.

## What this code does

The scripts perform **numerical experiments** exploring whether baryon masses (and their large hierarchy over leptons) can emerge geometrically from **topological solitons** (hedgehog Skyrmions) defined on a discrete causal set.

Main features:

- Generates 3+1D Poisson-sprinkled causal sets (N ≈ 20,000 nodes)
- Assigns SU(2)-valued hedgehog fields to nodes
- Computes discrete Skyrme-like energy (kinetic + quartic approximation) over causal links
- Compares soliton energy against a fluctuating zero-mode baseline
- Shows that the energy ratio varies strongly with core localization scale (λ_decay)
- Observed proton/electron mass ratio (≈1836) falls naturally within the computed geometric range (~100× to ~8800×)

This is **not** a production-level lattice QCD simulation — it is an exploratory proof-of-concept to test whether the qualitative hierarchy can appear from causal-set geometry alone.

## Main scripts

- `discrete_skyrmion_su2.py`  
  Single run: generates a causal set, places a hedgehog Skyrmion, computes energies and ratio.

- `skyrmion_ratio_scan.py`  
  Parameter scan: varies core decay length λ_decay and reports mean ratio ± std over multiple runs.

Dependencies:

- numpy, scipy, matplotlib, tqdm, numba (optional but recommended for speed)

## Requirements

```bash
pip install numpy scipy matplotlib tqdm numba
```
