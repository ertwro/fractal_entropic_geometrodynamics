# Discrete Skyrmion on Causal Sets – Exploratory Code

This repository contains Python code used in the preprint:

**"Fractal Entropic Geometrodynamics: Emergent Gravity and Standard Model Synthesis from Causal Set Information Geometry"**  
by Juan Pablo Silva Alvarado (metric engineer)

DOI (Zenodo): [https://doi.org/10.5281/zenodo.18526209]  
arXiv: [none – not endorsed yet]

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
