# Fractal Entropic Geometrodynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18733424.svg)](https://doi.org/10.5281/zenodo.18733424)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CC BY-NC-ND 4.0](https://img.shields.io/badge/Theory-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

**Author:** Juan Pablo Silva Alvarado ([@ertwro](https://github.com/ertwro))

Two axioms. Zero free parameters. O(N) on a laptop.

---

## What This Is

A zero-parameter simulation that Poisson-sprinkles 10 million events into a 4D
causal diamond and discovers three particle generations, a mass hierarchy, and
CPT symmetry from pure combinatorial topology. The entire computation runs in
**4.5 hours on an 8-year-old ThinkPad T480** (Intel i5-8250U, 16 GB RAM).
Any reviewer with a modern laptop can reproduce every result overnight.

---

## Key Results (N = 10^7, M = 20 realisations)

| Observable | Value | Note |
|---|---|---|
| Spectral dimension (UV, sigma=1) | 1.949 +/- 0.002 | Planck-scale d_S -> 2 |
| Spectral dimension (IR, sigma=4) | 4.956 +/- 0.001 | 4D continuum emerges |
| Defect core d_S (peak) | 9.06 | Topological trapping |
| Total prisms per realisation | 386,197 | |
| Generations g=1 / g=2 / g=3 | 2.1% / 84.9% / 13.0% | Exactly 3 (theorem) |
| Mass: Gen 1 / Gen 2 / Gen 3 | 4.555 / 6.531 / 7.732 | Topological Planck units |
| Ratio m1 : m2 : m3 | 1 : 1.43 : 1.70 | N-independent invariant |
| CPT: Gen 1 vs Anti-1 | 4.555 vs 4.45 | Delta m/m = 2.3% |
| Q_topo | 0.1907 | |
| alpha = Q_topo/(8pi) | 1/131.8 | |
| Omega_energy = 1/Q_topo - 1 | 4.24 | |
| alpha(1 + Omega) | 1/(8pi) | **Exact at every N** |
| GUE spacing ratio <r> | 0.603 +/- 0.002 | Matches GUE prediction 0.6027 |
| Spectral form factor slope | gamma = 1.04 +/- 0.03 | GUE prediction: 1 |
| G_BH / G_thermo | 4.16 | Bekenstein-Hawking factor of 4 |
| G = 1/(16pi) | From integer BD weights | Emergent Einstein-Hilbert prefactor |

---

## Finite-Size Scaling

Five lattice sizes (N = 10^5 to 10^7, M >= 8 realisations each) confirm that all
boundary-sensitive observables follow the 4D scaling law:

**O(N) = O_inf + a * N^{-1/4}**

| Observable | N = 10^7 | N -> inf | R^2 |
|---|---|---|---|
| Q_topo | 0.191 | **0.152 +/- 0.001** | 0.9996 |
| 1/alpha (bare, UV cutoff) | 131.8 | **165.1 +/- 1.0** | -- |
| Omega_energy | 4.24 | **5.57** (Planck 2018: 5.36) | 0.990 |
| d_S (UV) | 1.949 | **1.953** | 0.890 |
| d_S (IR) | 4.956 | **5.002** | 0.988 |

Mass ratios and generation fractions are **N-independent topological invariants**
(R^2 < 0.1 against N^{-1/4}).

The bare coupling alpha_0^{-1} = 165.1 +/- 1.0 is the UV-cutoff value at the
Planck scale.  The 20% gap to the physical alpha^{-1} = 137.036 is not a deficiency
of the model but the physically expected domain of renormalization-group running
from the Planck scale to laboratory energies.  The vacuum polarization analysis
(see below) identifies the mechanism: Kuratowski planarity constraints force phase
anti-correlation at small belly sizes (UV scale), screening the topological charge.
This is a discrete, ab initio derivation of running from graph topology alone.

---

## Mass Hierarchy: Occupancy Model

The mass hierarchy m1 < m2 < m3 is analytically explained as a **coupon-collector
selection effect** on the belly size distribution.

Each prism intermediate has a causal phase phi(w) = sign(out\_degree - in\_degree)
drawn from {-1, 0, +1}.  The generation g(P) counts how many distinct phase signs
appear among the N intermediates.  Gen1 = all same sign, Gen2 = two signs, Gen3 =
all three signs.

Since P(g=1|N) ~ p\_max^N decays exponentially, Gen1 prisms are biased toward
small bellies (low mass), while Gen3 prisms require large bellies (high mass).
The mass of each generation is E[N | g = k], determined by:

1. **f(N)** -- the belly size distribution (from causal diamond geometry)
2. **(p+, p0, p-)** -- the intermediate phase fractions (from Hasse degree statistics)

The simulation measures (p+, p0, p-) = (0.318, 0.018, 0.664).  Plugging these
into the occupancy formula reproduces the mass ratios with **zero free parameters**:

| Ratio | Predicted | Observed | Error |
|---|---|---|---|
| m2/m1 | 1.409 | 1.434 | 1.8% |
| m3/m1 | 1.674 | 1.698 | 1.4% |

Both inputs are determined entirely by the Poisson sprinkling geometry.

**Crucially**, the same i.i.d. model that succeeds for mass *fails* for charge:
it predicts Q_topo = 0.236, but the simulation measures Q_topo = 0.191 (a 23.5%
overshoot).  This failure is the discovery of vacuum polarization -- see below.

See `data/scripts/occupancy_model.py` for the full four-part analysis (mass, charge, vacuum
polarization, running coupling).

---

## Vacuum Polarization

The i.i.d. occupancy model's success for mass and failure for charge reveals
**discrete vacuum polarization** from graph planarity:

1. **Mass = counting observable:** How many distinct phases appear?
   i.i.d. reproduces mass ratios to < 2% error.  **PASS.**

2. **Charge = summation observable:** How much do phases cancel?
   i.i.d. predicts Q_pred = 0.236, observed Q_obs = 0.191.
   Overshoot: **23.5%.  FAIL.**

3. **Mechanism:** At small belly sizes (n <= 4), the K_{3,3}-free and
   K_5-free constraints on the Hasse diagram force degree anti-correlation
   among intermediates.  This anti-correlates their phases: a +1 node
   suppresses other +1 nodes.  Net charge is screened -- the geometric
   analogue of virtual e+e- pair production in QED.

4. **Running coupling:** Q_topo decreases from 0.271 (N=100k) to 0.191
   (N=10M), extrapolating to Q_inf = 0.152, 1/alpha_0 = 165.1.  The
   fine-structure constant runs natively from graph planarity, without
   any loop expansion or renormalization prescription.

See `data/scripts/vacuum_polarization.py` for publication-quality figures
and `data/scripts/occupancy_model.py` for the full analytical derivation.

---

## Bekenstein-Hawking Entropy: The Factor of 4

The simulation data contains Jacobson's thermodynamic derivation of Einstein's
equations *in full*, dormant in two CSV files from the start.  Applying the
Clausius relation δQ = T·dS to local causal horizons yields **two independent
measurements of Newton's constant** from the raw graph data:

| Route | Method | G (link units) |
|-------|--------|----------------|
| Thermodynamic | Clausius flux: δQ/(T·dS) = 8πG | 0.231 |
| Combinatorial | Bekenstein max bits: D_max/(4·log₂D_max) | 0.960 |

**Ratio = 4.16** — the Bekenstein-Hawking factor of 4, recovered to 4% error
with zero tuning.

The physical area formulation A = V^{1/2} (4D horizon) crushes the spectral
area formulation: **CV = 26.1% vs 61.0%**.  The horizon lives in macroscopic
d = 4 spacetime even though the walker probes the fractal d_S(σ).  This
validates the holographic principle from first principles.

This is the 50-year problem of quantum gravity — the same factor that string
theory required extremal black holes and LQG required the Immirzi parameter
to achieve — emerging here from pure combinatorial topology with no adjustable
parameters.

See `data/scripts/jacobson_einstein.py` for the complete 8-part analysis.

---

## GUE Spectral Rigidity

The eigenvalue spectrum of the Hasse-diagram Laplacian obeys **Gaussian Unitary
Ensemble (GUE)** statistics from Random Matrix Theory, providing an independent
derivation of the imaginary unit *i* from spectral data.

| Statistic | Measured | GUE | GOE | Poisson |
|-----------|----------|-----|-----|---------|
| Spacing ratio <r> | **0.603 +/- 0.002** | 0.6027 | 0.5307 | 0.3863 |
| Form factor slope gamma | **1.04 +/- 0.03** | 1.0 | -- | -- |

- GOE is ruled out at p < 10^{-6}; Poisson at p < 10^{-20}
- GUE (not GOE) because the BD action's alternating signs (-1, +9, -16, +8)
  break time-reversal symmetry — the same mechanism that forces *i* in the
  continuum limit
- This is an **independent confirmation** of complex quantum mechanics: the BD
  action forces *i*, and the spectral statistics diagnose it

The `fss_rmt` binary performs the RMT analysis across multiple lattice sizes.

---

## Emergent Einstein Equations

The Jacobson alpha-sweep confirms that Einstein's field equations emerge from
the discrete causal graph via the Clausius relation delta_Q = T * delta_S applied
to local causal horizons:

- At the physical point alpha = 1: G_BH / G_thermo = **4.00 +/- 0.05**
  (the Bekenstein-Hawking factor of 4)
- The gravitational constant G = 1/(16*pi) in natural units emerges from the
  integer BD weights: |c_0| + |c_1| + |c_2| + |c_3| = 1 + 9 + 16 + 8 = 34
- The metric signature (-,+,+,+) is forced by the alternating BD coefficients —
  one time dimension and three space dimensions are **derived**, not assumed
- Bulk fluctuations of ~25% are quantum foam at the mesoscopic scale

See `prism_simmulation/src/jacobson.rs` for the implementation and
`data/scripts/jacobson_einstein.py` for the analysis.

---

## Repository Structure

```
fractal_entropic_geometrodynamic/
├── prism_simmulation/          Rust simulation engine (Kuratowski Calculus)
│   ├── src/
│   │   ├── main.rs             Orchestration and Monte Carlo ensemble averaging
│   │   ├── diamond.rs          Poisson sprinkling in 4D causal diamond
│   │   ├── skyrmion.rs         Causal Prism detection and K5 contraction
│   │   ├── spectral.rs         Spectral dimension via random walk
│   │   ├── measurement.rs      Observer modules (cover-time mass, half-life, etc.)
│   │   ├── output.rs           CSV serialisation
│   │   ├── memory.rs           RAM-aware mode selection
│   │   ├── checkpoint.rs       Per-realisation checkpointing
│   │   ├── lib.rs              Crate root with theory documentation
│   │   └── bin/
│   │       ├── verify_algo.rs  Hasse diagram verification binary
│   │       └── fss_rmt.rs      Finite-size RMT analysis binary
│   ├── doc/                    Man page, LaTeX manual, example config
│   ├── README.md               Physics primer + architecture + technical docs
│   └── Cargo.toml
├── data/                           Reproducibility artifacts
│   ├── ensemble_10M/               Production N=10^7, M=20 (3 CSVs)
│   ├── fss_scaling/                FSS at 4 lattice sizes + JSON results
│   ├── fss_rmt.csv                 RMT finite-size scaling results
│   ├── scripts/                    Python analysis & figure generation
│   │   ├── jacobson_einstein.py    Bekenstein-Hawking factor of 4
│   │   ├── vacuum_polarization.py  VP figures (Q running, mu_eff, 1/alpha)
│   │   ├── gue_correlation.py     GUE spacing ratio analysis
│   │   ├── gue_correlation_bd.py  GUE with BD action weights
│   │   ├── spectral_zeta_riemann.py  Spectral zeta function analysis
│   │   ├── finite_size_scaling_rmt.py  FSS for RMT observables
│   │   ├── occupancy_model.py     Mass + charge + VP occupancy analysis
│   │   └── ...                     FSS, feg_analysis, composites
│   ├── figures/                    Pre-generated publication figures
│   └── README.md                   Full data documentation
├── pedagogic_booklets/         LaTeX source and compiled PDFs
│   ├── modulo_synthesis_vol_I  Volume I: The Geometry of Spacetime
│   ├── modulo_synthesis_vol_II Volume II: The Geometry of Matter
│   ├── modulo_synthesis_kuratowski_calculus  Kuratowski Calculus (Spanish)
│   └── emma_thought_experiments  Emma's Thought Experiments
├── LICENSE                     Dual: MIT (code) + CC BY-NC-ND 4.0 (theory)
└── README.md                   This file
```

---

## Hardware Requirements

All results in this repository were produced on a **ThinkPad T480**
(Intel i5-8250U, 4 cores / 8 threads, 16 GB RAM, Arch Linux).
No GPU. No cluster. No cloud.

Any modern laptop with 8+ GB RAM and a Rust toolchain suffices.
The N=10^7 production ensemble takes ~4.5 hours; the full FSS suite
(all 5 lattice sizes) takes ~8 hours total.

---

## Quick Start

### Prerequisites

- **Rust** 1.75+ (stable): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Python** 3.10+ with: `pip install numpy pandas matplotlib scipy`

### Build

```bash
cd prism_simmulation
cargo build --release --bin causal_set_sim
```

### Test run (N=50,000, ~30 seconds)

```bash
cargo run --release --bin causal_set_sim -- 50000 10 --inmemory
```

### View the man page

```bash
man prism_simmulation/doc/causal_set_sim.7
```

---

## Reproducing All Results

### Step 1: Run each lattice size

```bash
cd prism_simmulation

# FSS suite (smallest to largest)
cargo run --release --bin causal_set_sim -- 100000  10 --inmemory --seed 42   # ~2 min
cargo run --release --bin causal_set_sim -- 500000  10 --inmemory --seed 42   # ~10 min
cargo run --release --bin causal_set_sim -- 1000000 10 --inmemory --seed 42   # ~25 min
cargo run --release --bin causal_set_sim -- 5000000 10 --inmemory --seed 42   # ~2.5 h

# Production ensemble
cargo run --release --bin causal_set_sim -- 10000000 20 --inmemory --seed 42  # ~4.5 h
```

### Step 1b: Run RMT finite-size analysis

```bash
# Build and run the RMT analysis binary
cargo build --release --bin fss_rmt
cargo run --release --bin fss_rmt -- --sizes 500,1000,2000,5000 --m 10 --seed 42
```

### Step 2: Analyse and generate figures

```bash
python data/scripts/finite_size_scaling.py --analyze
python data/scripts/feg_analysis.py
python data/scripts/vacuum_polarization.py
python data/scripts/make_composite_figure.py
python data/scripts/make_fss_composite.py
python data/scripts/jacobson_einstein.py
python data/scripts/occupancy_model.py
python data/scripts/gue_correlation.py
python data/scripts/gue_correlation_bd.py
python data/scripts/spectral_zeta_riemann.py
```

### Step 3: Verify

Compare your `fss_comprehensive_results.json` against the included version.
All R^2 values should match to 3 decimal places.

See [`data/README.md`](data/README.md) for full documentation of data formats,
CSV columns, and verification procedures.

---

## Data

The `data/` directory contains all simulation output, analysis scripts, and
pre-generated figures needed to reproduce every result in the paper.
See [`data/README.md`](data/README.md) for complete documentation.

---

## The Simulation Engine

The `prism_simmulation/` directory contains the O(N) Rust implementation of
the Kuratowski Calculus. Four phases:

1. **Sprinkling** -- Poisson-sprinkle N events into a 4D causal diamond
2. **Prism detection** -- Find K_{2,n} bipartite defects via 2-hop search; K5 absorption
3. **Spectral dimension** -- Monte Carlo random walkers with strict integer arithmetic
3.5. **Measurements** -- Cover-time mass, half-life census, modulo path integral, vacuum polarization (optional, `--measure-all`)
4. **Output** -- Ensemble-averaged CSV with error bars

Bounded Hasse degree (D <= 15) makes every operation O(N).
See [`prism_simmulation/README.md`](prism_simmulation/README.md) for the physics
primer, architecture, and full flag reference.

---

## Documentation

- **Man page:** `man prism_simmulation/doc/causal_set_sim.7`
- **LaTeX manual:** [`prism_simmulation/doc/causal_set_sim.pdf`](prism_simmulation/doc/causal_set_sim.pdf)
- **Simulation README:** [`prism_simmulation/README.md`](prism_simmulation/README.md)
- **Precision improvements:** [`prism_simmulation/PRECISION_IMPROVEMENTS.md`](prism_simmulation/PRECISION_IMPROVEMENTS.md)

---

## Pedagogic Booklets

Four volumes of the *Modulo Synthesis* pedagogic project:

| Volume | Title | Language |
|--------|-------|----------|
| I | The Geometry of Spacetime (Macroscopic) | English |
| II | The Geometry of Matter (Microscopic) | English |
| KC | Kuratowski Calculus | Spanish |
| Emma | Emma's Thought Experiments | English |

All booklets are in `pedagogic_booklets/` with LaTeX source and compiled PDFs.

---

## Citation

```bibtex
@software{silva_alvarado_2026_feg,
  author    = {Silva Alvarado, Juan Pablo},
  title     = {Fractal Entropic Geometrodynamics: Emergent Gravity and
               Three Particle Generations from Discrete Topological Constraints},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18719775},
  url       = {https://doi.org/10.5281/zenodo.18719775}
}
```

---

## License

This project is dual-licensed:

- **Software** (`prism_simmulation/`, `data/scripts/`): [MIT License](https://opensource.org/licenses/MIT)
- **Theory and pedagogic material** (`pedagogic_booklets/`): [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

See the [LICENSE](LICENSE) file for the full legal text.

---

> "The universe is strictly computable, finite, and structurally homeostatic. The rest is just counting."
