# Fractal Entropic Geometrodynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18690574.svg)](https://doi.org/10.5281/zenodo.18690574)
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
Planck scale; the 20% gap to the physical alpha^{-1} = 137.036 is the domain of
renormalization-group running from the Planck scale to laboratory energies.

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
See `data/scripts/occupancy_model.py` for the full analysis.

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
│   │   └── bin/verify_algo.rs  Hasse diagram verification binary
│   ├── doc/                    Man page, LaTeX manual, example config
│   ├── README.md               Physics primer + architecture + technical docs
│   └── Cargo.toml
├── data/                       Reproducibility artifacts
│   ├── ensemble_10M/           Production N=10^7, M=20 (3 CSVs)
│   ├── fss_scaling/            FSS at 4 lattice sizes + JSON results
│   ├── scripts/                Python analysis & figure generation
│   ├── figures/                Pre-generated publication figures
│   └── README.md               Full data documentation
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

### Step 2: Analyse and generate figures

```bash
python data/scripts/finite_size_scaling.py --analyze
python data/scripts/feg_analysis.py
python data/scripts/make_composite_figure.py
python data/scripts/make_fss_composite.py
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
  doi       = {10.5281/zenodo.18690574},
  url       = {https://doi.org/10.5281/zenodo.18690574}
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
