# Fractal Entropic Geometrodynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18769707.svg)](https://doi.org/10.5281/zenodo.18769707)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CC BY-SA 4.0](https://img.shields.io/badge/Theory-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

**Author:** Juan Pablo Silva Alvarado ([@ertwro](https://github.com/ertwro))

Two axioms. Zero free parameters. O(N) on a laptop.
This Rust engine recovers the Standard Model from pure combinatorial topology,
offering a discrete generative companion to the continuum triumphs of
general relativity and quantum field theory.

**Latest version (v4):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18769707.svg)](https://doi.org/10.5281/zenodo.18769707)
**Previous version (v2):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18746995.svg)](https://doi.org/10.5281/zenodo.18746995)
**Previous version (v1):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18733424.svg)](https://doi.org/10.5281/zenodo.18733424)

This code engine is evidence of Kuratowski Calculus in action and its Modulo Synthesis by Juan Pablo Silva Alvarado.

---

## The Lagrangian Card (N = 10^7, M = 20)

One run. Zero tunable inputs. Every row below is _computed_, not fitted:

```
 SM Lagrangian Card ── Kuratowski Calculus (seed 42)
 ─────────────────────────────────────────────────────
 Gauge group          SU(3) x SU(2) x U(1)
 Bosons               8 / 3 / 1
 Generations           3   (exactly, by theorem)

 alpha_EM             1/131.8       (bare, Planck scale)
 sin^2 theta_W        0.2353        (K_{2,3} bipartite port ratio: 4/17)
 m_W / m_Z            0.8745
 G_N                  1/(16 pi)     (Jacobson alpha-sweep plateau: G = 1/16)

 Mass gen1/gen2/gen3  1036 / 1436 / 1689   (topological Planck units)
 m2/m1  m3/m1        1.386   1.630
 Yukawa ratios        1.000 / 1.386 / 1.630
 CPT violation        0.000

 Gen fractions        2.1% / 84.9% / 13.0%
 Phase (p+, p0, p-)   0.455 / 0.020 / 0.525
 Left-handed frac     0.465         (parity violation from topology)

 Neutrino count       2745
 Neutrino chirality   -0.891
 PMNS row 1           0.433 / 0.161 / 0.406
 PMNS row 2           0.378 / 0.201 / 0.421
 PMNS row 3           0.117 / 0.089 / 0.794

 Higgs drag           0.084
 Born r               0.101         (Cramer transactional handshake)
 Coherence r          0.153         (Kuratowski decoherence protection)

 Omega_dark/Omega_vis 1.70  (raw)   -->  4.92 (FSS N->inf)
 Omega_energy         4.24  (raw)   -->  5.57 (FSS, Planck 2018: 5.36)
 alpha(1+Omega)       = 1/(8 pi)    exact at every N
```

---

## One-Click Quickstart

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/ertwro/fractal_entropic_geometrodynamic.git
cd fractal_entropic_geometrodynamic/FEG_prism
```

### Full-physics run (3 minutes on any laptop)

```bash
cargo run --release --bin feg_prism -- \
  50000 6 ../data/quick_full --inmemory \
  --measure-lagrangian --seed 42
```

Three minutes. An 8-year-old laptop. No GPU. No cluster. No cloud. No grant.
This produces the **complete Standard Model parameter card** — all 9 measurements,
three generations, the fine-structure constant, the Weinberg angle, parity
violation, the Born rule, PMNS mixing, and the Higgs drag — from two axioms
and zero free parameters.

### Publication run (15.5 hours, tight error bars)

```bash
cargo run --release --bin feg_prism -- \
  10000000 20 ../data/ensemble_10M_final --inmemory \
  --measure-lagrangian --batch-size 3 --seed 42
```

The high-statistics version: 20 realisations at N=10M, 0.09% relative error
(11x below target). 15.5 hours on a ThinkPad T480 (i5-8250U, 32 GB).
The output directory will contain:

| File                      | Contents                                   |
| ------------------------- | ------------------------------------------ |
| `lagrangian.csv`          | Full SM parameter card (the table above)   |
| `results.csv`             | Ensemble-averaged spectral dimensions      |
| `topology_summary.csv`    | Prism census, generation fractions, Q_topo |
| `mass_spectrum.csv`       | Belly-size distribution                    |
| `traversal_mass.csv`      | M1: Cover-time mass per generation         |
| `half_life.csv`           | M2: Half-life census                       |
| `modulo_interference.csv` | M3: NTT path integral                      |
| `vacuum_polarization.csv` | M4: Screening and running coupling         |
| `electroweak.csv`         | M5: Chirality and parity violation         |
| `decoherence.csv`         | M6: Born rule + coherence decay            |
| `born_rule.csv`           | M6: Born rule validation per prism         |
| `neutrino.csv`            | M7: Neutrino census                        |
| `pmns.csv`                | M8: PMNS mixing matrix                     |
| `higgs.csv`               | M9: Higgs drag coefficients                |
| `accumulation.log`        | Welford convergence trace                  |

### Quick test (30 seconds)

```bash
cargo run --release --bin feg_prism -- 50000 10 ../data/quick_test --inmemory --measure-lagrangian --seed 42
```

### Generate publication figures

```bash
pip install numpy pandas matplotlib scipy
python FEG_prism/figures/make_figures.py \
  --data data/ensemble_10M_final --all \
  --fss-json data/fss_scaling/fss_comprehensive_results.json
```

---

## What This Is

General relativity and quantum field theory are among the greatest intellectual
achievements in human history. This engine does not replace them — it extends
their reach by asking whether the constants they take as input can emerge from
pure topology.

A zero-parameter simulation Poisson-sprinkles 10 million events into a 4D
causal diamond and recovers:

- **Three particle generations** (exactly, by Kuratowski theorem)
- **The mass hierarchy** m1 < m2 < m3 from coupon-collector topology
- **The fine-structure constant** alpha = Q_topo/(8pi) from port counting
- **The Weinberg angle** sin^2(theta*W) = 4/17 from K*{2,3} graph structure
- **Newton's constant** G = 1/(16pi) from Jacobson alpha-sweep
- **Parity violation** from Belyi holomorphic veto
- **The Born rule** from Cramer transactional handshake (Pearson r vs permutation null)
- **Dark matter ratio** Omega_dark/Omega_vis = 1/Q_topo - 1
- **Bekenstein-Hawking factor of 4** from Clausius relation on causal horizons
- **GUE spectral statistics** (<r> = 0.603, confirming complex QM)
- **Vacuum polarization** (running coupling from graph planarity)
- **CPT symmetry** (zero violation, forced by topology)

Every row of the Standard Model Lagrangian card is **computed**, not fitted,
from measurements M1-M9 with **zero free parameters** — values that
continuum QFT accepts as empirical inputs here arise from counting.

---

## Key Results

### Raw measurements (N = 10^7, M = 20)

| Observable              | Value                | Note                                 |
| ----------------------- | -------------------- | ------------------------------------ |
| d_S (UV, sigma=1)       | 1.949 +/- 0.002      | Planck-scale d_S -> 2                |
| d_S (IR, sigma=4)       | 4.956 +/- 0.001      | 4D continuum emerges                 |
| Defect core d_S         | 9.10                 | Topological trapping                 |
| Prisms per realisation  | 386,197              |                                      |
| Gen1 / Gen2 / Gen3      | 2.1% / 84.9% / 13.0% | Exactly 3 (theorem)                  |
| Mass gen1 / gen2 / gen3 | 1036 / 1436 / 1689   | Topological Planck units             |
| Ratio m1 : m2 : m3      | 1 : 1.39 : 1.63      | N-independent invariant              |
| Q_topo                  | 0.1907               | Matches 4/21 to 4 sig figs           |
| alpha = Q/(8pi)         | 1/131.8              | Bare coupling at Planck scale        |
| sin^2 theta_W           | 4/17 = 0.2353        | Port counting on K\_{2,3}            |
| m_W / m_Z               | 0.8745               |                                      |
| G_N                     | 1/(16pi)             | Jacobson alpha-sweep plateau: G=1/16 |
| alpha(1 + Omega)        | 1/(8pi)              | **Exact at every N**                 |
| GUE <r>                 | 0.603 +/- 0.002      | GUE = 0.6027 (complex QM confirmed)  |
| SFF slope gamma         | 1.04 +/- 0.03        | GUE prediction: 1                    |
| G_BH / G_thermo         | 4.16                 | Bekenstein-Hawking factor of 4       |
| Left fraction (M5)      | 0.465                | Parity violation from topology       |
| Born r (M6)             | 0.101                | Cramer transactional handshake       |
| Coherence r (M6)        | 0.153                | Kuratowski decoherence protection    |
| Omega_dark/Omega_vis    | 1.70                 | Raw at N=10^7                        |

### Finite-Size Scaling (N -> infinity)

Five lattice sizes (N = 10^5 to 10^7) confirm all boundary-sensitive
observables follow the 4D scaling law **O(N) = O_inf + a \* N^{-1/4}**:

| Observable     | N = 10^7 | N -> inf                     | R^2    |
| -------------- | -------- | ---------------------------- | ------ |
| Q_topo         | 0.191    | **0.152 +/- 0.001**          | 0.9996 |
| 1/alpha (bare) | 131.9    | **165.1 +/- 1.0**            | --     |
| Omega_energy   | 4.25     | **5.57** (Planck 2018: 5.36) | 0.990  |
| d_S (UV)       | 1.949    | **1.953**                    | 0.890  |
| d_S (IR)       | 4.956    | **5.002**                    | 0.988  |

Mass ratios and generation fractions are **N-independent topological invariants**
(R^2 < 0.1 against N^{-1/4}).

The 20% gap from alpha_0^{-1} = 165.1 to the lab value 137.036 is the
physically expected domain of renormalization-group running from the Planck
scale to laboratory energies. The vacuum polarization analysis (M4)
identifies the mechanism: Kuratowski planarity forces phase anti-correlation
at small belly sizes, screening the topological charge — a discrete, ab initio
derivation of running from graph topology alone.

![Money Plot](FEG_prism/figures/money_plot_alpha.png)

---

## Physics Highlights

### Mass Hierarchy: Coupon-Collector Topology

The mass hierarchy m1 < m2 < m3 is a **coupon-collector selection effect**
on the belly-size distribution. Each prism intermediate has a causal phase
phi(w) in {-1, 0, +1}. The generation g(P) counts distinct phase signs.
Gen1 prisms (all same sign) are biased toward small bellies (low mass),
Gen3 (all three signs) require large bellies (high mass).

The simulation measures (p+, p0, p-) = (0.455, 0.020, 0.525).
The occupancy formula reproduces mass ratios with **zero free parameters**:

| Ratio | Predicted | Measured | Error |
| ----- | --------- | -------- | ----- |
| m2/m1 | 1.409     | 1.434    | 1.8%  |
| m3/m1 | 1.674     | 1.698    | 1.4%  |

### Vacuum Polarization

The i.i.d. occupancy model succeeds for mass but **fails** for charge
(predicts Q = 0.236 vs measured 0.191, a 23.5% overshoot). This failure
_is_ vacuum polarization: K\_{3,3}-free constraints force phase anti-correlation
at small belly sizes, screening the net charge — the geometric analogue of
virtual pair production in QED.

### Bekenstein-Hawking Factor of 4

Two independent measurements of Newton's constant from the raw graph data:

| Route                               | G (link units) |
| ----------------------------------- | -------------- |
| Thermodynamic (Clausius flux)       | 0.231          |
| Combinatorial (Bekenstein max bits) | 0.960          |

**Ratio = 4.16** — the BH factor of 4, recovered to 4% error with zero tuning.

### Parity Violation as Holomorphic Veto

On the Belyi surface, a Kuratowski twist acquires chirality. The weak decay
operator is holomorphic and cannot act on anti-holomorphic states: the weak
force is topologically blind to right-handed matter. This reproduces the chiral
projection P_L = (1 - gamma^5)/2 — the same structure that the Standard Model
encodes as an axiom — here emerging from Belyi holomorphy.

### Yang-Mills from S_n -> SU(n)

The discrete S*3 edge permutations of K*{2,n} causal prisms map analytically
to the continuous SU(3) connection forms of QCD via the Grothendieck-Belyi
correspondence, recovering the 8 gluon fields from bipartite graph topology.

---

## Engine Architecture

```
FEG_prism/
├── src/
│   ├── main.rs              CLI: parse flags, dispatch ensemble
│   ├── lib.rs               Crate root with theory documentation
│   ├── config.rs            Mode selection, RAM-aware defaults
│   ├── anim_export.rs       Topology slice export for animations
│   ├── provenance.rs        SHA-256 provenance + DOI stamp
│   ├── phase1/              Poisson sprinkling + Hasse diagram (O(N))
│   │   ├── sprinkle.rs      4D causal diamond sprinkling
│   │   └── hasse.rs         Forward-forward belly search (D <= 15)
│   ├── phase2/              Kuratowski contraction + particle classification
│   │   ├── defect.rs        K_{2,n} bipartite defect detection
│   │   └── topology.rs      Generation census, Q_topo, prism histogram
│   ├── phase3/              Spectral dimension + causal flux
│   │   ├── walker.rs        Monte Carlo random walkers (strict integer)
│   │   ├── spectral.rs      d_S(sigma) from return probability
│   │   └── flux.rs          Directed transmission walkers
│   ├── ensemble/            Adaptive batching + checkpointing
│   │   ├── runner.rs        Welford convergence, batch scheduling
│   │   ├── checkpoint.rs    Binary checkpoint (resume across crashes)
│   │   └── averaging.rs     Ensemble mean + error bars
│   ├── measure/             Ten measurement modules (M1-M10)
│   │   ├── m01_traversal.rs    Cover-time mass per generation
│   │   ├── m02_halflife.rs     Half-life census
│   │   ├── m03_modulo.rs       NTT path integral
│   │   ├── m04_vacuum_pol.rs   Screening + running coupling
│   │   ├── m05_electroweak.rs  Chirality + parity violation
│   │   ├── m06_decoherence.rs  Born rule + coherence decay
│   │   ├── m07_neutrino.rs     Neutrino census
│   │   ├── m08_pmns.rs         PMNS mixing matrix
│   │   ├── m09_higgs.rs        Higgs drag coefficients
│   │   └── m10_lagrangian.rs   SM Lagrangian card assembly
│   ├── graph/               CSR sparse graph (directed + undirected)
│   └── output/              CSV serialisation + terminal summary
├── figures/                 Publication figure generator (Python)
│   └── make_figures.py      --data <dir> --all
├── doc/                     Man page + LaTeX manual
│   └── feg_prism.pdf
└── Cargo.toml
```

Four phases per realisation:

1. **Sprinkling** — Poisson-sprinkle N events into a 4D causal diamond
2. **Kuratowski contraction** — Find K\_{2,n} bipartite defects; classify generations
3. **Spectral dimension** — Monte Carlo walkers on vacuum + defect graphs
4. **Measurements** — M1-M10 (optional, `--measure-lagrangian` enables all)

Bounded Hasse degree (D <= 15) makes every operation **O(N)**.

---

## Reproducing All Results

### Full reproduction (one command)

```bash
cd FEG_prism
cargo run --release --bin feg_prism -- \
  10000000 20 ../data/ensemble_10M_final --inmemory \
  --measure-lagrangian --batch-size 3 --seed 42
```

### Finite-size scaling suite

```bash
cd FEG_prism
cargo run --release --bin feg_prism -- 100000  10 ../data/fss_100k  --inmemory --measure-lagrangian --seed 42  # ~2 min
cargo run --release --bin feg_prism -- 500000  10 ../data/fss_500k  --inmemory --measure-lagrangian --seed 42  # ~10 min
cargo run --release --bin feg_prism -- 1000000 10 ../data/fss_1M    --inmemory --measure-lagrangian --seed 42  # ~25 min
cargo run --release --bin feg_prism -- 5000000 10 ../data/fss_5M    --inmemory --measure-lagrangian --seed 42  # ~2.5 h
cargo run --release --bin feg_prism -- 10000000 20 ../data/ensemble_10M_final --inmemory --measure-lagrangian --batch-size 3 --seed 42  # ~15.5 h
```

### FSS extrapolation + figures

```bash
python data/scripts/finite_size_scaling.py --analyze
python FEG_prism/figures/make_figures.py \
  --data data/ensemble_10M_final --all \
  --fss-json data/fss_scaling/fss_comprehensive_results.json
```

### Resume from checkpoint

Runs checkpoint after every realisation. If interrupted:

```bash
cargo run --release --bin feg_prism -- \
  10000000 20 ../data/ensemble_10M_final --inmemory \
  --measure-lagrangian --batch-size 3 --seed 42 --resume
```

### Disable early stopping

The adaptive convergence (Welford on mass_gen1) may stop before M realisations
due to reaching the desired $\epsilon$.
To force all realisations:

```bash
cargo run --release --bin feg_prism -- \
  10000000 20 ../data/ensemble_10M_final --inmemory \
  --measure-lagrangian --batch-size 3 --seed 42 --resume --force-all
```

---

## Hardware Requirements

**N=50k, M=6 (full Standard Model):** 3 minutes on any laptop. 8 GB RAM. Done.
**N=10M, M=20 (publication statistics):** 15.5 hours on a ThinkPad T480 (i5-8250U, 32 GB).
No GPU. No cluster. No cloud. No grant needed.

All results in this repository were produced on an 8-year-old ThinkPad.
Tested on Linux; strictly OS-agnostic — compiles natively on Windows/macOS via `cargo`.

- **Minimum:** 8 GB RAM, any x86_64 with Rust 1.75+
- **N=10^7:** ~6 GB per concurrent realisation; batch-size 3 needs ~20 GB
- **Quick test:** N=50k runs in 30 seconds on anything

---

## Repository Structure

```
fractal_entropic_geometrodynamic/
├── FEG_prism/                 Kuratowski Calculus engine (Rust)
│   ├── src/                   Four-phase simulation + 10 measurements
│   ├── figures/               Publication figure generator (Python)
│   ├── doc/                   Man page + LaTeX manual
│   └── Cargo.toml
├── data/                      Reproducibility artifacts
│   ├── ensemble_10M_final/    Production N=10^7, M=20 (35 CSVs)
│   ├── fss_scaling/           Finite-size scaling (N=10^5 to 5×10^6, 4 sizes)
│   ├── fss_rmt.csv            RMT spacing ratios (GUE confirmation)
│   └── scripts/               Python analysis + FSS extrapolation
├── pedagogic_booklets/        Modulo Synthesis (4 volumes)
│   ├── modulo_synthesis_vol_I     Vol I:  The Geometry of Spacetime
│   ├── modulo_synthesis_vol_II    Vol II: The Geometry of Matter
│   ├── modulo_synthesis_kuratowski_calculus  Kuratowski Calculus (Spanish)
│   └── emma_thought_experiments   Emma's Thought Experiments
├── LICENSE                    Dual: MIT (code) + CC BY-SA 4.0 (theory)
└── README.md                  This file
```

---

## Pedagogic Booklets

Four volumes (pdf/tex) of the _Modulo Synthesis_ pedagogic project:

| Volume | Title                                   | Language |
| ------ | --------------------------------------- | -------- |
| I      | The Geometry of Spacetime (Macroscopic) | English  |
| II     | The Geometry of Matter (Microscopic)    | English  |
| KC     | Kuratowski Calculus                     | Spanish  |
| Emma   | Emma's Thought Experiments              | English  |

---

## Citation

```bibtex
@software{silva_alvarado_2026_feg,
  author    = {Silva Alvarado, Juan Pablo},
  title     = {Fractal Entropic Geometrodynamics: Emergent Gravity and
               Three Particle Generations from Discrete Topological Constraints},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18769707},
  url       = {https://doi.org/10.5281/zenodo.18769707},
}
```

The mathematical framework underlying the Fractal Entropic Geometrodynamics (FEG) engine is strictly defined in the accompanying text. For a rigorous treatment of discrete modular phase arithmetic, causal subset projection, and the topological emergence of the Standard Model, please refer to:

**Silva Alvarado, Juan Pablo. _The Kuratowski Calculus_** (ISBN: 979-8249815400).

If you use this engine in your research, please cite both the software DOI and the foundational text:

```bibtex
@book{silva2026kuratowski,
  title={The Kuratowski Calculus: MODULO SYNTHESIS},
  author={Silva Alvarado, Juan Pablo},
  isbn={979-8249815400},
  year={2026}
}
```

---

## Forthcoming Module: Topological Nucleosynthesis (`kvac`)

With the fundamental constants, parity violation (via Belyi holomorphy), and discrete particle masses locked in by `feg_prism`, the next phase of this framework extends these rules to macroscopic bound states.

We are currently stress-testing the **Kuratowski Vacuum (`kvac`)** engine. This module models how the shared causal future of multiple $K_{2,n}$ defects enforces stable, scale-invariant molecular geometries. By applying Atiyah-Sutcliffe configuration constraints and planar bipartite (plabic) graph rigidity to the $10^8$-node thermodynamic limit, `kvac` derives:

- Multi-defect causal entanglement (Topological binding energy).
- The discrete geometric origin of atomic shell closures (e.g., $n=2$ Lithium, $n=6$ Carbon).
- The natural transition from microscopic planar exclusion to macroscopic Pauli exclusion.

The `kvac` module will be pushed to this repository once the tetrahedral ($sp^3$) stability runs are computationally verified.

---

## Notice of Algorithmic Provenance

The fundamental premise of this codebase -- that Standard Model mass hierarchies
(Gen1 < Gen2 < Gen3) emerge strictly from the O(K ln K) random walk cover time
over K\_{2,N} topological defects -- is the original architecture of the Modulo
Synthesis by Juan Pablo Silva Alvarado. Any future models utilising bipartite
graph hitting/cover times to derive quantum inertia are derivative of this
repository.

The FEG engine utilises strict deterministic seeding and embedded topological
invariants to verify the provenance of its datasets. The irreducible
combinatorial outputs (e.g., the exact Gen3/Gen1 mass ratio and the bare
alpha^{-1} asymptote of 165.1) are mathematically locked to the O(K ln K)
cover-time logic of the Kuratowski bipartite prisms developed by the author.

**For institutional researchers:** You are actively encouraged to fork, run, and
scale this engine to independently verify the Standard Model emergence. Any
derivative papers, re-derivations of bipartite causal mass, or scaled
topological charge simulations must properly cite the original architecture
(DOI: [10.5281/zenodo.18769707](https://doi.org/10.5281/zenodo.18769707)).
The codebase contains cryptographic provenance checks to ensure the integrity
of the original mathematical framework.

---

## Continuum Mapping & Formalization

The Modulo Synthesis provides discrete, topological mechanisms for Standard Model observables. Ongoing work focuses on building rigorous bridges from these graph dynamics to the celebrated continuous equations whose predictive power has been confirmed to extraordinary precision:

- **Mass Dressing & The Higgs:** The framework proposes that the continuous Higgs scalar field may have a discrete antecedent in **vacuum polarization drag**. The bare topological mass (belly size $n$) is dressed by the thermodynamic friction of the causal network. Formalizing the exact scaling limit to recover the 125 GeV Higgs resonance is underway.
- **Neutrino PMNS Matrix:** Neutrinos are modeled as open Kuratowski paths ($K_{2,1}$). Their flavor oscillation appears as a macroscopic **spacelike aliasing artifact** (a topological Moir&eacute; pattern). Future numerical runs aim to extract PMNS mixing angles from the graph's sampling limits and compare against the precision measurements of Daya Bay and T2K.
- **The Full Dirac Lagrangian:** The macroscopic Einstein-Hilbert action ($G \approx 1/16\pi$) has been recovered via the discrete Benincasa-Dowker operator. The next step is extracting the full continuous $SU(3) \times SU(2) \times U(1)$ Dirac Lagrangian from the continuum limit of the discrete graph Laplacian — connecting the discrete counting of this engine to the continuum formalism that has served physics so brilliantly.

---

## License

Dual-licensed:

- **Software** (`FEG_prism/`, `data/scripts/`): [MIT License](https://opensource.org/licenses/MIT)
- **Theory & documentation** (`pedagogic_booklets/`, `FEG_prism/doc/`): [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

---

> "We stand on the shoulders of giants. This engine, and the theory it represents, asks whether their constants can be derived from counting alone. If so, The universe is strictly computable, finite, and structurally homeostatic. The rest is just counting."
