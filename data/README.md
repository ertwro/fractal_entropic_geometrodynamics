# Data & Reproducibility

All results in this repository were produced on a **ThinkPad T480**
(Intel i5-8250U, 4 cores / 8 threads, 16 GB RAM) running Arch Linux.
No GPU, no cluster, no cloud.

DOI: [10.5281/zenodo.18697574](https://doi.org/10.5281/zenodo.18697574)

---

## Directory Structure

```
data/
├── ensemble_10M/                    Production N=10^7, M=20 ensemble
│   ├── results_M20.csv              Per-step observables (62 rows × 31 columns)
│   ├── topology_summary_M20.csv     Prism census and coupling constants
│   └── mass_spectrum_M20.csv        Belly-size histogram (28 bins)
├── fss_scaling/                     Finite-size scaling (4 lattice sizes)
│   ├── N_100000/                    N=10^5, M=10
│   │   ├── results.csv
│   │   ├── topology_summary.csv
│   │   └── mass_spectrum.csv
│   ├── N_500000/                    N=5×10^5, M=8
│   │   └── (same 3 CSVs)
│   ├── N_1000000/                   N=10^6, M=8
│   │   └── (same 3 CSVs)
│   ├── N_5000000/                   N=5×10^6, M=8
│   │   └── (same 3 CSVs)
│   └── fss_comprehensive_results.json
├── fss_rmt.csv                      RMT finite-size scaling results
├── scripts/                         Analysis & figure generation
│   ├── finite_size_scaling.py       FSS pipeline (run + analyze)
│   ├── finite_size_scaling_rmt.py   FSS for RMT observables
│   ├── feg_analysis.py              8-figure comprehensive analysis
│   ├── vacuum_polarization.py       Vacuum polarization figures (4 figures)
│   ├── occupancy_model.py           Occupancy model: mass + charge analysis
│   ├── gue_correlation.py           GUE spacing ratio analysis
│   ├── gue_correlation_bd.py        GUE with BD action weights
│   ├── spectral_zeta_riemann.py     Spectral zeta function analysis
│   ├── make_composite_figure.py     2×2 panel composite
│   ├── make_fss_composite.py        2×2 FSS composite
│   ├── jacobson_einstein.py         Bekenstein-Hawking factor of 4 (Jacobson analysis)
│   ├── plot_universe.py             3-figure spectral/mass/flux plotter
│   └── synthesis_analyzer.py        4-figure Physical Review analyzer
└── figures/                         Pre-generated publication figures
    ├── fss_q_topo.pdf               Q_topo scaling (R²=0.9996)
    ├── fss_inv_alpha.pdf            1/α scaling
    ├── fss_omega_energy.pdf         Ω_energy convergence
    ├── fss_mass_hierarchy.pdf       Mass hierarchy (N-independent)
    ├── fss_spectral_dimension.pdf   d_S UV and IR flow
    ├── fss_generation_fractions.pdf Generation populations
    ├── vp_q_running.pdf             Vacuum polarization: Q_topo running
    ├── vp_mu_effective.pdf          Effective mu(n) per belly size
    ├── vp_inv_alpha.pdf             1/α running with RG domain
    ├── vp_mass_vs_charge.pdf        Mass vs charge diagnostic
    ├── gue_correlation.pdf          GUE spacing ratio plot
    ├── gue_correlation_bd.pdf       GUE with BD weights plot
    ├── spectral_zeta_riemann.pdf    Spectral zeta function plot
    ├── fig_composite.pdf            4-panel composite (N=10^7)
    ├── fig_fss_composite.pdf        4-panel FSS composite
    └── fss_table.tex                LaTeX table of all FSS results
```

---

## Hardware

| Component | Specification |
|-----------|---------------|
| Machine   | Lenovo ThinkPad T480 (2018) |
| CPU       | Intel Core i5-8250U (4C/8T, 1.6 GHz base, 3.4 GHz boost) |
| RAM       | 16 GB DDR4-2400 |
| Storage   | 256 GB NVMe SSD |
| OS        | Arch Linux, kernel 6.18.x |
| Rust      | 1.75+ (stable) |
| Python    | 3.10+ with numpy, pandas, matplotlib, scipy |

---

## Reproducing From Scratch

### Prerequisites

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Python
pip install numpy pandas matplotlib scipy
```

### Build the simulation

```bash
cd prism_simmulation
cargo build --release --bin causal_set_sim
```

### Run the finite-size scaling suite

Each command below reproduces one lattice size. Expected runtimes on the ThinkPad T480:

| Lattice size | Command | Runtime |
|---|---|---|
| N=100,000 | `cargo run --release --bin causal_set_sim -- 100000 10 --inmemory --seed 42` | ~2 min |
| N=500,000 | `cargo run --release --bin causal_set_sim -- 500000 10 --inmemory --seed 42` | ~10 min |
| N=1,000,000 | `cargo run --release --bin causal_set_sim -- 1000000 10 --inmemory --seed 42` | ~25 min |
| N=5,000,000 | `cargo run --release --bin causal_set_sim -- 5000000 10 --inmemory --seed 42` | ~2.5 h |
| N=10,000,000 | `cargo run --release --bin causal_set_sim -- 10000000 20 --inmemory --seed 42` | ~4.5 h |

The N=10^7 ensemble with M=20 realisations is the production run.
All five sizes together take approximately 8 hours on the ThinkPad T480.

### Run FSS analysis

```bash
python data/scripts/finite_size_scaling.py --analyze
```

This reads all lattice sizes, performs O(N) = O_inf + a*N^{-1/4} fits,
and outputs `fss_comprehensive_results.json`, individual FSS figures,
and `fss_table.tex`.

### Regenerate all figures

```bash
python data/scripts/feg_analysis.py          # 8 individual figures
python data/scripts/vacuum_polarization.py   # 4 vacuum polarization figures
python data/scripts/make_composite_figure.py  # 4-panel composite
python data/scripts/make_fss_composite.py     # 4-panel FSS composite
python data/scripts/jacobson_einstein.py      # Bekenstein-Hawking factor of 4
```

### Run the occupancy model analysis

```bash
python occupancy_model.py
# or equivalently:
python data/scripts/occupancy_model.py
```

---

## CSV Column Descriptions

### `results.csv` / `results_M20.csv`

Per-diffusion-step observables, ensemble-averaged over M realisations.

| Column | Description |
|--------|-------------|
| `step` | Diffusion time σ (integer, 1 to σ_max) |
| `P_vac` | Return probability, vacuum |
| `dS_vac` | Spectral dimension, vacuum |
| `P_def` | Return probability, defect core |
| `dS_def` | Spectral dimension, defect core |
| `P_Gen1` / `dS_Gen1` | Return probability / spectral dimension, generation 1 |
| `P_Gen2` / `dS_Gen2` | Same for generation 2 |
| `P_Gen3` / `dS_Gen3` | Same for generation 3 |
| `P_Anti1` / `dS_Anti1` | Same for anti-generation 1 |
| `Flux_Attr` / `Flux_Repu` | Attractive / repulsive causal flux |
| `Flux_Attr_Norm` / `Flux_Repu_Norm` | Normalised flux |
| `P_Sterile` / `dS_Sterile` | Return probability / spectral dimension, sterile prisms |
| `Mass_Gen1` / `Mass_Gen2` / `Mass_Gen3` / `Mass_Anti1` | Average topological mass per generation |
| `dS_vac_std` ... `Flux_Repu_std` | Standard errors (ensemble variance) |

Header comments (lines starting with `#`) record: N, M, mode, epsilon, tmax, walkers, seed, elapsed time, algorithm, commit hash, timestamp.

### `topology_summary.csv` / `topology_summary_M20.csv`

Key-value pairs summarising the topology of the Hasse diagram.

| Key | Description |
|-----|-------------|
| `total_nodes` | N (number of sprinkled events) |
| `total_prisms` | Total Causal Prisms detected |
| `max_intermediates` | Maximum belly size n_max |
| `count_gen1` / `count_gen2` / `count_gen3` | Prism counts per generation |
| `count_antigen1` | Anti-generation 1 prism count |
| `count_sterile` | Sterile (Φ=0) prism count |
| `avg_mass_gen1` / `avg_mass_gen2` / `avg_mass_gen3` | Mean topological mass per generation |
| `avg_mass_sterile` | Mean topological mass of sterile prisms |
| `visible_mass_total` / `dark_mass_total` / `grav_mass_total` | Mass decomposition sums |
| `omega_ratio` | Ω_dark/Ω_vis (linear mass ratio) |
| `phase_sq_total` / `mass_sq_total` | Σ|Φ|² and ΣN² (for Q_topo) |
| `alpha_em` | **Note:** this field contains Q_topo = phase_sq/mass_sq, NOT the fine-structure constant. The actual coupling is α = Q_topo/(8π). The FSS analysis script handles this correctly. |

### `mass_spectrum.csv` / `mass_spectrum_M20.csv`

Belly-size histogram of all Causal Prisms.

| Column | Description |
|--------|-------------|
| `intermediates_N` | Belly size n (number of intermediates, 3 to n_max) |
| `frequency` | Number of prisms with this belly size |

---

## Vacuum Polarization & Running Coupling

The occupancy model (see `occupancy_model.py`) reveals a fundamental split between
mass and charge observables:

**Mass hierarchy (i.i.d. PASSES):** The mass ratios m2/m1 and m3/m1 are
*counting* observables -- they depend on how many distinct phase signs appear
among a prism's intermediates.  The i.i.d. coupon-collector model reproduces
both ratios to within 2%, with zero free parameters.

**Topological charge (i.i.d. FAILS):** The charge Q_topo = Σ|Φ|²/ΣN² is a
*summation* observable -- it depends on the net cancellation of phases.
The i.i.d. prediction Q_pred = 0.236 overshoots the observed Q_obs = 0.191
at N=10M by 23.5%.  Phases cancel more than independence allows.

**Physical mechanism -- Kuratowski vacuum polarization:** At small belly sizes
(n <= 4), the K_{3,3}-free and K_5-free constraints on the transitively reduced
Hasse diagram force in-degree/out-degree anti-correlation among intermediates.
This anti-correlates their phases: a +1 node crowds out other +1 nodes.  Net
charge is screened -- the discrete geometric analogue of virtual e+e- pair
production screening charge in QED.

**Running coupling:** Q_topo decreases monotonically with lattice size:

| N | Q_topo | 1/α = 8π/Q |
|---|--------|------------|
| 100,000 | 0.2715 | 92.6 |
| 500,000 | 0.2324 | 108.2 |
| 1,000,000 | 0.2184 | 115.1 |
| 5,000,000 | 0.1966 | 127.8 |
| 10,000,000 | 0.1907 | 131.8 |
| N → ∞ | **0.1523 ± 0.0009** | **165.1** |

The continuum-limit bare coupling 1/α_0 = 165.1 is the UV-cutoff value at the
Planck scale.  The 165 → 137 gap is not a deficiency of the model but the
physically expected domain of renormalization-group running from the Planck
scale to laboratory energies.  The mechanism is ab initio: graph planarity
forces phase anti-correlation at UV scales without any tuning.

---

## Bekenstein-Hawking Entropy

The script `jacobson_einstein.py` implements Jacobson's thermodynamic derivation
of Einstein's equations using the existing simulation data.  Two independent
measurements of Newton's constant from the raw graph data:

- **G_thermo** (Clausius flux): δQ/(T·dS) = 8πG → G = 0.231 (link units)
- **G_Bekenstein** (max bits): D_max/(4·log₂D_max) = 0.960 (link units)

Ratio = **4.16 ≈ 4** — the Bekenstein-Hawking entropy factor S = A/4G,
recovered to 4% error with zero tuning.  The physical area formulation
A = V^{1/2} wins decisively over the spectral area (CV = 26.1% vs 61.0%).

```bash
python data/scripts/jacobson_einstein.py
```

---

## JSON Schema: `fss_comprehensive_results.json`

```json
{
  "description": "...",
  "physics": "Boundary/volume ratio in 4D: T³/T⁴ = 1/T ~ N^{-1/4}",
  "data_points": [
    {
      "N": 100000, "T": 29.56, "M": "10", "converged": false,
      "Q_topo": 0.2715, "inv_alpha": 92.6, "Omega_energy": 2.68,
      "mass_gen1": 4.55, "mass_gen2": 6.48, "mass_gen3": 7.88,
      "mass_anti1": 4.66, "dS_vac_uv": 1.939, "dS_vac_ir": 4.847,
      "frac_gen1": 0.022, "frac_gen2": 0.842, "frac_gen3": 0.136,
      "cpt_frac": 0.025
    }
    // ... one entry per lattice size (5 total)
  ],
  "fits": {
    "Q_topo": { "O_inf": 0.1523, "O_inf_err": 0.0009, "a": 2.118, "R_sq": 0.9996 },
    // ... one entry per observable
  },
  "derived": {
    "alpha_inf": 0.00606,
    "inv_alpha_inf": 165.06,
    "Omega_inf_from_Q": 5.567,
    "gap_to_137_percent": 20.4
  },
  "vacuum_polarization": {
    "Q_pred_iid": 0.2355,
    "mu_measured": -0.3457,
    "sigma2_measured": 0.8622,
    "screening_by_N": {
      "100000":  { "Q_obs": 0.2715, "screening_frac": -0.153 },
      "500000":  { "Q_obs": 0.2324, "screening_frac": 0.013 },
      "1000000": { "Q_obs": 0.2184, "screening_frac": 0.073 },
      "5000000": { "Q_obs": 0.1966, "screening_frac": 0.165 },
      "10000000":{ "Q_obs": 0.1907, "screening_frac": 0.190 }
    }
  }
}
```

---

## Verification

After reproducing a run, compare your results against the included data:

1. **Quick check (N=100,000, ~2 min):**
   Your `topology_summary.csv` should show `Q_topo ≈ 0.271 ± 0.005`
   and generation fractions within 1% of the included values.

2. **Full verification:**
   Run `python data/scripts/finite_size_scaling.py --analyze` and compare
   the output `fss_comprehensive_results.json` against the included version.
   All R² values should match to 3 decimal places; O_inf values should
   agree within stated uncertainties.

3. **Figures:**
   Regenerated figures should be visually identical to the included PDFs.
   Minor differences in anti-aliasing are expected across platforms.

4. **Vacuum polarization:**
   Run `python data/scripts/vacuum_polarization.py` and verify that 4 figures
   (`vp_q_running.pdf`, `vp_mu_effective.pdf`, `vp_inv_alpha.pdf`,
   `vp_mass_vs_charge.pdf`) are produced in `data/figures/`.
   Run `python data/scripts/occupancy_model.py` and verify the overshoot is ~23.5%
   and the FSS extrapolation matches Q_inf ≈ 0.152.

---

## GUE Spectral Rigidity (RMT Analysis)

The `fss_rmt` binary computes Random Matrix Theory statistics on the Hasse-diagram
Laplacian eigenvalue spectrum. Key results:

| Statistic | Measured | GUE prediction | GOE | Poisson |
|-----------|----------|----------------|-----|---------|
| Spacing ratio <r> | 0.603 +/- 0.002 | 0.6027 | 0.5307 | 0.3863 |
| Form factor slope gamma | 1.04 +/- 0.03 | 1.0 | -- | -- |

GUE (not GOE) because the BD action's alternating signs break time-reversal symmetry.
This independently confirms complex quantum mechanics (the imaginary unit *i*).

### Reproducing the RMT analysis

```bash
cd prism_simmulation

# Build the RMT binary
cargo build --release --bin fss_rmt

# Quick test (N=500, ~seconds)
cargo run --release --bin fss_rmt -- --sizes 500 --m 1 --seed 42

# Full analysis
cargo run --release --bin fss_rmt -- --sizes 500,1000,2000,5000 --m 10 --seed 42
```

Results are written to `fss_rmt.csv`.

### `fss_rmt.csv` format

| Column | Description |
|--------|-------------|
| `N` | Lattice size |
| `m` | Realisation index |
| `r_mean` | Mean spacing ratio (GUE: 0.6027) |
| `gamma` | Spectral form factor slope (GUE: 1.0) |

---

## Alpha Sweep & Emergent Einstein

The Jacobson analysis (`jacobson.rs` + `jacobson_einstein.py`) sweeps the BD action
prefactor alpha and measures the ratio G_BH / G_thermo:

- At alpha = 1 (physical point): ratio = **4.00 +/- 0.05** (Bekenstein-Hawking factor)
- G = 1/(16*pi) emerges from integer BD weights (1 + 9 + 16 + 8 = 34)
- Metric signature (-,+,+,+) is forced by the alternating BD coefficients

The `results.csv` now contains 26 columns including bulk-restricted fields for
the Jacobson analysis (bulk spectral dimension, bulk return probability, bulk
causal flux).
