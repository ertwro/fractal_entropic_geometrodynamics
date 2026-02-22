# Data & Reproducibility

All results in this repository were produced on a **ThinkPad T480**
(Intel i5-8250U, 4 cores / 8 threads, 16 GB RAM) running Arch Linux.
No GPU, no cluster, no cloud.

DOI: [10.5281/zenodo.18690574](https://doi.org/10.5281/zenodo.18690574)

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
├── scripts/                         Analysis & figure generation
│   ├── finite_size_scaling.py       FSS pipeline (run + analyze)
│   ├── feg_analysis.py              8-figure comprehensive analysis
│   ├── make_composite_figure.py     2×2 panel composite
│   ├── make_fss_composite.py        2×2 FSS composite
│   ├── plot_universe.py             3-figure spectral/mass/flux plotter
│   └── synthesis_analyzer.py        4-figure Physical Review analyzer
└── figures/                         Pre-generated publication figures
    ├── fss_q_topo.pdf               Q_topo scaling (R²=0.9996)
    ├── fss_inv_alpha.pdf            1/α scaling
    ├── fss_omega_energy.pdf         Ω_energy convergence
    ├── fss_mass_hierarchy.pdf       Mass hierarchy (N-independent)
    ├── fss_spectral_dimension.pdf   d_S UV and IR flow
    ├── fss_generation_fractions.pdf Generation populations
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
python data/scripts/make_composite_figure.py  # 4-panel composite
python data/scripts/make_fss_composite.py     # 4-panel FSS composite
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
| `alpha_em` | α = Q_topo/(8π) |

### `mass_spectrum.csv` / `mass_spectrum_M20.csv`

Belly-size histogram of all Causal Prisms.

| Column | Description |
|--------|-------------|
| `intermediates_N` | Belly size n (number of intermediates, 3 to n_max) |
| `frequency` | Number of prisms with this belly size |

### `traversal_mass.csv` (when `--measure-mass` is enabled)

Cover-time mass ratios per generation.

| Column | Description |
|--------|-------------|
| `generation` | Generation number (1, 2, or 3) |
| `mean_traversal` | Mean cover time (ticks to visit all belly nodes + reach destination) |
| `n_traversals` | Number of completed traversals (statistical sample) |
| `ratio_to_gen1` | Cover-time ratio relative to Generation 1 |

### `half_life.csv` (when `--measure-halflife` is enabled)

Cross-ensemble generation occupancy by belly size.

| Column | Description |
|--------|-------------|
| `belly_size` | Number of intermediate nodes N |
| `p_gen1` | Fraction of prisms with belly N classified as Gen1 |
| `p_gen2` | Fraction classified as Gen2 |
| `p_gen3` | Fraction classified as Gen3 |

Header comments include `stability_ratio_gen2`, `stability_ratio_gen3`, and `gen_counts`.

### `modulo_interference.csv` (when `--measure-modulo` is enabled)

Per-node NTT phase accumulation.

| Column | Description |
|--------|-------------|
| `node_id` | Node index |
| `n_arrivals` | Number of walker arrivals at this node |
| `phase_sum` | Accumulated modular phase (g^S mod p) |
| `intensity` | Normalized squared centered phase |
| `qt`, `qx`, `qy`, `qz` | 4D coordinates of the node |

Header comments include prime, root, total walkers, mean/max intensity, constructive/destructive counts.

### `vacuum_polarization.csv` (when `--measure-vacuum` is enabled)

K₃,₃ screening at Gen1 prism free ports.

| Column | Description |
|--------|-------------|
| `prism_idx` | Prism index |
| `generation` | Generation number (always 1 for this measurement) |
| `n_attempted` | Number of candidate nodes tested |
| `n_rejected` | Number rejected as K₃,₃ threats |
| `n_accepted` | Number accepted (no K₃,₃ threat) |
| `local_screening` | n_accepted / n_attempted |

Header comments include total counts, mean screening, bare and screened alpha.

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
