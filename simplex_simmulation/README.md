# Simplex Simulation: The Kuratowski Calculus Engine

This folder contains the core computational implementation of Fractal Entropic Geometrodynamics (FEG). Unlike traditional physics simulations based on floating-point approximations of continuous manifolds, this engine operates on pure integer logic and discrete combinatorial topology.

## Mission

The goal of this simulation is to demonstrate the emergence of baryonic matter as a topological necessity. By applying Kuratowski's Theorem to a Poisson-sprinkled Causal Set, we show that the $K_4$ simplex (the Skyrmion knot) acts as a structural anchor that prevents planar collapse and defines the spectral dimension of matter ($d_S \approx 8.36$).

## Project Structure

The simulation executes in four sequential phases, each implemented in a dedicated module:

### `main.rs` --- Orchestration and Ensemble Averaging

Entry point. Parses CLI arguments (`N` events, `M` realisations), dispatches each realisation through Phases 1--3, then performs ensemble averaging of the return probability $P(t)$ across all Monte Carlo realisations before recomputing the spectral dimension $d_S(t)$ from the mean. Automatically selects the computational tier: exact eigendecomposition for $N \leq 3{,}000$, sparse Monte Carlo walkers for $N > 3{,}000$. Sequential outer loop for large $N$ to prevent OOM; inner parallelism via `rayon` saturates all cores.

### `diamond.rs` --- Phase 1: Vacuum Generation (Poisson Sprinkling)

Implements Axiom 1 of the theory. Generates $N$ spacetime events via rejection sampling inside a 4D causal diamond ($|t| + r \leq T/2$) with volume $V = T^4/24$ and fundamental density $\rho \approx 1$. Constructs the Hasse diagram (transitive reduction of the causal partial order) using two tiers:

- **Tier 1** ($N \leq 15{,}000$): Builds the full causal closure as a sparse adjacency matrix $A$, computes $A^2$, and retains only edges where $A^2[i,j] = 0$ (no two-hop path exists). Exact.
- **Tier 2** ($N > 15{,}000$): Direct geometric construction. For each source node, iterates over future nodes in time order and rejects any edge $(i,j)$ for which an existing link-child $z$ of $i$ satisfies $z \prec j$. Avoids materialising the full closure.

### `skyrmion.rs` --- Phase 2: Kuratowski Contraction (Pure Integer Topology)

The heart of the Kuratowski Calculus. This module contains zero floating-point arithmetic. All operations --- core identification, $K_5$ threat detection, vertex contraction, $K_4$ completion --- are executed using integer degree ranking, boolean flags, and sorted index arrays.

The algorithm proceeds in seven steps:

1. **Undirected adjacency** from the directed Hasse edges (sorted, deduplicated).
2. **Core selection**: top 10% of nodes by undirected degree (the combinatorial centre of the diamond).
3. **Core-density ranking**: nodes sorted by number of core-to-core connections.
4. **Greedy $K_4$ formation**: starting from the highest-density seed, selects the three neighbours with maximal mutual connectivity to form a complete $K_4$ clique. Then scans remaining neighbours for **$K_5$ threats** --- any node connected to $\geq 3$ members of the $K_4$ clique. Threatening nodes are absorbed (vertex contraction) into the highest-degree clique member, preventing the formation of the forbidden $K_5$ subgraph.
5. **Transitive merge resolution**: chases pointer chains to resolve multi-step contractions.
6. **Edge list reconstruction**: applies merges to original edges, deduplicates via integer sort, then inserts completion edges to ensure each $K_4$ clique is fully connected.
7. **Core index vectors**: produces `vacuum_core` (all core indices, for control measurement on the unmodified graph) and `defect_core` (core minus merged nodes).

### `spectral.rs` --- Phase 3: Spectral Dimension via Random Walk Return Probability

Computes the spectral dimension $d_S(t) = -2 \, d(\ln P) / d(\ln t)$ where $P(t)$ is the return probability of a random walker after $t$ steps. Two computational tiers:

- **Eigendecomposition** ($N \leq 3{,}000$): constructs the symmetric normalised adjacency matrix $S = D^{-1/2} A D^{-1/2}$, computes all eigenvalues $\lambda_k$, and evaluates $P(t) = N^{-1} \sum_k \lambda_k^t$ exactly. Local $P(t)$ over core indices uses eigenvector-weighted sums.
- **Monte Carlo walkers** ($N > 3{,}000$): launches $W = 5{,}000$ independent lazy random walkers (stay with probability 0.5, move to uniform random neighbour with probability 0.5). Records returns at measurement steps $t = 2, 4, \ldots, 100$. Complexity is $O(W \cdot t_{\max})$, **independent of $N$**. This is the integer heart of Phase 3 --- adjacency traversal is pure index lookup; floating-point enters only at the final macroscopic averaging step.

Both tiers produce global $P(t)$ (walkers from uniformly random starts) and local $P(t)$ (walkers starting exclusively from core indices), enabling direct comparison of vacuum geometry versus defect topology.

### `output.rs` --- Phase 4: Data Serialisation

Writes ensemble-averaged results to `results.csv` with columns: step, global/local return probabilities, and spectral dimensions for both vacuum and defect graphs.

## Quick Start

Ensure the Rust toolchain is installed.

Compile for maximum performance:

```bash
cargo build --release
```

Execute a standard ensemble:

```bash
cargo run --release -- 50000 10
```

- First argument: number of events in the Causal Diamond (standard: 50,000).
- Second argument: number of independent Monte Carlo realisations (standard: 10).

## Expected Results

Upon completion, the simulation generates `results.csv` and prints a summary to stdout.

- **Local Vacuum**: the spectral dimension $d_S$ diverges toward $\approx 15.79$ at late diffusion times, reflecting boundary saturation of the unstructured bulk.
- **Baryonic Anchor**: the introduction of $K_4$ topological defects stabilises the local spectral dimension at $d_S \approx 8.36$, demonstrating that matter acts as a topological retention mechanism that traps the probability flow.
- **Coupling Constant**: the energy ratio between these two regimes yields the transitional coupling constant $\alpha_{\text{trans}} \approx 0.31$, consistent with the observed strong coupling at the tau mass scale.

## Dependencies

| Crate | Purpose |
|---|---|
| `nalgebra` | Dense eigendecomposition (Tier 1) |
| `sprs` | Sparse matrix algebra for Hasse construction |
| `rand` | Deterministic PRNG (seeded `StdRng`) |
| `rayon` | Data-parallel iteration across walkers and nodes |

## Troubleshooting

### Linux (Arch/Ubuntu/Debian)

If compilation fails due to missing system headers, ensure you have the essential build tools.

On Arch:

```bash
sudo pacman -S base-devel
```

On Ubuntu/Debian:

```bash
sudo apt update && sudo apt install build-essential
```

### macOS

You must have the Xcode Command Line Tools installed. If you see an error regarding `cc` or the linker, run:

```bash
xcode-select --install
```

Note: The simulation is optimised for Apple Silicon (M1/M2/M3) and will utilise the efficient cores for background Monte Carlo threads.

### Windows

Install Visual Studio Build Tools 2022 and ensure the "Desktop development with C++" workload is checked.

If you encounter path issues with `cargo`, ensure you are using the **x64 Native Tools Command Prompt for VS 2022**.

Performance note: Windows Defender may slow down file I/O during `.csv` generation. Adding an exclusion for the project folder is recommended.

## License and Ethics

This code is open-source. It was designed to run on consumer hardware (tested on a ThinkPad T480) to demonstrate that high-level theoretical physics should be accessible to anyone with a terminal.

> "The universe is strictly computable, finite, and structurally homeostatic. The rest is just counting."

**Author:** Juan Pablo Silva Alvarado
**Part of:** Modulo Synthesis / Fractal Entropic Geometrodynamics
