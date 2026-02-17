//! Phase 3 — Random Walk & Spectral Dimension
//!
//! Two tiers:
//!   - Eigendecomposition   (N ≤ 3k, exact, O(N³))
//!   - Monte Carlo walkers  (N > 3k, O(W·t), **independent of N**)

use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Spectral dimension measurements for one graph (vacuum or defect).
///
/// Contains return probabilities P(t) and their derived spectral dimensions
/// d_S(t) for global, local (core), per-generation, and causal flux observables.
pub struct SpectralResult {
    /// P(t) measured from uniformly random starting positions.
    /// Probes the global geometry of the graph (vacuum or defect).
    pub p_global: Vec<f64>,
    /// d_S(t) = −2 d(ln P_global)/d(ln t). Spectral dimension of the bulk.
    pub ds_global: Vec<f64>,
    /// P(t) measured from walkers starting exclusively on core nodes.
    /// Probes the local geometry near the combinatorial centre of the diamond.
    pub p_local: Vec<f64>,
    /// d_S(t) derived from P_local. Sensitive to topological defects.
    pub ds_local: Vec<f64>,

    // ── Modulo Synthesis Vol II: Particle Generations ──
    //
    // Causal Prisms are classified by their topological signature [M₀,M₁,M₂,M₃]
    // (normalised bulk momentum of each vertex, sorted by causal order).
    // The most frequent signature defines Generation 1 (electron-like),
    // the second most frequent Generation 2 (muon-like), etc.
    // Each Prism has topological mass N = number of intermediate nodes.

    /// P(t) for walkers starting on Generation 1 Causal Prism nodes (most abundant signature).
    pub p_gen1: Vec<f64>,
    /// d_S(t) for Generation 1.
    pub ds_gen1: Vec<f64>,
    /// P(t) for Generation 2 Causal Prism nodes (second most abundant signature).
    pub p_gen2: Vec<f64>,
    /// d_S(t) for Generation 2.
    pub ds_gen2: Vec<f64>,
    /// P(t) for Generation 3 Causal Prism nodes (third most abundant signature).
    pub p_gen3: Vec<f64>,
    /// d_S(t) for Generation 3.
    pub ds_gen3: Vec<f64>,
    /// P(t) for Anti-Generation 1: the CPT-conjugate signature (reversed + negated).
    pub p_anti1: Vec<f64>,
    /// d_S(t) for Anti-Generation 1 (antimatter).
    pub ds_anti1: Vec<f64>,

    // ── Causal Flux: Electromagnetism ──
    //
    // Directed walkers following the arrow of time (cause → effect).
    // Transmission probability from Gen1 sources to AntiGen1 targets (attraction)
    // and from Gen1 sources to Gen1 targets (repulsion).

    /// Directed transmission probability: Gen1 → AntiGen1 (opposite charge attraction).
    pub flux_attraction: Vec<f64>,
    /// Directed transmission probability: Gen1 → Gen1 (same charge repulsion).
    pub flux_repulsion: Vec<f64>,

    // ── Normalized Flux (per-node coupling strength, Conjecture C3) ──
    //
    // Raw flux is dominated by combinatorial factor |Gen1| >> |AntiGen1|.
    // Per-node normalization isolates the intrinsic coupling strength.

    /// Normalized attraction flux: flux_attraction / |targets_attraction|.
    pub flux_attr_norm: Vec<f64>,
    /// Normalized repulsion flux: flux_repulsion / |targets_repulsion|.
    pub flux_repu_norm: Vec<f64>,

    // ── Sterile Prism Spectral Data (Conjecture C6) ──
    //
    // Prisms with N > 5 intermediates are "sterile" — gravitationally active
    // but electromagnetically silent (dark matter candidates).

    /// P(t) for walkers starting on sterile prism nodes (N > 5).
    pub p_sterile: Vec<f64>,
    /// d_S(t) for sterile prism nodes.
    pub ds_sterile: Vec<f64>,

    // ── Mass Spectrum (Topological Inertia) ──
    //
    // Mass = N (number of intermediate nodes in the Causal Prism).
    // Static topological property — does not depend on diffusion time t.

    /// Average mass (N) for Generation 1 Prisms.
    pub mass_gen1: f64,
    /// Average mass (N) for Generation 2 Prisms.
    pub mass_gen2: f64,
    /// Average mass (N) for Generation 3 Prisms.
    pub mass_gen3: f64,
    /// Average mass (N) for Anti-Generation 1 Prisms.
    pub mass_anti1: f64,

    // ── Ensemble Error Bars ──
    //
    // Standard deviation across M realisations. Empty (vec![]) when M=1.

    /// Std dev of d_S vacuum global across realisations.
    pub ds_global_std: Vec<f64>,
    /// Std dev of d_S vacuum/defect local across realisations.
    pub ds_local_std: Vec<f64>,
    /// Std dev of d_S Generation 1 across realisations.
    pub ds_gen1_std: Vec<f64>,
    /// Std dev of d_S Generation 2 across realisations.
    pub ds_gen2_std: Vec<f64>,
    /// Std dev of d_S Generation 3 across realisations.
    pub ds_gen3_std: Vec<f64>,
    /// Std dev of d_S Anti-Generation 1 across realisations.
    pub ds_anti1_std: Vec<f64>,
    /// Std dev of d_S sterile across realisations.
    pub ds_sterile_std: Vec<f64>,
    /// Std dev of flux attraction across realisations.
    pub flux_attraction_std: Vec<f64>,
    /// Std dev of flux repulsion across realisations.
    pub flux_repulsion_std: Vec<f64>,
}

// ─── shared ──────────────────────────────────────────────────────────────────

/// d_S(t) = −2 d(ln P)/d(ln t)  via centred finite differences.
pub fn spectral_dimension(steps: &[u32], p_vals: &[f64]) -> Vec<f64> {
    let n = steps.len();
    if n < 2 { return vec![0.0; n]; }
    let ln_t: Vec<f64> = steps.iter().map(|&t| (t as f64).ln()).collect();
    let ln_p: Vec<f64> = p_vals.iter().map(|&p| p.max(1e-30).ln()).collect();
    let mut ds = vec![0.0; n];
    for i in 0..n {
        let d = if i == 0 {
            (ln_p[1] - ln_p[0]) / (ln_t[1] - ln_t[0])
        } else if i == n - 1 {
            (ln_p[n - 1] - ln_p[n - 2]) / (ln_t[n - 1] - ln_t[n - 2])
        } else {
            (ln_p[i + 1] - ln_p[i - 1]) / (ln_t[i + 1] - ln_t[i - 1])
        };
        ds[i] = -2.0 * d;
    }
    ds
}

/// Build sorted undirected adjacency list in CSR format.
pub fn build_adj_list(n: usize, rows: &[u32], cols: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut counts = vec![0u32; n];
    for (&r, &c) in rows.iter().zip(cols.iter()) {
        counts[r as usize] += 1;
        counts[c as usize] += 1;
    }

    let mut adj_head = vec![0u32; n + 1];
    for i in 0..n {
        adj_head[i+1] = adj_head[i] + counts[i];
    }

    let mut adj_data = vec![0u32; adj_head[n] as usize];
    let mut current_pos = adj_head.clone();

    for (&r, &c) in rows.iter().zip(cols.iter()) {
        adj_data[current_pos[r as usize] as usize] = c;
        current_pos[r as usize] += 1;
        adj_data[current_pos[c as usize] as usize] = r;
        current_pos[c as usize] += 1;
    }

    // Sort neighbors within CSR
    for i in 0..n {
        let start = adj_head[i] as usize;
        let end = adj_head[i+1] as usize;
        adj_data[start..end].sort_unstable();
    }

    (adj_head, adj_data)
}

// ═════════════════════════════════════════════════════════════════════════════
// Monte Carlo walker engine  (the integer heart)
// ═════════════════════════════════════════════════════════════════════════════

/// Distribute exactly `n_walkers` across `origins`, guaranteeing zero truncation.
///
/// **Strict Finitism**: uses integer `base = W / |origins|` with the remainder
/// `r = W mod |origins|` assigned round-robin to the first `r` nodes.  The
/// returned vector has length exactly `n_walkers`, satisfying the Causal
/// Resolution Theorem's exact-W requirement.
pub fn distribute_walkers(origins: &[usize], n_walkers: usize) -> Vec<usize> {
    if origins.is_empty() { return vec![]; }
    let k = origins.len();
    let base = n_walkers / k;
    let remainder = n_walkers % k;
    let mut starts = Vec::with_capacity(n_walkers);
    for (i, &node) in origins.iter().enumerate() {
        let count = base + if i < remainder { 1 } else { 0 };
        for _ in 0..count {
            starts.push(node);
        }
    }
    debug_assert_eq!(starts.len(), n_walkers, "distribute_walkers: exact W invariant");
    starts
}

/// Resolve a node through the Kuratowski contraction map.
///
/// After K₅ threat absorption, some nodes are logically merged but the CSR
/// graph retains the original vertex numbering.  This function chases the
/// `merge_into` chain (which may have depth > 1) until a fixed point is
/// reached, ensuring walkers never land on ghost (degree-0) nodes.
#[inline]
fn resolve(node: usize, merge: Option<&[usize]>) -> usize {
    match merge {
        None => node,
        Some(m) => {
            let mut cur = node;
            while m[cur] != cur { cur = m[cur]; }
            cur
        }
    }
}

/// Run W independent lazy-walk walkers in parallel.
///
/// **Strict Finitism**: all internal accumulation uses `u64` integer
/// arithmetic.  Each walker contributes exactly 0 or 1 (Kronecker delta)
/// per measurement step.  The reduction sums `u64` counts across all
/// walkers.  The single `f64` division `count / W` occurs only at the
/// final return, preserving exact integer semantics throughout the
/// Monte Carlo core.
///
/// **Ghost resolution**: when `merge_into` is `Some`, every position
/// (origin and each step) is resolved through the contraction map,
/// preventing walkers from landing on absorbed (degree-0) ghost nodes.
pub fn run_walkers(
    adj_head: &[u32],
    adj_data: &[u32],
    starts: &[usize],
    steps: &[u32],
    base_seed: u64,
    merge_into: Option<&[usize]>,
) -> Vec<f64> {
    let n_w = starts.len();
    if n_w == 0 { return vec![0.0; steps.len()]; }
    let n_s = steps.len();
    let max_t = *steps.last().unwrap_or(&0);

    let counts: Vec<u64> = starts
        .par_iter()
        .enumerate()
        .map(|(wi, &origin)| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(wi as u64));
            let origin = resolve(origin, merge_into);
            let mut pos = origin;
            let mut c = vec![0u64; n_s];
            let mut si = 0usize;

            for t in 1..=max_t {
                let start = adj_head[pos] as usize;
                let end = adj_head[pos+1] as usize;
                let len = end - start;

                if len > 0 && rng.gen_bool(0.5) {
                    let next = adj_data[start + rng.gen_range(0..len)] as usize;
                    pos = resolve(next, merge_into);
                }
                if si < n_s && t == steps[si] {
                    if pos == origin {
                        c[si] = 1;
                    }
                    si += 1;
                }
            }
            c
        })
        .reduce(
            || vec![0u64; n_s],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );

    counts.iter().map(|&c| c as f64 / n_w as f64).collect()
}

/// Run directed walkers to measure transmission between signatures.
/// Only moves past -> future in a DAG.
///
/// **Strict Finitism**: same `u64` integer accumulation as [`run_walkers`].
/// Each walker contributes 0 or 1 per step for attraction/repulsion hits.
/// The single `f64` division occurs only at the final return.
///
/// **Ghost resolution**: when `merge_into` is `Some`, every position is
/// resolved through the contraction map before target lookup.
pub fn run_transmission_walkers(
    adj_head: &[u32],
    adj_data: &[u32],
    starts: &[usize],
    targets_attraction: &[usize],
    targets_repulsion: &[usize],
    steps: &[u32],
    base_seed: u64,
    merge_into: Option<&[usize]>,
) -> (Vec<f64>, Vec<f64>) {
    let n_w = starts.len();
    if n_w == 0 { return (vec![0.0; steps.len()], vec![0.0; steps.len()]); }
    let n_s = steps.len();
    let max_t = *steps.last().unwrap_or(&0);

    // Sort targets for fast binary search
    let mut attr = targets_attraction.to_vec(); attr.sort_unstable();
    let mut repu = targets_repulsion.to_vec(); repu.sort_unstable();

    let (attr_counts, repu_counts): (Vec<u64>, Vec<u64>) = starts
        .par_iter()
        .enumerate()
        .map(|(wi, &origin)| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(wi as u64 + 1000));
            let mut pos = resolve(origin, merge_into);
            let mut ca = vec![0u64; n_s];
            let mut cr = vec![0u64; n_s];
            let mut si = 0usize;

            for t in 1..=max_t {
                let start = adj_head[pos] as usize;
                let end = adj_head[pos+1] as usize;
                let len = end - start;

                if len == 0 {
                    // Walker escapes (ends of causal set)
                    break;
                }
                // Mandatory move in directed graph (causal flux)
                let next = adj_data[start + rng.gen_range(0..len)] as usize;
                pos = resolve(next, merge_into);

                if si < n_s && t == steps[si] {
                    if attr.binary_search(&pos).is_ok() { ca[si] = 1; }
                    if repu.binary_search(&pos).is_ok() { cr[si] = 1; }
                    si += 1;
                }
            }
            (ca, cr)
        })
        .reduce(
            || (vec![0u64; n_s], vec![0u64; n_s]),
            |mut a, b| {
                for i in 0..n_s { a.0[i] += b.0[i]; a.1[i] += b.1[i]; }
                a
            },
        );

    (
        attr_counts.iter().map(|&c| c as f64 / n_w as f64).collect(),
        repu_counts.iter().map(|&c| c as f64 / n_w as f64).collect()
    )
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 1 — Eigendecomposition  (N ≤ 3k, exact)
// ═════════════════════════════════════════════════════════════════════════════

/// Exact spectral dimension via eigendecomposition of the symmetric
/// normalised adjacency S = D^{-1/2} A D^{-1/2}.
///
/// Complexity: O(N³). Used only for small graphs (N ≤ 3k).
///
/// Global P(t) = N⁻¹ Σ λₖᵗ. Local P(t) uses eigenvector-weighted sums
/// over core indices, probing the geometry near the diamond's centre.
pub fn compute_eigen(
    n: usize,
    edge_rows: &[u32],
    edge_cols: &[u32],
    steps: &[u32],
    core_indices: &[usize],
) -> SpectralResult {
    let mut degree = vec![0.0_f64; n];
    for (&r, &c) in edge_rows.iter().zip(edge_cols.iter()) {
        degree[r as usize] += 1.0;
        degree[c as usize] += 1.0;
    }
    let inv_sqrt_d: Vec<f64> = degree
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let mut s = DMatrix::<f64>::zeros(n, n);
    for (&r, &c) in edge_rows.iter().zip(edge_cols.iter()) {
        let (ri, ci) = (r as usize, c as usize);
        let val = inv_sqrt_d[ri] * inv_sqrt_d[ci];
        s[(ri, ci)] += val;
        s[(ci, ri)] += val;
    }

    println!("  Eigendecomposition ({n}×{n}) …");
    let eigen = SymmetricEigen::new(s);
    let eigenvalues: Vec<f64> = eigen
        .eigenvalues
        .iter()
        .map(|&v| v.clamp(-1.0, 1.0))
        .collect();

    let p_global: Vec<f64> = steps
        .iter()
        .map(|&t| eigenvalues.iter().map(|&l| l.powi(t as i32)).sum::<f64>() / n as f64)
        .collect();
    let ds_global = spectral_dimension(steps, &p_global);

    let (p_local, ds_local) = if core_indices.is_empty() {
        (p_global.clone(), ds_global.clone())
    } else {
        let nc = core_indices.len();
        let w: Vec<f64> = (0..n)
            .map(|k| core_indices.iter().map(|&i| eigen.eigenvectors[(i, k)].powi(2)).sum())
            .collect();
        let p_loc: Vec<f64> = steps
            .iter()
            .map(|&t| {
                eigenvalues
                    .iter()
                    .zip(w.iter())
                    .map(|(&l, &wk)| wk * l.powi(t as i32))
                    .sum::<f64>()
                    / nc as f64
            })
            .collect();
        let ds_loc = spectral_dimension(steps, &p_loc);
        (p_loc, ds_loc)
    };

    SpectralResult {
        p_global, ds_global, p_local, ds_local,
        p_gen1: vec![], ds_gen1: vec![],
        p_gen2: vec![], ds_gen2: vec![],
        p_gen3: vec![], ds_gen3: vec![],
        p_anti1: vec![], ds_anti1: vec![],
        flux_attraction: vec![], flux_repulsion: vec![],
        flux_attr_norm: vec![], flux_repu_norm: vec![],
        p_sterile: vec![], ds_sterile: vec![],
        mass_gen1: 0.0, mass_gen2: 0.0, mass_gen3: 0.0, mass_anti1: 0.0,
        ds_global_std: vec![], ds_local_std: vec![],
        ds_gen1_std: vec![], ds_gen2_std: vec![], ds_gen3_std: vec![],
        ds_anti1_std: vec![], ds_sterile_std: vec![],
        flux_attraction_std: vec![], flux_repulsion_std: vec![],
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 2 — Monte Carlo walkers  (N > 3k, O(W·t))
// ═════════════════════════════════════════════════════════════════════════════

/// Monte Carlo spectral dimension from edge lists.
///
/// Builds CSR adjacency internally, then launches lazy random walkers.
/// Complexity: O(W·t) where W = number of walkers, independent of N.
/// Used for large graphs (N > 3k) where eigendecomposition is infeasible.
pub fn compute_monte_carlo(
    n: usize,
    edge_rows: &[u32],
    edge_cols: &[u32],
    steps: &[u32],
    core_indices: &[usize],
    n_walkers: usize,
    rng: &mut impl Rng,
) -> SpectralResult {
    let (adj_head, adj_data) = build_adj_list(n, edge_rows, edge_cols);

    // ── Global ───────────────────────────────────────────────────────
    let global_starts: Vec<usize> = (0..n_walkers).map(|_| rng.gen_range(0..n)).collect();
    let seed_g: u64 = rng.gen();
    println!("[Phase 3] MC walkers (global, W={n_walkers}) …");
    let p_global = run_walkers(&adj_head, &adj_data, &global_starts, steps, seed_g, None);
    let ds_global = spectral_dimension(steps, &p_global);

    // ── Local ────────────────────────────────────────────────────────
    let (p_local, ds_local) = if core_indices.is_empty() {
        (p_global.clone(), ds_global.clone())
    } else {
        let local_starts = distribute_walkers(core_indices, n_walkers);
        let seed_l: u64 = rng.gen();
        println!(
            "[Phase 3] MC walkers (core, W={}, |core|={}) …",
            n_walkers,
            core_indices.len()
        );
        let p_loc = run_walkers(&adj_head, &adj_data, &local_starts, steps, seed_l, None);
        let ds_loc = spectral_dimension(steps, &p_loc);
        (p_loc, ds_loc)
    };

    SpectralResult {
        p_global, ds_global, p_local, ds_local,
        p_gen1: vec![], ds_gen1: vec![],
        p_gen2: vec![], ds_gen2: vec![],
        p_gen3: vec![], ds_gen3: vec![],
        p_anti1: vec![], ds_anti1: vec![],
        flux_attraction: vec![], flux_repulsion: vec![],
        flux_attr_norm: vec![], flux_repu_norm: vec![],
        p_sterile: vec![], ds_sterile: vec![],
        mass_gen1: 0.0, mass_gen2: 0.0, mass_gen3: 0.0, mass_anti1: 0.0,
        ds_global_std: vec![], ds_local_std: vec![],
        ds_gen1_std: vec![], ds_gen2_std: vec![], ds_gen3_std: vec![],
        ds_anti1_std: vec![], ds_sterile_std: vec![],
        flux_attraction_std: vec![], flux_repulsion_std: vec![],
    }
}

/// Monte Carlo spectral dimension from pre-built CSR adjacency.
///
/// Zero-copy variant — avoids rebuilding adjacency when CSR is already
/// available from Phase 2. This is the hot path for large N in-memory runs.
pub fn compute_monte_carlo_csr(
    n: usize,
    adj_head: &[u32],
    adj_data: &[u32],
    steps: &[u32],
    core_indices: &[usize],
    n_walkers: usize,
    rng: &mut impl Rng,
) -> SpectralResult {
    // ── Global ───────────────────────────────────────────────────────
    let global_starts: Vec<usize> = (0..n_walkers).map(|_| rng.gen_range(0..n)).collect();
    let seed_g: u64 = rng.gen();
    // println!("[Phase 3] MC walkers (global, W={n_walkers}) …"); // Reduced verbosity
    let p_global = run_walkers(adj_head, adj_data, &global_starts, steps, seed_g, None);
    let ds_global = spectral_dimension(steps, &p_global);

    // ── Local ────────────────────────────────────────────────────────
    let (p_local, ds_local) = if core_indices.is_empty() {
        (p_global.clone(), ds_global.clone())
    } else {
        let local_starts = distribute_walkers(core_indices, n_walkers);
        let seed_l: u64 = rng.gen();
        // println!("[Phase 3] MC walkers (core, W={}, |core|={}) …", n_walkers, core_indices.len());
        let p_loc = run_walkers(adj_head, adj_data, &local_starts, steps, seed_l, None);
        let ds_loc = spectral_dimension(steps, &p_loc);
        (p_loc, ds_loc)
    };

    SpectralResult {
        p_global, ds_global, p_local, ds_local,
        p_gen1: vec![], ds_gen1: vec![],
        p_gen2: vec![], ds_gen2: vec![],
        p_gen3: vec![], ds_gen3: vec![],
        p_anti1: vec![], ds_anti1: vec![],
        flux_attraction: vec![], flux_repulsion: vec![],
        flux_attr_norm: vec![], flux_repu_norm: vec![],
        p_sterile: vec![], ds_sterile: vec![],
        mass_gen1: 0.0, mass_gen2: 0.0, mass_gen3: 0.0, mass_anti1: 0.0,
        ds_global_std: vec![], ds_local_std: vec![],
        ds_gen1_std: vec![], ds_gen2_std: vec![], ds_gen3_std: vec![],
        ds_anti1_std: vec![], ds_sterile_std: vec![],
        flux_attraction_std: vec![], flux_repulsion_std: vec![],
    }
}
