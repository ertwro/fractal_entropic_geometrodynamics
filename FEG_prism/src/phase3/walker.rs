// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Random walk engine — eigendecomposition (Tier 1) and Monte Carlo (Tier 2).
//!
//! Two tiers:
//!   - Eigendecomposition   (N <= 3k, exact, O(N^3))
//!   - Monte Carlo walkers  (N > 3k, O(W*t), **independent of N**)
//!
//! All Monte Carlo accumulation uses `u64` integer arithmetic (Strict Finitism).
//! The single `f64` division `count / W` occurs only at the final return.
//!
//! # Spectral structure of the lazy walk
//!
//! The lazy walk (50% stay, 50% random neighbor) has transition matrix:
//!
//! ```text
//!   T = ½(I + D⁻¹A)
//! ```
//!
//! where A is the adjacency matrix and D the degree matrix.  Eigenvalues
//! of T lie in \[0, 1\] with λ₀ = 1 for the stationary mode.  The return
//! probability decomposes as:
//!
//! ```text
//!   P(t) = (1/N) Tr(T^t) = (1/N) Σ_k  λ_k^t
//! ```
//!
//! Three eigenvalue bands contribute at different timescales:
//!
//! | Band | Eigenvalues | Decay timescale | Information |
//! |------|-------------|-----------------|-------------|
//! | Continuum | λ ≈ 1 | O(N^{2/d_s}) | P ~ t^{−d_s/2} (signal) |
//! | Lattice | λ ≈ 1 − O(1/D_max) | O(D_max) = O(15) | Degree echoes, parity |
//! | High-freq | λ ≈ 0 | O(1) | Irrelevant for t ≥ 2 |
//!
//! The d_S(t) plateau emerges at t ≈ D_max = 15 when lattice modes have
//! decayed by ~1/e.  The full measurement schedule (t = 1..100) captures
//! the short-time transient, the scaling plateau, and the approach to
//! finite-size saturation P(t) → 1/N.

use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::spectral::{spectral_dimension, WalkResult};

// ─── Adjacency builder ──────────────────────────────────────────────────────

/// Build sorted undirected adjacency list in CSR format from edge lists.
///
/// Each undirected edge (r, c) is stored in both directions.
/// Neighbor lists within each row are sorted ascending for binary search.
fn build_adj_list(n: usize, rows: &[u32], cols: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut counts = vec![0u32; n];
    for (&r, &c) in rows.iter().zip(cols.iter()) {
        counts[r as usize] += 1;
        counts[c as usize] += 1;
    }

    let mut adj_head = vec![0u32; n + 1];
    for i in 0..n {
        adj_head[i + 1] = adj_head[i] + counts[i];
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
        let end = adj_head[i + 1] as usize;
        adj_data[start..end].sort_unstable();
    }

    (adj_head, adj_data)
}

// ─── Walker distribution ─────────────────────────────────────────────────────

/// Distribute exactly `n_walkers` across `origins`, guaranteeing zero truncation.
///
/// **Strict Finitism**: uses integer `base = W / |origins|` with the remainder
/// `r = W mod |origins|` assigned round-robin to the first `r` nodes.  The
/// returned vector has length exactly `n_walkers`, satisfying the Causal
/// Resolution Theorem's exact-W requirement.
pub fn distribute_walkers(origins: &[usize], n_walkers: usize) -> Vec<usize> {
    if origins.is_empty() {
        return vec![];
    }
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

/// Distribute exactly `n_walkers` across `origins` with seeded shuffle.
///
/// Identical to [`distribute_walkers`] except the origin order is shuffled
/// with a deterministic PRNG before the base+remainder round-robin.  This
/// eliminates cell-sort bias: `build_hasse_direct` (N > 3k) produces nodes
/// sorted by (qt, qx, qy, qz), so cycling in index order preferentially
/// samples early time slices when W < N.  Shuffling ensures uniform temporal
/// coverage regardless of index ordering.
///
/// **Deterministic**: seeded RNG, exact W invariant, reproducible.
pub fn distribute_walkers_shuffled(
    origins: &[usize], n_walkers: usize, seed: u64,
) -> Vec<usize> {
    if origins.is_empty() {
        return vec![];
    }
    use rand::seq::SliceRandom;
    let mut shuffled = origins.to_vec();
    let mut rng = StdRng::seed_from_u64(seed);
    shuffled.shuffle(&mut rng);

    let k = shuffled.len();
    let base = n_walkers / k;
    let remainder = n_walkers % k;
    let mut starts = Vec::with_capacity(n_walkers);
    for (i, &node) in shuffled.iter().enumerate() {
        let count = base + if i < remainder { 1 } else { 0 };
        for _ in 0..count {
            starts.push(node);
        }
    }
    debug_assert_eq!(starts.len(), n_walkers, "distribute_walkers_shuffled: exact W invariant");
    starts
}

/// Count how many walkers started in each category.
///
/// Given the `starts` array (walker origins) and a `category_map` (u8 bitmask
/// per node), returns a `Vec<usize>` of length `n_categories` where entry `c`
/// is the number of walkers whose resolved origin has bit `c` set.
///
/// Needed for the final `count / W_cat` division in the unified walk.
pub fn count_category_walkers(
    starts: &[usize], category_map: &[u8], n_categories: usize,
) -> Vec<usize> {
    let mut counts = vec![0usize; n_categories];
    for &origin in starts {
        let mask = category_map[origin];
        for c in 0..n_categories {
            if mask & (1u8 << c) != 0 {
                counts[c] += 1;
            }
        }
    }
    counts
}

// ─── Ghost resolution ────────────────────────────────────────────────────────

/// Resolve a node through the Kuratowski contraction map.
///
/// After K_5 threat absorption, some nodes are logically merged but the CSR
/// graph retains the original vertex numbering.  This function chases the
/// `merge_into` chain (which may have depth > 1) until a fixed point is
/// reached, ensuring walkers never land on ghost (degree-0) nodes.
#[inline]
fn resolve(node: usize, merge: Option<&[usize]>) -> usize {
    match merge {
        None => node,
        Some(m) => {
            let mut cur = node;
            while m[cur] != cur {
                cur = m[cur];
            }
            cur
        }
    }
}

// ─── Monte Carlo lazy-walk engine ────────────────────────────────────────────

/// Run W independent lazy-walk walkers in parallel.
///
/// Implements the transition matrix T = ½(I + D⁻¹A) stochastically:
/// at each step the walker stays with probability ½ or hops to a
/// uniformly random neighbor with probability ½.  The return indicator
/// X_i ∈ {0,1} at each measurement step is a Bernoulli(P(t)) variable;
/// the CLT gives RSE = 1/√(W · P(t)), which drives the CRT budget.
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
    if n_w == 0 {
        return vec![0.0; steps.len()];
    }
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
                let end = adj_head[pos + 1] as usize;
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

/// Run W independent lazy-walk walkers with unified per-category binning.
///
/// Identical walk operator to [`run_walkers`] (T = ½(I + D⁻¹A)), but each
/// walker's return is credited to **all** categories matching the origin's
/// bitmask in `category_map`.  This allows a single walk on the full graph
/// to produce P(t) for every sub-population simultaneously — no separate
/// convergence loops, no redundant computation.
///
/// **Trap 2 fix**: the walker's origin is resolved through `merge_into`
/// *before* category lookup, preventing ghost nodes from silently dropping
/// returns.
///
/// **Strict Finitism**: all internal accumulation uses `u64` integer
/// arithmetic.  The single `f64` division occurs only at the caller's
/// discretion (not inside this function).
///
/// # Returns
/// `(global_counts, per_category_counts)`:
/// - `global_counts: Vec<u64>` — length `steps.len()`, total returns across all walkers
/// - `per_category_counts: Vec<Vec<u64>>` — `n_categories` vectors, each length `steps.len()`
pub fn run_walkers_unified(
    adj_head: &[u32],
    adj_data: &[u32],
    starts: &[usize],
    steps: &[u32],
    base_seed: u64,
    merge_into: Option<&[usize]>,
    category_map: &[u8],
    n_categories: usize,
) -> (Vec<u64>, Vec<Vec<u64>>) {
    let n_w = starts.len();
    let n_s = steps.len();
    if n_w == 0 {
        return (vec![0u64; n_s], vec![vec![0u64; n_s]; n_categories]);
    }
    let max_t = *steps.last().unwrap_or(&0);

    let (global, cats): (Vec<u64>, Vec<Vec<u64>>) = starts
        .par_iter()
        .enumerate()
        .map(|(wi, &origin)| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(wi as u64));
            // Trap 2 fix: resolve origin before category lookup
            let origin = resolve(origin, merge_into);
            let mask = category_map[origin];
            let mut pos = origin;
            let mut g = vec![0u64; n_s];
            let mut c = vec![vec![0u64; n_s]; n_categories];
            let mut si = 0usize;

            for t in 1..=max_t {
                let start = adj_head[pos] as usize;
                let end = adj_head[pos + 1] as usize;
                let len = end - start;

                if len > 0 && rng.gen_bool(0.5) {
                    let next = adj_data[start + rng.gen_range(0..len)] as usize;
                    pos = resolve(next, merge_into);
                }
                if si < n_s && t == steps[si] {
                    if pos == origin {
                        g[si] = 1;
                        for cat in 0..n_categories {
                            if mask & (1u8 << cat) != 0 {
                                c[cat][si] = 1;
                            }
                        }
                    }
                    si += 1;
                }
            }
            (g, c)
        })
        .reduce(
            || (vec![0u64; n_s], vec![vec![0u64; n_s]; n_categories]),
            |mut a, b| {
                for i in 0..n_s {
                    a.0[i] += b.0[i];
                }
                for cat in 0..n_categories {
                    for i in 0..n_s {
                        a.1[cat][i] += b.1[cat][i];
                    }
                }
                a
            },
        );

    (global, cats)
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 1 — Eigendecomposition  (N <= 3k, exact)
// ═════════════════════════════════════════════════════════════════════════════

/// Exact spectral dimension via eigendecomposition of the symmetric
/// normalised adjacency S = D^{-1/2} A D^{-1/2}.
///
/// Complexity: O(N^3). Used only for small graphs (N <= 3k).
///
/// Global P(t) = N^{-1} sum_k lambda_k^t.  Local P(t) uses eigenvector-weighted
/// sums over core indices, probing the geometry near the diamond's centre.
///
/// # Returns
/// `(global, core)` pair of [`WalkResult`]. If `core_indices` is empty,
/// `core` is a clone of `global`.
pub fn compute_eigen(
    n: usize,
    edge_rows: &[u32],
    edge_cols: &[u32],
    steps: &[u32],
    core_indices: &[usize],
) -> (WalkResult, WalkResult) {
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

    println!("  Eigendecomposition ({n}x{n}) ...");
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

    let global = WalkResult {
        p: p_global.clone(),
        ds: ds_global.clone(),
        ds_std: vec![],
    };

    let core = if core_indices.is_empty() {
        global.clone()
    } else {
        let nc = core_indices.len();
        let w: Vec<f64> = (0..n)
            .map(|k| {
                core_indices
                    .iter()
                    .map(|&i| eigen.eigenvectors[(i, k)].powi(2))
                    .sum()
            })
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
        WalkResult {
            p: p_loc,
            ds: ds_loc,
            ds_std: vec![],
        }
    };

    (global, core)
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 2 — Monte Carlo walkers  (N > 3k, O(W*t))
// ═════════════════════════════════════════════════════════════════════════════

/// Monte Carlo spectral dimension from pre-built CSR adjacency.
///
/// Zero-copy variant — avoids rebuilding adjacency when CSR is already
/// available from Phase 2. This is the hot path for large N in-memory runs.
///
/// # Returns
/// `(global, core)` pair of [`WalkResult`]. If `core_indices` is empty,
/// `core` is a clone of `global`.
pub fn compute_monte_carlo_csr(
    n: usize,
    adj_head: &[u32],
    adj_data: &[u32],
    steps: &[u32],
    core_indices: &[usize],
    n_walkers: usize,
    rng: &mut impl Rng,
) -> (WalkResult, WalkResult) {
    // ── Global ───────────────────────────────────────────────────────
    let global_starts: Vec<usize> = (0..n_walkers).map(|_| rng.gen_range(0..n)).collect();
    let seed_g: u64 = rng.gen();
    let p_global = run_walkers(adj_head, adj_data, &global_starts, steps, seed_g, None);
    let ds_global = spectral_dimension(steps, &p_global);

    let global = WalkResult {
        p: p_global,
        ds: ds_global,
        ds_std: vec![],
    };

    // ── Local ────────────────────────────────────────────────────────
    let core = if core_indices.is_empty() {
        global.clone()
    } else {
        let local_starts = distribute_walkers(core_indices, n_walkers);
        let seed_l: u64 = rng.gen();
        let p_loc = run_walkers(adj_head, adj_data, &local_starts, steps, seed_l, None);
        let ds_loc = spectral_dimension(steps, &p_loc);
        WalkResult {
            p: p_loc,
            ds: ds_loc,
            ds_std: vec![],
        }
    };

    (global, core)
}

/// Monte Carlo spectral dimension from edge lists.
///
/// Builds CSR adjacency internally, then launches lazy random walkers.
/// Complexity: O(W*t) where W = number of walkers, independent of N.
/// Used for large graphs (N > 3k) where eigendecomposition is infeasible.
///
/// # Returns
/// `(global, core)` pair of [`WalkResult`]. If `core_indices` is empty,
/// `core` is a clone of `global`.
pub fn compute_monte_carlo(
    n: usize,
    edge_rows: &[u32],
    edge_cols: &[u32],
    steps: &[u32],
    core_indices: &[usize],
    n_walkers: usize,
    rng: &mut impl Rng,
) -> (WalkResult, WalkResult) {
    let (adj_head, adj_data) = build_adj_list(n, edge_rows, edge_cols);
    compute_monte_carlo_csr(n, &adj_head, &adj_data, steps, core_indices, n_walkers, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Triangle graph: 0-1, 0-2, 1-2
    fn triangle_edges() -> (Vec<u32>, Vec<u32>) {
        (vec![0, 0, 1], vec![1, 2, 2])
    }

    #[test]
    fn distribute_walkers_exact_count() {
        let origins = vec![0, 1, 2];
        let starts = distribute_walkers(&origins, 10);
        assert_eq!(starts.len(), 10);
        // first remainder=1 node gets base+1 = 4, rest get 3
        assert_eq!(starts.iter().filter(|&&s| s == 0).count(), 4);
        assert_eq!(starts.iter().filter(|&&s| s == 1).count(), 3);
        assert_eq!(starts.iter().filter(|&&s| s == 2).count(), 3);
    }

    #[test]
    fn distribute_walkers_empty() {
        let starts = distribute_walkers(&[], 100);
        assert!(starts.is_empty());
    }

    #[test]
    fn build_adj_list_triangle() {
        let (rows, cols) = triangle_edges();
        let (head, data) = build_adj_list(3, &rows, &cols);
        // Each node has degree 2 in a triangle
        assert_eq!(head.len(), 4);
        assert_eq!((head[1] - head[0]) as usize, 2); // node 0: neighbors 1,2
        assert_eq!((head[2] - head[1]) as usize, 2); // node 1: neighbors 0,2
        assert_eq!((head[3] - head[2]) as usize, 2); // node 2: neighbors 0,1
        // Neighbors are sorted
        let n0 = &data[head[0] as usize..head[1] as usize];
        assert_eq!(n0, &[1, 2]);
    }

    #[test]
    fn run_walkers_returns_correct_length() {
        let (rows, cols) = triangle_edges();
        let (head, data) = build_adj_list(3, &rows, &cols);
        let steps = [1, 2, 4, 8];
        let starts = vec![0; 100];
        let p = run_walkers(&head, &data, &starts, &steps, 42, None);
        assert_eq!(p.len(), 4);
        for &val in &p {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn run_walkers_empty_starts() {
        let steps = [1, 2, 4];
        let p = run_walkers(&[0], &[], &[], &steps, 0, None);
        assert_eq!(p, vec![0.0; 3]);
    }

    #[test]
    fn compute_eigen_triangle() {
        let (rows, cols) = triangle_edges();
        let steps = [1, 2, 4];
        let (global, core) = compute_eigen(3, &rows, &cols, &steps, &[]);
        assert_eq!(global.p.len(), 3);
        assert_eq!(global.ds.len(), 3);
        assert!(global.ds_std.is_empty());
        // With empty core_indices, core == global
        assert_eq!(core.p, global.p);
    }

    #[test]
    fn compute_eigen_with_core() {
        let (rows, cols) = triangle_edges();
        let steps = [1, 2, 4];
        let (global, core) = compute_eigen(3, &rows, &cols, &steps, &[0]);
        // Core should differ from global when core subset is specified
        assert_eq!(core.p.len(), 3);
        assert_eq!(core.ds.len(), 3);
        // Global and core may differ
        assert_eq!(global.p.len(), core.p.len());
    }

    #[test]
    fn compute_monte_carlo_csr_triangle() {
        let (rows, cols) = triangle_edges();
        let (head, data) = build_adj_list(3, &rows, &cols);
        let steps = [1, 2, 4, 8];
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        let (global, core) = compute_monte_carlo_csr(
            3, &head, &data, &steps, &[0, 1], 200, &mut rng,
        );
        assert_eq!(global.p.len(), 4);
        assert_eq!(core.p.len(), 4);
    }

    #[test]
    fn resolve_identity_without_merge() {
        assert_eq!(resolve(5, None), 5);
    }

    #[test]
    fn resolve_chases_chain() {
        // 0 -> 1 -> 2 -> 2 (fixed point)
        let merge = vec![1, 2, 2];
        assert_eq!(resolve(0, Some(&merge)), 2);
        assert_eq!(resolve(1, Some(&merge)), 2);
        assert_eq!(resolve(2, Some(&merge)), 2);
    }

    #[test]
    fn distribute_walkers_shuffled_exact_count() {
        let origins = vec![0, 1, 2, 3, 4];
        let starts = distribute_walkers_shuffled(&origins, 13, 42);
        assert_eq!(starts.len(), 13);
        // Every origin should appear (13/5 = 2 base + 3 remainder)
        for &o in &origins {
            assert!(starts.contains(&o));
        }
    }

    #[test]
    fn distribute_walkers_shuffled_empty() {
        let starts = distribute_walkers_shuffled(&[], 100, 0);
        assert!(starts.is_empty());
    }

    #[test]
    fn distribute_walkers_shuffled_deterministic() {
        let origins: Vec<usize> = (0..100).collect();
        let a = distribute_walkers_shuffled(&origins, 50, 999);
        let b = distribute_walkers_shuffled(&origins, 50, 999);
        assert_eq!(a, b);
    }

    #[test]
    fn count_category_walkers_basic() {
        // 3 nodes: node 0 = cat 0+1, node 1 = cat 0, node 2 = cat 1
        let cat_map = vec![0x03u8, 0x01, 0x02];
        let starts = vec![0, 0, 1, 2, 2, 2];
        let counts = count_category_walkers(&starts, &cat_map, 2);
        // Cat 0: node 0 (×2) + node 1 (×1) = 3
        assert_eq!(counts[0], 3);
        // Cat 1: node 0 (×2) + node 2 (×3) = 5
        assert_eq!(counts[1], 5);
    }

    #[test]
    fn run_walkers_unified_triangle() {
        let (rows, cols) = triangle_edges();
        let (head, data) = build_adj_list(3, &rows, &cols);
        let steps = [1, 2, 4, 8];
        // Node 0,1 = cat 0 (0x01), node 2 = cat 1 (0x02), all = cat 0 (global)
        let cat_map = vec![0x01u8, 0x01, 0x02];
        let starts = vec![0, 0, 1, 1, 2, 2];
        let (global, cats) = run_walkers_unified(
            &head, &data, &starts, &steps, 42, None, &cat_map, 2,
        );
        assert_eq!(global.len(), 4);
        assert_eq!(cats.len(), 2);
        assert_eq!(cats[0].len(), 4);
        assert_eq!(cats[1].len(), 4);
        // Global counts should be >= sum of cat counts (a node may be in multiple cats)
        // Actually global counts = total returns regardless of category
        for i in 0..4 {
            assert!(global[i] >= cats[0][i]);
            assert!(global[i] >= cats[1][i]);
        }
    }

    #[test]
    fn run_walkers_unified_empty() {
        let cat_map = vec![0x01u8];
        let (global, cats) = run_walkers_unified(
            &[0, 0], &[], &[], &[1, 2], 0, None, &cat_map, 1,
        );
        assert_eq!(global, vec![0u64; 2]);
        assert_eq!(cats, vec![vec![0u64; 2]]);
    }

    #[test]
    fn run_walkers_unified_ghost_resolution() {
        // Triangle with merge: node 1 merged into node 0
        let (rows, cols) = triangle_edges();
        let (head, data) = build_adj_list(3, &rows, &cols);
        let merge = vec![0, 0, 2]; // node 1 -> node 0
        // Category: node 0 = cat 0, node 2 = cat 1
        let cat_map = vec![0x01u8, 0x00, 0x02]; // ghost node 1 has no category
        // Walker starts at ghost node 1 — should resolve to node 0, get cat 0
        let starts = vec![1; 100];
        let (global, cats) = run_walkers_unified(
            &head, &data, &starts, &[2, 4], 77, Some(&merge), &cat_map, 2,
        );
        assert_eq!(global.len(), 2);
        // All walkers resolved to node 0 (cat 0), so cat 0 should have returns
        // and cat 1 should have zero (walkers start at node 0, not node 2)
        // (cat 1 returns come only from walkers whose origin is node 2)
        assert_eq!(cats[1], vec![0u64; 2]);
    }
}
