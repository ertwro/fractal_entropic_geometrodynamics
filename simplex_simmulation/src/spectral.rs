//! Phase 3 — Random Walk & Spectral Dimension
//!
//! Two tiers:
//!   - Eigendecomposition   (N ≤ 3k, exact, O(N³))
//!   - Monte Carlo walkers  (N > 3k, O(W·t), **independent of N**)

use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

pub struct SpectralResult {
    pub p_global: Vec<f64>,
    pub ds_global: Vec<f64>,
    pub p_local: Vec<f64>,
    pub ds_local: Vec<f64>,
}

// ─── shared ──────────────────────────────────────────────────────────────────

/// d_S(t) = −2 d(ln P)/d(ln t)  via centred finite differences.
pub fn spectral_dimension(steps: &[u32], p_vals: &[f64]) -> Vec<f64> {
    let n = steps.len();
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

/// Build sorted undirected adjacency lists (u32 indices, cache-compact).
fn build_adj_list(n: usize, rows: &[u32], cols: &[u32]) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (&r, &c) in rows.iter().zip(cols.iter()) {
        adj[r as usize].push(c);
        adj[c as usize].push(r);
    }
    for list in adj.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    adj
}

// ═════════════════════════════════════════════════════════════════════════════
// Monte Carlo walker engine  (the integer heart)
// ═════════════════════════════════════════════════════════════════════════════

/// Run W independent lazy-walk walkers in parallel.
///
/// Each walker starts at `starts[i]`, takes steps using the lazy rule:
///   - with prob 0.5: stay
///   - with prob 0.5: move to uniform random neighbour
///
/// Returns `P(t)` = fraction of walkers back at their origin at step `t`.
/// Complexity: **O(W × t_max)**, independent of N.
fn run_walkers(
    adj: &[Vec<u32>],
    starts: &[usize],
    steps: &[u32],
    base_seed: u64,
) -> Vec<f64> {
    let n_w = starts.len();
    let n_s = steps.len();
    let max_t = *steps.last().unwrap();

    let counts: Vec<usize> = starts
        .par_iter()
        .enumerate()
        .map(|(wi, &origin)| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(wi as u64));
            let mut pos = origin;
            let mut c = vec![0usize; n_s];
            let mut si = 0usize;

            for t in 1..=max_t {
                // Lazy walk: integer adjacency traversal
                let nbs = &adj[pos];
                if !nbs.is_empty() && rng.gen_bool(0.5) {
                    pos = nbs[rng.gen_range(0..nbs.len())] as usize;
                }
                // Record return at measurement steps
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
            || vec![0usize; n_s],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );

    // f64 only here, at the very end: macroscopic averaging
    counts.iter().map(|&c| c as f64 / n_w as f64).collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 1 — Eigendecomposition  (N ≤ 3k, exact)
// ═════════════════════════════════════════════════════════════════════════════

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

    SpectralResult { p_global, ds_global, p_local, ds_local }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tier 2 — Monte Carlo walkers  (N > 3k, O(W·t))
// ═════════════════════════════════════════════════════════════════════════════

/// Stochastic spectral dimension via parallel Monte Carlo walkers.
///
/// - Global: `n_walkers` from uniformly random starting nodes.
/// - Local:  `n_walkers` distributed across `core_indices`.
///
/// Complexity: O(W × t_max) per graph.  **Independent of N.**
pub fn compute_monte_carlo(
    n: usize,
    edge_rows: &[u32],
    edge_cols: &[u32],
    steps: &[u32],
    core_indices: &[usize],
    n_walkers: usize,
    rng: &mut impl Rng,
) -> SpectralResult {
    let adj = build_adj_list(n, edge_rows, edge_cols);

    // ── Global ───────────────────────────────────────────────────────
    let global_starts: Vec<usize> = (0..n_walkers).map(|_| rng.gen_range(0..n)).collect();
    let seed_g: u64 = rng.gen();
    println!("[Phase 3] MC walkers (global, W={}) …", n_walkers);
    let p_global = run_walkers(&adj, &global_starts, steps, seed_g);
    let ds_global = spectral_dimension(steps, &p_global);

    // ── Local ────────────────────────────────────────────────────────
    let (p_local, ds_local) = if core_indices.is_empty() {
        (p_global.clone(), ds_global.clone())
    } else {
        let local_starts: Vec<usize> = (0..n_walkers)
            .map(|i| core_indices[i % core_indices.len()])
            .collect();
        let seed_l: u64 = rng.gen();
        println!(
            "[Phase 3] MC walkers (core, W={}, |core|={}) …",
            n_walkers,
            core_indices.len()
        );
        let p_loc = run_walkers(&adj, &local_starts, steps, seed_l);
        let ds_loc = spectral_dimension(steps, &p_loc);
        (p_loc, ds_loc)
    };

    SpectralResult { p_global, ds_global, p_local, ds_local }
}
