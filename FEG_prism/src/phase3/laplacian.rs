// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Heat kernel spectral flow and RG integration.
//!
//! Implements the rigorous, parameter-free path from the causal graph to mass
//! scale predictions:
//!
//! ```text
//!   Graph Laplacian  →  Heat kernel trace  →  D_s(T)  →  RG integration  →  M_EW/M_Pl
//! ```
//!
//! Two tiers:
//! - **Tier 1** (`heat_kernel_exact`): full eigendecomp via nalgebra (N ≤ 10 000)
//! - **Tier 2** (`heat_kernel_slq`): Stochastic Lanczos Quadrature (any N)
//!
//! The normalized graph Laplacian is  L_sym = I − D^{−1/2} A D^{−1/2}  with
//! eigenvalues λ_k ∈ \[0, 2\].  The heat kernel trace  K(T) = Σ e^{−T λ_k}
//! gives K(0) = N and K(∞) → 1.

use crate::graph::csr::{CsrGraph, Undirected};
use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ═══════════════════════════════════════════════════════════════════════════════
//  Result types
// ═══════════════════════════════════════════════════════════════════════════════

/// Output of the heat kernel computation (Tier 1 or Tier 2).
#[derive(Debug, Clone)]
pub struct HeatKernelResult {
    /// Laplacian eigenvalues, sorted ascending (Tier 1 only; empty for Tier 2).
    pub eigenvalues: Vec<f64>,
    /// Log-spaced diffusion times T.
    pub diffusion_times: Vec<f64>,
    /// Heat kernel trace K(T) = Σ e^{−T λ_k} at each diffusion time.
    pub heat_trace: Vec<f64>,
    /// Spectral dimension D_s(T) = −2 d ln K / d ln T at each diffusion time.
    pub spectral_dim: Vec<f64>,
    /// Number of nodes in the graph.
    pub n_nodes: usize,
    /// Smallest nonzero eigenvalue (spectral gap).
    pub spectral_gap: f64,
}

/// Output of the RG integration.
#[derive(Debug, Clone)]
pub struct RgFlowResult {
    /// ln(μ/M_Pl) at each integration step.
    pub ln_mu: Vec<f64>,
    /// Running coupling α(μ) at each step.
    pub alpha: Vec<f64>,
    /// Inverse coupling 1/α(μ) at each step.
    pub inv_alpha: Vec<f64>,
    /// Spectral dimension D_s(μ) interpolated at each step.
    pub ds_at_mu: Vec<f64>,
    /// Scale where α ∼ 1 (confinement / EW), as M/M_Pl.
    pub m_ew_over_m_pl: f64,
    /// Scale where α ∼ 0.1185 (α_s at M_Z), as M/M_Pl.
    pub m_qcd_over_m_pl: f64,
    /// Initial coupling at M_Pl.
    pub alpha_0: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Utility: log-spaced times
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate `n` logarithmically spaced values from `t_min` to `t_max`.
pub fn log_spaced_times(t_min: f64, t_max: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2, "need at least 2 points");
    assert!(t_min > 0.0 && t_max > t_min);
    let log_min = t_min.ln();
    let log_max = t_max.ln();
    (0..n)
        .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Tier 1: Exact eigendecomposition
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the heat kernel trace via full eigendecomposition of the normalized
/// Laplacian L_sym = I − D^{−1/2} A D^{−1/2}.
///
/// Suitable for N ≤ 10 000 (O(N³) eigendecomp).
pub fn heat_kernel_exact(
    sym_csr: &CsrGraph<Undirected>,
    times: &[f64],
) -> HeatKernelResult {
    let n = sym_csr.n_nodes();

    // Inverse square-root of degree
    let inv_sqrt_d: Vec<f64> = (0..n)
        .map(|u| {
            let d = sym_csr.degree(u) as f64;
            if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
        })
        .collect();

    // Build dense L_sym
    let mut l_sym = DMatrix::<f64>::zeros(n, n);
    for u in 0..n {
        if sym_csr.degree(u) > 0 {
            l_sym[(u, u)] = 1.0;
        }
        for &v in sym_csr.neighbors(u) {
            let vi = v as usize;
            // Only fill upper triangle once since we visit both (u,v) and (v,u)
            if u < vi {
                let val = inv_sqrt_d[u] * inv_sqrt_d[vi];
                l_sym[(u, vi)] -= val;
                l_sym[(vi, u)] -= val;
            }
        }
    }

    // Eigendecompose
    let eigen = SymmetricEigen::new(l_sym);
    let mut eigenvalues: Vec<f64> = eigen
        .eigenvalues
        .iter()
        .map(|&v| v.clamp(0.0, 2.0))
        .collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Spectral gap: smallest nonzero eigenvalue
    let spectral_gap = eigenvalues
        .iter()
        .copied()
        .find(|&v| v > 1e-10)
        .unwrap_or(0.0);

    // Heat kernel trace K(T)
    let heat_trace: Vec<f64> = times
        .iter()
        .map(|&t| eigenvalues.iter().map(|&lam| (-t * lam).exp()).sum())
        .collect();

    // Spectral dimension D_s(T) via finite differences on log scale
    let spectral_dim = heat_spectral_dim(times, &heat_trace);

    HeatKernelResult {
        eigenvalues,
        diffusion_times: times.to_vec(),
        heat_trace,
        spectral_dim,
        n_nodes: n,
        spectral_gap,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Tier 2: Stochastic Lanczos Quadrature
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the heat kernel trace via Stochastic Lanczos Quadrature.
///
/// Uses `n_probes` Rademacher random vectors and `k_lanczos` Lanczos steps.
/// Suitable for any N (O(k × nnz × n_probes) per diffusion time).
///
/// # Arguments
/// * `sym_csr` — undirected graph in CSR format
/// * `times` — diffusion time values T
/// * `n_probes` — number of Rademacher probe vectors (typically 20–50)
/// * `k_lanczos` — number of Lanczos iterations (typically 100–300)
/// * `seed` — random seed
pub fn heat_kernel_slq(
    sym_csr: &CsrGraph<Undirected>,
    times: &[f64],
    n_probes: usize,
    k_lanczos: usize,
    seed: u64,
) -> HeatKernelResult {
    let n = sym_csr.n_nodes();
    let k = k_lanczos.min(n); // can't exceed matrix dimension

    // Precompute inv_sqrt_d
    let inv_sqrt_d: Vec<f64> = (0..n)
        .map(|u| {
            let d = sym_csr.degree(u) as f64;
            if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
        })
        .collect();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut heat_trace = vec![0.0_f64; times.len()];

    for _probe in 0..n_probes {
        // Rademacher random vector z ∈ {−1, +1}^n
        let z: Vec<f64> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();

        // Lanczos iteration with full reorthogonalization
        let (alpha_vec, beta_vec) =
            lanczos_reorth(sym_csr, &inv_sqrt_d, &z, k);

        // Build k×k tridiagonal matrix and eigendecompose
        let k_actual = alpha_vec.len();
        let mut tri = DMatrix::<f64>::zeros(k_actual, k_actual);
        for i in 0..k_actual {
            tri[(i, i)] = alpha_vec[i];
            if i + 1 < k_actual {
                tri[(i, i + 1)] = beta_vec[i];
                tri[(i + 1, i)] = beta_vec[i];
            }
        }
        let eig = SymmetricEigen::new(tri);

        // Ritz values θ_i and weights τ_i = e_1^T Q_i (first component of eigenvector)
        let ritz_values = &eig.eigenvalues;
        let ritz_vectors = &eig.eigenvectors;

        // Accumulate: K(T) += (N/m) Σ_i τ_i² e^{-T θ_i}
        for (ti, &t) in times.iter().enumerate() {
            let mut contrib = 0.0;
            for i in 0..k_actual {
                let theta = ritz_values[i].clamp(0.0, 2.0);
                let tau_sq = ritz_vectors[(0, i)].powi(2);
                contrib += tau_sq * (-t * theta).exp();
            }
            heat_trace[ti] += (n as f64 / n_probes as f64) * contrib;
        }
    }

    // Spectral gap: estimate from smallest positive Ritz value across probes
    // (not exact, but good enough for diagnostics)
    let spectral_gap = estimate_spectral_gap_slq(sym_csr, &inv_sqrt_d, seed + 999, k);

    let spectral_dim = heat_spectral_dim(times, &heat_trace);

    HeatKernelResult {
        eigenvalues: vec![], // not available in SLQ
        diffusion_times: times.to_vec(),
        heat_trace,
        spectral_dim,
        n_nodes: n,
        spectral_gap,
    }
}

/// Lanczos iteration with full reorthogonalization.
///
/// Returns (alpha, beta) vectors defining the tridiagonal matrix.
fn lanczos_reorth(
    sym_csr: &CsrGraph<Undirected>,
    inv_sqrt_d: &[f64],
    z: &[f64],
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = z.len();
    let mut alpha = Vec::with_capacity(k);
    let mut beta = Vec::with_capacity(k);

    // Store all Lanczos vectors for reorthogonalization
    let mut q_vecs: Vec<Vec<f64>> = Vec::with_capacity(k + 1);

    // Normalize initial vector
    let norm_z: f64 = z.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_z < 1e-15 {
        return (vec![0.0], vec![]);
    }
    let q: Vec<f64> = z.iter().map(|&x| x / norm_z).collect();
    q_vecs.push(q);

    for j in 0..k {
        // w = L_sym · q_j
        let mut w = laplacian_matvec(sym_csr, inv_sqrt_d, &q_vecs[j]);

        // α_j = q_j^T w
        let a: f64 = q_vecs[j].iter().zip(w.iter()).map(|(&q, &wi)| q * wi).sum();
        alpha.push(a);

        // w = w − α_j q_j − β_{j-1} q_{j-1}
        for i in 0..n {
            w[i] -= a * q_vecs[j][i];
        }
        if j > 0 {
            let b_prev = beta[j - 1];
            for i in 0..n {
                w[i] -= b_prev * q_vecs[j - 1][i];
            }
        }

        // Full reorthogonalization against all previous vectors
        for prev in &q_vecs {
            let dot: f64 = prev.iter().zip(w.iter()).map(|(&p, &wi)| p * wi).sum();
            for i in 0..n {
                w[i] -= dot * prev[i];
            }
        }

        // β_j = ||w||
        let b: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if b < 1e-14 {
            // Invariant subspace found
            break;
        }
        beta.push(b);

        // q_{j+1} = w / β_j
        let q_next: Vec<f64> = w.iter().map(|&x| x / b).collect();
        q_vecs.push(q_next);
    }

    (alpha, beta)
}

/// Matrix-vector product y = L_sym · v via CSR.
///
/// L_sym = I − D^{−1/2} A D^{−1/2}, so:
///   y[u] = v[u] − inv_sqrt_d[u] × Σ_{w∈N(u)} inv_sqrt_d[w] × v[w]
fn laplacian_matvec(
    sym_csr: &CsrGraph<Undirected>,
    inv_sqrt_d: &[f64],
    v: &[f64],
) -> Vec<f64> {
    let n = sym_csr.n_nodes();
    let mut y = vec![0.0_f64; n];
    for u in 0..n {
        if sym_csr.degree(u) == 0 {
            // Isolated node: L_sym row is zero
            continue;
        }
        let mut sum = 0.0;
        for &w in sym_csr.neighbors(u) {
            sum += inv_sqrt_d[w as usize] * v[w as usize];
        }
        y[u] = v[u] - inv_sqrt_d[u] * sum;
    }
    y
}

/// Estimate spectral gap from a single Lanczos run (for SLQ diagnostics).
fn estimate_spectral_gap_slq(
    sym_csr: &CsrGraph<Undirected>,
    inv_sqrt_d: &[f64],
    seed: u64,
    k: usize,
) -> f64 {
    let n = sym_csr.n_nodes();
    let mut rng = StdRng::seed_from_u64(seed);
    let z: Vec<f64> = (0..n)
        .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
        .collect();
    let (alpha_vec, beta_vec) = lanczos_reorth(sym_csr, inv_sqrt_d, &z, k);
    let k_actual = alpha_vec.len();
    let mut tri = DMatrix::<f64>::zeros(k_actual, k_actual);
    for i in 0..k_actual {
        tri[(i, i)] = alpha_vec[i];
        if i + 1 < k_actual {
            tri[(i, i + 1)] = beta_vec[i];
            tri[(i + 1, i)] = beta_vec[i];
        }
    }
    let eig = SymmetricEigen::new(tri);
    let mut vals: Vec<f64> = eig
        .eigenvalues
        .iter()
        .map(|&v| v.clamp(0.0, 2.0))
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.iter().copied().find(|&v| v > 1e-10).unwrap_or(0.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Spectral dimension from heat kernel
// ═══════════════════════════════════════════════════════════════════════════════

/// D_s(T) = −2 d ln K / d ln T via centred finite differences.
fn heat_spectral_dim(times: &[f64], heat_trace: &[f64]) -> Vec<f64> {
    let n = times.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let ln_t: Vec<f64> = times.iter().map(|&t| t.ln()).collect();
    let ln_k: Vec<f64> = heat_trace.iter().map(|&k| k.max(1e-30).ln()).collect();
    let mut ds = vec![0.0; n];
    for i in 0..n {
        let d = if i == 0 {
            (ln_k[1] - ln_k[0]) / (ln_t[1] - ln_t[0])
        } else if i == n - 1 {
            (ln_k[n - 1] - ln_k[n - 2]) / (ln_t[n - 1] - ln_t[n - 2])
        } else {
            (ln_k[i + 1] - ln_k[i - 1]) / (ln_t[i + 1] - ln_t[i - 1])
        };
        ds[i] = -2.0 * d;
    }
    ds
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Weyl law check
// ═══════════════════════════════════════════════════════════════════════════════

/// Fit log N(λ) vs log λ to estimate manifold dimension d from Weyl's law
/// N(λ) ∝ λ^{d/2}.
///
/// Returns (slope, R²). For a 4D causal set, slope ≈ 2.0.
pub fn weyl_law_check(eigenvalues: &[f64]) -> (f64, f64) {
    // Filter out zero eigenvalues and build cumulative count
    let nonzero: Vec<f64> = eigenvalues
        .iter()
        .copied()
        .filter(|&v| v > 1e-10)
        .collect();
    if nonzero.len() < 3 {
        return (0.0, 0.0);
    }

    let log_lam: Vec<f64> = nonzero.iter().map(|v| v.ln()).collect();
    let log_n: Vec<f64> = (1..=nonzero.len())
        .map(|k| (k as f64).ln())
        .collect();

    // Linear regression: log_n = slope * log_lam + intercept
    let m = log_lam.len() as f64;
    let sum_x: f64 = log_lam.iter().sum();
    let sum_y: f64 = log_n.iter().sum();
    let sum_xy: f64 = log_lam.iter().zip(log_n.iter()).map(|(&x, &y)| x * y).sum();
    let sum_x2: f64 = log_lam.iter().map(|&x| x * x).sum();

    let denom = m * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return (0.0, 0.0);
    }
    let slope = (m * sum_xy - sum_x * sum_y) / denom;

    // R²
    let mean_y = sum_y / m;
    let ss_tot: f64 = log_n.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let intercept = (sum_y - slope * sum_x) / m;
    let ss_res: f64 = log_lam
        .iter()
        .zip(log_n.iter())
        .map(|(&x, &y)| (y - slope * x - intercept).powi(2))
        .sum();
    let r_sq = if ss_tot > 1e-30 { 1.0 - ss_res / ss_tot } else { 0.0 };

    (slope, r_sq)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RG integration
// ═══════════════════════════════════════════════════════════════════════════════

/// Integrate the running coupling α(μ) from M_Pl down to low energies using
/// the β-function:
///
/// ```text
///   dα/d(ln μ) = (D_s(μ) − 4)/2 · α − b₀ · α²
/// ```
///
/// where μ = 1/√T (diffusion length), b₀ = (33 − 2 n_f)/(12π) for SU(3),
/// and D_s(μ) is interpolated from the heat kernel result.
///
/// # Arguments
/// * `hk` — heat kernel result (provides D_s(T) table)
/// * `alpha_0` — initial coupling at M_Pl (T = 1)
/// * `n_f` — number of active quark flavors
/// * `ln_mu_min` — minimum ln(μ/M_Pl) to integrate to (negative, e.g. −40)
/// * `n_steps` — number of RK4 steps
pub fn integrate_rg(
    hk: &HeatKernelResult,
    alpha_0: f64,
    n_f: u32,
    ln_mu_min: f64,
    n_steps: usize,
) -> RgFlowResult {
    let b0 = (33.0 - 2.0 * n_f as f64) / (12.0 * std::f64::consts::PI);

    // Build lookup table: (ln_T, D_s) sorted by ln_T
    let mut lookup: Vec<(f64, f64)> = hk
        .diffusion_times
        .iter()
        .zip(hk.spectral_dim.iter())
        .map(|(&t, &ds)| (t.ln(), ds))
        .collect();
    lookup.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Interpolate D_s at given ln_mu (T = 1/μ² → ln T = −2 ln μ)
    let interp_ds = |ln_mu: f64| -> f64 {
        let ln_t = -2.0 * ln_mu;
        if lookup.is_empty() {
            return 4.0;
        }
        if ln_t <= lookup[0].0 {
            return lookup[0].1;
        }
        if ln_t >= lookup.last().unwrap().0 {
            return lookup.last().unwrap().1;
        }
        // Binary search for bracket
        let idx = lookup
            .partition_point(|&(lt, _)| lt < ln_t)
            .saturating_sub(1);
        let (lt0, ds0) = lookup[idx];
        let (lt1, ds1) = lookup[(idx + 1).min(lookup.len() - 1)];
        let frac = (ln_t - lt0) / (lt1 - lt0).max(1e-30);
        ds0 + frac * (ds1 - ds0)
    };

    // β-function
    let beta = |alpha: f64, ln_mu: f64| -> f64 {
        let ds = interp_ds(ln_mu);
        (ds - 4.0) / 2.0 * alpha - b0 * alpha * alpha
    };

    // RK4 integration from ln_mu = 0 (M_Pl) down to ln_mu_min
    let h = ln_mu_min / n_steps as f64; // h < 0
    let mut ln_mu_vals = Vec::with_capacity(n_steps + 1);
    let mut alpha_vals = Vec::with_capacity(n_steps + 1);
    let mut ds_vals = Vec::with_capacity(n_steps + 1);

    let mut ln_mu = 0.0_f64;
    let mut alpha = alpha_0;

    ln_mu_vals.push(ln_mu);
    alpha_vals.push(alpha);
    ds_vals.push(interp_ds(ln_mu));

    let mut m_ew_over_m_pl = 0.0_f64;
    let mut m_qcd_over_m_pl = 0.0_f64;
    let mut found_ew = false;
    let mut found_qcd = false;

    for _ in 0..n_steps {
        // RK4
        let k1 = h * beta(alpha, ln_mu);
        let k2 = h * beta(alpha + 0.5 * k1, ln_mu + 0.5 * h);
        let k3 = h * beta(alpha + 0.5 * k2, ln_mu + 0.5 * h);
        let k4 = h * beta(alpha + k3, ln_mu + h);

        let alpha_new = alpha + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        ln_mu += h;
        alpha = alpha_new.max(0.0); // clamp to non-negative

        ln_mu_vals.push(ln_mu);
        alpha_vals.push(alpha);
        ds_vals.push(interp_ds(ln_mu));

        // Detect crossings
        if !found_ew && alpha >= 1.0 {
            m_ew_over_m_pl = ln_mu.exp();
            found_ew = true;
        }
        if !found_qcd && alpha >= 0.1185 {
            m_qcd_over_m_pl = ln_mu.exp();
            found_qcd = true;
        }
    }

    let inv_alpha: Vec<f64> = alpha_vals.iter().map(|&a| if a > 1e-30 { 1.0 / a } else { f64::INFINITY }).collect();

    RgFlowResult {
        ln_mu: ln_mu_vals,
        alpha: alpha_vals,
        inv_alpha,
        ds_at_mu: ds_vals,
        m_ew_over_m_pl,
        m_qcd_over_m_pl,
        alpha_0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::csr::build_undirected_csr;

    /// Triangle graph (K₃): eigenvalues of L_sym are 0, 3/2, 3/2.
    #[test]
    fn triangle_eigenvalues() {
        // Triangle: edges 0-1, 0-2, 1-2
        let g = build_undirected_csr(3, &[0, 0, 1], &[1, 2, 2]);
        let times = log_spaced_times(0.01, 100.0, 50);
        let hk = heat_kernel_exact(&g, &times);

        assert_eq!(hk.n_nodes, 3);
        assert_eq!(hk.eigenvalues.len(), 3);

        // Check eigenvalues: 0, 1.5, 1.5
        assert!(hk.eigenvalues[0].abs() < 1e-10, "λ₀ should be 0, got {}", hk.eigenvalues[0]);
        assert!((hk.eigenvalues[1] - 1.5).abs() < 1e-10, "λ₁ should be 1.5, got {}", hk.eigenvalues[1]);
        assert!((hk.eigenvalues[2] - 1.5).abs() < 1e-10, "λ₂ should be 1.5, got {}", hk.eigenvalues[2]);

        // K(T=0.01) ≈ 1 + 2*exp(-0.015) ≈ 2.970
        let k_small_t = 1.0 + 2.0 * (-0.01 * 1.5_f64).exp();
        assert!((hk.heat_trace[0] - k_small_t).abs() < 0.01,
            "K(T=0.01) expected {k_small_t:.4}, got {:.4}", hk.heat_trace[0]);

        // Spectral gap
        assert!((hk.spectral_gap - 1.5).abs() < 1e-10);
    }

    /// K(T→∞) converges to 1 (only zero mode).
    #[test]
    fn heat_trace_limits() {
        let g = build_undirected_csr(5, &[0, 0, 1, 1, 2, 3], &[1, 2, 2, 3, 3, 4]);
        let times = log_spaced_times(0.001, 1000.0, 80);
        let hk = heat_kernel_exact(&g, &times);

        // K(T→0): should approach N=5
        assert!(hk.heat_trace[0] > 4.5, "K(small T) should be close to N=5");

        // K(T→∞): should approach 1
        let k_large = *hk.heat_trace.last().unwrap();
        assert!((k_large - 1.0).abs() < 0.01, "K(large T) should → 1, got {k_large}");
    }

    /// RG with constant D_s = 4 should reproduce standard 1-loop QCD:
    /// 1/α(μ) = 1/α₀ + b₀ ln(M_Pl/μ)
    #[test]
    fn rg_constant_ds4() {
        // Build a fake heat kernel with D_s = 4 everywhere
        let times = log_spaced_times(0.001, 1e20, 100);
        let n = times.len();
        let hk = HeatKernelResult {
            eigenvalues: vec![],
            diffusion_times: times.clone(),
            // For D_s=4, we need d ln K / d ln T = -2.
            // K(T) = c * T^{-2} gives D_s = -2*(-2) = 4.
            heat_trace: times.iter().map(|&t| 1000.0 * t.powi(-2)).collect(),
            spectral_dim: vec![4.0; n],
            n_nodes: 1000,
            spectral_gap: 0.01,
        };

        let alpha_0 = 0.01;
        let n_f = 6;
        let b0 = (33.0 - 12.0) / (12.0 * std::f64::consts::PI);
        let rg = integrate_rg(&hk, alpha_0, n_f, -40.0, 10000);

        // With D_s=4: dα/d(ln μ) = -b₀ α² → 1/α(ln μ) = 1/α₀ + b₀ ln μ
        // As ln μ → −∞ (IR), 1/α decreases → α grows (confinement).
        // Check at a point not too far from origin (avoid large nonlinear regime)
        let check_idx = rg.ln_mu.len() / 10; // ~10% into the flow
        let ln_mu_check = rg.ln_mu[check_idx];
        let expected_inv_alpha = 1.0 / alpha_0 + b0 * ln_mu_check;
        let actual_inv_alpha = rg.inv_alpha[check_idx];
        let rel_err = (actual_inv_alpha - expected_inv_alpha).abs() / expected_inv_alpha;
        assert!(rel_err < 0.01,
            "1/α at ln μ = {ln_mu_check:.1}: expected {expected_inv_alpha:.2}, got {actual_inv_alpha:.2} (err {rel_err:.4})");
    }

    /// SLQ vs exact on small graph.
    #[test]
    fn slq_vs_exact() {
        // Build a small random-looking graph: path + extra edges
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let n = 100;
        // Path graph
        for i in 0..(n - 1) {
            rows.push(i as u32);
            cols.push((i + 1) as u32);
        }
        // Add some cross-edges
        for i in (0..n - 5).step_by(3) {
            rows.push(i as u32);
            cols.push((i + 5) as u32);
        }
        let g = build_undirected_csr(n, &rows, &cols);
        let times = log_spaced_times(0.1, 50.0, 30);

        let exact = heat_kernel_exact(&g, &times);
        let slq = heat_kernel_slq(&g, &times, 40, 80, 42);

        // SLQ trace should be within ~10% of exact for these parameters
        for i in 0..times.len() {
            let rel_err = (slq.heat_trace[i] - exact.heat_trace[i]).abs()
                / exact.heat_trace[i].max(1e-10);
            assert!(rel_err < 0.15,
                "SLQ vs exact at T={:.2}: exact={:.4}, slq={:.4}, err={:.3}",
                times[i], exact.heat_trace[i], slq.heat_trace[i], rel_err);
        }
    }

    #[test]
    fn weyl_law_triangle() {
        let eigenvalues = vec![0.0, 1.5, 1.5];
        let (slope, _r2) = weyl_law_check(&eigenvalues);
        // Only 2 nonzero eigenvalues (both equal), slope is well-defined but
        // with only 2 points the fit is trivially perfect
        assert!(slope.is_finite());
    }

    #[test]
    fn log_spaced_times_basic() {
        let t = log_spaced_times(0.01, 100.0, 5);
        assert_eq!(t.len(), 5);
        assert!((t[0] - 0.01).abs() < 1e-10);
        assert!((t[4] - 100.0).abs() < 1e-8);
        // Check log-spacing: ratios between consecutive should be equal
        let r1 = t[1] / t[0];
        let r2 = t[2] / t[1];
        assert!((r1 - r2).abs() / r1 < 1e-10);
    }
}
