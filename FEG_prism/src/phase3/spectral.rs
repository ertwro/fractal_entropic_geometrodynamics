// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Spectral dimension types and the central d_S(t) finite-difference routine.
//!
//! This module defines the composed output types [`WalkResult`] and
//! [`SpectralOutput`] that replace the monolithic `SpectralResult` from the
//! original simulation.  The spectral dimension formula
//!
//!   d_S(t) = -2 d(ln P) / d(ln t)
//!
//! is computed via centred finite differences, with forward/backward
//! differences at the endpoints.

use serde::{Deserialize, Serialize};

/// Return probability P(t), spectral dimension d_S(t), and optional ensemble
/// standard deviation for a single walk category (global, core, generation, etc.).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WalkResult {
    /// P(t) return probability at each measurement step.
    pub p: Vec<f64>,
    /// d_S(t) spectral dimension derived from P(t).
    pub ds: Vec<f64>,
    /// Ensemble standard deviation of d_S across M realisations.
    /// Empty (`vec![]`) when M = 1.
    pub ds_std: Vec<f64>,
}

/// Composed spectral output for a complete simulation run.
///
/// Replaces the flat `SpectralResult` struct with logically grouped
/// [`WalkResult`] fields for vacuum, defect, generations, sterile,
/// and causal flux observables.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpectralOutput {
    /// Vacuum global: walkers start from uniformly random positions on the
    /// vacuum (Hasse) graph.
    pub vacuum: WalkResult,
    /// Vacuum core (local): walkers start exclusively on core nodes
    /// (combinatorial centre of the Alexandrov diamond).
    pub vac_core: WalkResult,
    /// Defect global: walkers on the Kuratowski-contracted graph.
    pub defect: WalkResult,
    /// Defect core (local): walkers starting on core nodes of the defect graph.
    pub def_core: WalkResult,

    /// Per-generation spectral data: [gen1, gen2, gen3, anti1].
    ///
    /// - gen1: most abundant Causal Prism topological signature (electron-like)
    /// - gen2: second most abundant (muon-like)
    /// - gen3: third most abundant (tau-like)
    /// - anti1: CPT-conjugate of gen1 (positron-like)
    pub generations: [WalkResult; 4],
    /// Sterile Prism spectral data (phase flux Phi = 0).
    /// Gravitationally active but electromagnetically silent (dark matter).
    pub sterile: WalkResult,

    /// Causal flux attraction: directed transmission Gen1 -> AntiGen1
    /// (opposite charge attraction).
    pub flux_attr: WalkResult,
    /// Causal flux repulsion: directed transmission Gen1 -> Gen1
    /// (same charge repulsion).
    pub flux_repu: WalkResult,

    /// Normalized attraction flux: flux_attr / |targets_attraction|.
    /// Isolates intrinsic coupling per unit charge (E = F/q).
    pub flux_attr_norm: Vec<f64>,
    /// Normalized repulsion flux: flux_repu / |targets_repulsion|.
    pub flux_repu_norm: Vec<f64>,

    /// Average topological mass (N = intermediate nodes in Causal Prism)
    /// for [gen1, gen2, gen3, anti1].
    pub mass: [f64; 4],
}

/// Compute spectral dimension d_S(t) = -2 d(ln P)/d(ln t) via centred
/// finite differences.
///
/// Uses forward difference at i = 0, backward difference at i = n-1,
/// and centred difference for all interior points.
///
/// # Arguments
/// * `steps` - Measurement times t (must be positive integers).
/// * `p_vals` - Return probabilities P(t) at each step.
///
/// # Returns
/// Vector of d_S values, same length as `steps`.
pub fn spectral_dimension(steps: &[u32], p_vals: &[f64]) -> Vec<f64> {
    let n = steps.len();
    if n < 2 {
        return vec![0.0; n];
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_dimension_single_point() {
        let ds = spectral_dimension(&[1], &[0.5]);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds[0], 0.0);
    }

    #[test]
    fn spectral_dimension_empty() {
        let ds = spectral_dimension(&[], &[]);
        assert!(ds.is_empty());
    }

    #[test]
    fn spectral_dimension_three_points() {
        // P(t) = 1/t  =>  ln P = -ln t  =>  d(ln P)/d(ln t) = -1  =>  d_S = 2
        let steps = [1, 2, 4];
        let p_vals = [1.0, 0.5, 0.25];
        let ds = spectral_dimension(&steps, &p_vals);
        assert_eq!(ds.len(), 3);
        for &d in &ds {
            assert!((d - 2.0).abs() < 1e-10, "expected ~2.0, got {d}");
        }
    }

    #[test]
    fn walk_result_default() {
        let wr = WalkResult::default();
        assert!(wr.p.is_empty());
        assert!(wr.ds.is_empty());
        assert!(wr.ds_std.is_empty());
    }

    #[test]
    fn spectral_output_default() {
        let so = SpectralOutput::default();
        assert!(so.vacuum.p.is_empty());
        assert!(so.flux_attr_norm.is_empty());
        assert_eq!(so.mass, [0.0; 4]);
    }
}
