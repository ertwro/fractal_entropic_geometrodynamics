// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M14 — Exact Mass Formula Verification
//!
//! Computes the genus of every K_{2,n} prism via the Grothendieck-Euler
//! formula (phase2/writhe.rs) and verifies the exact mass formula:
//!
//!   M(g) = M_e × 2^g × (32π)^g / C(max(3, 2g+1), 3)
//!
//! where g is the genus (0 = electron, 1 = muon, 2 = tau) and C(n,k)
//! is the binomial coefficient.
//!
//! Zero free parameters.  Predictions:
//!   m_μ/m_e = 201.06   (SM: 206.77, -2.76%)
//!   m_τ/m_e = 4042.6   (SM: 3477.5, +16.2%)

use super::context::MeasureContext;
use crate::output::CsvWriter;
use crate::phase2::writhe::compute_writhe;

const PI: f64 = std::f64::consts::PI;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MassFormulaResult {
    /// Number of prisms at each genus level [g=0, g=1, g=2, g>=3].
    pub genus_counts: [usize; 4],
    /// Mean belly size per genus [g=0, g=1, g=2].
    pub mean_belly: [f64; 3],
    /// Predicted mass ratios: [m_μ/m_e, m_τ/m_e].
    pub predicted_ratios: [f64; 2],
    /// Total prisms scanned.
    pub total_prisms: usize,
}

/// Binomial coefficient C(n, k) for small n.
fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Exact mass ratio M(g) / M_e from the genus ladder formula.
///
/// M(g) = M_e × 2^g × (32π)^g / C(max(3, 2g+1), 3)
pub fn mass_ratio(g: usize) -> f64 {
    if g == 0 {
        return 1.0;
    }
    let two_g = 2f64.powi(g as i32);
    let alpha_inv_g = (32.0 * PI).powi(g as i32);
    let n_min = (2 * g + 1).max(3);
    let denom = binom(n_min, 3) as f64;
    if denom == 0.0 {
        return 0.0;
    }
    two_g * alpha_inv_g / denom
}

pub fn run(ctx: &MeasureContext) -> MassFormulaResult {
    let mut genus_counts = [0usize; 4];
    let mut belly_sum = [0.0f64; 3];
    let mut belly_count = [0usize; 3];

    for prism in ctx.prisms {
        if prism.intermediates.len() < 2 {
            continue;
        }

        let stats = compute_writhe(prism, ctx.sorted_coords);
        let g = stats.genus;
        let belly = stats.belly_size;

        if g <= 2 {
            genus_counts[g] += 1;
            belly_sum[g] += belly as f64;
            belly_count[g] += 1;
        } else {
            genus_counts[3] += 1;
        }
    }

    let mean_belly = [
        if belly_count[0] > 0 { belly_sum[0] / belly_count[0] as f64 } else { 0.0 },
        if belly_count[1] > 0 { belly_sum[1] / belly_count[1] as f64 } else { 0.0 },
        if belly_count[2] > 0 { belly_sum[2] / belly_count[2] as f64 } else { 0.0 },
    ];

    let predicted_ratios = [mass_ratio(1), mass_ratio(2)];
    let total = genus_counts.iter().sum();

    MassFormulaResult {
        genus_counts,
        mean_belly,
        predicted_ratios,
        total_prisms: total,
    }
}

pub fn aggregate(results: &[MassFormulaResult]) -> MassFormulaResult {
    let mut genus_counts = [0usize; 4];
    let mut total_prisms = 0usize;
    let mut belly_wsum = [0.0f64; 3];
    let mut belly_wcount = [0usize; 3];

    for r in results {
        for i in 0..4 {
            genus_counts[i] += r.genus_counts[i];
        }
        total_prisms += r.total_prisms;
        for i in 0..3 {
            if r.genus_counts[i] > 0 {
                belly_wsum[i] += r.mean_belly[i] * r.genus_counts[i] as f64;
                belly_wcount[i] += r.genus_counts[i];
            }
        }
    }

    let mean_belly = [
        if belly_wcount[0] > 0 { belly_wsum[0] / belly_wcount[0] as f64 } else { 0.0 },
        if belly_wcount[1] > 0 { belly_wsum[1] / belly_wcount[1] as f64 } else { 0.0 },
        if belly_wcount[2] > 0 { belly_wsum[2] / belly_wcount[2] as f64 } else { 0.0 },
    ];

    MassFormulaResult {
        genus_counts,
        mean_belly,
        predicted_ratios: [mass_ratio(1), mass_ratio(2)],
        total_prisms,
    }
}

pub fn write_csv(result: &MassFormulaResult, w: &mut CsvWriter) {
    w.comment("M14 Exact Mass Formula (genus-ladder, zero free parameters)");
    w.header(&["key", "value"]);
    w.row_fmt(format_args!("total_prisms,{}", result.total_prisms));
    w.row_fmt(format_args!("genus_0_count,{}", result.genus_counts[0]));
    w.row_fmt(format_args!("genus_1_count,{}", result.genus_counts[1]));
    w.row_fmt(format_args!("genus_2_count,{}", result.genus_counts[2]));
    w.row_fmt(format_args!("genus_3plus_count,{}", result.genus_counts[3]));
    w.row_fmt(format_args!("mean_belly_g0,{:.2}", result.mean_belly[0]));
    w.row_fmt(format_args!("mean_belly_g1,{:.2}", result.mean_belly[1]));
    w.row_fmt(format_args!("mean_belly_g2,{:.2}", result.mean_belly[2]));
    w.row_fmt(format_args!("predicted_mu_e,{:.4}", result.predicted_ratios[0]));
    w.row_fmt(format_args!("predicted_tau_e,{:.4}", result.predicted_ratios[1]));
    w.row_fmt(format_args!("SM_mu_e,206.77"));
    w.row_fmt(format_args!("SM_tau_e,3477.5"));
}

pub fn print_summary(result: &MassFormulaResult) {
    println!("  [M14] Exact Mass Formula:");
    println!("    Prisms scanned:  {}", result.total_prisms);
    println!(
        "    Genus histogram: g=0: {}  g=1: {}  g=2: {}  g>=3: {}",
        result.genus_counts[0], result.genus_counts[1],
        result.genus_counts[2], result.genus_counts[3]
    );
    println!(
        "    Mean belly:      g=0: {:.1}  g=1: {:.1}  g=2: {:.1}",
        result.mean_belly[0], result.mean_belly[1], result.mean_belly[2]
    );
    println!("    Predicted m_mu/m_e:  {:.2}  (SM: 206.77)", result.predicted_ratios[0]);
    println!("    Predicted m_tau/m_e: {:.1}  (SM: 3477.5)", result.predicted_ratios[1]);
}
