// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Phase 2 topology summary and ensemble aggregation.
//!
//! [`TopologySummary`] captures all prism statistics, generation counts,
//! phase-coherence mass decomposition, and intermediate phase census from
//! a single realisation.  [`aggregate_topology`] merges summaries across
//! M realisations for ensemble averaging.

use serde::{Serialize, Deserialize};

/// Aggregate topology data from Phase 2 (prism detection + classification).
///
/// Exported alongside spectral results so that output writers can serialize
/// `topology_summary.csv` and `mass_spectrum.csv` without reaching back
/// into the defect graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologySummary {
    pub total_nodes: usize,
    pub total_prisms: usize,
    pub max_intermediates: usize,
    pub count_gen1: usize,
    pub count_gen2: usize,
    pub count_gen3: usize,
    pub count_antigen1: usize,
    /// Number of sterile prism nodes (Phi = 0, fully phase-cancelled).
    pub count_sterile: usize,
    pub avg_mass_gen1: f64,
    pub avg_mass_gen2: f64,
    pub avg_mass_gen3: f64,
    /// Average topological mass for sterile prisms.
    pub avg_mass_sterile: f64,
    /// Histogram of committed prisms by belly size: (N_intermediates, frequency).
    pub prism_histogram: Vec<(usize, usize)>,
    /// Phase-coherence mass decomposition (Theorem: zero free parameters).
    /// Sum |Phi(P)| — total visible (EM) mass across all prisms.
    pub visible_mass_total: usize,
    /// Sum (N - |Phi(P)|) — total dark mass across all prisms.
    pub dark_mass_total: usize,
    /// Sum N — total gravitational mass across all prisms.
    pub grav_mass_total: usize,
    /// Omega_dark / Omega_vis = Sum(N - |Phi|) / Sum|Phi| — linear mass ratio.
    pub omega_ratio: f64,
    /// Sum |Phi(P)|^2 — total EM self-energy (numerator of alpha).
    pub phase_sq_total: usize,
    /// Sum N^2 — total gravitational self-energy (denominator of alpha).
    pub mass_sq_total: usize,
    /// alpha = Q_topo / (8 pi) — emergent fine structure constant.
    pub alpha_em: f64,
    /// Omega_energy = 1/Q_topo - 1 = (Sum N^2 - Sum|Phi|^2) / Sum|Phi|^2
    /// — self-energy dark matter ratio.
    /// Satisfies the exact identity alpha(1 + Omega_energy) = 1/(8 pi).
    pub omega_energy: f64,
    /// Intermediate phase census: count of intermediate nodes with phi = +1 (source-like).
    pub phase_pos_count: usize,
    /// Intermediate phase census: count of intermediate nodes with phi = 0 (balanced).
    pub phase_zero_count: usize,
    /// Intermediate phase census: count of intermediate nodes with phi = -1 (sink-like).
    pub phase_neg_count: usize,
    /// Number of prisms per generation (prism count, not node count).
    pub prisms_gen1: usize,
    pub prisms_gen2: usize,
    pub prisms_gen3: usize,
    /// Number of K_5 minors detected during threat contraction.
    pub k5_count: usize,
    /// Mean Z_5 tree level of K_5 threat nodes.
    pub mean_k5_z5_level: f64,
}

/// Aggregate topology summaries across M realisations.
///
/// Sums prism counts and histogram frequencies, takes max of max_intermediates,
/// and averages generation abundances and masses.  Phase-coherence totals are
/// summed across realisations and ratios are recomputed from the sums.
pub fn aggregate_topology(topos: &[TopologySummary]) -> TopologySummary {
    let m = topos.len();
    if m == 1 {
        return topos[0].clone();
    }

    let mut hist_map: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut total_prisms = 0usize;
    let mut max_inter = 0usize;
    for t in topos {
        total_prisms += t.total_prisms;
        max_inter = max_inter.max(t.max_intermediates);
        for &(n, freq) in &t.prism_histogram {
            *hist_map.entry(n).or_insert(0) += freq;
        }
    }
    let mut hist: Vec<(usize, usize)> = hist_map.into_iter().collect();
    hist.sort_unstable_by_key(|&(n, _)| n);

    // Phase-coherence: SUM across realizations, then recompute ratios
    let vis_total: usize = topos.iter().map(|t| t.visible_mass_total).sum();
    let dark_total: usize = topos.iter().map(|t| t.dark_mass_total).sum();
    let grav_total: usize = topos.iter().map(|t| t.grav_mass_total).sum();
    let psq_total: usize = topos.iter().map(|t| t.phase_sq_total).sum();
    let msq_total: usize = topos.iter().map(|t| t.mass_sq_total).sum();
    let omega = if vis_total > 0 {
        dark_total as f64 / vis_total as f64
    } else {
        f64::INFINITY
    };
    let q_topo = if msq_total > 0 {
        psq_total as f64 / msq_total as f64
    } else {
        0.0
    };
    let alpha = q_topo / (8.0 * std::f64::consts::PI);
    let omega_energy = if q_topo > 0.0 {
        1.0 / q_topo - 1.0
    } else {
        f64::INFINITY
    };

    TopologySummary {
        total_nodes: topos[0].total_nodes,
        total_prisms,
        max_intermediates: max_inter,
        count_gen1: topos.iter().map(|t| t.count_gen1).sum::<usize>() / m,
        count_gen2: topos.iter().map(|t| t.count_gen2).sum::<usize>() / m,
        count_gen3: topos.iter().map(|t| t.count_gen3).sum::<usize>() / m,
        count_antigen1: topos.iter().map(|t| t.count_antigen1).sum::<usize>() / m,
        count_sterile: topos.iter().map(|t| t.count_sterile).sum::<usize>() / m,
        avg_mass_gen1: topos.iter().map(|t| t.avg_mass_gen1).sum::<f64>() / m as f64,
        avg_mass_gen2: topos.iter().map(|t| t.avg_mass_gen2).sum::<f64>() / m as f64,
        avg_mass_gen3: topos.iter().map(|t| t.avg_mass_gen3).sum::<f64>() / m as f64,
        avg_mass_sterile: topos.iter().map(|t| t.avg_mass_sterile).sum::<f64>() / m as f64,
        prism_histogram: hist,
        visible_mass_total: vis_total,
        dark_mass_total: dark_total,
        grav_mass_total: grav_total,
        omega_ratio: omega,
        phase_sq_total: psq_total,
        mass_sq_total: msq_total,
        alpha_em: alpha,
        omega_energy,
        phase_pos_count: topos.iter().map(|t| t.phase_pos_count).sum(),
        phase_zero_count: topos.iter().map(|t| t.phase_zero_count).sum(),
        phase_neg_count: topos.iter().map(|t| t.phase_neg_count).sum(),
        prisms_gen1: topos.iter().map(|t| t.prisms_gen1).sum::<usize>() / m,
        prisms_gen2: topos.iter().map(|t| t.prisms_gen2).sum::<usize>() / m,
        prisms_gen3: topos.iter().map(|t| t.prisms_gen3).sum::<usize>() / m,
        k5_count: topos.iter().map(|t| t.k5_count).sum(),
        mean_k5_z5_level: topos.iter().map(|t| t.mean_k5_z5_level).sum::<f64>() / m as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_topo(total_prisms: usize, vis: usize, dark: usize) -> TopologySummary {
        TopologySummary {
            total_nodes: 1000,
            total_prisms,
            max_intermediates: 5,
            count_gen1: 10,
            count_gen2: 5,
            count_gen3: 2,
            count_antigen1: 3,
            count_sterile: 1,
            avg_mass_gen1: 3.0,
            avg_mass_gen2: 4.0,
            avg_mass_gen3: 5.0,
            avg_mass_sterile: 3.5,
            prism_histogram: vec![(3, total_prisms)],
            visible_mass_total: vis,
            dark_mass_total: dark,
            grav_mass_total: vis + dark,
            omega_ratio: dark as f64 / vis.max(1) as f64,
            phase_sq_total: vis * vis,
            mass_sq_total: (vis + dark) * (vis + dark),
            alpha_em: 0.0,
            omega_energy: 0.0,
            phase_pos_count: 10,
            phase_zero_count: 5,
            phase_neg_count: 3,
            prisms_gen1: 6,
            prisms_gen2: 3,
            prisms_gen3: 1,
            k5_count: 0,
            mean_k5_z5_level: 0.0,
        }
    }

    #[test]
    fn single_topology_passthrough() {
        let t = make_topo(10, 20, 30);
        let agg = aggregate_topology(&[t.clone()]);
        assert_eq!(agg.total_prisms, 10);
        assert_eq!(agg.visible_mass_total, 20);
    }

    #[test]
    fn two_realisations_sum_prisms() {
        let t1 = make_topo(10, 20, 30);
        let t2 = make_topo(15, 25, 35);
        let agg = aggregate_topology(&[t1, t2]);
        assert_eq!(agg.total_prisms, 25);
        assert_eq!(agg.visible_mass_total, 45);
        assert_eq!(agg.dark_mass_total, 65);
    }
}
