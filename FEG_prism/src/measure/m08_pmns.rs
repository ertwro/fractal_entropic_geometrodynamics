// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M8 — PMNS Mixing Matrix (Layered Front Propagation)
//!
//! For each open prism (neutrino) detected by M7, a BFS wavefront is
//! propagated through the directed vacuum CSR.  At each causal depth d,
//! the wavefront width determines the instantaneous generation (flavor).
//! Tracking generation transitions layer-by-layer gives the discrete
//! topological analogue of |U_{PMNS}|^2.
//!
//! Calculo de Kuratowski, Vol II, section 8: flavor oscillation from wavefront width.

use super::context::MeasureContext;
use super::m07_neutrino::NeutrinoResult;
use crate::output::CsvWriter;
use std::collections::HashSet;

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PMNSResult {
    pub transition_matrix: [[f64; 3]; 3],
    pub raw_counts: [[usize; 3]; 3],
    pub total_transitions: usize,
    pub max_depth: usize,
    pub mean_survival_depth: f64,
    pub depth_gen_fractions: Vec<[f64; 3]>,
    pub depth_mean_width: Vec<f64>,
    pub depth_survival: Vec<f64>,
    pub n_neutrinos: usize,
}

// ── Utilities ────────────────────────────────────────────────────────────────

/// Classify wavefront width -> generation (same thresholds as M7).
fn classify_wavefront_gen(width: usize) -> u8 {
    if width < 3 { 0 }
    else if width <= 4 { 1 }
    else if width <= 6 { 2 }
    else { 3 }
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext, neutrino: &NeutrinoResult) -> PMNSResult {
    let n = ctx.n_points;
    let (vac_head, vac_data) = ctx.vacuum_csr.raw();
    let max_depth = 10usize;

    // Step 1: Build is_placed from committed closed prisms
    let mut is_placed = vec![false; n];
    for p in ctx.prisms {
        if p.origin < n { is_placed[p.origin] = true; }
        if p.destination < n { is_placed[p.destination] = true; }
        for &w in &p.intermediates {
            if w < n { is_placed[w] = true; }
        }
    }

    let mut raw_counts = [[0usize; 3]; 3];
    let mut total_transitions = 0usize;

    // Per-depth accumulators
    let mut depth_width_sums = vec![0.0f64; max_depth];
    let mut depth_gen_counts = vec![[0usize; 3]; max_depth];
    let mut depth_alive = vec![0usize; max_depth];
    let mut survival_depths = Vec::new();
    let mut n_valid = 0usize;

    // Step 2: For each neutrino candidate, propagate wavefront
    for cand in &neutrino.candidates {
        let origin = cand.origin;
        if origin >= n { continue; }

        let mut visited = HashSet::new();
        visited.insert(origin);

        // Layer 1: children of origin, filtered
        let s = vac_head[origin] as usize;
        let e = vac_head[origin + 1] as usize;
        let mut layer: Vec<usize> = Vec::new();
        for &v in &vac_data[s..e] {
            let vi = v as usize;
            if vi < n && !is_placed[vi] && !visited.contains(&vi) {
                layer.push(vi);
            }
        }
        layer.sort_unstable();
        layer.dedup();
        for &v in &layer { visited.insert(v); }

        if layer.len() < 3 { continue; }
        n_valid += 1;

        let mut prev_gen = classify_wavefront_gen(layer.len());
        if prev_gen == 0 { continue; }

        // Record depth 0 (layer 1)
        depth_width_sums[0] += layer.len() as f64;
        depth_gen_counts[0][(prev_gen - 1) as usize] += 1;
        depth_alive[0] += 1;

        let mut final_depth = 1usize;

        // Propagate layers 2..=max_depth
        for d in 1..max_depth {
            let mut child_parent_count: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();

            for &w in &layer {
                let ws = vac_head[w] as usize;
                let we = vac_head[w + 1] as usize;
                for &v in &vac_data[ws..we] {
                    let vi = v as usize;
                    if vi < n && !is_placed[vi] && !visited.contains(&vi) {
                        *child_parent_count.entry(vi).or_insert(0) += 1;
                    }
                }
            }

            // Convergence filter: keep children with < 3 parents in current layer
            let mut next_layer: Vec<usize> = child_parent_count
                .into_iter()
                .filter(|&(_, count)| count < 3)
                .map(|(v, _)| v)
                .collect();
            next_layer.sort_unstable();
            next_layer.dedup();

            for &v in &next_layer { visited.insert(v); }

            if next_layer.len() < 3 {
                final_depth = d;
                break;
            }

            let curr_gen = classify_wavefront_gen(next_layer.len());
            if curr_gen == 0 {
                final_depth = d;
                break;
            }

            // Record transition
            raw_counts[(prev_gen - 1) as usize][(curr_gen - 1) as usize] += 1;
            total_transitions += 1;

            // Record depth statistics
            depth_width_sums[d] += next_layer.len() as f64;
            depth_gen_counts[d][(curr_gen - 1) as usize] += 1;
            depth_alive[d] += 1;

            prev_gen = curr_gen;
            layer = next_layer;
            final_depth = d + 1;
        }

        survival_depths.push(final_depth);
    }

    // Step 3: Normalize raw_counts -> transition_matrix (row-stochastic)
    let mut transition_matrix = [[0.0f64; 3]; 3];
    for i in 0..3 {
        let row_sum: usize = raw_counts[i].iter().sum();
        if row_sum > 0 {
            for j in 0..3 {
                transition_matrix[i][j] = raw_counts[i][j] as f64 / row_sum as f64;
            }
        }
    }

    // Step 4: Compute depth statistics
    let mean_survival_depth = if survival_depths.is_empty() {
        0.0
    } else {
        survival_depths.iter().sum::<usize>() as f64 / survival_depths.len() as f64
    };

    let n_valid_f = n_valid.max(1) as f64;
    let depth_mean_width: Vec<f64> = depth_width_sums.iter().enumerate().map(|(d, &sum)| {
        if depth_alive[d] > 0 { sum / depth_alive[d] as f64 } else { 0.0 }
    }).collect();

    let depth_gen_fractions: Vec<[f64; 3]> = depth_gen_counts.iter().enumerate().map(|(d, counts)| {
        let total = depth_alive[d] as f64;
        if total > 0.0 {
            [counts[0] as f64 / total, counts[1] as f64 / total, counts[2] as f64 / total]
        } else {
            [0.0; 3]
        }
    }).collect();

    let depth_survival: Vec<f64> = depth_alive.iter().map(|&alive| {
        alive as f64 / n_valid_f
    }).collect();

    PMNSResult {
        transition_matrix,
        raw_counts,
        total_transitions,
        max_depth,
        mean_survival_depth,
        depth_gen_fractions,
        depth_mean_width,
        depth_survival,
        n_neutrinos: neutrino.candidates.len(),
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[PMNSResult]) -> PMNSResult {
    let m = results.len() as f64;
    let mut avg_matrix = [[0.0f64; 3]; 3];
    let mut sum_counts = [[0usize; 3]; 3];
    for p in results {
        for i in 0..3 { for j in 0..3 {
            avg_matrix[i][j] += p.transition_matrix[i][j];
            sum_counts[i][j] += p.raw_counts[i][j];
        }}
    }
    for i in 0..3 { for j in 0..3 { avg_matrix[i][j] /= m; } }
    // Re-normalize the averaged matrix
    for i in 0..3 {
        let row_sum: f64 = avg_matrix[i].iter().sum();
        if row_sum > 0.0 {
            for j in 0..3 { avg_matrix[i][j] /= row_sum; }
        }
    }
    PMNSResult {
        transition_matrix: avg_matrix,
        raw_counts: sum_counts,
        total_transitions: results.iter().map(|p| p.total_transitions).sum(),
        max_depth: results[0].max_depth,
        mean_survival_depth: results.iter().map(|p| p.mean_survival_depth).sum::<f64>() / m,
        depth_gen_fractions: vec![],
        depth_mean_width: vec![],
        depth_survival: vec![],
        n_neutrinos: (results.iter().map(|p| p.n_neutrinos).sum::<usize>() as f64 / m) as usize,
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &PMNSResult, w: &mut CsvWriter) {
    w.comment("M8 PMNS Mixing Matrix (layered front propagation)");
    w.header(&["from_gen", "to_gen", "raw_count", "probability"]);
    for i in 0..3 {
        for j in 0..3 {
            w.row_fmt(format_args!(
                "{},{},{},{:.6}",
                i + 1, j + 1, result.raw_counts[i][j], result.transition_matrix[i][j]
            ));
        }
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &PMNSResult) {
    println!("  [M8] PMNS Mixing Matrix:");
    println!("    Neutrinos:       {}", result.n_neutrinos);
    println!("    Transitions:     {}", result.total_transitions);
    println!("    Survival depth:  {:.2}", result.mean_survival_depth);
    println!("    |U_PMNS|^2:");
    for i in 0..3 {
        println!("      [{:.4}  {:.4}  {:.4}]",
            result.transition_matrix[i][0],
            result.transition_matrix[i][1],
            result.transition_matrix[i][2]);
    }
}
