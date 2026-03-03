// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M1 — Traversal Mass Ratios (Prism-Confined Random Walk)
//!
//! Walkers start at prism origins and perform a lazy random walk strictly
//! confined to the prism's internal subgraph (using the defect CSR).
//! Traversal time from the origin pole to the destination pole directly
//! measures the topological mass delay (belly size).
//!
//! Calculo de Kuratowski, Vol II, Def 3.1: topological mass = N.

use super::context::MeasureContext;
use crate::convergence::{AutoConverge, ConvergeState};
use crate::output::CsvWriter;
use crate::phase2::defect::CausalPrism;
use crate::phase3::walker::distribute_walkers;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TraversalRecord {
    pub prism_idx: usize,
    pub generation: u8,
    pub n_belly: usize,
    pub traversal_ticks: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TraversalMassResult {
    pub mean_traversal: [f64; 3],
    pub ratio_gen2_gen1: f64,
    pub ratio_gen3_gen1: f64,
    pub n_traversals: [usize; 3],
    pub records: Vec<TraversalRecord>,
}

// ── Utilities ────────────────────────────────────────────────────────────────

/// Chase the merge-into contraction map until reaching a fixed point.
#[inline]
fn resolve(node: usize, merge: &[usize]) -> usize {
    let mut cur = node;
    while merge[cur] != cur {
        cur = merge[cur];
    }
    cur
}

/// Build lookup table: node -> generation (1/2/3/4=anti1, 0=unclassified).
fn build_gen_lookup(n: usize, ctx: &MeasureContext) -> Vec<u8> {
    let mut lookup = vec![0u8; n];
    for &node in &ctx.defect.generations.gen1 {
        if node < n { lookup[node] = 1; }
    }
    for &node in &ctx.defect.generations.gen2 {
        if node < n { lookup[node] = 2; }
    }
    for &node in &ctx.defect.generations.gen3 {
        if node < n { lookup[node] = 3; }
    }
    for &node in &ctx.defect.generations.anti1 {
        if node < n { lookup[node] = 4; }
    }
    lookup
}

/// Classify a prism's generation by checking which gen list its nodes belong to.
fn classify_prism_generation(prism: &CausalPrism, gen_lookup: &[u8]) -> u8 {
    for &node in &prism.intermediates {
        let g = gen_lookup[node];
        if g >= 1 && g <= 3 {
            return g;
        }
    }
    let g = gen_lookup[prism.origin];
    if g >= 1 && g <= 3 {
        return g;
    }
    let g = gen_lookup[prism.destination];
    if g >= 1 && g <= 3 {
        return g;
    }
    0
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> TraversalMassResult {
    let n = ctx.n_points;
    let merge = &ctx.defect.merge_map;
    let gen_lookup = build_gen_lookup(n, ctx);

    // Symmetric defect CSR for walk
    let (sym_def_head, sym_def_data) = ctx.defect.defect_csr.raw();

    struct PrismInfo {
        origin: usize,
        destination: usize,
        generation: u8,
        belly: usize,
        intermediates_resolved: Vec<usize>,
    }

    let prism_info: Vec<PrismInfo> = ctx.prisms
        .iter()
        .map(|p| {
            let origin = resolve(p.origin, merge);
            let dest = resolve(p.destination, merge);
            let gen = classify_prism_generation(p, &gen_lookup);
            let ints: Vec<usize> = p.intermediates.iter()
                .map(|&i| resolve(i, merge))
                .collect();
            PrismInfo {
                origin,
                destination: dest,
                generation: gen,
                belly: p.intermediates.len(),
                intermediates_resolved: ints,
            }
        })
        .collect();

    // Build node -> prism index mapping
    let mut node_to_prism: Vec<Option<usize>> = vec![None; n];
    let mut is_origin: Vec<bool> = vec![false; n];

    for (pi, info) in prism_info.iter().enumerate() {
        node_to_prism[info.origin] = Some(pi);
        is_origin[info.origin] = true;
        node_to_prism[info.destination] = Some(pi);
        for &ri in &info.intermediates_resolved {
            if ri < n {
                node_to_prism[ri] = Some(pi);
            }
        }
    }

    let origins: Vec<usize> = prism_info
        .iter()
        .filter(|p| p.generation >= 1 && p.generation <= 3)
        .map(|p| p.origin)
        .collect();

    if origins.is_empty() {
        return TraversalMassResult {
            mean_traversal: [0.0; 3],
            ratio_gen2_gen1: 0.0,
            ratio_gen3_gen1: 0.0,
            n_traversals: [0; 3],
            records: vec![],
        };
    }

    // Cover time budget: O(N_belly * log(N_belly) * ambient_dilution).
    let max_cover_steps: u32 = 5000;

    let num_def_nodes = sym_def_head.len() - 1;

    // Auto-convergence: batch walkers until mean traversal time stabilises
    let ac = AutoConverge::new(origins.len() * 200, 2048, ctx.epsilon);
    let mut conv_state = ConvergeState::new();
    let mut all_records: Vec<TraversalRecord> = Vec::new();

    loop {
        let starts = distribute_walkers(&origins, ac.batch_size);
        let seed_offset = conv_state.total_walkers;

        let batch_records: Vec<Vec<TraversalRecord>> = starts
            .par_iter()
            .enumerate()
            .map(|(wi, &start_pos)| {
                let mut rng = StdRng::seed_from_u64(
                    ctx.seed.wrapping_add((seed_offset + wi) as u64),
                );
                let mut pos = start_pos;
                let mut local_records = Vec::new();

                let mut in_prism = false;
                let mut current_prism_idx = 0usize;
                let mut entry_tick = 0u32;
                let mut visited_belly: HashSet<usize> = HashSet::new();

                if let Some(pi) = node_to_prism[pos] {
                    if is_origin[pos] {
                        in_prism = true;
                        current_prism_idx = pi;
                        entry_tick = 0;
                        visited_belly.clear();
                    }
                }

                for t in 1..=max_cover_steps {
                    let s = if pos < num_def_nodes { sym_def_head[pos] as usize } else { 0 };
                    let e = if pos < num_def_nodes { sym_def_head[pos + 1] as usize } else { 0 };
                    let deg = e - s;

                    // Lazy walk with strict prism confinement
                    if deg > 0 && rng.gen_bool(0.5) {
                        let candidate_next = sym_def_data[s + rng.gen_range(0..deg)] as usize;
                        let resolved_next = resolve(candidate_next, merge);

                        if in_prism {
                            if node_to_prism.get(resolved_next) == Some(&Some(current_prism_idx)) {
                                pos = resolved_next;
                                let info = &prism_info[current_prism_idx];
                                if pos != info.origin && pos != info.destination {
                                    visited_belly.insert(pos);
                                }
                            }
                        } else {
                            pos = resolved_next;
                        }
                    }

                    if in_prism {
                        let info = &prism_info[current_prism_idx];
                        let at_dest = pos == info.destination;

                        if at_dest && visited_belly.len() >= info.belly {
                            local_records.push(TraversalRecord {
                                prism_idx: current_prism_idx,
                                generation: info.generation,
                                n_belly: info.belly,
                                traversal_ticks: t - entry_tick,
                            });
                            in_prism = false;
                            visited_belly.clear();
                        } else if at_dest {
                            pos = info.origin;
                        }
                    } else if let Some(pi) = node_to_prism.get(pos).copied().flatten() {
                        if is_origin[pos] {
                            in_prism = true;
                            current_prism_idx = pi;
                            entry_tick = t;
                            visited_belly.clear();
                        }
                    }
                }

                local_records
            })
            .collect();

        // Batch-only observable for Welford (independent sample)
        let batch_obs = {
            let batch_flat: Vec<&TraversalRecord> = batch_records.iter()
                .flat_map(|v| v.iter()).collect();
            if !batch_flat.is_empty() {
                batch_flat.iter().map(|r| r.traversal_ticks as f64).sum::<f64>()
                    / batch_flat.len() as f64
            } else {
                0.0
            }
        };

        // Still extend cumulative records for final output
        all_records.extend(batch_records.into_iter().flatten());

        // Guard: do not feed zero-traversal batches to Welford.  If no walker
        // completes a traversal, batch_obs = 0.0.  Three consecutive zeros
        // would collapse the variance to 0, triggering premature convergence
        // with mean_traversal = [0,0,0].  Require at least one successful
        // traversal before updating the convergence check.
        if batch_obs > 0.0 {
            if conv_state.update(batch_obs, &ac) { break; }
        } else {
            // Still count walkers dispatched so the cap fires correctly
            conv_state.total_walkers += ac.batch_size;
        }
        if conv_state.at_limit(&ac) {
            eprintln!("[M1] WARNING: {} walkers without convergence", conv_state.total_walkers);
            break;
        }
    }
    println!("  [M1] converged at {} walkers", conv_state.total_walkers);

    let mut sum = [0.0f64; 3];
    let mut count = [0usize; 3];

    for r in &all_records {
        let g = r.generation as usize;
        if g >= 1 && g <= 3 {
            sum[g - 1] += r.traversal_ticks as f64;
            count[g - 1] += 1;
        }
    }

    let mean = [
        if count[0] > 0 { sum[0] / count[0] as f64 } else { 0.0 },
        if count[1] > 0 { sum[1] / count[1] as f64 } else { 0.0 },
        if count[2] > 0 { sum[2] / count[2] as f64 } else { 0.0 },
    ];

    TraversalMassResult {
        mean_traversal: mean,
        ratio_gen2_gen1: if mean[0] > 0.0 { mean[1] / mean[0] } else { 0.0 },
        ratio_gen3_gen1: if mean[0] > 0.0 { mean[2] / mean[0] } else { 0.0 },
        n_traversals: count,
        records: all_records,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[TraversalMassResult]) -> TraversalMassResult {
    let mut sum = [0.0f64; 3];
    let mut count = [0usize; 3];
    for t in results {
        for i in 0..3 {
            sum[i] += t.mean_traversal[i] * t.n_traversals[i] as f64;
            count[i] += t.n_traversals[i];
        }
    }
    let mean = [
        if count[0] > 0 { sum[0] / count[0] as f64 } else { 0.0 },
        if count[1] > 0 { sum[1] / count[1] as f64 } else { 0.0 },
        if count[2] > 0 { sum[2] / count[2] as f64 } else { 0.0 },
    ];
    TraversalMassResult {
        mean_traversal: mean,
        ratio_gen2_gen1: if mean[0] > 0.0 { mean[1] / mean[0] } else { 0.0 },
        ratio_gen3_gen1: if mean[0] > 0.0 { mean[2] / mean[0] } else { 0.0 },
        n_traversals: count,
        records: vec![],
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &TraversalMassResult, w: &mut CsvWriter) {
    w.comment("M1 Traversal Mass Ratios (prism-confined random walk)");
    w.header(&["prism_idx", "generation", "n_belly", "traversal_ticks"]);
    for r in &result.records {
        w.row_fmt(format_args!("{},{},{},{}", r.prism_idx, r.generation, r.n_belly, r.traversal_ticks));
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &TraversalMassResult) {
    println!("  [M1] Traversal Mass Ratios:");
    println!("    Mean traversal:  Gen1={:.2}  Gen2={:.2}  Gen3={:.2}",
        result.mean_traversal[0], result.mean_traversal[1], result.mean_traversal[2]);
    println!("    N traversals:    Gen1={}  Gen2={}  Gen3={}",
        result.n_traversals[0], result.n_traversals[1], result.n_traversals[2]);
    println!("    Ratios:          m2/m1={:.4}  m3/m1={:.4}",
        result.ratio_gen2_gen1, result.ratio_gen3_gen1);
}
