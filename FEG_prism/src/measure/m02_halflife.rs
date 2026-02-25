// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M2 — Half-Life Census (Cross-Ensemble Stability Statistics)
//!
//! For each prism, records generation, belly size, and net phase.
//! Computes occupancy fractions p(gen_k | N) per belly size
//! and stability ratios tau(gen2)/tau(gen1), tau(gen3)/tau(gen1).
//!
//! Calculo de Kuratowski, Vol II, Thm 5.1: generation persistence.

use super::context::MeasureContext;
use crate::output::CsvWriter;
use crate::phase2::defect::CausalPrism;

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct HalfLifeResult {
    pub prism_census: Vec<(usize, u8, usize, i32)>,  // (idx, gen, belly, phase)
    pub gen_counts: [usize; 4],                       // [gen1, gen2, gen3, anti1]
    pub occupancy_by_belly: Vec<(usize, [f64; 3])>,   // (belly_N, [p_gen1, p_gen2, p_gen3])
    pub stability_ratio_gen2: f64,
    pub stability_ratio_gen3: f64,
}

// ── Utilities ────────────────────────────────────────────────────────────────

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

fn classify_prism_generation(prism: &CausalPrism, gen_lookup: &[u8]) -> u8 {
    for &node in &prism.intermediates {
        let g = gen_lookup[node];
        if g >= 1 && g <= 3 { return g; }
    }
    let g = gen_lookup[prism.origin];
    if g >= 1 && g <= 3 { return g; }
    let g = gen_lookup[prism.destination];
    if g >= 1 && g <= 3 { return g; }
    0
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> HalfLifeResult {
    let n = ctx.n_points;
    let gen_lookup = build_gen_lookup(n, ctx);

    let mut census: Vec<(usize, u8, usize, i32)> = Vec::with_capacity(ctx.prisms.len());
    let mut gen_counts = [0usize; 4];

    for (pi, prism) in ctx.prisms.iter().enumerate() {
        let gen = classify_prism_generation(prism, &gen_lookup);
        let belly = prism.intermediates.len();

        // Net phase: sum of bulk momentum signs for intermediates
        let phase: i32 = prism
            .intermediates
            .iter()
            .filter_map(|&i| ctx.momentum.get(i))
            .map(|&m| m.signum())
            .sum();

        census.push((pi, gen, belly, phase));

        match gen {
            1 => gen_counts[0] += 1,
            2 => gen_counts[1] += 1,
            3 => gen_counts[2] += 1,
            4 => gen_counts[3] += 1,
            _ => {}
        }
    }

    // Build occupancy histogram: per belly size, fraction in each generation
    let mut belly_counts: std::collections::HashMap<usize, [usize; 3]> =
        std::collections::HashMap::new();
    for &(_, gen, belly, _) in &census {
        if gen >= 1 && gen <= 3 {
            belly_counts.entry(belly).or_insert([0; 3])[gen as usize - 1] += 1;
        }
    }

    let mut occupancy: Vec<(usize, [f64; 3])> = belly_counts
        .iter()
        .map(|(&belly, counts)| {
            let total = (counts[0] + counts[1] + counts[2]) as f64;
            if total > 0.0 {
                (
                    belly,
                    [
                        counts[0] as f64 / total,
                        counts[1] as f64 / total,
                        counts[2] as f64 / total,
                    ],
                )
            } else {
                (belly, [0.0; 3])
            }
        })
        .collect();
    occupancy.sort_by_key(|&(b, _)| b);

    // Stability ratios: p(gen2)/p(gen1) and p(gen3)/p(gen1) at shared belly sizes
    let (mut p1_sum, mut p2_sum, mut p3_sum) = (0.0f64, 0.0f64, 0.0f64);
    for &(_, probs) in &occupancy {
        if probs[0] > 0.0 {
            p1_sum += probs[0];
            p2_sum += probs[1];
            p3_sum += probs[2];
        }
    }
    let stability_gen2 = if p1_sum > 0.0 { p2_sum / p1_sum } else { 0.0 };
    let stability_gen3 = if p1_sum > 0.0 { p3_sum / p1_sum } else { 0.0 };

    HalfLifeResult {
        prism_census: census,
        gen_counts,
        occupancy_by_belly: occupancy,
        stability_ratio_gen2: stability_gen2,
        stability_ratio_gen3: stability_gen3,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[HalfLifeResult]) -> HalfLifeResult {
    let mut all_census: Vec<(usize, u8, usize, i32)> = Vec::new();
    let mut gen_counts = [0usize; 4];
    for hl in results {
        all_census.extend_from_slice(&hl.prism_census);
        for i in 0..4 {
            gen_counts[i] += hl.gen_counts[i];
        }
    }

    let mut belly_counts: std::collections::HashMap<usize, [usize; 3]> =
        std::collections::HashMap::new();
    for &(_, gen, belly, _) in &all_census {
        if gen >= 1 && gen <= 3 {
            belly_counts.entry(belly).or_insert([0; 3])[gen as usize - 1] += 1;
        }
    }
    let mut occupancy: Vec<(usize, [f64; 3])> = belly_counts
        .iter()
        .map(|(&belly, counts)| {
            let total = (counts[0] + counts[1] + counts[2]) as f64;
            if total > 0.0 {
                (
                    belly,
                    [
                        counts[0] as f64 / total,
                        counts[1] as f64 / total,
                        counts[2] as f64 / total,
                    ],
                )
            } else {
                (belly, [0.0; 3])
            }
        })
        .collect();
    occupancy.sort_by_key(|&(b, _)| b);

    let (mut p1_sum, mut p2_sum, mut p3_sum) = (0.0f64, 0.0f64, 0.0f64);
    for &(_, probs) in &occupancy {
        if probs[0] > 0.0 {
            p1_sum += probs[0];
            p2_sum += probs[1];
            p3_sum += probs[2];
        }
    }

    HalfLifeResult {
        prism_census: all_census,
        gen_counts,
        occupancy_by_belly: occupancy,
        stability_ratio_gen2: if p1_sum > 0.0 { p2_sum / p1_sum } else { 0.0 },
        stability_ratio_gen3: if p1_sum > 0.0 { p3_sum / p1_sum } else { 0.0 },
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &HalfLifeResult, w: &mut CsvWriter) {
    w.comment("M2 Half-Life Census (generation stability)");
    w.header(&["prism_idx", "generation", "belly_size", "net_phase"]);
    for &(idx, gen, belly, phase) in &result.prism_census {
        w.row_fmt(format_args!("{},{},{},{}", idx, gen, belly, phase));
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &HalfLifeResult) {
    println!("  [M2] Half-Life Census:");
    println!("    Gen counts:      Gen1={}  Gen2={}  Gen3={}  Anti1={}",
        result.gen_counts[0], result.gen_counts[1], result.gen_counts[2], result.gen_counts[3]);
    println!("    Stability:       tau2/tau1={:.4}  tau3/tau1={:.4}",
        result.stability_ratio_gen2, result.stability_ratio_gen3);
    println!("    Belly sizes:     {} distinct", result.occupancy_by_belly.len());
}
