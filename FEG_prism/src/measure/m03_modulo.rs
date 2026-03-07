// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M3 — Modulo Path Integral (NTT-Based Interference Fringes)
//!
//! Walkers accumulate a modular phase g^S mod p (where S = cumulative moves)
//! as they traverse the symmetric vacuum graph.  Per-node phase coherence
//! reveals constructive/destructive interference patterns.
//!
//! Calculo de Kuratowski, Vol I, section 6: modular arithmetic on causal paths.

use super::context::MeasureContext;
use crate::convergence::{AutoConverge, ConvergeState};
use crate::output::CsvWriter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NodeInterference {
    pub node_id: usize,
    pub n_arrivals: u64,
    pub phase_sum: u64,
    pub intensity: f64,
    pub coords: [f32; 4],
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModuloPathResult {
    pub nodes: Vec<NodeInterference>,
    pub total_walkers: usize,
    pub mean_intensity: f64,
    pub max_intensity: f64,
    pub constructive_count: usize,
    pub destructive_count: usize,
    pub prime: u64,
    pub root: u64,
}

impl ModuloPathResult {
    /// Drop per-node detail, keeping only summary statistics.
    /// Frees ~440 MB per realization at N=10M.
    pub fn compact(&mut self) {
        self.nodes = Vec::new();
    }
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

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> ModuloPathResult {
    let n = ctx.n_points;
    let p = ctx.modulo_config.prime;
    let g = ctx.modulo_config.root;
    let merge = &ctx.defect.merge_map;
    let sym = match ctx.sym_vacuum {
        Some(s) => s,
        None => return ModuloPathResult {
            nodes: vec![], total_walkers: 0, mean_intensity: 0.0,
            max_intensity: 0.0, constructive_count: 0, destructive_count: 0,
            prime: p, root: g,
        },
    };
    let (sym_vac_head, sym_vac_data) = sym.raw();
    let n_steps = ctx.modulo_config.steps;

    let arrivals: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let phase_acc: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();

    // Auto-convergence: batch walkers until mean intensity stabilises
    let ac = AutoConverge::new(ctx.walkers, 50_000, ctx.epsilon);
    let mut conv_state = ConvergeState::new();
    let half_p = p / 2;
    let mut is_first_batch = true;

    loop {
        // Snapshot cumulative intensity before batch (skip on first — all zeros)
        let (pre_sum_int, pre_count) = if is_first_batch {
            (0.0f64, 0usize)
        } else {
            (0..n).into_par_iter()
                .fold(|| (0.0f64, 0usize), |(s, c), i| {
                    let arr = arrivals[i].load(Ordering::Relaxed);
                    if arr == 0 { return (s, c); }
                    let pacc = phase_acc[i].load(Ordering::Relaxed) % p;
                    let sym = if pacc > half_p {
                        pacc as i64 - p as i64
                    } else {
                        pacc as i64
                    };
                    let intensity = (sym as f64).powi(2) / (half_p as f64).powi(2);
                    (s + intensity, c + 1)
                })
                .reduce(|| (0.0, 0), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2))
        };

        let base = conv_state.total_walkers;
        (base..base + ac.batch_size).into_par_iter().for_each(|wi| {
            let mut rng = StdRng::seed_from_u64(ctx.seed.wrapping_add(wi as u64));
            let mut pos = resolve(rng.gen_range(0..n), merge);
            let mut phase: u64 = 1; // g^0 mod p = 1

            for _t in 0..n_steps {
                let start_idx = sym_vac_head[pos] as usize;
                let end_idx = sym_vac_head[pos + 1] as usize;
                let deg = end_idx - start_idx;

                if deg > 0 && rng.gen_bool(0.5) {
                    let next = sym_vac_data[start_idx + rng.gen_range(0..deg)] as usize;
                    pos = resolve(next, merge);
                    phase = ((phase as u128 * g as u128) % p as u128) as u64;
                }

                arrivals[pos].fetch_add(1, Ordering::Relaxed);
                phase_acc[pos].fetch_add(phase, Ordering::Relaxed);
            }
        });
        is_first_batch = false;

        // Snapshot cumulative intensity after batch, compute batch-only observable
        let (post_sum_int, post_count) = (0..n).into_par_iter()
            .fold(|| (0.0f64, 0usize), |(s, c), i| {
                let arr = arrivals[i].load(Ordering::Relaxed);
                if arr == 0 { return (s, c); }
                let pacc = phase_acc[i].load(Ordering::Relaxed) % p;
                let sym = if pacc > half_p {
                    pacc as i64 - p as i64
                } else {
                    pacc as i64
                };
                let intensity = (sym as f64).powi(2) / (half_p as f64).powi(2);
                (s + intensity, c + 1)
            })
            .reduce(|| (0.0, 0), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2));
        let batch_obs = if post_count > pre_count {
            (post_sum_int - pre_sum_int) / (post_count - pre_count) as f64
        } else {
            if post_count > 0 { post_sum_int / post_count as f64 } else { 0.0 }
        };

        if conv_state.update(batch_obs, &ac) { break; }
        if conv_state.at_limit(&ac) {
            eprintln!("[M3] WARNING: {} walkers without convergence", conv_state.total_walkers);
            break;
        }
    }
    println!("  [M3] converged at {} walkers", conv_state.total_walkers);
    let n_walkers = conv_state.total_walkers;

    // Collect per-node interference results
    let mut nodes = Vec::new();

    for i in 0..n {
        let arr = arrivals[i].load(Ordering::Relaxed);
        if arr == 0 {
            continue;
        }
        let pacc = phase_acc[i].load(Ordering::Relaxed);
        let centered = pacc % p;
        let sym = if centered > half_p {
            centered as i64 - p as i64
        } else {
            centered as i64
        };
        let intensity = (sym as f64).powi(2) / (half_p as f64).powi(2);

        nodes.push(NodeInterference {
            node_id: i,
            n_arrivals: arr,
            phase_sum: pacc,
            intensity,
            coords: [
                ctx.sorted_coords.get(i).map_or(0.0, |c| c[0] as f32),
                ctx.sorted_coords.get(i).map_or(0.0, |c| c[1] as f32),
                ctx.sorted_coords.get(i).map_or(0.0, |c| c[2] as f32),
                ctx.sorted_coords.get(i).map_or(0.0, |c| c[3] as f32),
            ],
        });
    }

    // Statistics
    let intensities: Vec<f64> = nodes.iter().map(|nd| nd.intensity).collect();
    let mean_int = if !intensities.is_empty() {
        intensities.iter().sum::<f64>() / intensities.len() as f64
    } else {
        0.0
    };
    let max_int = intensities.iter().cloned().fold(0.0f64, f64::max);

    let variance = if intensities.len() > 1 {
        intensities
            .iter()
            .map(|&x| (x - mean_int).powi(2))
            .sum::<f64>()
            / intensities.len() as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    let constructive = intensities
        .iter()
        .filter(|&&x| x > mean_int + 2.0 * std_dev)
        .count();
    let destructive = intensities
        .iter()
        .filter(|&&x| x < (mean_int - 2.0 * std_dev).max(0.0))
        .count();

    ModuloPathResult {
        nodes,
        total_walkers: n_walkers,
        mean_intensity: mean_int,
        max_intensity: max_int,
        constructive_count: constructive,
        destructive_count: destructive,
        prime: p,
        root: g,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[ModuloPathResult]) -> ModuloPathResult {
    let m = results.len() as f64;
    ModuloPathResult {
        nodes: vec![],
        total_walkers: results.iter().map(|i| i.total_walkers).sum(),
        mean_intensity: results.iter().map(|i| i.mean_intensity).sum::<f64>() / m,
        max_intensity: results
            .iter()
            .map(|i| i.max_intensity)
            .fold(0.0f64, f64::max),
        constructive_count: results.iter().map(|i| i.constructive_count).sum(),
        destructive_count: results.iter().map(|i| i.destructive_count).sum(),
        prime: results[0].prime,
        root: results[0].root,
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &ModuloPathResult, w: &mut CsvWriter) {
    w.comment("M3 Modulo Path Integral (NTT interference fringes)");
    w.header(&["node_id", "n_arrivals", "phase_sum", "intensity", "qt", "qx", "qy", "qz"]);
    for nd in &result.nodes {
        w.row_fmt(format_args!(
            "{},{},{},{:.6},{:.4},{:.4},{:.4},{:.4}",
            nd.node_id, nd.n_arrivals, nd.phase_sum, nd.intensity,
            nd.coords[0], nd.coords[1], nd.coords[2], nd.coords[3]
        ));
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &ModuloPathResult) {
    println!("  [M3] Modulo Path Integral:");
    println!("    Walkers:         {}", result.total_walkers);
    println!("    Mean intensity:  {:.6}", result.mean_intensity);
    println!("    Max intensity:   {:.6}", result.max_intensity);
    println!("    Constructive:    {}  Destructive: {}",
        result.constructive_count, result.destructive_count);
    println!("    NTT config:      p={}, g={}", result.prime, result.root);
}
