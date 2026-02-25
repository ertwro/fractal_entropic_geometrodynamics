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
use crate::output::CsvWriter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct NodeInterference {
    pub node_id: usize,
    pub n_arrivals: u64,
    pub phase_sum: u64,
    pub intensity: f64,
    pub coords: [f32; 4],
}

#[derive(Clone, Debug)]
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

// ── Utilities ────────────────────────────────────────────────────────────────

/// Modular exponentiation via repeated squaring: base^exp mod modulus.
fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

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
    let (sym_vac_head, sym_vac_data) = ctx.sym_vacuum.raw();
    let n_walkers = ctx.walkers;
    let n_steps = ctx.modulo_config.steps;

    let arrivals: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let phase_acc: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();

    (0..n_walkers).into_par_iter().for_each(|wi| {
        let mut rng = StdRng::seed_from_u64(ctx.seed.wrapping_add(wi as u64));
        let mut pos = resolve(rng.gen_range(0..n), merge);
        let mut s: u64 = 0;

        for _t in 0..n_steps {
            let start_idx = sym_vac_head[pos] as usize;
            let end_idx = sym_vac_head[pos + 1] as usize;
            let deg = end_idx - start_idx;

            if deg > 0 && rng.gen_bool(0.5) {
                let next = sym_vac_data[start_idx + rng.gen_range(0..deg)] as usize;
                pos = resolve(next, merge);
                s += 1;
            }

            let phase = pow_mod(g, s, p);
            arrivals[pos].fetch_add(1, Ordering::Relaxed);
            phase_acc[pos].fetch_add(phase, Ordering::Relaxed);
        }
    });

    // Collect per-node interference results
    let half_p = p / 2;
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
                ctx.pts.get(i).map_or(0.0, |c| c[0] as f32),
                ctx.pts.get(i).map_or(0.0, |c| c[1] as f32),
                ctx.pts.get(i).map_or(0.0, |c| c[2] as f32),
                ctx.pts.get(i).map_or(0.0, |c| c[3] as f32),
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
