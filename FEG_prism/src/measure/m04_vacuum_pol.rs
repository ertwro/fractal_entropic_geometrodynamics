// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M4 — Combinatorial RG Flow: Q_topo(V) in Alexandrov Intervals
//!
//! Samples random Alexandrov intervals A(p,q) = {r : p ≺ r ≺ q} of varying
//! volume V = |A(p,q)|, counts which prisms fall entirely inside each interval,
//! and computes the local Q_topo(V).  This is the running coupling constant —
//! purely combinatorial, zero free parameters.

use super::context::MeasureContext;
use crate::graph::grid::is_causal;
use crate::output::CsvWriter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Number of Alexandrov interval samples per realization.
const N_SAMPLES: usize = 1000;

/// Maximum random-walk depth when picking the future endpoint q.
const MAX_DEPTH: usize = 15;

/// Minimum interval volume to keep (skip degenerate intervals).
const MIN_VOLUME: usize = 10;

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AlexandrovSample {
    pub volume: usize,
    pub n_prisms: usize,
    pub local_phase_sq: usize,
    pub local_mass_sq: usize,
    pub local_q: f64,
    pub local_alpha: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VacuumPolResult {
    pub samples: Vec<AlexandrovSample>,
    pub n_sampled: usize,
    pub global_q: f64,
    pub global_alpha: f64,
    pub mean_local_q: f64,
}

// ── Pre-computed prism info ──────────────────────────────────────────────────

struct PrismInfo {
    all_nodes: Vec<usize>,
    phi_abs: usize,
    n_inter: usize,
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> VacuumPolResult {
    let n = ctx.n_points;
    let pts = ctx.pts;
    let (vac_head, vac_data) = ctx.vacuum_csr.raw();

    // Global Q_topo from topology summary
    let global_q = if ctx.topology.mass_sq_total > 0 {
        ctx.topology.phase_sq_total as f64 / ctx.topology.mass_sq_total as f64
    } else {
        0.0
    };
    let global_alpha = global_q / (8.0 * std::f64::consts::PI);

    if ctx.prisms.is_empty() {
        return VacuumPolResult {
            samples: vec![],
            n_sampled: 0,
            global_q,
            global_alpha,
            mean_local_q: 0.0,
        };
    }

    // ── Pre-compute prism info ───────────────────────────────────────────
    let prism_info: Vec<PrismInfo> = ctx
        .prisms
        .iter()
        .map(|p| {
            let mut all_nodes = Vec::with_capacity(2 + p.intermediates.len());
            all_nodes.push(p.origin);
            all_nodes.push(p.destination);
            all_nodes.extend_from_slice(&p.intermediates);

            let net_phase: i32 = p
                .intermediates
                .iter()
                .map(|&w| ctx.momentum[w].signum())
                .sum();
            PrismInfo {
                all_nodes,
                phi_abs: net_phase.unsigned_abs() as usize,
                n_inter: p.intermediates.len(),
            }
        })
        .collect();

    // ── Build node → prism index map ─────────────────────────────────────
    let mut node_to_prisms: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (pi, info) in prism_info.iter().enumerate() {
        for &node in &info.all_nodes {
            if node < n {
                node_to_prisms[node].push(pi);
            }
        }
    }

    // ── Pre-sort nodes by time coordinate ────────────────────────────────
    let mut time_sorted: Vec<usize> = (0..n).collect();
    time_sorted.sort_unstable_by(|&a, &b| {
        pts[a][0].partial_cmp(&pts[b][0]).unwrap()
    });

    // Extract sorted time values for binary search
    let sorted_times: Vec<f64> = time_sorted.iter().map(|&i| pts[i][0]).collect();

    // ── Sample Alexandrov intervals in parallel ──────────────────────────
    let samples: Vec<AlexandrovSample> = (0..N_SAMPLES)
        .into_par_iter()
        .filter_map(|sample_idx| {
            let mut rng = StdRng::seed_from_u64(ctx.seed.wrapping_add(sample_idx as u64));

            // Pick random starting node p
            let p = rng.gen_range(0..n);

            // Random walk forward to find q
            let mut current = p;
            let depth = rng.gen_range(1..=MAX_DEPTH);
            for _ in 0..depth {
                let s = vac_head[current] as usize;
                let e = vac_head[current + 1] as usize;
                if s == e {
                    break; // no forward neighbors
                }
                current = vac_data[rng.gen_range(s..e)] as usize;
            }
            let q = current;
            if q == p {
                return None; // degenerate: couldn't move forward
            }

            // Verify p ≺ q (should hold by construction, but be safe)
            if !is_causal(&pts[p], &pts[q]) {
                return None;
            }

            let t_p = pts[p][0];
            let t_q = pts[q][0];

            // Binary search for time window [t_p, t_q] in sorted_times
            let lo = sorted_times.partition_point(|&t| t <= t_p);
            let hi = sorted_times.partition_point(|&t| t < t_q);

            // Build interval membership set: r ∈ A(p,q) iff p ≺ r ≺ q
            let mut interval: Vec<usize> = Vec::new();
            for &idx in &time_sorted[lo..hi] {
                if idx == p || idx == q {
                    continue;
                }
                if is_causal(&pts[p], &pts[idx]) && is_causal(&pts[idx], &pts[q]) {
                    interval.push(idx);
                }
            }

            let volume = interval.len();
            if volume < MIN_VOLUME {
                return None;
            }

            // Build fast membership lookup
            let mut in_interval = vec![false; n];
            for &idx in &interval {
                in_interval[idx] = true;
            }
            // Also include endpoints for prism containment check
            in_interval[p] = true;
            in_interval[q] = true;

            // Collect candidate prism indices from interval nodes
            let mut candidate_prisms: Vec<usize> = Vec::new();
            for &idx in &interval {
                candidate_prisms.extend_from_slice(&node_to_prisms[idx]);
            }
            // Also check prisms touching p and q
            candidate_prisms.extend_from_slice(&node_to_prisms[p]);
            candidate_prisms.extend_from_slice(&node_to_prisms[q]);
            candidate_prisms.sort_unstable();
            candidate_prisms.dedup();

            // A prism is interior iff ALL its nodes are in the interval
            let mut local_phase_sq: usize = 0;
            let mut local_mass_sq: usize = 0;
            let mut n_prisms: usize = 0;
            for &pi in &candidate_prisms {
                let info = &prism_info[pi];
                if info.all_nodes.iter().all(|&nd| in_interval[nd]) {
                    local_phase_sq += info.phi_abs * info.phi_abs;
                    local_mass_sq += info.n_inter * info.n_inter;
                    n_prisms += 1;
                }
            }

            let local_q = if local_mass_sq > 0 {
                local_phase_sq as f64 / local_mass_sq as f64
            } else {
                return None; // no prisms with mass
            };
            let local_alpha = local_q / (8.0 * std::f64::consts::PI);

            Some(AlexandrovSample {
                volume,
                n_prisms,
                local_phase_sq,
                local_mass_sq,
                local_q,
                local_alpha,
            })
        })
        .collect();

    let n_sampled = samples.len();
    let mean_local_q = if n_sampled > 0 {
        samples.iter().map(|s| s.local_q).sum::<f64>() / n_sampled as f64
    } else {
        0.0
    };

    VacuumPolResult {
        samples,
        n_sampled,
        global_q,
        global_alpha,
        mean_local_q,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[VacuumPolResult]) -> VacuumPolResult {
    let mut all_samples: Vec<AlexandrovSample> = Vec::new();
    for r in results {
        all_samples.extend(r.samples.iter().cloned());
    }
    let n_sampled = all_samples.len();
    let mean_local_q = if n_sampled > 0 {
        all_samples.iter().map(|s| s.local_q).sum::<f64>() / n_sampled as f64
    } else {
        0.0
    };
    let global_q = results.iter().map(|r| r.global_q).sum::<f64>() / results.len() as f64;
    let global_alpha = global_q / (8.0 * std::f64::consts::PI);

    VacuumPolResult {
        samples: all_samples,
        n_sampled,
        global_q,
        global_alpha,
        mean_local_q,
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &VacuumPolResult, w: &mut CsvWriter) {
    w.comment("M4 Combinatorial RG Flow (Q_topo in Alexandrov intervals)");
    w.header(&[
        "volume",
        "n_prisms",
        "local_phase_sq",
        "local_mass_sq",
        "local_q",
        "local_alpha",
    ]);
    for s in &result.samples {
        w.row_fmt(format_args!(
            "{},{},{},{},{:.6},{:.6}",
            s.volume, s.n_prisms, s.local_phase_sq, s.local_mass_sq, s.local_q, s.local_alpha
        ));
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &VacuumPolResult) {
    println!("  [M4] Combinatorial RG Flow:");
    println!("    Samples:         {}", result.n_sampled);
    println!(
        "    Global Q_topo:   {:.4}  (1/alpha = {:.1})",
        result.global_q,
        if result.global_alpha > 0.0 {
            1.0 / result.global_alpha
        } else {
            0.0
        }
    );
    println!("    Mean local Q:    {:.4}", result.mean_local_q);
    if !result.samples.is_empty() {
        let v_min = result.samples.iter().map(|s| s.volume).min().unwrap();
        let v_max = result.samples.iter().map(|s| s.volume).max().unwrap();
        println!("    Volume range:    [{}, {}]", v_min, v_max);
    }
}
