// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M13 — Topological Particle Collider
//!
//! Controlled scattering experiments on K_{2,2} diamonds in the cold vacuum.
//!
//! A K_{2,2} diamond has two tops sharing exactly 2 children.  Any seam node
//! `z` (child of one top, not the other) will push that count to 3 when
//! bridged — a K_{2,3} formation is a *mathematical certainty*, not a
//! probabilistic event.  The hit rate is tautologically 1.0.
//!
//! The real observable is the Markov blanket cross-section:
//!
//!   Q_topo = seam_size / blanket_size
//!
//! Measured: Q_topo = 1/4 (exact, locked asymptote).
//! α₀ = Q_topo / (8π) = 1/(32π) ≈ 1/100.5 (bare coupling at Planck scale).
//! Vacuum polarization screens bare → observed α ≈ 1/137.

use super::context::MeasureContext;
use crate::graph::csr::{CsrGraph, Directed};
use crate::output::CsvWriter;
use rayon::prelude::*;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ColliderResult {
    pub diamonds_found: usize,
    pub diamonds_tested: usize,
    pub total_seam: usize,
    pub total_blanket: usize,
    pub q_topo: f64,
    pub inv_alpha_bare: f64,
}

/// Per-diamond measurement (transient, not stored).
#[derive(Debug, Clone, Copy)]
struct DiamondMeasurement {
    seam: usize,
    blanket: usize,
}

// ── Sorted-slice set operations ─────────────────────────────────────────────

fn intersect_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

fn difference_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() {
        if j >= b.len() || a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out
}

// ── Streaming collider: fused extraction + measurement (O(1) memory) ────────
//
// At N=10M the graph contains ~900M K_{2,2} diamonds.  Collecting them into a
// Vec would require ~16 GB; the dedup FxHashSet another ~24 GB → OOM on 32 GB
// machines.  Instead we stream: discover each diamond, measure it immediately,
// and accumulate running totals.  Peak extra memory: O(D²) per node ≈ tiny.
//
// Deduplication without a HashSet: for each node u we find candidate partners v
// through shared children.  Since CSR neighbor lists are sorted, we collect
// candidate v's via a single pass through children(u), and for each (u,v) pair
// (u < v) we compute intersect_sorted once.  The key insight: we only reach
// (u,v) through children of u that are also children of v, so the pair is
// discovered exactly |shared_children(u,v)| times.  We gate on the *first*
// shared child (smallest index) to process each pair exactly once.

fn measure_diamond_inline(
    forward: &CsrGraph<Directed>,
    backward: &CsrGraph<Directed>,
    u: u32,
    v: u32,
    _bot0: u32,
    _bot1: u32,
) -> DiamondMeasurement {
    let children_u = forward.neighbors(u as usize);
    let children_v = forward.neighbors(v as usize);
    let seam_u_only = difference_sorted(children_u, children_v);
    let seam_v_only = difference_sorted(children_v, children_u);

    let valid_seam_u = seam_u_only.iter().filter(|&&z| z > v).count();
    let valid_seam_v = seam_v_only.iter().filter(|&&z| z > u).count();
    let seam = valid_seam_u + valid_seam_v;

    let diamond_nodes = [u, v, _bot0, _bot1];
    let mut blanket = 0usize;
    for &node in &diamond_nodes {
        for &child in forward.neighbors(node as usize) {
            if !diamond_nodes.contains(&child) {
                blanket += 1;
            }
        }
        for &parent in backward.neighbors(node as usize) {
            if !diamond_nodes.contains(&parent) {
                blanket += 1;
            }
        }
    }

    DiamondMeasurement { seam, blanket }
}

// ── Collider driver (streaming, O(1) diamond memory) ────────────────────────

pub fn run(ctx: &MeasureContext, rev_csr: &CsrGraph<Directed>) -> ColliderResult {
    let forward = ctx.vacuum_csr;
    let backward = rev_csr;
    let n = forward.n_nodes();

    // Parallel: partition nodes into chunks, each thread accumulates locally
    use std::sync::atomic::{AtomicUsize, Ordering};
    let found = AtomicUsize::new(0);
    let tested = AtomicUsize::new(0);
    let seam_total = AtomicUsize::new(0);
    let blanket_total = AtomicUsize::new(0);

    (0..n).into_par_iter().for_each(|u| {
        let children_u = forward.neighbors(u);
        if children_u.len() < 2 {
            return;
        }
        // Collect candidate partners v > u that share children with u
        // Use a small local vec (bounded by D² ≈ 225) to deduplicate v's
        let mut candidates: smallvec::SmallVec<[u32; 64]> = smallvec::SmallVec::new();
        for &x in children_u {
            let parents_x = backward.neighbors(x as usize);
            for &v in parents_x {
                if v as usize <= u {
                    continue;
                }
                // Insert unique (linear scan on tiny vec, D² ≤ 225 entries)
                if !candidates.contains(&v) {
                    candidates.push(v);
                }
            }
        }
        // For each unique candidate pair (u, v), check if exactly 2 shared children
        for &v in &candidates {
            let children_v = forward.neighbors(v as usize);
            let shared = intersect_sorted(children_u, children_v);
            if shared.len() == 2 {
                found.fetch_add(1, Ordering::Relaxed);
                let m = measure_diamond_inline(
                    forward, backward,
                    u as u32, v, shared[0], shared[1],
                );
                if m.seam > 0 && m.blanket > 0 {
                    tested.fetch_add(1, Ordering::Relaxed);
                }
                seam_total.fetch_add(m.seam, Ordering::Relaxed);
                blanket_total.fetch_add(m.blanket, Ordering::Relaxed);
            }
        }
    });

    let n_found = found.load(Ordering::Relaxed);
    let n_tested = tested.load(Ordering::Relaxed);
    let total_seam = seam_total.load(Ordering::Relaxed);
    let total_blanket = blanket_total.load(Ordering::Relaxed);

    let q_topo = if total_blanket > 0 {
        total_seam as f64 / total_blanket as f64
    } else {
        0.0
    };

    let inv_alpha = if q_topo > 0.0 {
        8.0 * std::f64::consts::PI / q_topo
    } else {
        0.0
    };

    ColliderResult {
        diamonds_found: n_found,
        diamonds_tested: n_tested,
        total_seam,
        total_blanket,
        q_topo,
        inv_alpha_bare: inv_alpha,
    }
}

pub fn aggregate(results: &[ColliderResult]) -> ColliderResult {
    let total_seam: usize = results.iter().map(|r| r.total_seam).sum();
    let total_blanket: usize = results.iter().map(|r| r.total_blanket).sum();
    let total_found: usize = results.iter().map(|r| r.diamonds_found).sum();
    let total_tested: usize = results.iter().map(|r| r.diamonds_tested).sum();

    let q_topo = if total_blanket > 0 {
        total_seam as f64 / total_blanket as f64
    } else {
        0.0
    };
    let inv_alpha = if q_topo > 0.0 {
        8.0 * std::f64::consts::PI / q_topo
    } else {
        0.0
    };

    ColliderResult {
        diamonds_found: total_found,
        diamonds_tested: total_tested,
        total_seam,
        total_blanket,
        q_topo,
        inv_alpha_bare: inv_alpha,
    }
}

pub fn write_csv(result: &ColliderResult, w: &mut CsvWriter) {
    w.comment("M13 Topological Collider (Q_topo and bare coupling)");
    w.header(&["key", "value"]);
    w.row_fmt(format_args!("diamonds_found,{}", result.diamonds_found));
    w.row_fmt(format_args!("diamonds_tested,{}", result.diamonds_tested));
    w.row_fmt(format_args!("total_seam,{}", result.total_seam));
    w.row_fmt(format_args!("total_blanket,{}", result.total_blanket));
    w.row_fmt(format_args!("Q_topo,{:.8}", result.q_topo));
    w.row_fmt(format_args!("inv_alpha_bare,{:.4}", result.inv_alpha_bare));
    w.row_fmt(format_args!("Q_topo_exact,0.250000"));
    w.row_fmt(format_args!("inv_alpha_exact,{:.6}", 32.0 * std::f64::consts::PI));
}

pub fn print_summary(result: &ColliderResult) {
    let q_exact = 0.25;
    let inv_a_exact = 32.0 * std::f64::consts::PI;

    println!("  [M13] Topological Collider:");
    println!("    Diamonds found:    {}", result.diamonds_found);
    println!("    Diamonds tested:   {}", result.diamonds_tested);
    println!("    Q_topo (measured): {:.6}  (exact: {:.6})", result.q_topo, q_exact);
    println!(
        "    1/alpha_0:         {:.2}   (exact: {:.2} = 32pi)",
        result.inv_alpha_bare, inv_a_exact
    );
    if result.diamonds_tested > 0 {
        println!(
            "    Mean seam/diamond:    {:.2}",
            result.total_seam as f64 / result.diamonds_tested as f64
        );
        println!(
            "    Mean blanket/diamond: {:.2}",
            result.total_blanket as f64 / result.diamonds_tested as f64
        );
    }
}
