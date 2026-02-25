// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M4 — Vacuum Polarization (K_{3,3} Screening)
//!
//! For each Gen1 prism, gathers candidate neighbor nodes and checks whether
//! connecting them would create a K_{3,3} subgraph (forbidden in planar graphs).
//! The screening factor = fraction of candidates NOT blocked by K_{3,3}.
//!
//! Calculo de Kuratowski, Vol II, Thm 6.3: K_{3,3} obstruction as charge screening.

use super::context::MeasureContext;
use crate::output::CsvWriter;
use crate::phase2::defect::CausalPrism;
use rayon::prelude::*;
use std::collections::HashSet;

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PrismScreening {
    pub prism_idx: usize,
    pub generation: u8,
    pub n_attempted: usize,
    pub n_rejected_k33: usize,
    pub n_accepted: usize,
    pub local_screening: f64,
}

#[derive(Clone, Debug)]
pub struct VacuumPolResult {
    pub per_prism: Vec<PrismScreening>,
    pub total_attempted: usize,
    pub total_rejected: usize,
    pub total_accepted: usize,
    pub mean_screening: f64,
    pub bare_alpha: f64,
    pub screened_alpha: f64,
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

/// Check if there is an edge between nodes c and x in either direction
/// using both forward and reverse CSR (sorted, binary-searchable).
fn has_edge_bidirectional(
    c: usize,
    x: usize,
    fwd_head: &[u32],
    fwd_data: &[u32],
    rev_head: &[u32],
    rev_data: &[u32],
) -> bool {
    let x32 = x as u32;
    // Forward: c -> x
    let fs = fwd_head[c] as usize;
    let fe = fwd_head[c + 1] as usize;
    if fwd_data[fs..fe].binary_search(&x32).is_ok() {
        return true;
    }
    // Reverse: x -> c (c has predecessor x)
    let rs = rev_head[c] as usize;
    let re = rev_head[c + 1] as usize;
    rev_data[rs..re].binary_search(&(x as u32)).is_ok()
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> VacuumPolResult {
    let n = ctx.n_points;
    let gen_lookup = build_gen_lookup(n, ctx);
    let (vac_head, vac_data) = ctx.vacuum_csr.raw();
    let bare_alpha = ctx.topology.alpha_em;

    // Build reverse CSR
    let mut rev_deg = vec![0u32; n];
    for u in 0..n {
        let s = vac_head[u] as usize;
        let e = vac_head[u + 1] as usize;
        for &v in &vac_data[s..e] {
            let vi = v as usize;
            if vi < n {
                rev_deg[vi] += 1;
            }
        }
    }

    let mut rev_head = vec![0u32; n + 1];
    for i in 0..n {
        rev_head[i + 1] = rev_head[i] + rev_deg[i];
    }

    let total_rev = rev_head[n] as usize;
    let mut rev_data = vec![0u32; total_rev];
    let mut rev_pos = rev_head[..n].to_vec();
    for u in 0..n {
        let s = vac_head[u] as usize;
        let e = vac_head[u + 1] as usize;
        for &v in &vac_data[s..e] {
            let vi = v as usize;
            if vi < n {
                rev_data[rev_pos[vi] as usize] = u as u32;
                rev_pos[vi] += 1;
            }
        }
    }

    // Sort reverse adjacency lists for binary search
    for u in 0..n {
        let s = rev_head[u] as usize;
        let e = rev_head[u + 1] as usize;
        rev_data[s..e].sort_unstable();
    }

    // Build prism membership set
    let mut is_prism_member = vec![false; n];
    for p in ctx.prisms {
        if p.origin < n {
            is_prism_member[p.origin] = true;
        }
        if p.destination < n {
            is_prism_member[p.destination] = true;
        }
        for &i in &p.intermediates {
            if i < n {
                is_prism_member[i] = true;
            }
        }
    }

    // Identify Gen1 prisms with >= 3 intermediates
    let gen1_prisms: Vec<(usize, &CausalPrism)> = ctx.prisms
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            classify_prism_generation(p, &gen_lookup) == 1 && p.intermediates.len() >= 3
        })
        .collect();

    if gen1_prisms.is_empty() {
        return VacuumPolResult {
            per_prism: vec![],
            total_attempted: 0,
            total_rejected: 0,
            total_accepted: 0,
            mean_screening: 1.0,
            bare_alpha,
            screened_alpha: bare_alpha,
        };
    }

    // Process each Gen1 prism in parallel
    let per_prism: Vec<PrismScreening> = gen1_prisms
        .par_iter()
        .map(|&(pi, prism)| {
            // Gather all prism node indices
            let prism_nodes: Vec<usize> = std::iter::once(prism.origin)
                .chain(std::iter::once(prism.destination))
                .chain(prism.intermediates.iter().copied())
                .collect();

            // Collect candidate neighbors (not prism members)
            let mut candidates: HashSet<usize> = HashSet::new();
            for &pn in &prism_nodes {
                if pn >= n {
                    continue;
                }
                // Forward neighbors
                let fs = vac_head[pn] as usize;
                let fe = vac_head[pn + 1] as usize;
                for &v in &vac_data[fs..fe] {
                    let vi = v as usize;
                    if vi < n && !is_prism_member[vi] {
                        candidates.insert(vi);
                    }
                }
                // Reverse neighbors
                let rs = rev_head[pn] as usize;
                let re = rev_head[pn + 1] as usize;
                for &v in &rev_data[rs..re] {
                    let vi = v as usize;
                    if vi < n && !is_prism_member[vi] {
                        candidates.insert(vi);
                    }
                }
            }

            let n_attempted = candidates.len();
            let mut n_rejected = 0usize;

            let poles = [prism.origin, prism.destination];

            for &c in &candidates {
                // K_{3,3} check: C connects to both poles AND >= 3 intermediates
                let connects_pole0 =
                    has_edge_bidirectional(c, poles[0], vac_head, vac_data, &rev_head, &rev_data);
                let connects_pole1 =
                    has_edge_bidirectional(c, poles[1], vac_head, vac_data, &rev_head, &rev_data);

                if connects_pole0 && connects_pole1 {
                    let int_connections = prism
                        .intermediates
                        .iter()
                        .filter(|&&i| {
                            has_edge_bidirectional(
                                c, i, vac_head, vac_data, &rev_head, &rev_data,
                            )
                        })
                        .count();
                    if int_connections >= 3 {
                        n_rejected += 1;
                    }
                }
            }

            let n_accepted = n_attempted - n_rejected;
            PrismScreening {
                prism_idx: pi,
                generation: 1,
                n_attempted,
                n_rejected_k33: n_rejected,
                n_accepted,
                local_screening: if n_attempted > 0 {
                    n_accepted as f64 / n_attempted as f64
                } else {
                    1.0
                },
            }
        })
        .collect();

    let total_attempted: usize = per_prism.iter().map(|p| p.n_attempted).sum();
    let total_rejected: usize = per_prism.iter().map(|p| p.n_rejected_k33).sum();
    let total_accepted: usize = per_prism.iter().map(|p| p.n_accepted).sum();
    let mean_screening = if !per_prism.is_empty() {
        per_prism.iter().map(|p| p.local_screening).sum::<f64>() / per_prism.len() as f64
    } else {
        1.0
    };

    VacuumPolResult {
        per_prism,
        total_attempted,
        total_rejected,
        total_accepted,
        mean_screening,
        bare_alpha,
        screened_alpha: bare_alpha * mean_screening,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[VacuumPolResult]) -> VacuumPolResult {
    let m = results.len() as f64;
    let mean_scr = results.iter().map(|v| v.mean_screening).sum::<f64>() / m;
    let bare_a = results.iter().map(|v| v.bare_alpha).sum::<f64>() / m;
    VacuumPolResult {
        per_prism: vec![],
        total_attempted: results.iter().map(|v| v.total_attempted).sum(),
        total_rejected: results.iter().map(|v| v.total_rejected).sum(),
        total_accepted: results.iter().map(|v| v.total_accepted).sum(),
        mean_screening: mean_scr,
        bare_alpha: bare_a,
        screened_alpha: bare_a * mean_scr,
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &VacuumPolResult, w: &mut CsvWriter) {
    w.comment("M4 Vacuum Polarization (K_3,3 screening)");
    w.header(&["prism_idx", "generation", "n_attempted", "n_rejected_k33", "n_accepted", "local_screening"]);
    for ps in &result.per_prism {
        w.row_fmt(format_args!(
            "{},{},{},{},{},{:.6}",
            ps.prism_idx, ps.generation, ps.n_attempted,
            ps.n_rejected_k33, ps.n_accepted, ps.local_screening
        ));
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &VacuumPolResult) {
    println!("  [M4] Vacuum Polarization:");
    println!("    Attempted:       {}", result.total_attempted);
    println!("    Rejected (K33):  {}", result.total_rejected);
    println!("    Accepted:        {}", result.total_accepted);
    println!("    Mean screening:  {:.6}", result.mean_screening);
    println!("    Bare alpha:      {:.6}  (1/alpha={:.1})",
        result.bare_alpha, if result.bare_alpha > 0.0 { 1.0 / result.bare_alpha } else { 0.0 });
    println!("    Screened alpha:  {:.6}  (1/alpha={:.1})",
        result.screened_alpha, if result.screened_alpha > 0.0 { 1.0 / result.screened_alpha } else { 0.0 });
}
