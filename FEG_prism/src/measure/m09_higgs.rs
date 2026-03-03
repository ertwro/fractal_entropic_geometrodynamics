// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M9 — Higgs Mechanism (Topological Drag via Gauge Propagation Delay)
//!
//! The photon (U(1)) propagates freely on the symmetric vacuum CSR.
//! The W boson (SU(2)) propagates only through left-handed nodes (chi < 0).
//! The drag coefficient mu = (d_chiral - d_base) / d_base is the emergent
//! mass of the vector boson -- the topological cost of chiral confinement.
//!
//! Calculo de Kuratowski, Vol II, section 9: gauge boson mass from chiral confinement.

use super::context::MeasureContext;
use crate::output::CsvWriter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HiggsPairResult {
    pub origin_idx: usize,
    pub dest_idx: usize,
    pub d_base: u32,
    pub d_chiral: u32,
    pub drag: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HiggsResult {
    pub mean_drag: f64,
    pub median_drag: f64,
    pub std_drag: f64,
    pub mean_d_base: f64,
    pub mean_d_chiral: f64,
    pub n_pairs_sampled: usize,
    pub n_reachable_both: usize,
    pub n_reachable_photon: usize,
    pub n_reachable_weak: usize,
    pub left_fraction: f64,
    pub per_pair: Vec<HiggsPairResult>,
    pub cdf_n_sources: usize,
    pub cdf_photon_frontier: Vec<f64>,
    pub cdf_weak_frontier: Vec<f64>,
    pub cdf_photon_cumulative: Vec<f64>,
    pub cdf_weak_cumulative: Vec<f64>,
    pub cdf_area_ratio: Vec<f64>,
}

// ── Utilities ────────────────────────────────────────────────────────────────

/// BFS shortest path from `src` to `dst` on a CSR graph with optional node filter.
fn bfs_shortest_path(
    src: usize, dst: usize,
    head: &[u32], data: &[u32],
    n: usize,
    filter: Option<&[bool]>,
    max_dist: u32,
) -> Option<u32> {
    if src == dst { return Some(0); }
    if src >= n || dst >= n { return None; }
    let mut visited = vec![false; n];
    visited[src] = true;
    let mut current = vec![src];

    for d in 1..=max_dist {
        let mut next = Vec::new();
        for &u in &current {
            let s = head[u] as usize;
            let e = head[u + 1] as usize;
            for &v in &data[s..e] {
                let vi = v as usize;
                if vi >= n || visited[vi] { continue; }
                // Destination exempt from filter -- always absorb the boson
                if vi == dst { return Some(d); }
                // Intermediate nodes must satisfy chirality constraint
                if let Some(f) = filter {
                    if !f[vi] { continue; }
                }
                visited[vi] = true;
                next.push(vi);
            }
        }
        if next.is_empty() { return None; }
        current = next;
    }
    None
}

/// Full BFS ball volume: expand from `src` up to `max_depth`.
fn bfs_ball_volume(
    src: usize,
    head: &[u32], data: &[u32],
    n: usize,
    filter: Option<&[bool]>,
    max_depth: usize,
) -> Vec<usize> {
    let mut visited = vec![false; n];
    visited[src] = true;
    let mut current = vec![src];
    let mut frontier_sizes = Vec::with_capacity(max_depth + 1);
    frontier_sizes.push(1);

    for _d in 1..=max_depth {
        let mut next = Vec::new();
        for &u in &current {
            if u >= n { continue; }
            let s = head[u] as usize;
            let e = head[u + 1] as usize;
            for &v in &data[s..e] {
                let vi = v as usize;
                if vi >= n || visited[vi] { continue; }
                if let Some(f) = filter {
                    if !f[vi] { continue; }
                }
                visited[vi] = true;
                next.push(vi);
            }
        }
        frontier_sizes.push(next.len());
        if next.is_empty() { break; }
        current = next;
    }
    while frontier_sizes.len() < max_depth + 1 {
        frontier_sizes.push(0);
    }
    frontier_sizes
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> HiggsResult {
    let n = ctx.n_points;
    let (sym_vac_head, sym_vac_data) = ctx.sym_vacuum.raw();
    let (vac_head, _vac_data) = ctx.vacuum_csr.raw();
    let n_pairs = 200usize;

    let empty = HiggsResult {
        mean_drag: 0.0, median_drag: 0.0, std_drag: 0.0,
        mean_d_base: 0.0, mean_d_chiral: 0.0,
        n_pairs_sampled: 0, n_reachable_both: 0,
        n_reachable_photon: 0, n_reachable_weak: 0,
        left_fraction: 0.0, per_pair: vec![],
        cdf_n_sources: 0,
        cdf_photon_frontier: vec![], cdf_weak_frontier: vec![],
        cdf_photon_cumulative: vec![], cdf_weak_cumulative: vec![],
        cdf_area_ratio: vec![],
    };
    if ctx.prisms.len() < 8 { return empty; }

    // Step 1: Compute node chirality
    let mut is_left = vec![false; n];
    let mut n_left = 0usize;
    for v in 0..n {
        let out_deg = vac_head[v + 1] - vac_head[v];
        let sym_deg = sym_vac_head[v + 1] - sym_vac_head[v];
        if sym_deg > 0 && 2 * out_deg < sym_deg {
            is_left[v] = true;
            n_left += 1;
        }
    }
    let left_fraction = n_left as f64 / n as f64;

    // Step 2: Select distant prism pairs (early x late temporal quartiles)
    let mut prism_by_time: Vec<(usize, f64)> = ctx.prisms.iter().enumerate()
        .map(|(i, p)| (i, ctx.pts[p.origin][0]))
        .collect();
    prism_by_time.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let q_size = prism_by_time.len() / 4;
    if q_size == 0 { return empty; }
    let early: Vec<usize> = prism_by_time[..q_size].iter().map(|&(i, _)| i).collect();
    let late: Vec<usize> = prism_by_time[prism_by_time.len() - q_size..].iter()
        .map(|&(i, _)| i).collect();

    let mut rng = StdRng::seed_from_u64(ctx.seed);
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n_pairs);
    for _ in 0..(n_pairs * 10) {
        if pairs.len() >= n_pairs { break; }
        let a = early[rng.gen_range(0..early.len())];
        let b = late[rng.gen_range(0..late.len())];
        pairs.push((a, b));
    }
    let n_pairs_sampled = pairs.len();

    // Step 3: Parallel BFS for each pair
    let max_dist = 200u32;
    let pair_results: Vec<(bool, bool, Option<HiggsPairResult>)> = pairs
        .par_iter()
        .map(|&(a_idx, b_idx)| {
            let src = ctx.prisms[a_idx].origin;
            let dst = ctx.prisms[b_idx].origin;

            let d_base = match bfs_shortest_path(
                src, dst, sym_vac_head, sym_vac_data, n, None, max_dist,
            ) {
                Some(d) => d,
                None => return (false, false, None),
            };
            if d_base == 0 { return (true, true, None); }

            let d_chiral = match bfs_shortest_path(
                src, dst, sym_vac_head, sym_vac_data, n, Some(&is_left), max_dist,
            ) {
                Some(d) => d,
                None => return (true, false, None),
            };

            let drag = (d_chiral as f64 - d_base as f64) / d_base as f64;
            (true, true, Some(HiggsPairResult {
                origin_idx: a_idx,
                dest_idx: b_idx,
                d_base,
                d_chiral,
                drag,
            }))
        })
        .collect();

    // Step 4: Aggregate statistics
    let n_reachable_photon = pair_results.iter().filter(|r| r.0).count();
    let n_reachable_weak = pair_results.iter().filter(|r| r.1).count();
    let valid: Vec<HiggsPairResult> = pair_results.into_iter()
        .filter_map(|(_, _, r)| r)
        .collect();
    let n_reachable_both = valid.len();

    if valid.is_empty() {
        return HiggsResult {
            left_fraction,
            n_pairs_sampled,
            n_reachable_photon,
            n_reachable_weak,
            n_reachable_both: 0,
            ..empty
        };
    }

    let mean_drag = valid.iter().map(|p| p.drag).sum::<f64>() / valid.len() as f64;
    let mean_d_base = valid.iter().map(|p| p.d_base as f64).sum::<f64>() / valid.len() as f64;
    let mean_d_chiral = valid.iter().map(|p| p.d_chiral as f64).sum::<f64>() / valid.len() as f64;

    let variance = valid.iter()
        .map(|p| (p.drag - mean_drag).powi(2))
        .sum::<f64>() / valid.len() as f64;
    let std_drag = variance.sqrt();

    let mut drags: Vec<f64> = valid.iter().map(|p| p.drag).collect();
    drags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_drag = if drags.len() % 2 == 0 {
        (drags[drags.len() / 2 - 1] + drags[drags.len() / 2]) / 2.0
    } else {
        drags[drags.len() / 2]
    };

    // Step 5: CDF diagnostic -- BFS ball volume for area-law verification
    let cdf_max_depth = 15usize;
    let cdf_n_sources = 20.min(prism_by_time.len());
    let stride = if cdf_n_sources > 0 { prism_by_time.len() / cdf_n_sources } else { 1 };
    let cdf_sources: Vec<usize> = (0..cdf_n_sources)
        .map(|i| ctx.prisms[prism_by_time[i * stride].0].origin)
        .collect();

    let cdf_results: Vec<(Vec<usize>, Vec<usize>)> = cdf_sources
        .par_iter()
        .map(|&src| {
            let photon = bfs_ball_volume(
                src, sym_vac_head, sym_vac_data, n, None, cdf_max_depth,
            );
            let weak = bfs_ball_volume(
                src, sym_vac_head, sym_vac_data, n, Some(&is_left), cdf_max_depth,
            );
            (photon, weak)
        })
        .collect();

    let nd = cdf_max_depth + 1;
    let m_cdf = cdf_results.len().max(1) as f64;
    let mut cdf_photon_frontier = vec![0.0f64; nd];
    let mut cdf_weak_frontier = vec![0.0f64; nd];
    for (pf, wf) in &cdf_results {
        for d in 0..nd {
            cdf_photon_frontier[d] += *pf.get(d).unwrap_or(&0) as f64;
            cdf_weak_frontier[d] += *wf.get(d).unwrap_or(&0) as f64;
        }
    }
    for d in 0..nd {
        cdf_photon_frontier[d] /= m_cdf;
        cdf_weak_frontier[d] /= m_cdf;
    }

    // Cumulative sums
    let mut cdf_photon_cumulative = vec![0.0f64; nd];
    let mut cdf_weak_cumulative = vec![0.0f64; nd];
    let mut pc = 0.0;
    let mut wc = 0.0;
    for d in 0..nd {
        pc += cdf_photon_frontier[d];
        wc += cdf_weak_frontier[d];
        cdf_photon_cumulative[d] = pc;
        cdf_weak_cumulative[d] = wc;
    }

    // Area ratio: weak / photon at each depth
    let cdf_area_ratio: Vec<f64> = (0..nd)
        .map(|d| {
            if cdf_photon_cumulative[d] > 0.0 {
                cdf_weak_cumulative[d] / cdf_photon_cumulative[d]
            } else {
                1.0
            }
        })
        .collect();

    HiggsResult {
        mean_drag,
        median_drag,
        std_drag,
        mean_d_base,
        mean_d_chiral,
        n_pairs_sampled,
        n_reachable_both,
        n_reachable_photon,
        n_reachable_weak,
        left_fraction,
        per_pair: valid,
        cdf_n_sources,
        cdf_photon_frontier,
        cdf_weak_frontier,
        cdf_photon_cumulative,
        cdf_weak_cumulative,
        cdf_area_ratio,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[HiggsResult]) -> HiggsResult {
    let m = results.len() as f64;
    HiggsResult {
        mean_drag: results.iter().map(|h| h.mean_drag).sum::<f64>() / m,
        median_drag: results.iter().map(|h| h.median_drag).sum::<f64>() / m,
        std_drag: results.iter().map(|h| h.std_drag).sum::<f64>() / m,
        mean_d_base: results.iter().map(|h| h.mean_d_base).sum::<f64>() / m,
        mean_d_chiral: results.iter().map(|h| h.mean_d_chiral).sum::<f64>() / m,
        n_pairs_sampled: (results.iter().map(|h| h.n_pairs_sampled).sum::<usize>() as f64 / m) as usize,
        n_reachable_both: (results.iter().map(|h| h.n_reachable_both).sum::<usize>() as f64 / m) as usize,
        n_reachable_photon: (results.iter().map(|h| h.n_reachable_photon).sum::<usize>() as f64 / m) as usize,
        n_reachable_weak: (results.iter().map(|h| h.n_reachable_weak).sum::<usize>() as f64 / m) as usize,
        left_fraction: results.iter().map(|h| h.left_fraction).sum::<f64>() / m,
        per_pair: vec![],
        cdf_n_sources: 0,
        cdf_photon_frontier: vec![],
        cdf_weak_frontier: vec![],
        cdf_photon_cumulative: vec![],
        cdf_weak_cumulative: vec![],
        cdf_area_ratio: vec![],
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &HiggsResult, w: &mut CsvWriter) {
    w.comment("M9 Higgs Mechanism (topological drag)");
    w.header(&["origin_idx", "dest_idx", "d_base", "d_chiral", "drag"]);
    for pr in &result.per_pair {
        w.row_fmt(format_args!(
            "{},{},{},{},{:.6}",
            pr.origin_idx, pr.dest_idx, pr.d_base, pr.d_chiral, pr.drag
        ));
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &HiggsResult) {
    println!("  [M9] Higgs Mechanism:");
    println!("    Pairs sampled:   {}", result.n_pairs_sampled);
    println!("    Reachable (both):{}", result.n_reachable_both);
    println!("    Mean drag:       {:.6}", result.mean_drag);
    println!("    Median drag:     {:.6}", result.median_drag);
    println!("    Std drag:        {:.6}", result.std_drag);
    println!("    Mean d_base:     {:.2}  d_chiral: {:.2}", result.mean_d_base, result.mean_d_chiral);
    println!("    Left fraction:   {:.4}", result.left_fraction);
    if !result.cdf_area_ratio.is_empty() {
        println!("    CDF area ratio:  {:.4} (depth {})",
            result.cdf_area_ratio.last().unwrap_or(&0.0),
            result.cdf_area_ratio.len() - 1);
    }
}
