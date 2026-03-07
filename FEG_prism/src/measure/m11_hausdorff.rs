// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M11 — Hausdorff Dimension via BFS Volume Growth
//!
//! Measures the Hausdorff dimension d_H by sampling random source nodes and
//! performing BFS on both the directed (forward-only) and undirected
//! (symmetric) vacuum CSR.  Volume growth V(r) ~ r^{d_H} gives d_H via
//! log-log linear regression.
//!
//! Expected results:
//! - d_H (directed) ≈ 4.0  (the true geometric dimension)
//! - d_H (undirected) ≈ 4.0–4.8  (UV ≈ 4.8, IR → 4.0)
//!
//! The spectral dimension d_S ≈ 2 at UV is a lazy-walk artifact on the
//! tree-like Hasse DAG; Hausdorff dimension via volume growth is the
//! correct geometric observable.

use super::context::MeasureContext;
use crate::graph::csr::CsrGraph;
use crate::output::CsvWriter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

const MAX_SOURCES: usize = 500;
const MAX_DEPTH: usize = 30;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HausdorffResult {
    pub d_h_directed: f64,
    pub d_h_undirected: f64,
    pub n_sources: usize,
    pub mean_volume_directed: Vec<f64>,
    pub mean_volume_undirected: Vec<f64>,
}

/// BFS volume growth on a directed CSR.  Returns cumulative V(r) for r=0..=max_depth.
fn bfs_volume<D>(csr: &CsrGraph<D>, source: usize, max_depth: usize) -> Vec<usize> {
    let n = csr.n_nodes();
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    let mut shell = vec![0usize; max_depth + 1];

    visited[source] = true;
    queue.push_back((source, 0u32));
    shell[0] = 1;

    while let Some((node, depth)) = queue.pop_front() {
        if depth as usize >= max_depth {
            continue;
        }
        for &nb in csr.neighbors(node) {
            let nb = nb as usize;
            if nb < n && !visited[nb] {
                visited[nb] = true;
                shell[(depth + 1) as usize] += 1;
                queue.push_back((nb, depth + 1));
            }
        }
    }

    // Cumulative volume
    let mut volumes = vec![0usize; max_depth + 1];
    let mut cum = 0;
    for r in 0..=max_depth {
        cum += shell[r];
        volumes[r] = cum;
    }
    volumes
}

/// Fit d_H = slope of log V(r) vs log r via least-squares regression.
/// Uses points where V(r) > 1 and not yet saturated (growth stalled).
fn fit_hausdorff(mean_volumes: &[f64], _n_points: usize) -> f64 {
    let mut log_r = Vec::new();
    let mut log_v = Vec::new();

    // Detect saturation: volume growth < 5% from previous step
    for r in 1..mean_volumes.len() {
        let v = mean_volumes[r];
        if v <= 1.0 {
            continue;
        }
        // Stop if saturated (growth stalled)
        if r > 1 && mean_volumes[r - 1] > 1.0 {
            let growth = (v - mean_volumes[r - 1]) / mean_volumes[r - 1];
            if growth < 0.05 {
                break;
            }
        }
        log_r.push((r as f64).ln());
        log_v.push(v.ln());
    }

    if log_r.len() < 2 {
        return 0.0;
    }

    // Least-squares slope
    let n = log_r.len() as f64;
    let sx: f64 = log_r.iter().sum();
    let sy: f64 = log_v.iter().sum();
    let sxy: f64 = log_r.iter().zip(log_v.iter()).map(|(x, y)| x * y).sum();
    let sx2: f64 = log_r.iter().map(|x| x * x).sum();

    let denom = n * sx2 - sx * sx;
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    (n * sxy - sx * sy) / denom
}

pub fn run(ctx: &MeasureContext) -> HausdorffResult {
    let n = ctx.n_points;
    let n_sources = MAX_SOURCES.min(n / 20).max(10);
    let max_depth = MAX_DEPTH.min(n / 2);

    // Sample from middle 60% of time range to avoid boundary effects
    let mut rng = StdRng::seed_from_u64(ctx.seed.wrapping_add(1100));
    let t_min = ctx.sorted_coords.iter().map(|c| c[0]).fold(f64::INFINITY, f64::min);
    let t_max = ctx.sorted_coords.iter().map(|c| c[0]).fold(f64::NEG_INFINITY, f64::max);
    let t_range = t_max - t_min;
    let t_lo = t_min + 0.2 * t_range;
    let t_hi = t_max - 0.2 * t_range;

    let eligible: Vec<usize> = (0..n)
        .filter(|&i| ctx.sorted_coords[i][0] >= t_lo && ctx.sorted_coords[i][0] <= t_hi)
        .collect();

    let sources: Vec<usize> = if eligible.len() >= n_sources {
        (0..n_sources)
            .map(|_| eligible[rng.gen_range(0..eligible.len())])
            .collect()
    } else {
        (0..n_sources).map(|_| rng.gen_range(0..n)).collect()
    };

    // Directed BFS
    let mut sum_dir = vec![0.0f64; max_depth + 1];
    for &src in &sources {
        let vol = bfs_volume(ctx.vacuum_csr, src, max_depth);
        for r in 0..=max_depth {
            sum_dir[r] += vol[r] as f64;
        }
    }
    let mean_dir: Vec<f64> = sum_dir.iter().map(|&s| s / n_sources as f64).collect();

    // Undirected BFS (skipped when sym_vacuum unavailable, e.g. topology-only mode)
    let (d_h_undir, mean_undir) = if let Some(sym) = ctx.sym_vacuum {
        let mut sum_undir = vec![0.0f64; max_depth + 1];
        for &src in &sources {
            let vol = bfs_volume(sym, src, max_depth);
            for r in 0..=max_depth {
                sum_undir[r] += vol[r] as f64;
            }
        }
        let mean: Vec<f64> = sum_undir.iter().map(|&s| s / n_sources as f64).collect();
        let d_h = fit_hausdorff(&mean, n);
        (d_h, mean)
    } else {
        (0.0, vec![0.0; max_depth + 1])
    };

    let d_h_dir = fit_hausdorff(&mean_dir, n);

    HausdorffResult {
        d_h_directed: d_h_dir,
        d_h_undirected: d_h_undir,
        n_sources,
        mean_volume_directed: mean_dir,
        mean_volume_undirected: mean_undir,
    }
}

pub fn aggregate(results: &[HausdorffResult]) -> HausdorffResult {
    let n = results.len() as f64;
    let d_h_dir = results.iter().map(|r| r.d_h_directed).sum::<f64>() / n;
    let d_h_undir = results.iter().map(|r| r.d_h_undirected).sum::<f64>() / n;
    let total_sources: usize = results.iter().map(|r| r.n_sources).sum();

    let max_len = results.iter().map(|r| r.mean_volume_directed.len()).max().unwrap_or(0);
    let mut avg_dir = vec![0.0f64; max_len];
    let mut avg_undir = vec![0.0f64; max_len];
    for r in results {
        for (i, &v) in r.mean_volume_directed.iter().enumerate() {
            avg_dir[i] += v;
        }
        for (i, &v) in r.mean_volume_undirected.iter().enumerate() {
            avg_undir[i] += v;
        }
    }
    for v in avg_dir.iter_mut() { *v /= n; }
    for v in avg_undir.iter_mut() { *v /= n; }

    HausdorffResult {
        d_h_directed: d_h_dir,
        d_h_undirected: d_h_undir,
        n_sources: total_sources,
        mean_volume_directed: avg_dir,
        mean_volume_undirected: avg_undir,
    }
}

pub fn write_csv(result: &HausdorffResult, w: &mut CsvWriter) {
    w.comment("M11 Hausdorff Dimension (BFS volume growth)");
    w.header(&["key", "value"]);
    w.row_fmt(format_args!("d_H_directed,{:.6}", result.d_h_directed));
    w.row_fmt(format_args!("d_H_undirected,{:.6}", result.d_h_undirected));
    w.row_fmt(format_args!("n_sources,{}", result.n_sources));
    for (r, &v) in result.mean_volume_directed.iter().enumerate() {
        w.row_fmt(format_args!("V_dir_r{},{:.2}", r, v));
    }
    for (r, &v) in result.mean_volume_undirected.iter().enumerate() {
        w.row_fmt(format_args!("V_undir_r{},{:.2}", r, v));
    }
}

pub fn print_summary(result: &HausdorffResult) {
    println!("  [M11] Hausdorff Dimension:");
    println!("    d_H (directed BFS):   {:.3}  (expected: 4.0)", result.d_h_directed);
    println!("    d_H (undirected BFS): {:.3}  (expected: 4.0-4.8)", result.d_h_undirected);
    println!("    Sources sampled:      {}", result.n_sources);
}
