// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M12 — Zigzag Kaluza-Klein Dimension
//!
//! Measures d_zig by alternating forward and backward BFS on the directed
//! vacuum CSR.  This mimics the traversal of the compactified K_{2,2}
//! circle: each diamond has loop length L=4, so alternating BFS steps
//! explore the extra Kaluza-Klein dimension.
//!
//! Expected: d_zig ≈ 5.10
//!
//! The screening ratio S_4 / S_{5.10} = 32π/137 ≈ 0.733 bridges the
//! bare coupling α₀ = 1/(32π) to the observed α ≈ 1/137 via volume
//! dilution in the zigzag geometry.

use super::context::MeasureContext;
use crate::graph::csr::{CsrGraph, Directed};
use crate::output::CsvWriter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const MAX_SOURCES: usize = 500;
const MAX_ZIGZAG_DEPTH: usize = 20;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ZigzagResult {
    pub d_zig: f64,
    pub n_sources: usize,
    pub mean_volume: Vec<f64>,
    /// Screening ratio: S_4 / S_{d_zig} = Γ(d_zig/2+1)/π × π²/Γ(3) ratio
    pub screening_ratio: f64,
}

/// Zigzag BFS: alternate forward/backward steps from a source node.
/// Returns cumulative volume V(d) for d=0..=max_depth.
fn zigzag_bfs(
    forward: &CsrGraph<Directed>,
    backward: &CsrGraph<Directed>,
    source: usize,
    max_depth: usize,
) -> Vec<usize> {
    let n = forward.n_nodes();
    let mut visited = vec![false; n];
    let mut volumes = vec![0usize; max_depth + 1];

    visited[source] = true;
    let mut frontier: Vec<usize> = vec![source];
    volumes[0] = 1;

    for depth in 1..=max_depth {
        // Odd steps = forward, even steps = backward
        let csr: &CsrGraph<Directed> = if depth % 2 == 1 { forward } else { backward };

        let mut new_frontier = Vec::new();
        for &node in &frontier {
            for &nb in csr.neighbors(node) {
                let nb = nb as usize;
                if nb < n && !visited[nb] {
                    visited[nb] = true;
                    new_frontier.push(nb);
                }
            }
        }
        frontier = new_frontier;
        let prev = volumes[depth - 1];
        volumes[depth] = prev + frontier.len();
    }

    volumes
}

/// Fit d_zig from log V(d) vs log d via least-squares.
fn fit_dimension(mean_volumes: &[f64], _n_points: usize) -> f64 {
    let mut log_r = Vec::new();
    let mut log_v = Vec::new();

    for d in 1..mean_volumes.len() {
        let v = mean_volumes[d];
        if v <= 1.0 {
            continue;
        }
        if d > 1 && mean_volumes[d - 1] > 1.0 {
            let growth = (v - mean_volumes[d - 1]) / mean_volumes[d - 1];
            if growth < 0.05 {
                break;
            }
        }
        log_r.push((d as f64).ln());
        log_v.push(v.ln());
    }

    if log_r.len() < 2 {
        return 0.0;
    }

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

/// Compute S_d = 2π^{d/2} / Γ(d/2) (surface area of unit (d-1)-sphere).
fn sphere_surface(d: f64) -> f64 {
    use std::f64::consts::PI;
    let half_d = d / 2.0;
    2.0 * PI.powf(half_d) / gamma(half_d)
}

/// Lanczos approximation of Γ(z) for z > 0.
fn gamma(z: f64) -> f64 {
    if z < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * z).sin() * gamma(1.0 - z));
    }
    let z = z - 1.0;
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    let mut x = 0.999_999_999_999_809_3_f64;
    for (i, &pi) in p.iter().enumerate() {
        x += pi / (z + i as f64 + 1.0);
    }
    let t = z + p.len() as f64 - 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

pub fn run(ctx: &MeasureContext, rev_csr: &CsrGraph<Directed>) -> ZigzagResult {
    let n = ctx.n_points;
    let n_sources = MAX_SOURCES.min(n / 20).max(10);
    let max_depth = MAX_ZIGZAG_DEPTH.min(n / 2);

    // Sample from middle 60% of time range
    let mut rng = StdRng::seed_from_u64(ctx.seed.wrapping_add(1200));
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

    let mut sum_vol = vec![0.0f64; max_depth + 1];
    for &src in &sources {
        let vol = zigzag_bfs(ctx.vacuum_csr, rev_csr, src, max_depth);
        for d in 0..=max_depth {
            sum_vol[d] += vol[d] as f64;
        }
    }
    let mean_vol: Vec<f64> = sum_vol.iter().map(|&s| s / n_sources as f64).collect();

    let d_zig = fit_dimension(&mean_vol, n);

    // Screening ratio: S_4 / S_{d_zig}
    let screening = if d_zig > 0.0 {
        sphere_surface(4.0) / sphere_surface(d_zig)
    } else {
        0.0
    };

    ZigzagResult {
        d_zig,
        n_sources,
        mean_volume: mean_vol,
        screening_ratio: screening,
    }
}

pub fn aggregate(results: &[ZigzagResult]) -> ZigzagResult {
    let n = results.len() as f64;
    let d_zig = results.iter().map(|r| r.d_zig).sum::<f64>() / n;
    let total_sources: usize = results.iter().map(|r| r.n_sources).sum();

    let max_len = results.iter().map(|r| r.mean_volume.len()).max().unwrap_or(0);
    let mut avg = vec![0.0f64; max_len];
    for r in results {
        for (i, &v) in r.mean_volume.iter().enumerate() {
            avg[i] += v;
        }
    }
    for v in avg.iter_mut() { *v /= n; }

    let screening = if d_zig > 0.0 {
        sphere_surface(4.0) / sphere_surface(d_zig)
    } else {
        0.0
    };

    ZigzagResult {
        d_zig,
        n_sources: total_sources,
        mean_volume: avg,
        screening_ratio: screening,
    }
}

pub fn write_csv(result: &ZigzagResult, w: &mut CsvWriter) {
    w.comment("M12 Zigzag Kaluza-Klein Dimension");
    w.header(&["key", "value"]);
    w.row_fmt(format_args!("d_zig,{:.6}", result.d_zig));
    w.row_fmt(format_args!("screening_ratio,{:.6}", result.screening_ratio));
    w.row_fmt(format_args!("n_sources,{}", result.n_sources));
    for (d, &v) in result.mean_volume.iter().enumerate() {
        w.row_fmt(format_args!("V_zig_d{},{:.2}", d, v));
    }
}

pub fn print_summary(result: &ZigzagResult) {
    println!("  [M12] Zigzag KK Dimension:");
    println!("    d_zig:             {:.3}  (expected: ~5.10)", result.d_zig);
    println!("    S_4/S_{{d_zig}}:     {:.4}  (expected: 32pi/137 = {:.4})",
        result.screening_ratio, 32.0 * std::f64::consts::PI / 137.036);
    println!("    Sources sampled:   {}", result.n_sources);
}
