//! Volume growth d_H measurement via BFS on the undirected Hasse graph.
//!
//! Measures V(r) = number of nodes within graph distance r from random origins.
//! Fits V(r) ~ r^{d_H} to extract the Hausdorff dimension.
//!
//! Unlike the spectral dimension d_S (which uses a lazy random walk and suffers
//! from lattice artifacts at short times), d_H directly probes the graph's
//! geometric expansion rate and is insensitive to walk parity issues.
//!
//! Usage: cargo run --release --example volume_growth [N] [n_origins] [seed]
//! Default: N=100000, n_origins=500, seed=42

use feg_prism::phase1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// BFS from a single origin on the undirected CSR graph.
/// Returns shell_counts[r] = number of nodes at exactly distance r.
/// r=0 is the origin itself.
fn bfs_shells(
    head: &[u32],
    data: &[u32],
    origin: usize,
    n: usize,
) -> Vec<usize> {
    let mut dist = vec![u32::MAX; n];
    let mut queue = VecDeque::new();
    dist[origin] = 0;
    queue.push_back(origin);

    let mut max_r = 0u32;

    while let Some(u) = queue.pop_front() {
        let d = dist[u];
        let start = head[u] as usize;
        let end = head[u + 1] as usize;
        for &v in &data[start..end] {
            let vi = v as usize;
            if dist[vi] == u32::MAX {
                dist[vi] = d + 1;
                if d + 1 > max_r {
                    max_r = d + 1;
                }
                queue.push_back(vi);
            }
        }
    }

    let mut shells = vec![0usize; (max_r + 1) as usize];
    for &d in &dist {
        if d != u32::MAX {
            shells[d as usize] += 1;
        }
    }
    shells
}

/// Least-squares fit of log V vs log r (excluding r=0).
/// Returns (d_H, intercept, R²).
fn fit_power_law(r_vals: &[f64], v_vals: &[f64]) -> (f64, f64, f64) {
    let n = r_vals.len() as f64;
    let sum_x: f64 = r_vals.iter().sum();
    let sum_y: f64 = v_vals.iter().sum();
    let sum_xx: f64 = r_vals.iter().map(|x| x * x).sum();
    let sum_xy: f64 = r_vals.iter().zip(v_vals.iter()).map(|(x, y)| x * y).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    // R²
    let mean_y = sum_y / n;
    let ss_tot: f64 = v_vals.iter().map(|y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = r_vals
        .iter()
        .zip(v_vals.iter())
        .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
        .sum();
    let r_sq = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    (slope, intercept, r_sq)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let n_origins: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(42);

    println!("=== Volume Growth (Hausdorff Dimension) ===");
    println!("  N = {},  origins = {},  seed = {}\n", n, n_origins, seed);

    // Phase 1: sprinkle + Hasse
    let mut rng = StdRng::seed_from_u64(seed);
    let (pts_raw, _half_t) = phase1::sprinkle(n, &mut rng);
    let (_sorted_coords, vacuum_csr, _momentum) = phase1::build_hasse_direct(&pts_raw);
    drop(pts_raw);

    // Degree statistics on the directed Hasse
    println!("[1] Directed Hasse degree statistics...");
    {
        let mut out_degrees = vec![0u32; n];
        let mut in_degrees = vec![0u32; n];
        for u in 0..n {
            let deg = vacuum_csr.degree(u);
            out_degrees[u] = deg as u32;
            for &v in vacuum_csr.neighbors(u) {
                in_degrees[v as usize] += 1;
            }
        }
        let total_edges: usize = out_degrees.iter().map(|&d| d as usize).sum();
        let avg_out: f64 = total_edges as f64 / n as f64;
        let avg_in: f64 = in_degrees.iter().map(|&d| d as f64).sum::<f64>() / n as f64;
        let max_in = in_degrees.iter().max().copied().unwrap_or(0);
        let max_out = out_degrees.iter().max().copied().unwrap_or(0);

        // Interior nodes: middle 50% by index
        let lo = n / 4;
        let hi = 3 * n / 4;
        let int_out: f64 = out_degrees[lo..hi].iter().map(|&d| d as f64).sum::<f64>()
            / (hi - lo) as f64;
        let int_in: f64 = in_degrees[lo..hi].iter().map(|&d| d as f64).sum::<f64>()
            / (hi - lo) as f64;

        println!("  Total directed edges: {}", total_edges);
        println!("  Global:  avg_out={:.2}, avg_in={:.2}, max_out={}, max_in={}",
                 avg_out, avg_in, max_out, max_in);
        println!("  Interior (25-75%): avg_out={:.2}, avg_in={:.2}",
                 int_out, int_in);
        println!("  Interior undirected degree: {:.2}", int_out + int_in);

        // In-degree histogram (buckets of 5)
        let mut in_hist = vec![0usize; 20];
        for &d in &in_degrees {
            let bucket = (d as usize / 5).min(19);
            in_hist[bucket] += 1;
        }
        println!("  In-degree histogram (bucket size 5):");
        for (i, &count) in in_hist.iter().enumerate() {
            if count > 0 {
                println!("    {:3}-{:3}: {} ({:.1}%)",
                         i * 5, (i + 1) * 5 - 1, count,
                         100.0 * count as f64 / n as f64);
            }
        }
    }

    // Symmetrize for undirected BFS
    println!("\n[2] Symmetrizing Hasse diagram...");
    let sym = vacuum_csr.symmetrize();

    // Directed BFS (causal, forward-only)
    println!("[3] Directed (causal) volume growth...");
    let (dir_head, dir_data) = vacuum_csr.raw();

    let margin = n / 10;
    let dir_origins: Vec<usize> = {
        let mut rng2 = StdRng::seed_from_u64(seed + 1);
        // For directed BFS, start from early nodes (first 30%) to have room to grow
        (0..n_origins.min(200))
            .map(|_| rng2.gen_range(margin..n / 3))
            .collect()
    };

    let mut dir_max_shells = 0usize;
    let dir_all_shells: Vec<Vec<usize>> = dir_origins
        .iter()
        .map(|&o| {
            let s = bfs_shells(dir_head, dir_data, o, n);
            if s.len() > dir_max_shells {
                dir_max_shells = s.len();
            }
            s
        })
        .collect();

    let mut dir_avg_shell = vec![0.0f64; dir_max_shells];
    let mut dir_count = vec![0usize; dir_max_shells];
    for shells in &dir_all_shells {
        for (r, &c) in shells.iter().enumerate() {
            dir_avg_shell[r] += c as f64;
            dir_count[r] += 1;
        }
    }
    for r in 0..dir_max_shells {
        if dir_count[r] > 0 { dir_avg_shell[r] /= dir_count[r] as f64; }
    }
    let mut dir_cum = vec![0.0f64; dir_max_shells];
    dir_cum[0] = dir_avg_shell[0];
    for r in 1..dir_max_shells { dir_cum[r] = dir_cum[r - 1] + dir_avg_shell[r]; }

    println!("\n  Directed (causal) volume growth:");
    println!("  {:>5} {:>12} {:>12} {:>10}", "r", "shell(r)", "V(r)", "d_H(r)");
    for r in 0..dir_max_shells.min(30) {
        let d_h = if r >= 2 && dir_cum[r] > 0.0 && dir_cum[r - 1] > 0.0 {
            (dir_cum[r].ln() - dir_cum[r - 1].ln())
                / ((r as f64).ln() - ((r - 1) as f64).ln())
        } else {
            0.0
        };
        println!("  {:5} {:12.1} {:12.1} {:10.3}", r, dir_avg_shell[r], dir_cum[r], d_h);
    }

    // Undirected BFS
    let (head, data) = sym.raw();
    let origins: Vec<usize> = (0..n_origins)
        .map(|_| rng.gen_range(margin..n - margin))
        .collect();

    println!("[3] Running BFS from {} origins...", n_origins);

    // BFS from each origin
    let mut max_shells = 0usize;
    let all_shells: Vec<Vec<usize>> = origins
        .iter()
        .map(|&o| {
            let s = bfs_shells(head, data, o, n);
            if s.len() > max_shells {
                max_shells = s.len();
            }
            s
        })
        .collect();

    // Average shell counts and cumulative volume
    let mut avg_shell = vec![0.0f64; max_shells];
    let mut count_at_r = vec![0usize; max_shells]; // how many origins reached this r
    for shells in &all_shells {
        for (r, &c) in shells.iter().enumerate() {
            avg_shell[r] += c as f64;
            count_at_r[r] += 1;
        }
    }
    for r in 0..max_shells {
        if count_at_r[r] > 0 {
            avg_shell[r] /= count_at_r[r] as f64;
        }
    }

    // Cumulative volume
    let mut cum_vol = vec![0.0f64; max_shells];
    cum_vol[0] = avg_shell[0];
    for r in 1..max_shells {
        cum_vol[r] = cum_vol[r - 1] + avg_shell[r];
    }

    // Print raw data
    println!("\n{:>5} {:>12} {:>12} {:>12} {:>8}", "r", "shell(r)", "V(r)", "ln V", "origins");
    for r in 0..max_shells.min(60) {
        let lnv = if cum_vol[r] > 0.0 { cum_vol[r].ln() } else { 0.0 };
        println!(
            "{:5} {:12.1} {:12.1} {:12.4} {:8}",
            r, avg_shell[r], cum_vol[r], lnv, count_at_r[r]
        );
    }

    // Fit d_H in different r-windows
    println!("\n=== Power-law fits V(r) ~ r^d_H ===");
    println!("{:>12} {:>8} {:>12} {:>8}", "r-range", "d_H", "intercept", "R²");

    let fit_windows = [
        (1, 5),
        (2, 8),
        (3, 10),
        (5, 15),
        (3, 15),
        (5, 20),
        (3, 20),
        (10, 30),
    ];

    for &(r_lo, r_hi) in &fit_windows {
        let r_hi = r_hi.min(max_shells - 1);
        if r_lo >= r_hi || r_lo >= max_shells {
            continue;
        }

        let log_r: Vec<f64> = (r_lo..=r_hi)
            .filter(|&r| cum_vol[r] > 0.0 && count_at_r[r] > n_origins / 2)
            .map(|r| (r as f64).ln())
            .collect();
        let log_v: Vec<f64> = (r_lo..=r_hi)
            .filter(|&r| cum_vol[r] > 0.0 && count_at_r[r] > n_origins / 2)
            .map(|r| cum_vol[r].ln())
            .collect();

        if log_r.len() < 3 {
            continue;
        }

        let (d_h, intercept, r_sq) = fit_power_law(&log_r, &log_v);
        println!(
            "{:>5}-{:<5} {:8.3} {:12.3} {:8.4}",
            r_lo, r_hi, d_h, intercept, r_sq
        );
    }

    // Also compute local d_H = d(ln V)/d(ln r) at each r
    println!("\n=== Local d_H(r) = Δ(ln V)/Δ(ln r) ===");
    println!("{:>5} {:>10}", "r", "d_H(r)");
    for r in 2..max_shells.min(40) {
        if cum_vol[r] > 0.0 && cum_vol[r - 1] > 0.0 {
            let d_h = (cum_vol[r].ln() - cum_vol[r - 1].ln())
                / ((r as f64).ln() - ((r - 1) as f64).ln());
            println!("{:5} {:10.3}", r, d_h);
        }
    }

    println!("\nDone.");
}
