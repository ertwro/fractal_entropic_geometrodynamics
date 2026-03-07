//! Per-prism causal diamond embedding diagnostic.
//!
//! Usage: cargo run --release --example diamond_diag [N] [seed]
//! Default: N=100000, seed=42

use feg_prism::{phase1, phase2};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(42);

    println!("=== Causal Diamond Embedding Diagnostic ===");
    println!("  N = {},  seed = {}\n", n, seed);

    // Phase 1: sprinkle + Hasse diagram (keep sorted_coords)
    let mut rng = StdRng::seed_from_u64(seed);
    let (pts_raw, _half_t) = phase1::sprinkle(n, &mut rng);
    let (sorted_coords, vacuum_csr, momentum) = phase1::build_hasse_direct(&pts_raw);
    drop(pts_raw);

    // Phase 2: Kuratowski contraction
    let (defect, _topo, prisms) = phase2::apply_defect(n, vacuum_csr, momentum);

    if prisms.is_empty() {
        println!("No prisms detected — nothing to measure.");
        return;
    }

    println!("[Diamond] Computing causal diamonds for {} prisms...", prisms.len());

    // Density ρ = 1.0 by construction (V = N for Poisson sprinkling)
    let density = 1.0;
    let diamonds = phase2::compute_all_diamonds(
        &prisms,
        &defect.vacuum_csr,
        &sorted_coords,
        density,
        &defect.generations,
    );

    // Partition by generation
    let mut by_gen: [Vec<&phase2::DiamondStats>; 4] = Default::default();
    for d in &diamonds {
        let g = (d.generation as usize).min(3);
        by_gen[g].push(d);
    }

    // ── 1. Per-generation summary ──
    println!("\n{}", "─".repeat(78));
    println!("  Per-Generation Summary");
    println!("{}", "─".repeat(78));
    println!(
        "{:>5} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Gen", "Count", "τ mean", "τ median", "τ std",
        "Vol mean", "Chain mean", "Paths mean"
    );

    let gen_labels = ["?", "1", "2", "3"];
    let mut gen_means: [[f64; 4]; 4] = [[0.0; 4]; 4]; // [gen][tau, vol, chain, paths]

    for g in 1..=3 {
        let group = &by_gen[g];
        if group.is_empty() {
            println!("{:>5} {:>6}", gen_labels[g], 0);
            continue;
        }
        let count = group.len();

        let mut taus: Vec<f64> = group.iter().map(|d| d.tau).collect();
        let mut vols: Vec<f64> = group.iter().map(|d| d.diamond_vol as f64).collect();
        let mut chains: Vec<f64> = group.iter().map(|d| d.longest_chain as f64).collect();
        let paths: Vec<f64> = group.iter().map(|d| d.chain_count as f64).collect();

        taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
        chains.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let tau_mean = mean(&taus);
        let tau_med = median(&taus);
        let tau_std = std_dev(&taus);
        let vol_mean = mean(&vols);
        let chain_mean = mean(&chains);
        let paths_mean = mean(&paths);

        gen_means[g] = [tau_mean, vol_mean, chain_mean, paths_mean];

        println!(
            "{:>5} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>10.1} {:>10.2} {:>12.1}",
            gen_labels[g], count, tau_mean, tau_med, tau_std,
            vol_mean, chain_mean, paths_mean
        );
    }

    // ── 2. Generation separation ratios ──
    println!("\n{}", "─".repeat(78));
    println!("  Generation Separation (ratio to Gen1)");
    println!("{}", "─".repeat(78));

    let qty_names = ["τ", "Vol", "Chain", "Paths"];
    if gen_means[1][0] > 0.0 {
        println!("{:>10} {:>12} {:>12}", "Quantity", "Gen2/Gen1", "Gen3/Gen1");
        for (i, name) in qty_names.iter().enumerate() {
            let r21 = if gen_means[1][i] > 0.0 {
                gen_means[2][i] / gen_means[1][i]
            } else {
                0.0
            };
            let r31 = if gen_means[1][i] > 0.0 {
                gen_means[3][i] / gen_means[1][i]
            } else {
                0.0
            };
            println!("{:>10} {:>12.4} {:>12.4}", name, r21, r31);
        }
    } else {
        println!("  (No Gen1 prisms — cannot compute ratios)");
    }

    // ── 3. Top-10 largest diamonds ──
    println!("\n{}", "─".repeat(78));
    println!("  Top-10 Largest Diamonds");
    println!("{}", "─".repeat(78));
    println!(
        "{:>6} {:>5} {:>8} {:>10} {:>8} {:>12} {:>10}",
        "Prism", "Gen", "Belly", "Volume", "Chain", "Paths", "τ"
    );

    let mut sorted_by_vol: Vec<&phase2::DiamondStats> = diamonds.iter().collect();
    sorted_by_vol.sort_by(|a, b| b.diamond_vol.cmp(&a.diamond_vol));
    for d in sorted_by_vol.iter().take(10) {
        println!(
            "{:>6} {:>5} {:>8} {:>10} {:>8} {:>12} {:>10.4}",
            d.prism_idx, d.generation, d.belly_size, d.diamond_vol,
            d.longest_chain, d.chain_count, d.tau
        );
    }

    // ── 4. Chain count statistics by generation ──
    println!("\n{}", "─".repeat(78));
    println!("  Chain Count (Paths) Statistics by Generation");
    println!("{}", "─".repeat(78));
    println!(
        "{:>5} {:>12} {:>12} {:>12} {:>12}",
        "Gen", "Min", "Median", "Mean", "Max"
    );

    for g in 1..=3 {
        let group = &by_gen[g];
        if group.is_empty() {
            continue;
        }
        let mut paths: Vec<u64> = group.iter().map(|d| d.chain_count).collect();
        paths.sort();
        let min = paths[0];
        let max = *paths.last().unwrap();
        let med = paths[paths.len() / 2];
        let avg = paths.iter().map(|&p| p as f64).sum::<f64>() / paths.len() as f64;
        println!(
            "{:>5} {:>12} {:>12} {:>12.1} {:>12}",
            gen_labels[g], min, med, avg, max
        );
    }

    // ── 5. Density ratio statistics ──
    println!("\n{}", "─".repeat(78));
    println!("  Density Ratio (measured/flat) by Generation");
    println!("{}", "─".repeat(78));
    println!("{:>5} {:>12} {:>12} {:>12}", "Gen", "Mean", "Median", "Std");

    for g in 1..=3 {
        let group = &by_gen[g];
        if group.is_empty() {
            continue;
        }
        let mut ratios: Vec<f64> = group.iter().map(|d| d.density_ratio).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{:>5} {:>12.4} {:>12.4} {:>12.4}",
            gen_labels[g], mean(&ratios), median(&ratios), std_dev(&ratios)
        );
    }

    // ── Sanity checks ──
    println!("\n{}", "─".repeat(78));
    println!("  Sanity Checks");
    println!("{}", "─".repeat(78));

    let n_tau_zero = diamonds.iter().filter(|d| d.tau <= 0.0).count();
    let n_vol_small = diamonds
        .iter()
        .filter(|d| d.diamond_vol < d.belly_size + 2)
        .count();
    let n_chain_short = diamonds.iter().filter(|d| d.longest_chain < 2).count();

    println!("  tau <= 0:            {} / {}", n_tau_zero, diamonds.len());
    println!("  vol < belly+2:       {} / {}", n_vol_small, diamonds.len());
    println!("  longest_chain < 2:   {} / {}", n_chain_short, diamonds.len());

    if n_tau_zero == 0 && n_vol_small == 0 && n_chain_short == 0 {
        println!("  All sanity checks PASSED.");
    } else {
        println!("  WARNING: some sanity checks failed.");
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    // Assumes v is already sorted
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}
