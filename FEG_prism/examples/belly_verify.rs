//! Verify the zero-parameter predictions from FEG theory:
//!   1. Belly distribution f(N) ∝ e^{-cN} with c ≈ 0.53
//!   2. Coupon-collector topological mass ratios ≈ 1.434, 1.698
//!   3. WKB tunneling at theoretical c vs fitted c
//!   4. Cover-time dressing M_dyn = κ N H_N

use feg_prism::{phase1, phase2};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    // Run at multiple N to check scaling
    for &n in &[100_000usize, 200_000, 500_000] {
        run_at_n(n);
    }
}

fn run_at_n(n: usize) {
    let seed = 42u64;
    println!("\n{}", "=".repeat(70));
    println!("=== N = {} ===", n);
    println!("{}\n", "=".repeat(70));

    let mut rng = StdRng::seed_from_u64(seed);
    let (pts_raw, _half_t) = phase1::sprinkle(n, &mut rng);
    let (euclidean, vacuum_csr, momentum) = phase1::build_hasse_direct(&pts_raw);
    drop(pts_raw);
    drop(euclidean);

    let (_defect, topo, prisms) = phase2::apply_defect(n, vacuum_csr, momentum);

    // ── 1. Belly size distribution & exponential fit ──
    let mut belly_hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for p in &prisms {
        *belly_hist.entry(p.intermediates.len()).or_insert(0) += 1;
    }

    println!("── Belly Distribution f(N) ∝ e^{{-cN}} ──\n");
    println!("{:>6} {:>8} {:>10} {:>10}", "N", "count", "ln(count)", "c_local");

    let entries: Vec<(usize, usize)> = belly_hist.iter().map(|(&k, &v)| (k, v)).collect();
    let mut c_estimates: Vec<f64> = Vec::new();

    for i in 0..entries.len() {
        let (n_belly, count) = entries[i];
        let ln_count = (count as f64).ln();
        let c_local = if i > 0 && entries[i-1].1 > 0 && count > 0 {
            let prev_ln = (entries[i-1].1 as f64).ln();
            let dn = (n_belly as f64) - (entries[i-1].0 as f64);
            if dn > 0.0 {
                let c = (prev_ln - ln_count) / dn;
                if c > 0.0 && count > 5 { c_estimates.push(c); }
                c
            } else { 0.0 }
        } else { 0.0 };
        if count > 0 {
            println!("{:>6} {:>8} {:>10.3} {:>10.4}", n_belly, count, ln_count,
                if c_local > 0.0 { format!("{c_local:.4}") } else { "—".to_string() });
        }
    }

    // Fit c via linear regression on ln(count) vs N for the tail (N >= 5)
    let tail: Vec<(f64, f64)> = entries.iter()
        .filter(|(n, count)| *n >= 5 && *count > 2)
        .map(|(n, count)| (*n as f64, (*count as f64).ln()))
        .collect();

    let (c_fit, r_sq) = if tail.len() >= 3 {
        let n_t = tail.len() as f64;
        let sum_x: f64 = tail.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = tail.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = tail.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = tail.iter().map(|(x, _)| x * x).sum();
        let slope = (n_t * sum_xy - sum_x * sum_y) / (n_t * sum_x2 - sum_x * sum_x);
        let mean_y = sum_y / n_t;
        let ss_tot: f64 = tail.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
        let intercept = (sum_y - slope * sum_x) / n_t;
        let ss_res: f64 = tail.iter().map(|(x, y)| (y - (slope * x + intercept)).powi(2)).sum();
        let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
        (-slope, r2)
    } else {
        (0.0, 0.0)
    };

    let c_median = if !c_estimates.is_empty() {
        let mut sorted = c_estimates.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    } else { 0.0 };

    println!("\n  c (linear regression, N≥5): {c_fit:.4}  (R² = {r_sq:.4})");
    println!("  c (median of consecutive):  {c_median:.4}");
    println!("  c (theory prediction):      0.53");
    println!("  Deviation from theory:      {:.1}%", ((c_fit - 0.53) / 0.53 * 100.0).abs());

    // ── 2. Coupon-collector topological mass ratios ──
    println!("\n── Coupon-Collector Topological Ratios ──\n");

    let g1 = topo.prisms_gen1;
    let g2 = topo.prisms_gen2;
    let g3 = topo.prisms_gen3;

    let mut sorted_bellies: Vec<usize> = prisms.iter().map(|p| p.intermediates.len()).collect();
    sorted_bellies.sort();
    let total = sorted_bellies.len();

    let mean_g1 = if g1 > 0 { sorted_bellies[..g1].iter().sum::<usize>() as f64 / g1 as f64 } else { 1.0 };
    let mean_g2 = if g2 > 0 && g1+g2 <= total {
        sorted_bellies[g1..g1+g2].iter().sum::<usize>() as f64 / g2 as f64
    } else { 1.0 };
    let mean_g3 = if g3 > 0 && g1+g2 < total {
        sorted_bellies[g1+g2..].iter().sum::<usize>() as f64 / g3 as f64
    } else { 1.0 };

    let r21_topo = mean_g2 / mean_g1;
    let r31_topo = mean_g3 / mean_g1;

    println!("  Mean belly: Gen1={mean_g1:.2}, Gen2={mean_g2:.2}, Gen3={mean_g3:.2}");
    println!("  Topological ratios:");
    println!("    ⟨N⟩₂/⟨N⟩₁ = {r21_topo:.4}  (theory: 1.434, i.i.d: 1.409)");
    println!("    ⟨N⟩₃/⟨N⟩₁ = {r31_topo:.4}  (theory: 1.698, i.i.d: 1.674)");

    // Engine's existing topological mass (from Phase 2)
    let topo_r21 = topo.avg_mass_gen2 / topo.avg_mass_gen1.max(0.01);
    let topo_r31 = topo.avg_mass_gen3 / topo.avg_mass_gen1.max(0.01);
    println!("\n  Engine avg_mass ratios: m₂/m₁={topo_r21:.4}, m₃/m₁={topo_r31:.4}");
    println!("  (avg_mass_gen1={:.2}, gen2={:.2}, gen3={:.2})",
        topo.avg_mass_gen1, topo.avg_mass_gen2, topo.avg_mass_gen3);

    // ── 3. WKB at theoretical c vs fitted c ──
    println!("\n── WKB Tunneling: m ∝ e^{{-c/α}} ──\n");

    let max_belly = sorted_bellies.last().copied().unwrap_or(1) as f64;
    let a1 = mean_g1 / max_belly;
    let a2 = mean_g2 / max_belly;
    let a3 = mean_g3 / max_belly;
    let diff_21 = 1.0/a1 - 1.0/a2;
    let diff_31 = 1.0/a1 - 1.0/a3;

    println!("  α_k = belly_k / max_belly: α₁={a1:.4}, α₂={a2:.4}, α₃={a3:.4}");
    println!("  1/α₁ - 1/α₂ = {diff_21:.4}");
    println!("  1/α₁ - 1/α₃ = {diff_31:.4}\n");

    let sm_21: f64 = 206.768;
    let sm_31: f64 = 3477.23;

    // At c = 0.53 (theory)
    let r21_053 = (0.53 * diff_21).exp();
    let r31_053 = (0.53 * diff_31).exp();
    println!("  c = 0.53 (theory):  m₂/m₁ = {r21_053:.2}, m₃/m₁ = {r31_053:.2}");

    // At c = measured from belly distribution
    let r21_meas = (c_fit * diff_21).exp();
    let r31_meas = (c_fit * diff_31).exp();
    println!("  c = {c_fit:.3} (measured): m₂/m₁ = {r21_meas:.2}, m₃/m₁ = {r31_meas:.2}");

    // At c_fit (best fit to SM)
    let c_best_21 = sm_21.ln() / diff_21;
    let c_best_31 = sm_31.ln() / diff_31;
    let c_best = (c_best_21 + c_best_31) / 2.0;
    let r21_best = (c_best * diff_21).exp();
    let r31_best = (c_best * diff_31).exp();
    println!("  c = {c_best:.3} (SM fit):   m₂/m₁ = {r21_best:.1}, m₃/m₁ = {r31_best:.1}  (c_21={c_best_21:.3}, c_31={c_best_31:.3})");
    println!("  SM target:            m₂/m₁ = {sm_21:.1}, m₃/m₁ = {sm_31:.1}");

    // ── 4. Cover-time dressing ──
    println!("\n── Cover-Time Dressing: M_dyn = κ N H_N ──\n");

    let h = |n: f64| -> f64 {
        // Harmonic number approximation
        if n <= 0.5 { return 0.0; }
        n.ln() + 0.5772 + 1.0/(2.0*n)
    };

    let m1_cover = mean_g1 * h(mean_g1);
    let m2_cover = mean_g2 * h(mean_g2);
    let m3_cover = mean_g3 * h(mean_g3);
    let r21_cover = m2_cover / m1_cover;
    let r31_cover = m3_cover / m1_cover;
    println!("  N·H_N: Gen1={m1_cover:.2}, Gen2={m2_cover:.2}, Gen3={m3_cover:.2}");
    println!("  M_dyn ratios: m₂/m₁={r21_cover:.4}, m₃/m₁={r31_cover:.4}");
    println!("  (superlogarithmic amplification over linear: {:.2}× for Gen3)",
        r31_cover / r31_topo);

    // N · H_N^2 (stronger dressing)
    let m1_c2 = mean_g1 * h(mean_g1).powi(2);
    let m2_c2 = mean_g2 * h(mean_g2).powi(2);
    let m3_c2 = mean_g3 * h(mean_g3).powi(2);
    println!("  N·H_N²: ratios m₂/m₁={:.4}, m₃/m₁={:.4}",
        m2_c2 / m1_c2, m3_c2 / m1_c2);

    // WKB on cover-time dressed masses
    let ac1 = m1_cover / m3_cover;
    let ac2 = m2_cover / m3_cover;
    let ac3 = 1.0;
    let dc_21 = 1.0/ac1 - 1.0/ac2;
    let dc_31 = 1.0/ac1 - 1.0/ac3;
    let cc_21 = sm_21.ln() / dc_21;
    let cc_31 = sm_31.ln() / dc_31;
    println!("\n  WKB on cover-time α (= N·H_N / max):");
    println!("    c_fit_21 = {cc_21:.4}, c_fit_31 = {cc_31:.4}, ratio = {:.4}", cc_21/cc_31);

    // WKB with c_theory on raw belly, then apply H_N correction multiplicatively
    println!("\n── Composed: e^{{-c_theory/α}} × (H_N₃/H_N₁) correction ──");
    let hn_ratio_21 = h(mean_g2) / h(mean_g1);
    let hn_ratio_31 = h(mean_g3) / h(mean_g1);
    let composed_21 = r21_053 * hn_ratio_21;
    let composed_31 = r31_053 * hn_ratio_31;
    println!("  H_N correction factors: ×{hn_ratio_21:.3} (Gen2), ×{hn_ratio_31:.3} (Gen3)");
    println!("  Composed m₂/m₁ = {composed_21:.2}, m₃/m₁ = {composed_31:.2}");

    println!();
}
