//! Finite-Size Scaling sweep for the BD effective Hamiltonian.
//!
//! Generates Poisson-sprinkled causal diamonds at multiple lattice sizes,
//! builds the Benincasa–Dowker matrix, extracts H_eff = (B − Bᵀ)/(2i)
//! eigenvalues via Schur decomposition, and computes the spacing ratio
//! ⟨r⟩(N) for each size.
//!
//! Usage:
//!     cargo run --release --bin fss_rmt [-- OPTIONS]
//!
//! Options:
//!     --sizes 500,1000,1500,2000,2500,3000   Lattice sizes (comma-separated)
//!     --m 20                                  Ensemble size per N
//!     --seed 42                               Base RNG seed

use causal_set_sim::diamond;
use causal_set_sim::jacobson;
use causal_set_sim::rmt;

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;
use std::time::Instant;

/// Above this threshold, use coarse-grained macro pipeline instead of exact BD.
const COARSE_THRESHOLD: usize = 3_000;
/// Default number of macro voxels for the coarse-grained pipeline.
const DEFAULT_N_MACRO: usize = 3_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // ── Parse CLI ────────────────────────────────────────────────────
    let m: usize = parse_flag(&args, "--m").unwrap_or(20);
    let base_seed: u64 = parse_flag(&args, "--seed").unwrap_or(42);

    let lattice_sizes: Vec<usize> = if let Some(pos) = args.iter().position(|a| a == "--sizes") {
        args.get(pos + 1)
            .map(|s| {
                s.split(',')
                    .filter_map(|tok| tok.trim().parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    } else {
        vec![500, 1000, 1500, 2000, 2500, 3000]
    };

    if lattice_sizes.is_empty() {
        eprintln!("No lattice sizes specified.");
        return;
    }

    // ── Header ───────────────────────────────────────────────────────
    println!("════════════════════════════════════════════════════════════════════");
    println!("  Benincasa–Dowker Effective Hamiltonian — GUE Finite-Size Scaling");
    println!("  H_eff = (B − Bᵀ)/(2i)    BD weights = [+1, −9, +16, −8]");
    println!("  Ensemble: M = {} realisations per lattice size", m);
    println!("  Base seed: {}", base_seed);
    println!("════════════════════════════════════════════════════════════════════\n");

    println!(
        "  {:>6}  {:>10}  {:>10}  {:>8}  {:>8}  {:>8}",
        "N", "⟨r⟩", "± SE", "evals", "spacings", "time"
    );
    println!("  {}", "─".repeat(60));

    let mut results: Vec<FssPoint> = Vec::new();
    let t_total = Instant::now();

    for &n in &lattice_sizes {
        let t0 = Instant::now();
        let mut per_real_r: Vec<f64> = Vec::new();
        let mut total_evals: usize = 0;
        let mut total_spacings: usize = 0;
        let mut last_jacobson: Option<jacobson::TensorAnalysis> = None;

        for i in 0..m {
            // Unique seed per (N, i): avoids correlation across lattice sizes
            let seed = base_seed.wrapping_add(i as u64 * 7919 + n as u64);
            let mut rng = StdRng::seed_from_u64(seed);

            print!("\r  N={:>5}  [{:>2}/{}]", n, i + 1, m);
            std::io::stdout().flush().ok();

            // Phase 1: sprinkle + Hasse + BD eigenvalues
            let (pts_raw, _) = diamond::sprinkle(n, &mut rng);

            let (evals, jacobson_result) = if n > COARSE_THRESHOLD {
                // Large-N: voxelize → collapse → directed BD on macro graph
                let (pts, vac_head, vac_data, _momentum) =
                    diamond::build_hasse_direct(&pts_raw);
                let (micro_to_macro, n_macro) =
                    rmt::voxelize(&pts, DEFAULT_N_MACRO);
                let (macro_head, macro_data) = rmt::collapse_hasse_to_macro(
                    &vac_head, &vac_data, n, &micro_to_macro, n_macro,
                );
                let b = rmt::build_bd_matrix_directed(
                    &macro_head, &macro_data, n_macro,
                );

                // ── Jacobson: horizon areas ──
                let areas = jacobson::horizon_areas(
                    &vac_head, &vac_data,
                    &macro_head, &macro_data,
                    &micro_to_macro, n, n_macro,
                );

                // ── Jacobson: multi-depth BD tensor (v3) ──
                let centroids = jacobson::macro_centroids(
                    &pts, &micro_to_macro, n, n_macro,
                );
                let metrics = jacobson::fisher_covariances_multidepth(
                    &centroids, &macro_head, &macro_data, n_macro,
                );
                // ── Alpha sweep: inscribed diamond at multiple margins ──
                let alphas: &[f64] = &[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35];
                println!("\n         ┌─────────────────────────────────────────────────────────────────┐");
                println!("         │  α     Nodes   (-+++)%    G=|λ_t|/Σλ_s    CV       λ_t (mean)  │");
                println!("         ├─────────────────────────────────────────────────────────────────┤");
                for &alpha in alphas {
                    let ta_a = jacobson::tensor_analysis_raw(&metrics, &areas, &centroids, alpha);
                    if ta_a.bulk_n > 0 {
                        println!(
                            "         │ {:.2}   {:>5}    {:>5.1}%     {:.4}        {:.4}   {:>12.1} │",
                            alpha, ta_a.bulk_n, ta_a.bulk_lorentzian_frac * 100.0,
                            ta_a.bulk_g_mean, ta_a.bulk_g_cv,
                            ta_a.bulk_mean_eigenvalues[0]
                        );
                    }
                }
                println!("         └─────────────────────────────────────────────────────────────────┘");

                let ta = jacobson::tensor_analysis_raw(&metrics, &areas, &centroids, 0.15);

                let ev = rmt::effective_hamiltonian_eigenvalues(b);
                (ev, Some(ta))
            } else {
                // Exact path: time-sorted Hasse + standard BD
                let (_pts, vac_head, vac_data, _momentum) =
                    diamond::build_hasse_sparse(&pts_raw);
                let b = rmt::build_bd_matrix(&vac_head, &vac_data, n);
                let ev = rmt::effective_hamiltonian_eigenvalues(b);
                (ev, None)
            };
            if jacobson_result.is_some() {
                last_jacobson = jacobson_result;
            }
            total_evals += evals.len();

            let ratios = rmt::spacing_ratios(&evals);
            if !ratios.is_empty() {
                let (r_i, _) = rmt::mean_se(&ratios);
                per_real_r.push(r_i);
                total_spacings += ratios.len();
            }
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let (r_mean, r_se) = rmt::mean_se(&per_real_r);

        // Overwrite progress line
        print!("\r");
        println!(
            "  {:>6}  {:>10.6}  {:>10.6}  {:>8}  {:>8}  {:>7.1}s",
            n, r_mean, r_se, total_evals, total_spacings, elapsed
        );

        if let Some(ref ta) = last_jacobson {
            let ev = &ta.mean_eigenvalues;
            println!(
                "         Jacobson tensor: signature (-+++) in {}/{} nodes ({:.1}%)",
                ta.n_lorentzian,
                ta.n_valid,
                ta.lorentzian_frac * 100.0
            );
            println!(
                "         Emergent G: lambda_max/dA = {:.6} +/- {:.6}, CV = {:.4}",
                ta.g_mean, ta.g_se, ta.g_cv
            );
            println!(
                "         Mean eigenvalues: [{:.4}, {:.4}, {:.4}, {:.4}]",
                ev[0], ev[1], ev[2], ev[3]
            );
            if ta.bulk_n > 0 {
                let bev = &ta.bulk_mean_eigenvalues;
                println!(
                    "         Primary (alpha=0.15): (-+++) {}/{} ({:.1}%), G={:.6}+-{:.6}, CV={:.4}",
                    ta.bulk_n_lorentzian, ta.bulk_n,
                    ta.bulk_lorentzian_frac * 100.0,
                    ta.bulk_g_mean, ta.bulk_g_se, ta.bulk_g_cv
                );
                println!(
                    "         Bulk eigenvalues: [{:.1}, {:.1}, {:.1}, {:.1}]",
                    bev[0], bev[1], bev[2], bev[3]
                );
            }
        }

        results.push(FssPoint {
            n,
            r_mean,
            r_se,
            total_evals,
            total_spacings,
            elapsed,
            jacobson: last_jacobson,
        });
    }

    let total_elapsed = t_total.elapsed().as_secs_f64();

    // ── Summary ──────────────────────────────────────────────────────
    println!("\n  {}", "═".repeat(60));
    println!("  Theoretical spacing ratios:");
    println!("    Poisson  ⟨r⟩ = 0.3863");
    println!("    GOE      ⟨r⟩ = 0.5307");
    println!("    GUE      ⟨r⟩ = 0.5996");

    // ── Finite-size fit ──────────────────────────────────────────────
    // Model: r_GUE − ⟨r⟩(N) = c · N^(−γ)
    // Linear regression: ln(r_GUE − r) = ln(c) − γ·ln(N)
    let r_gue = 0.5996;
    let valid: Vec<(f64, f64)> = results
        .iter()
        .filter(|p| p.r_mean < r_gue && p.r_se.is_finite())
        .map(|p| ((p.n as f64).ln(), (r_gue - p.r_mean).ln()))
        .collect();

    if valid.len() >= 3 {
        let n_v = valid.len() as f64;
        let sx: f64 = valid.iter().map(|(x, _)| x).sum();
        let sy: f64 = valid.iter().map(|(_, y)| y).sum();
        let sxy: f64 = valid.iter().map(|(x, y)| x * y).sum();
        let sx2: f64 = valid.iter().map(|(x, _)| x * x).sum();

        let denom = n_v * sx2 - sx * sx;
        if denom.abs() > 1e-15 {
            let slope = (n_v * sxy - sx * sy) / denom;
            let intercept = (sy - slope * sx) / n_v;
            let gamma = -slope;
            let c = intercept.exp();

            // R² for the log-log fit
            let y_mean = sy / n_v;
            let ss_tot: f64 = valid.iter().map(|(_, y)| (y - y_mean).powi(2)).sum();
            let ss_res: f64 = valid
                .iter()
                .map(|(x, y)| {
                    let y_hat = intercept + slope * x;
                    (y - y_hat).powi(2)
                })
                .sum();
            let r_sq = if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                f64::NAN
            };

            println!("\n  Finite-size fit (log-log, {} points):", valid.len());
            println!("    r_GUE − ⟨r⟩(N) = {:.4} × N^(−{:.3})", c, gamma);
            println!("    R² = {:.4}", r_sq);

            if gamma > 0.0 {
                println!(
                    "    γ = {:.3}  →  ⟨r⟩(N) converges to {:.4} as N → ∞",
                    gamma, r_gue
                );
            } else {
                println!(
                    "    γ = {:.3}  →  does NOT converge to GUE",
                    gamma
                );
            }
        }
    } else {
        println!("\n  (insufficient points for finite-size fit)");
    }

    println!("\n  Total wall time: {:.0}s", total_elapsed);
    println!();

    // ── CSV output ───────────────────────────────────────────────────
    let csv_path = "fss_rmt.csv";
    match std::fs::File::create(csv_path) {
        Ok(mut f) => {
            let _ = writeln!(
                f,
                "# BD Effective Hamiltonian FSS + Jacobson Tensor\n\
                 # M = {} realisations per N, seed = {}\n\
                 # BD weights = [+1, -9, +16, -8]\n\
                 N,r_mean,r_se,total_evals,total_spacings,elapsed_s,\
                 lorentz_frac,n_lorentz,n_valid,G_mean,G_se,G_cv,\
                 ev0,ev1,ev2,ev3,\
                 bulk_n,bulk_n_lorentz,bulk_lorentz_frac,\
                 bulk_ev0,bulk_ev1,bulk_ev2,bulk_ev3,\
                 bulk_G_mean,bulk_G_se,bulk_G_cv",
                m, base_seed
            );
            for p in &results {
                if let Some(ref j) = p.jacobson {
                    let ev = &j.mean_eigenvalues;
                    let bev = &j.bulk_mean_eigenvalues;
                    let _ = writeln!(
                        f,
                        "{},{:.8},{:.8},{},{},{:.1},{:.6},{},{},{:.8},{:.8},{:.6},\
                         {:.6},{:.6},{:.6},{:.6},\
                         {},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.8},{:.8},{:.6}",
                        p.n, p.r_mean, p.r_se, p.total_evals, p.total_spacings,
                        p.elapsed, j.lorentzian_frac, j.n_lorentzian, j.n_valid,
                        j.g_mean, j.g_se, j.g_cv,
                        ev[0], ev[1], ev[2], ev[3],
                        j.bulk_n, j.bulk_n_lorentzian, j.bulk_lorentzian_frac,
                        bev[0], bev[1], bev[2], bev[3],
                        j.bulk_g_mean, j.bulk_g_se, j.bulk_g_cv
                    );
                } else {
                    let _ = writeln!(
                        f,
                        "{},{:.8},{:.8},{},{},{:.1},,,,,,,,,,,,,,,,,,,",
                        p.n, p.r_mean, p.r_se, p.total_evals, p.total_spacings,
                        p.elapsed
                    );
                }
            }
            println!("  Saved {csv_path}");
        }
        Err(e) => eprintln!("  Failed to write {csv_path}: {e}"),
    }
}

struct FssPoint {
    n: usize,
    r_mean: f64,
    r_se: f64,
    total_evals: usize,
    total_spacings: usize,
    elapsed: f64,
    jacobson: Option<jacobson::TensorAnalysis>,
}

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}
