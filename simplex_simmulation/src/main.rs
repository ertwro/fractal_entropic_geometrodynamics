mod diamond;
mod output;
mod skyrmion;
mod spectral;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use spectral::SpectralResult;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const BASE_SEED: u64 = 42;
/// N ≤ this value uses eigendecomp; above uses exact lazy power iteration.
/// Lowered to 3k so that N=5000+ runs use the much faster sparse path
/// (~30× speedup over dense O(N³) eigendecomp on a T480).
const EIGEN_CUTOFF: usize = 3_000;
const DEFAULT_ENSEMBLE: usize = 10;

// ─────────────────────────────────────────────────────────────────────────────
// Single realisation
// ─────────────────────────────────────────────────────────────────────────────

/// Run phases 1-3 for one universe with a given seed.
/// Returns (vacuum_result, defect_result).
fn run_realization(
    n_points: usize,
    seed: u64,
    steps: &[u32],
) -> (SpectralResult, SpectralResult) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Phase 1
    let (pts, _big_t) = diamond::sprinkle(n_points, &mut rng);
    let (rows_vac, cols_vac) = if n_points <= EIGEN_CUTOFF {
        diamond::build_hasse_sparse(&pts)
    } else {
        diamond::build_hasse_direct(&pts)
    };

    // Phase 2 — Kuratowski contraction (pure integer topology)
    let (rows_def, cols_def, vacuum_core, defect_core) =
        skyrmion::apply_defect(n_points, &rows_vac, &cols_vac);

    // Phase 3 — both vacuum and defect compute local P over core_indices
    //          so we can compare P_loc_vac vs P_loc_def (geometry vs topology).
    if n_points <= EIGEN_CUTOFF {
        let vac =
            spectral::compute_eigen(n_points, &rows_vac, &cols_vac, steps, &vacuum_core);
        let def =
            spectral::compute_eigen(n_points, &rows_def, &cols_def, steps, &defect_core);
        (vac, def)
    } else {
        let n_walkers = 5_000;
        let vac = spectral::compute_monte_carlo(
            n_points, &rows_vac, &cols_vac, steps, &vacuum_core, n_walkers, &mut rng,
        );
        let def = spectral::compute_monte_carlo(
            n_points, &rows_def, &cols_def, steps, &defect_core, n_walkers, &mut rng,
        );
        (vac, def)
    }
}
// ─────────────────────────────────────────────────────────────────────────────
// Ensemble averaging
// ─────────────────────────────────────────────────────────────────────────────

/// Average P(t) across realisations, then recompute d_S from the mean P.
fn average_ensemble(
    results: &[(SpectralResult, SpectralResult)],
    steps: &[u32],
) -> (SpectralResult, SpectralResult) {
    let m = results.len() as f64;
    let ns = steps.len();

    let mut vp_g = vec![0.0; ns]; // vacuum P global
    let mut vp_l = vec![0.0; ns]; // vacuum P local
    let mut dp_g = vec![0.0; ns]; // defect P global
    let mut dp_l = vec![0.0; ns]; // defect P local

    for (vac, def) in results {
        for i in 0..ns {
            vp_g[i] += vac.p_global[i];
            vp_l[i] += vac.p_local[i];
            dp_g[i] += def.p_global[i];
            dp_l[i] += def.p_local[i];
        }
    }
    for i in 0..ns {
        vp_g[i] /= m;
        vp_l[i] /= m;
        dp_g[i] /= m;
        dp_l[i] /= m;
    }

    let vac = SpectralResult {
        ds_global: spectral::spectral_dimension(steps, &vp_g),
        ds_local: spectral::spectral_dimension(steps, &vp_l),
        p_global: vp_g,
        p_local: vp_l,
    };
    let def = SpectralResult {
        ds_global: spectral::spectral_dimension(steps, &dp_g),
        ds_local: spectral::spectral_dimension(steps, &dp_l),
        p_global: dp_g,
        p_local: dp_l,
    };
    (vac, def)
}

// ─────────────────────────────────────────────────────────────────────────────
// Time formatting
// ─────────────────────────────────────────────────────────────────────────────

fn fmt_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.0}s", secs)
    } else if secs < 3600.0 {
        format!("{}m {:02}s", secs as u64 / 60, secs as u64 % 60)
    } else {
        let h = secs as u64 / 3600;
        let m = (secs as u64 % 3600) / 60;
        format!("{}h {:02}m", h, m)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_points: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);
    let m_ensemble: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ENSEMBLE);

    let tier = if n_points <= EIGEN_CUTOFF {
        "eigendecomp"
    } else {
        "mc-walkers"
    };
    println!(
        "Causal Set Spectral Dimension  (N={}, M={}, tier={})\n",
        n_points, m_ensemble, tier
    );

    let steps: Vec<u32> = (1..=50).map(|i| i * 2).collect(); // 2, 4, …, 100
    let t0 = Instant::now();

    // ── Ensemble ────────────────────────────────────────────────────────
    // For large N, run realizations sequentially to avoid OOM — each
    // realization builds ~800MB sparse matrices, and rayon's inner
    // parallelism over core_indices already saturates all cores.
    // For small N (eigendecomp tier), outer parallelism is safe.
    let use_par = n_points <= EIGEN_CUTOFF;
    if !use_par {
        println!("  (sequential ensemble — inner rayon parallelises core nodes)\n");
    }

    let done = AtomicUsize::new(0);
    let run_one = |i: usize| {
        let seed = BASE_SEED + i as u64;
        let result = run_realization(n_points, seed, &steps);
        let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed = t0.elapsed().as_secs_f64();
        let remaining = m_ensemble - completed;
        if completed == 1 && remaining > 0 {
            let eta = elapsed * remaining as f64;
            println!(
                "  [done {}/{}]  first realization took {} — ETA remaining: ~{}",
                completed, m_ensemble, fmt_duration(elapsed), fmt_duration(eta)
            );
        } else if remaining > 0 {
            let rate = elapsed / completed as f64;
            let eta = rate * remaining as f64;
            println!(
                "  [done {}/{}]  elapsed {} — ETA remaining: ~{}",
                completed, m_ensemble, fmt_duration(elapsed), fmt_duration(eta)
            );
        } else {
            println!(
                "  [done {}/{}]  elapsed {}",
                completed, m_ensemble, fmt_duration(elapsed)
            );
        }
        result
    };

    let results: Vec<(SpectralResult, SpectralResult)> = if use_par {
        (0..m_ensemble).into_par_iter().map(run_one).collect()
    } else {
        (0..m_ensemble).map(run_one).collect()
    };

    // ── Average P(t), then derive d_S ───────────────────────────────────
    let (vac_avg, def_avg) = average_ensemble(&results, &steps);

    // ── Phase 4 ─────────────────────────────────────────────────────────
    println!("\n[Phase 4] Output …");
    output::write_csv("results.csv", &steps, &vac_avg, &def_avg);

    // ── Summary ─────────────────────────────────────────────────────────
    let mid = steps.len() / 2;
    let last = steps.len() - 1;
    let elapsed = t0.elapsed().as_secs_f64();
    println!("\n── Summary ({} realisations) ────────────────────────", m_ensemble);
    println!(
        "  d_S vacuum global  (t={}): {:.2}",
        steps[mid], vac_avg.ds_global[mid]
    );
    println!(
        "  d_S defect global  (t={}): {:.2}",
        steps[mid], def_avg.ds_global[mid]
    );
    println!(
        "  d_S core on vacuum (t={}): {:.2}",
        steps[mid], vac_avg.ds_local[mid]
    );
    println!(
        "  d_S core on defect (t={}): {:.2}",
        steps[mid], def_avg.ds_local[mid]
    );
    println!("  ── P saturation (t={}) ──", steps[last]);
    println!(
        "  P_loc_vac = {:.6e}  P_loc_def = {:.6e}  ratio = {:.4}",
        vac_avg.p_local[last],
        def_avg.p_local[last],
        def_avg.p_local[last] / vac_avg.p_local[last]
    );
    println!("  Total time: {}", fmt_duration(elapsed));
    println!("────────────────────────────────────────────────────");
}
