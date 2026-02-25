// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Ensemble runner: adaptive batching with Welford convergence and checkpointing.
//!
//! Runs M independent Poisson sprinklings through Phases 1-3, checkpoints
//! after each realisation, and checks convergence via the Welford online
//! algorithm on mass_gen1.  Intermediate CSV snapshots are produced after
//! each batch for monitoring long runs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::config::ExecMode;
use crate::ensemble::averaging::average_ensemble;
use crate::ensemble::checkpoint;
use crate::graph::csr::{CsrGraph, Undirected};
use crate::measure::{self, MeasureContext, MeasureFlags, MeasureResults};
use crate::output::{self, Metadata};
use crate::phase1;
use crate::phase2::{self, topology::aggregate_topology, TopologySummary};
use crate::phase3::{self, SpectralOutput, WalkResult, spectral_dimension};
use crate::provenance;

/// Result of a full ensemble run.
pub struct EnsembleResult {
    pub spectral: SpectralOutput,
    pub topology: TopologySummary,
    pub measurements: Option<MeasureResults>,
    pub actual_m: usize,
    pub converged: bool,
}

/// Format a duration in seconds into a human-readable string.
fn fmt_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{secs:.0}s")
    } else if secs < 3600.0 {
        format!("{}m {:02}s", secs as u64 / 60, secs as u64 % 60)
    } else {
        let h = secs as u64 / 3600;
        let m = (secs as u64 % 3600) / 60;
        format!("{h}h {m:02}m")
    }
}

/// Run Phases 1-3 for one Monte Carlo universe.
///
/// Sprinkle -> Hasse -> Kuratowski -> spectral dimensions + generation walkers
/// + causal flux + optional measurements.
fn run_realization(
    n_points: usize,
    seed: u64,
    steps: &[u32],
    walkers: usize,
    eigen_cutoff: usize,
    measure: &MeasureFlags,
) -> (SpectralOutput, TopologySummary, Option<MeasureResults>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Phase 1 — Poisson sprinkling + Hasse diagram
    let (pts_raw, _big_t) = phase1::sprinkle(n_points, &mut rng);
    let (pts, vacuum_csr, momentum) = if n_points <= eigen_cutoff {
        phase1::build_hasse_sparse(&pts_raw)
    } else {
        phase1::build_hasse_direct(&pts_raw)
    };
    drop(pts_raw);

    // Phase 2 — Kuratowski contraction + particle classification
    println!("  [In-Memory] Phase 1 CSR: {} edges", vacuum_csr.n_edge_slots());
    let momentum_clone = if measure.halflife || measure.electroweak || measure.neutrino {
        momentum.clone()
    } else {
        vec![]
    };
    let (defect, topology, prisms) = phase2::apply_defect(n_points, vacuum_csr, momentum);

    // Phase 3a — vacuum + defect spectral dimensions
    let sym_vac: CsrGraph<Undirected> = defect.vacuum_csr.symmetrize();
    let (sym_head, sym_data) = sym_vac.raw();

    let (vac_walk, def_walk) = if n_points <= eigen_cutoff {
        // Eigendecomp path: reconstruct edge lists from symmetric CSR
        let mut vac_rows = Vec::new();
        let mut vac_cols = Vec::new();
        for u in 0..n_points {
            for &v in sym_vac.neighbors(u) {
                if (u as u32) < v {
                    vac_rows.push(u as u32);
                    vac_cols.push(v);
                }
            }
        }
        let mut def_rows = Vec::new();
        let mut def_cols = Vec::new();
        let n_def = defect.defect_csr.n_nodes();
        for u in 0..n_def {
            for &v in defect.defect_csr.neighbors(u) {
                if (u as u32) < v {
                    def_rows.push(u as u32);
                    def_cols.push(v);
                }
            }
        }
        let (vac_global, vac_core) = phase3::compute_eigen(
            n_points, &vac_rows, &vac_cols, steps, &defect.vacuum_core,
        );
        let (def_global, def_core) = phase3::compute_eigen(
            n_def, &def_rows, &def_cols, steps, &defect.defect_core,
        );
        ((vac_global, vac_core), (def_global, def_core))
    } else {
        // Monte Carlo walkers
        let mut rng_mc = StdRng::seed_from_u64(seed + 2);
        let (vac_global, vac_core) = phase3::compute_monte_carlo_csr(
            n_points, sym_head, sym_data, steps, &defect.vacuum_core,
            walkers, &mut rng_mc,
        );

        let n_def = defect.defect_csr.n_nodes();
        let (def_head, def_data) = defect.defect_csr.raw();
        let (def_global, def_core) = if n_def <= eigen_cutoff {
            println!("  [Defect graph <={eigen_cutoff}: using eigendecomp (exact, zero noise)]");
            let mut def_rows = Vec::new();
            let mut def_cols = Vec::new();
            for u in 0..n_def {
                for &v in defect.defect_csr.neighbors(u) {
                    if (u as u32) < v {
                        def_rows.push(u as u32);
                        def_cols.push(v);
                    }
                }
            }
            phase3::compute_eigen(n_def, &def_rows, &def_cols, steps, &defect.defect_core)
        } else {
            phase3::compute_monte_carlo_csr(
                n_def, def_head, def_data, steps, &defect.defect_core,
                walkers, &mut rng_mc,
            )
        };
        ((vac_global, vac_core), (def_global, def_core))
    };

    // ── Measurements ──────────────────────────────────────────────────
    let meas = if measure.any_active() {
        let ctx = MeasureContext {
            n_points,
            pts: &pts,
            vacuum_csr: &defect.vacuum_csr,
            sym_vacuum: &sym_vac,
            defect: &defect,
            prisms: &prisms,
            momentum: &momentum_clone,
            topology: &topology,
            walkers,
            seed,
            modulo_config: &measure.modulo_config,
        };
        Some(measure::run_all(measure, &ctx))
    } else {
        None
    };

    // Phase 3b — generation walkers + causal flux
    let (def_head, def_data) = defect.defect_csr.raw();

    let run_gen = |nodes: &[usize], seed_offset: u64| -> WalkResult {
        if nodes.is_empty() {
            return WalkResult::default();
        }
        let resolved: Vec<usize> = nodes.iter().map(|&n| defect.merge_map[n]).collect();
        let starts = phase3::distribute_walkers(&resolved, walkers);
        let p = phase3::run_walkers(
            def_head, def_data, &starts, steps,
            seed.wrapping_add(seed_offset), Some(&defect.merge_map),
        );
        let ds = spectral_dimension(steps, &p);
        WalkResult { p, ds, ds_std: vec![] }
    };

    let gen0 = run_gen(&defect.generations.gen1, 100);
    let gen1 = run_gen(&defect.generations.gen2, 200);
    let gen2 = run_gen(&defect.generations.gen3, 300);
    let gen3 = run_gen(&defect.generations.anti1, 400);

    // Causal flux: directed adjacency (past -> future)
    let flux_csr = phase3::build_flux_csr(
        &defect.vacuum_csr, &pts, &defect.merge_map, n_points,
    );
    let (flux_head, flux_data) = flux_csr.raw();

    let flux_starts: Vec<usize> = defect.generations.gen1.iter()
        .map(|&s| defect.merge_map[s])
        .collect();
    let flux_attr_targets: Vec<usize> = defect.generations.anti1.iter()
        .map(|&s| defect.merge_map[s])
        .collect();
    let flux_repu_targets: Vec<usize> = defect.generations.gen1.iter()
        .map(|&s| defect.merge_map[s])
        .collect();

    let (flux_attr_p, flux_repu_p) = phase3::run_transmission_walkers(
        flux_head, flux_data, &flux_starts, &flux_attr_targets, &flux_repu_targets,
        steps, seed.wrapping_add(500), Some(&defect.merge_map),
    );

    // Normalized flux (per-charge coupling)
    let n_attr = flux_attr_targets.len().max(1) as f64;
    let n_repu = flux_repu_targets.len().max(1) as f64;
    let flux_attr_norm: Vec<f64> = flux_attr_p.iter().map(|&f| f / n_attr).collect();
    let flux_repu_norm: Vec<f64> = flux_repu_p.iter().map(|&f| f / n_repu).collect();

    // Sterile walkers (Phi = 0)
    let sterile = run_gen(&defect.generations.sterile, 600);

    // Compose SpectralOutput
    let spectral = SpectralOutput {
        vacuum: vac_walk.0,
        vac_core: vac_walk.1,
        defect: def_walk.0,
        def_core: def_walk.1,
        generations: [gen0, gen1, gen2, gen3],
        sterile,
        flux_attr: WalkResult {
            ds: spectral_dimension(steps, &flux_attr_p),
            p: flux_attr_p,
            ds_std: vec![],
        },
        flux_repu: WalkResult {
            ds: spectral_dimension(steps, &flux_repu_p),
            p: flux_repu_p,
            ds_std: vec![],
        },
        flux_attr_norm,
        flux_repu_norm,
        mass: defect.generations.mass,
    };

    (spectral, topology, meas)
}

/// Run the full adaptive ensemble.
///
/// Processes realizations in batches of `max_concurrent_runs`.  Each batch
/// runs in parallel via Rayon; between batches the heavy CSR graphs are
/// freed, keeping only the lightweight SpectralOutput accumulator.
///
/// After each batch, results are drained one at a time.  Each individual
/// result triggers a Welford update AND a checkpoint write.  This means
/// a crash mid-batch loses the in-flight realizations, but every fully
/// completed realization is persisted immediately.
pub fn run_ensemble(
    n_points: usize,
    steps: &[u32],
    walkers: usize,
    eigen_cutoff: usize,
    measure: &MeasureFlags,
    exec_mode: ExecMode,
    seed_base: u64,
    max_ensemble: usize,
    min_ensemble: usize,
    target_error: f64,
    batch_size: usize,
    max_concurrent_runs: usize,
    output_dir: &str,
    resume: bool,
    force_all: bool,
) -> EnsembleResult {
    if exec_mode == ExecMode::Streaming {
        eprintln!("ERROR: --stream mode is not yet implemented in FEG_prism.");
        eprintln!("       Use --inmemory or reduce N to fit in RAM.");
        std::process::exit(1);
    }

    let t0 = Instant::now();
    let done = AtomicUsize::new(0);
    let max_concurrent_runs = max_concurrent_runs.min(batch_size);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_concurrent_runs)
        .build()
        .unwrap();

    let run_one = |i: usize| -> (SpectralOutput, TopologySummary, Option<MeasureResults>) {
        let seed = seed_base + i as u64;
        let result = run_realization(n_points, seed, steps, walkers, eigen_cutoff, measure);

        let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed = t0.elapsed().as_secs_f64();
        let remaining = max_ensemble.saturating_sub(completed);

        let msg = if completed == 1 && remaining > 0 {
            let eta = elapsed * remaining as f64;
            format!(
                "  [done {completed}/{max_ensemble}]  first realization took {} \
                 -- ETA remaining: ~{}",
                fmt_duration(elapsed), fmt_duration(eta),
            )
        } else if remaining > 0 {
            let rate = elapsed / completed as f64 / max_concurrent_runs as f64;
            let eta = rate * remaining as f64;
            format!(
                "  [done {completed}/{max_ensemble}]  elapsed {} -- ETA remaining: ~{}",
                fmt_duration(elapsed), fmt_duration(eta),
            )
        } else {
            format!(
                "  [done {completed}/{max_ensemble}]  elapsed {}",
                fmt_duration(elapsed),
            )
        };

        println!("{msg}");
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true).append(true).open("simulation.log")
        {
            use std::io::Write;
            writeln!(file, "{msg}").ok();
        }

        result
    };

    // ── Checkpoint: parameter fingerprint + try resume ──────────────
    let mode_str = if exec_mode == ExecMode::Streaming { "streaming" } else { "inmemory" };
    let tmax = *steps.last().unwrap_or(&15) as usize;
    let epsilon = (tmax as f64) / (walkers as f64).sqrt();
    let run_params = checkpoint::make_params(
        n_points, seed_base, epsilon, tmax, eigen_cutoff, mode_str,
    );

    let mut spectral_vec: Vec<SpectralOutput> = Vec::with_capacity(max_ensemble);
    let mut topo_vec: Vec<TopologySummary> = Vec::with_capacity(max_ensemble);
    let mut measure_vec: Vec<MeasureResults> = Vec::with_capacity(max_ensemble);
    let mut welford_mean: f64 = 0.0;
    let mut welford_m2: f64 = 0.0;

    let start_from = if resume {
        match checkpoint::load(output_dir, &run_params) {
            Ok(ckpt) => {
                println!(
                    "  Resuming: {}/{} realizations completed",
                    ckpt.completed, max_ensemble,
                );
                spectral_vec = ckpt.spectral_vec;
                topo_vec = ckpt.topo_vec;
                welford_mean = ckpt.welford_mean;
                welford_m2 = ckpt.welford_m2;
                done.store(ckpt.completed, Ordering::Relaxed);
                ckpt.completed
            }
            Err(msg) => {
                eprintln!("  --resume: {msg}");
                0
            }
        }
    } else {
        0
    };

    // ── Adaptive Batching with Per-Realization Checkpointing ─────────
    let mut current = start_from;
    let mut converged = false;

    while current < max_ensemble && (force_all || !converged) {
        let batch_end = (current + max_concurrent_runs).min(max_ensemble);
        let chunk: Vec<usize> = (current..batch_end).collect();

        println!(
            "  \u{2500}\u{2500} Batch (realizations {}-{}/{}) \u{2500}\u{2500}",
            current + 1, batch_end, max_ensemble,
        );

        let batch_results: Vec<(SpectralOutput, TopologySummary, Option<MeasureResults>)> =
            pool.install(|| {
                chunk.par_iter()
                    .with_max_len(1)
                    .map(|&i| run_one(i))
                    .collect()
            });

        // ── Backup checkpoint before combining new batch ──────────
        if current > start_from {
            let ckpt_path = std::path::Path::new(output_dir).join(".checkpoint.bin");
            if ckpt_path.exists() {
                let backup = std::path::Path::new(output_dir)
                    .join(format!(".checkpoint.bin.bak.{current}"));
                if let Err(e) = std::fs::copy(&ckpt_path, &backup) {
                    eprintln!("  [backup] warning: {e}");
                } else {
                    println!("  [backup] .checkpoint.bin -> {}", backup.display());
                }
            }
        }

        // Drain batch: Welford update + checkpoint after EACH realization
        for (spec, topo, meas) in batch_results {
            let x = spec.mass[0]; // mass_gen1
            spectral_vec.push(spec);
            topo_vec.push(topo);
            if let Some(m) = meas {
                measure_vec.push(m);
            }

            // Welford online update
            let count = spectral_vec.len();
            let delta = x - welford_mean;
            welford_mean += delta / count as f64;
            let delta2 = x - welford_mean;
            welford_m2 += delta * delta2;

            // Checkpoint after EACH realization (atomic write)
            if let Err(e) = checkpoint::save(
                output_dir, &run_params, &spectral_vec, &topo_vec,
                welford_mean, welford_m2,
            ) {
                eprintln!("  [checkpoint] warning: {e}");
            }
        }

        // Convergence check
        let count = spectral_vec.len();
        if count >= min_ensemble && welford_mean.abs() > 1e-12 {
            let variance = welford_m2 / (count - 1) as f64; // Bessel correction
            let std_error = (variance / count as f64).sqrt();
            let rel_error = std_error / welford_mean.abs();

            println!(
                "  [convergence] M={count}, mass_gen1={welford_mean:.4}, \
                 SE={std_error:.4}, rel_err={rel_error:.6}",
            );

            if rel_error <= target_error {
                println!(
                    "  \u{2713} Statistical convergence reached \
                     (rel_err={rel_error:.6} \u{2264} {target_error})",
                );
                converged = true;
            }
        } else {
            println!("  [checkpoint: {count}/{max_ensemble}]");
        }

        // ── Accumulation metadata log ──────────────────────────────
        {
            use std::io::Write;
            let log_path = format!("{output_dir}/accumulation.log");
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true).append(true).open(&log_path)
            {
                let count = spectral_vec.len();
                let variance = if count > 1 {
                    welford_m2 / (count - 1) as f64
                } else {
                    0.0
                };
                let std_error = if count > 1 {
                    (variance / count as f64).sqrt()
                } else {
                    0.0
                };
                let elapsed = t0.elapsed().as_secs();
                let _ = writeln!(
                    f,
                    "batch {}-{}/{} | realizations={} | mass_gen1_mean={:.4} | \
                     SE={:.6} | rel_err={:.6} | converged={} | elapsed={}s",
                    current + 1, batch_end, max_ensemble, count,
                    welford_mean, std_error,
                    if welford_mean.abs() > 1e-12 {
                        std_error / welford_mean.abs()
                    } else {
                        f64::NAN
                    },
                    converged, elapsed,
                );
            }
        }

        // ── Intermediate CSV snapshot ──────────────────────────────
        {
            let count = spectral_vec.len();
            let snap_spec = average_ensemble(&spectral_vec, steps);
            let snap_topo = aggregate_topology(&topo_vec);
            let elapsed = t0.elapsed().as_secs();
            let snap_meta = Metadata {
                n_points,
                actual_m: count,
                converged: false,
                min_ensemble,
                max_ensemble,
                mode: mode_str.to_string(),
                epsilon,
                tmax,
                walkers,
                seed: seed_base,
                timestamp: provenance::utc_timestamp(),
                commit: provenance::commit_string(),
            };

            output::write_spectral_csv(
                &format!("{output_dir}/results_M{count:02}.csv"),
                steps, &snap_spec, &snap_meta,
            );
            output::write_topology_csv(
                &format!("{output_dir}/topology_summary_M{count:02}.csv"),
                &snap_topo, &snap_meta,
            );
            output::write_mass_spectrum_csv(
                &format!("{output_dir}/mass_spectrum_M{count:02}.csv"),
                &snap_topo.prism_histogram, &snap_meta,
            );

            // Measurement snapshots
            if !measure_vec.is_empty() {
                let meas_agg = measure::aggregate_all(&measure_vec);
                measure::write_all_csv(&meas_agg, output_dir, &snap_meta);
            }

            println!(
                "  [snapshot] wrote *_M{count:02}.csv ({count} realizations, {elapsed}s)",
            );
        }

        current = batch_end;
    }

    if !converged && current >= max_ensemble {
        println!("  [warning] Safety cap M={max_ensemble} reached before convergence");
    }

    // ── Final ensemble average ─────────────────────────────────────
    let actual_m = spectral_vec.len();
    let spectral = average_ensemble(&spectral_vec, steps);
    drop(spectral_vec);
    let topology = aggregate_topology(&topo_vec);

    let measurements = if measure_vec.len() == 1 {
        Some(measure_vec.remove(0))
    } else if !measure_vec.is_empty() {
        Some(measure::aggregate_all(&measure_vec))
    } else {
        None
    };
    drop(measure_vec);

    EnsembleResult {
        spectral,
        topology,
        measurements,
        actual_m,
        converged,
    }
}
