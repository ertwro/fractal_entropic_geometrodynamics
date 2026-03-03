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
use crate::convergence::{AutoConverge, ConvergeState};
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

/// Run a single unified walk on a graph, producing global + per-category WalkResults.
///
/// Replaces the per-category convergence loops (`run_converge_loop` ×10) with
/// a single walk that bins returns by origin category.  A walker dropped at
/// any origin samples P(t) for all categories simultaneously — the
/// thermodynamics of the random walk does not care about category labels.
///
/// **CRT budget**: W_bulk = ceil(1/(ε² · P_bulk)), where P_bulk = t_max^{-2}.
/// Per-category P(t) is pure binning of the bulk walk; no separate convergence
/// loop is needed.  Small categories receive RSE = ε × √(N/|S|) — honest
/// statistical uncertainty, not a convergence failure.
///
/// **Trap 1 fix**: guards per-category division against W_cat = 0.
/// **Trap 2 fix**: resolves walker origins through merge_map before category lookup.
/// **Cell-sort fix**: shuffles origins to avoid temporal sampling bias.
fn run_unified_walk(
    adj_head: &[u32],
    adj_data: &[u32],
    all_origins: &[usize],
    steps: &[u32],
    seed: u64,
    seed_offset: u64,
    epsilon: f64,
    tmax: usize,
    merge_map: Option<&[usize]>,
    category_map: &[u8],
    n_categories: usize,
    category_labels: &[&str],
    label: &str,
) -> (WalkResult, Vec<WalkResult>) {
    if all_origins.is_empty() {
        let empty = WalkResult::default();
        return (empty, vec![WalkResult::default(); n_categories]);
    }

    // CRT bulk budget: W_bulk = ceil(1/(ε² · P_bulk(t_max)))
    let p_bulk = (tmax as f64).powi(-2);
    let w_bulk = (1.0 / (epsilon * epsilon * p_bulk)).ceil() as usize;
    // 1.25× headroom for Welford k=3 consecutive-pass trigger
    let max_walkers = ((w_bulk as f64) * 1.25).ceil() as usize;
    let max_walkers = max_walkers.max(all_origins.len());
    println!("  [Phase3] {label}: W_bulk = {max_walkers} (P_bulk = {p_bulk:.2e}, N = {})", all_origins.len());

    let ac = AutoConverge::new(max_walkers, 2048, epsilon);
    let mut state = ConvergeState::new();
    let n_steps = steps.len();
    let mut cum_global = vec![0u64; n_steps];
    let mut cum_cats: Vec<Vec<u64>> = vec![vec![0u64; n_steps]; n_categories];

    loop {
        let starts = phase3::distribute_walkers_shuffled(
            all_origins, ac.batch_size,
            seed.wrapping_add(seed_offset).wrapping_add(state.total_walkers as u64),
        );
        let batch_seed = seed.wrapping_add(seed_offset)
            .wrapping_add(state.total_walkers as u64)
            .wrapping_add(0x1000_0000); // distinct seed space from shuffle

        let (batch_global, batch_cats) = phase3::run_walkers_unified(
            adj_head, adj_data, &starts, steps,
            batch_seed, merge_map, category_map, n_categories,
        );

        // Accumulate u64 counts (Strict Finitism)
        for i in 0..n_steps {
            cum_global[i] += batch_global[i];
        }
        for c in 0..n_categories {
            for i in 0..n_steps {
                cum_cats[c][i] += batch_cats[c][i];
            }
        }

        // Convergence tracking on global P(t_max)
        let tmax_idx = steps.iter().rposition(|&s| (s as usize) <= tmax)
            .unwrap_or(n_steps - 1);
        let total_w = state.total_walkers + ac.batch_size;
        let obs = batch_global[tmax_idx] as f64 / ac.batch_size as f64;
        if state.update(obs, &ac) { break; }
        if state.at_limit(&ac) {
            eprintln!("[Phase3/{label}] WARNING: {total_w} walkers without convergence");
            break;
        }
    }

    let total_walkers = state.total_walkers;
    println!("  [Phase3] {label} converged at {total_walkers} walkers");

    // Count per-category walkers from all accumulated starts
    // (approximated as total_walkers × population fraction)
    // For exact counts, rebuild starts at the final total — but since
    // distribute_walkers_shuffled preserves the W invariant per batch,
    // the category fractions are deterministic.
    let cat_walker_fracs: Vec<f64> = {
        let sample = phase3::distribute_walkers_shuffled(all_origins, ac.batch_size.min(total_walkers), seed.wrapping_add(seed_offset));
        let sample_counts = phase3::count_category_walkers(&sample, category_map, n_categories);
        sample_counts.iter().map(|&c| c as f64 / sample.len() as f64).collect()
    };

    // Single f64 division (Strict Finitism: all accumulation was u64)
    let global_p: Vec<f64> = cum_global.iter()
        .map(|&c| c as f64 / total_walkers as f64)
        .collect();
    let global_ds = spectral_dimension(steps, &global_p);
    let global_result = WalkResult { p: global_p, ds: global_ds, ds_std: vec![] };

    // Per-category results with NaN guard (Trap 1 fix)
    let cat_results: Vec<WalkResult> = (0..n_categories).map(|c| {
        let w_cat = (cat_walker_fracs[c] * total_walkers as f64).round() as usize;
        let cat_label = if c < category_labels.len() { category_labels[c] } else { "?" };
        println!("    {cat_label}: W_cat ~ {w_cat} ({:.1}%)", cat_walker_fracs[c] * 100.0);
        if w_cat == 0 {
            // Trap 1: no walkers sampled this category — return zeros, not NaN
            WalkResult {
                p: vec![0.0; n_steps],
                ds: vec![0.0; n_steps],
                ds_std: vec![],
            }
        } else {
            let w_cat_f = cat_walker_fracs[c] * total_walkers as f64;
            let cat_p: Vec<f64> = cum_cats[c].iter()
                .map(|&count| {
                    if w_cat_f > 0.0 { count as f64 / w_cat_f } else { 0.0 }
                })
                .collect();
            let cat_ds = spectral_dimension(steps, &cat_p);
            WalkResult { p: cat_p, ds: cat_ds, ds_std: vec![] }
        }
    }).collect();

    (global_result, cat_results)
}

/// Run Phases 1-3 for one Monte Carlo universe.
///
/// Sprinkle -> Hasse -> Kuratowski -> spectral dimensions + generation walkers
/// + causal flux + optional measurements.
fn run_realization(
    n_points: usize,
    seed: u64,
    steps: &[u32],
    epsilon: f64,
    tmax: usize,
    eigen_cutoff: usize,
    measure: &MeasureFlags,
    realization_idx: usize,
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

    // CRT bulk budget (used by unified walk and flux circuit breaker)
    let p_bulk = (tmax as f64).powi(-2);
    let w_bulk = (1.0 / (epsilon * epsilon * p_bulk)).ceil() as usize;

    let (vac_walk, def_walk, gen_walks, sterile_walk) = if n_points <= eigen_cutoff {
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
        // Eigendecomp doesn't produce per-generation results — use defaults
        let gen_walks = [WalkResult::default(), WalkResult::default(),
                         WalkResult::default(), WalkResult::default()];
        ((vac_global, vac_core), (def_global, def_core),
         gen_walks, WalkResult::default())
    } else {
        // ── Unified Monte Carlo walk ────────────────────────────────────
        //
        // One walk per graph (vacuum, defect), binning returns by category.
        // Replaces 10 separate convergence loops with 2 unified walks.
        println!("  [Phase3] W_bulk = {w_bulk} (unified walk)");
        let all_nodes: Vec<usize> = (0..n_points).collect();
        let n_def = defect.defect_csr.n_nodes();
        let (def_head, def_data) = defect.defect_csr.raw();

        // Check if defect graph is small enough for eigendecomp
        let def_eigen = n_def <= eigen_cutoff;

        // ── Vacuum category map (2 bits) ────────────────────────────────
        // Bit 0 (0x01): global — all nodes
        // Bit 1 (0x02): core  — vacuum_core nodes
        let mut vac_cat_map = vec![0x01u8; n_points]; // all nodes are global
        for &ci in &defect.vacuum_core {
            vac_cat_map[ci] |= 0x02;
        }

        // ── Defect category map (7 bits) ────────────────────────────────
        // Bit 0 (0x01): def_global — all defect nodes
        // Bit 1 (0x02): def_core   — defect_core nodes
        // Bit 2 (0x04): gen1
        // Bit 3 (0x08): gen2
        // Bit 4 (0x10): gen3
        // Bit 5 (0x20): anti1
        // Bit 6 (0x40): sterile
        let mut def_cat_map = vec![0x01u8; n_def]; // all defect nodes are global
        for &ci in &defect.defect_core {
            if ci < n_def { def_cat_map[ci] |= 0x02; }
        }
        for &ni in &defect.generations.gen1 {
            let ri = defect.merge_map[ni];
            if ri < n_def { def_cat_map[ri] |= 0x04; }
        }
        for &ni in &defect.generations.gen2 {
            let ri = defect.merge_map[ni];
            if ri < n_def { def_cat_map[ri] |= 0x08; }
        }
        for &ni in &defect.generations.gen3 {
            let ri = defect.merge_map[ni];
            if ri < n_def { def_cat_map[ri] |= 0x10; }
        }
        for &ni in &defect.generations.anti1 {
            let ri = defect.merge_map[ni];
            if ri < n_def { def_cat_map[ri] |= 0x20; }
        }
        for &ni in &defect.generations.sterile {
            let ri = defect.merge_map[ni];
            if ri < n_def { def_cat_map[ri] |= 0x40; }
        }

        let vac_labels: &[&str] = &["vac_global", "vac_core"];
        let def_labels: &[&str] = &[
            "def_global", "def_core", "gen1", "gen2", "gen3", "anti1", "sterile",
        ];

        // ── Unified walks: vacuum + defect concurrent ───────────────────
        let (vac_results, def_results) = std::thread::scope(|s| {
            let h_vac = s.spawn(|| {
                run_unified_walk(
                    sym_head, sym_data, &all_nodes, steps,
                    seed, 2, epsilon, tmax,
                    None, &vac_cat_map, 2, vac_labels, "vacuum",
                )
            });

            let h_def = s.spawn(|| {
                if def_eigen {
                    None
                } else {
                    let all_def: Vec<usize> = (0..n_def).collect();
                    Some(run_unified_walk(
                        def_head, def_data, &all_def, steps,
                        seed, 4, epsilon, tmax,
                        None, &def_cat_map, 7, def_labels, "defect",
                    ))
                }
            });

            let (vac_global_result, vac_cat_results) = h_vac.join().unwrap();
            let def_result = h_def.join().unwrap();

            // Extract vacuum results
            let vac_global = vac_global_result;
            let vac_core = vac_cat_results.into_iter().nth(1)
                .unwrap_or_else(|| vac_global.clone());

            // Extract defect results
            let (def_global, def_core, gen1, gen2, gen3, anti1, sterile_wr) = if def_eigen {
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
                let (dg, dc) = phase3::compute_eigen(
                    n_def, &def_rows, &def_cols, steps, &defect.defect_core,
                );
                // Eigendecomp doesn't give per-generation results — use defaults
                (dg, dc,
                 WalkResult::default(), WalkResult::default(),
                 WalkResult::default(), WalkResult::default(),
                 WalkResult::default())
            } else {
                let (def_global_result, mut def_cat_results) = def_result.unwrap();
                // cat indices: 0=def_global, 1=def_core, 2=gen1, 3=gen2,
                //              4=gen3, 5=anti1, 6=sterile
                let sterile_wr = def_cat_results.pop().unwrap_or_default(); // 6
                let anti1 = def_cat_results.pop().unwrap_or_default();      // 5
                let gen3 = def_cat_results.pop().unwrap_or_default();       // 4
                let gen2 = def_cat_results.pop().unwrap_or_default();       // 3
                let gen1 = def_cat_results.pop().unwrap_or_default();       // 2
                let def_core = def_cat_results.pop()                        // 1
                    .unwrap_or_else(|| def_global_result.clone());
                let _def_global_cat = def_cat_results.pop();               // 0 (redundant with global)
                (def_global_result, def_core, gen1, gen2, gen3, anti1, sterile_wr)
            };

            (
                (vac_global, vac_core),
                (def_global, def_core, gen1, gen2, gen3, anti1, sterile_wr),
            )
        });

        let vac_global = vac_results.0;
        let vac_core = vac_results.1;
        let def_global = def_results.0;
        let def_core = def_results.1;
        let gen1 = def_results.2;
        let gen2 = def_results.3;
        let gen3 = def_results.4;
        let anti1 = def_results.5;
        let sterile = def_results.6;

        ((vac_global, vac_core), (def_global, def_core),
         [gen1, gen2, gen3, anti1], sterile)
    };

    // ── Measurements ──────────────────────────────────────────────────
    let meas = if measure.any_active() {
        // Gate M6: only run on every N-th realization
        let mut flags = measure.clone();
        if measure.decoherence_every > 1 && realization_idx % measure.decoherence_every != 0 {
            flags.decoherence = false;
        }
        // CRT bulk budget for MeasureContext (global walk, |S| = N → P_S = P_bulk)
        let walkers = ((tmax as f64 / epsilon).powi(2)).ceil() as usize;
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
            epsilon,
            seed,
            modulo_config: &flags.modulo_config,
        };
        let mut m = measure::run_all(&flags, &ctx);
        // Free per-node M3 data (~440 MB at N=10M) — summary stats are sufficient
        // for ensemble aggregation.  Without this, measure_vec accumulates
        // 440 MB × M realizations → OOM.
        if let Some(ref mut mod_result) = m.modulo {
            mod_result.compact();
        }
        Some(m)
    } else {
        None
    };

    // Free sym_vac (2.28 GB at N=10M) — no longer needed after measurements.
    // NLL: sym_head/sym_data borrows expired after thread::scope;
    //      MeasureContext borrow expired after run_all returned.
    drop(sym_vac);

    // Phase 3b — causal flux (auto-converged, unchanged)
    // Build flux CSR (trivial, <1s)
    let n_def = defect.defect_csr.n_nodes();
    let flux_csr = phase3::build_flux_csr(
        &defect.vacuum_csr, &pts, &defect.merge_map, n_points,
    );
    // Free pts (320 MB at N=10M) — no longer needed after flux CSR construction.
    drop(pts);
    let (flux_head, flux_data) = flux_csr.raw();

    let flux_origins: Vec<usize> = defect.generations.gen1.iter()
        .map(|&s| defect.merge_map[s])
        .collect();
    let flux_attr_targets: Vec<usize> = defect.generations.anti1.iter()
        .map(|&s| defect.merge_map[s])
        .collect();
    let flux_repu_targets: Vec<usize> = defect.generations.gen1.iter()
        .map(|&s| defect.merge_map[s])
        .collect();

    // ── Flux walk (unchanged — different CSR, different semantics) ────
    let (flux_attr_p, flux_repu_p) = if flux_origins.is_empty() {
        (vec![0.0; steps.len()], vec![0.0; steps.len()])
    } else {
        // Trap 3 fix: flux circuit breaker = W_bulk × 50
        let p_bulk = (tmax as f64).powi(-2);
        let w_bulk = (1.0 / (epsilon * epsilon * p_bulk)).ceil() as usize;
        let flux_circuit_breaker = w_bulk * 50;
        let n_fl = flux_origins.len();
        let p_s = p_bulk * (n_fl as f64) / (n_def.max(1) as f64);
        let flux_max = (1.0 / (epsilon * epsilon * p_s)).ceil() as usize;
        let flux_max = ((flux_max as f64) * 1.25).ceil() as usize;
        let flux_max = flux_max.max(n_fl).min(flux_circuit_breaker);
        println!("  [Phase3] flux: W_S = {flux_max} (P_S = {p_s:.2e}, |S| = {n_fl})");
        let ac = AutoConverge::new(flux_max, 2048, epsilon);
        let mut state = ConvergeState::new();
        let mut cum_attr = vec![0.0f64; steps.len()];
        let mut cum_repu = vec![0.0f64; steps.len()];
        loop {
            let starts = phase3::distribute_walkers(&flux_origins, ac.batch_size);
            let batch_seed = seed.wrapping_add(500)
                .wrapping_add(state.total_walkers as u64);
            let (batch_attr, batch_repu) = phase3::run_transmission_walkers(
                flux_head, flux_data, &starts,
                &flux_attr_targets, &flux_repu_targets,
                steps, batch_seed, Some(&defect.merge_map),
            );
            let prev = state.total_walkers;
            let new_total = prev + ac.batch_size;
            for i in 0..steps.len() {
                cum_attr[i] = (cum_attr[i] * prev as f64
                    + batch_attr[i] * ac.batch_size as f64) / new_total as f64;
                cum_repu[i] = (cum_repu[i] * prev as f64
                    + batch_repu[i] * ac.batch_size as f64) / new_total as f64;
            }
            let tmax_idx = steps.iter().rposition(|&s| (s as usize) <= tmax)
                .unwrap_or(steps.len() - 1);
            let obs = batch_attr[tmax_idx];
            if state.update(obs, &ac) { break; }
            if state.at_limit(&ac) {
                eprintln!("[Phase3/flux] WARNING: {} walkers without convergence",
                    state.total_walkers);
                break;
            }
        }
        println!("  [Phase3] flux converged at {} walkers", state.total_walkers);
        (cum_attr, cum_repu)
    };

    // Normalized flux (per-charge coupling)
    let n_attr = flux_attr_targets.len().max(1) as f64;
    let n_repu = flux_repu_targets.len().max(1) as f64;
    let flux_attr_norm: Vec<f64> = flux_attr_p.iter().map(|&f| f / n_attr).collect();
    let flux_repu_norm: Vec<f64> = flux_repu_p.iter().map(|&f| f / n_repu).collect();

    // Compose SpectralOutput
    let spectral = SpectralOutput {
        vacuum: vac_walk.0,
        vac_core: vac_walk.1,
        defect: def_walk.0,
        def_core: def_walk.1,
        generations: gen_walks,
        sterile: sterile_walk,
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
    epsilon: f64,
    tmax: usize,
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
        let result = run_realization(n_points, seed, steps, epsilon, tmax, eigen_cutoff, measure, i);

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
                measure_vec = ckpt.measure_vec;
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
                &measure_vec, welford_mean, welford_m2,
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

            // Measurement snapshots — deferred to final output.
            // aggregate_all() clones every MeasureResults via collect_and_agg(),
            // and M3 (ModuloPathResult.nodes) holds one NodeInterference per
            // node (~440 MB at N=10M).  Cloning M realizations mid-ensemble
            // would spike to M × 440 MB, causing OOM at batch 2+.
            // Spectral + topology snapshots (small) are still written above.

            println!(
                "  [snapshot] wrote *_M{count:02}.csv ({count} realizations, {elapsed}s)",
            );
        }

        current = batch_end;
    }

    if !converged && current >= max_ensemble {
        println!("  [warning] Safety cap M={max_ensemble} reached before convergence");
    }

    // ── Write per-realization alpha values (Money Plot) ────────────
    {
        use std::io::Write;
        let path = format!("{output_dir}/per_realization_alpha.csv");
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = writeln!(f, "realization,phase_sq,mass_sq,q_topo,inv_alpha");
            for (i, t) in topo_vec.iter().enumerate() {
                let q = if t.mass_sq_total > 0 {
                    t.phase_sq_total as f64 / t.mass_sq_total as f64
                } else {
                    0.0
                };
                let inv_a = if q > 0.0 { 8.0 * std::f64::consts::PI / q } else { 0.0 };
                let _ = writeln!(
                    f,
                    "{},{},{},{:.8},{:.4}",
                    i + 1,
                    t.phase_sq_total,
                    t.mass_sq_total,
                    q,
                    inv_a
                );
            }
        }
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
