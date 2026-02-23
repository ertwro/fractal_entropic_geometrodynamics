// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18746995

//! Orchestration layer for the Kuratowski Calculus engine.
//!
//! Runs an ensemble of independent Poisson sprinklings (Monte Carlo universes),
//! executes each realisation through Phases 1–3 in parallel, then ensemble-averages
//! the return probability P(t) before recomputing d_S from the mean.
//!
//! Both execution modes share the same parallel Rayon infrastructure, dense step
//! sampling, ensemble averaging, and output — only the data path differs:
//!
//! - **In-memory**: full N-node CSR in RAM.  Best for N ≤ ~500k.
//! - **Streaming (sparse)**: Two-pass sparse scanning with HashMap grid,
//!   zero disk I/O.  ~1.5 GB per realization at N=100M.  Realizations run
//!   concurrently via Rayon thread pool.

use causal_set_sim::checkpoint;
use causal_set_sim::diamond;
use causal_set_sim::measurement;
use causal_set_sim::memory::{self, ExecMode};
use causal_set_sim::output;
use causal_set_sim::skyrmion;
use causal_set_sim::spectral;
use causal_set_sim::spectral::SpectralResult;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Default ensemble seed fallback: use system clock nanos for real entropy.
fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}
/// Default eigendecomp cutoff: N <= 3000 uses exact eigendecomp,
/// above uses Monte Carlo walkers.  Overridable via `--eigen-cutoff`.
const DEFAULT_EIGEN_CUTOFF: usize = 3_000;
/// Default safety cap for ensemble realisations in adaptive mode.
const DEFAULT_MAX_ENSEMBLE: usize = 50;
/// Default relative standard error target for mass_gen1 convergence.
const DEFAULT_TARGET_ERROR: f64 = 0.01;
/// Maximum concurrent realizations per sequential batch.
/// Limits peak memory to ~BATCH_SIZE × 3 GB of CSR graphs at N=10M.
/// Override via `--batch-size`.
const DEFAULT_BATCH_SIZE: usize = 4;
/// Minimum realizations before convergence checking.
/// With fewer samples, the Welford variance estimator is unstable
/// (t-distribution with < 7 d.f. — SE underestimates true variance).
/// Override via `--min-ensemble`.
const DEFAULT_MIN_ENSEMBLE: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Measurement flags
// ─────────────────────────────────────────────────────────────────────────────

struct MeasureFlags {
    mass: bool,
    halflife: bool,
    modulo: bool,
    vacuum: bool,
    modulo_config: measurement::ModuloConfig,
}

impl MeasureFlags {
    fn any_active(&self) -> bool {
        self.mass || self.halflife || self.modulo || self.vacuum
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single realisation
// ─────────────────────────────────────────────────────────────────────────────

/// Run Phases 1–3 for one Monte Carlo universe: sprinkle → Hasse → Kuratowski → spectral.
///
/// A single realisation generates one Poisson-sprinkled causal set, builds its
/// Hasse diagram, applies Kuratowski contraction (Causal Prism detection + K₅ absorption),
/// classifies particles into generations, measures spectral dimensions via
/// random walkers, and computes causal flux between generations.
///
/// Returns `(vacuum_result, defect_result)` with full particle classification.
fn run_realization(
    n_points: usize,
    seed: u64,
    steps: &[u32],
    walkers: usize,
    eigen_cutoff: usize,
    measure: &MeasureFlags,
) -> (SpectralResult, SpectralResult, skyrmion::TopologySummary,
      Option<measurement::MeasurementResult>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Phase 1
    let (pts_raw, _big_t) = diamond::sprinkle(n_points, &mut rng);
    let (pts, vac_head, vac_data, momentum) = if n_points <= eigen_cutoff {
        diamond::build_hasse_sparse(&pts_raw)
    } else {
        diamond::build_hasse_direct(&pts_raw)
    };

    // Phase 2 — Kuratowski contraction + particle classification
    println!("  [In-Memory] Phase 1 CSR: {} edges", vac_data.len());
    let momentum_clone = if measure.halflife { momentum.clone() } else { vec![] };
    let (mut defect, topology, prisms) = skyrmion::apply_defect(n_points, vac_head, vac_data, momentum);

    // ── Measurements M2 + M4 (need defect.vac_head/vac_data, before drop) ──
    let half_life_result = if measure.halflife {
        println!("  [M2] Half-life census...");
        Some(measurement::measure_half_life_census(&prisms, &defect, &momentum_clone))
    } else {
        None
    };
    let vacuum_pol_result = if measure.vacuum {
        println!("  [M4] Vacuum polarization...");
        Some(measurement::measure_vacuum_polarization(n_points, &defect, &prisms, topology.alpha_em))
    } else {
        None
    };

    // Phase 3a — vacuum + defect spectral dimensions
    //
    // The vacuum CSR from Phase 1 is a DIRECTED Hasse diagram (DAG).
    // Spectral walkers need the UNDIRECTED (symmetric) graph so they can
    // step both past→future and future→past, probing the full 4D manifold.
    // Without symmetrization, walkers hit the future boundary immediately
    // and d_S collapses to ~1.6 (fractal/1D) instead of ~4 (Minkowski).
    let (sym_vac_head, sym_vac_data) =
        diamond::make_symmetric(n_points, &defect.vac_head, &defect.vac_data);

    let (vac, mut def) = if n_points <= eigen_cutoff {
        // Reconstruct Vac Edges for Eigen (only small N)
        // Symmetric CSR already built above — extract unique edges for eigendecomp.
        let mut vac_rows = Vec::new();
        let mut vac_cols = Vec::new();
        for u in 0..n_points {
            let start = sym_vac_head[u] as usize;
            let end = sym_vac_head[u + 1] as usize;
            for &v in &sym_vac_data[start..end] {
                if (u as u32) < v {   // symmetric CSR: dedup by u < v
                    vac_rows.push(u as u32);
                    vac_cols.push(v);
                }
            }
        }

        // Reconstruct Defect Edges for Eigen (only small N)
        // Defect CSR is already undirected (skyrmion.rs pushes both directions).
        let mut def_rows = Vec::new();
        let mut def_cols = Vec::new();
        for u in 0..defect.def_head.len() - 1 {
            let start = defect.def_head[u] as usize;
            let end = defect.def_head[u+1] as usize;
            for &v in &defect.def_data[start..end] {
                if (u as u32) < v { // Undirected unique
                    def_rows.push(u as u32);
                    def_cols.push(v);
                }
            }
        }

        let vac = spectral::compute_eigen(n_points, &vac_rows, &vac_cols, steps, &defect.vacuum_core);
        let def = spectral::compute_eigen(defect.def_head.len() - 1, &def_rows, &def_cols, steps, &defect.defect_core);
        (vac, def)
    } else {
        // Monte Carlo — use the symmetric vacuum CSR directly
        let mut rng_mc = StdRng::seed_from_u64(seed + 2);
        let vac = spectral::compute_monte_carlo_csr(
            n_points,
            &sym_vac_head,
            &sym_vac_data,
            steps,
            &defect.vacuum_core,
            walkers,
            &mut rng_mc,
        );

        // Defect — Use exact eigendecomp for small graphs (eliminates MC noise)
        let num_def_nodes = defect.def_head.len() - 1;
        let def = if num_def_nodes <= eigen_cutoff {
            println!("  [Defect graph ≤{eigen_cutoff}: using eigendecomp (exact, zero noise)]");
            // Reconstruct edge lists from defect CSR for eigendecomp
            let mut def_rows = Vec::new();
            let mut def_cols = Vec::new();
            for u in 0..num_def_nodes {
                let start = defect.def_head[u] as usize;
                let end = defect.def_head[u + 1] as usize;
                for &v in &defect.def_data[start..end] {
                    if u < v as usize {  // Undirected: only store u < v
                        def_rows.push(u as u32);
                        def_cols.push(v);
                    }
                }
            }
            spectral::compute_eigen(
                num_def_nodes, &def_rows, &def_cols, steps, &defect.defect_core
            )
        } else {
            spectral::compute_monte_carlo_csr(
                num_def_nodes,
                &defect.def_head,
                &defect.def_data,
                steps,
                &defect.defect_core,
                walkers,
                &mut rng_mc
            )
        };
        (vac, def)
    };

    // ── Measurements M1 (defect CSR, prism-confined) + M3 (symmetric vac CSR) ──
    let traversal_result = if measure.mass {
        println!("  [M1] Traversal mass ratios (prism-confined)...");
        Some(measurement::measure_traversal_mass(
            n_points, &defect.def_head, &defect.def_data, &defect, &prisms,
            walkers, *steps.last().unwrap(), seed.wrapping_add(700),
        ))
    } else {
        None
    };
    let modulo_result = if measure.modulo {
        println!("  [M3] Modulo path integral...");
        Some(measurement::measure_modulo_interference(
            n_points, &sym_vac_head, &sym_vac_data, &pts,
            walkers, *steps.last().unwrap(), seed.wrapping_add(800),
            &defect.merge_into, &measure.modulo_config,
        ))
    } else {
        None
    };

    // Free symmetric vacuum CSR — no longer needed after spectral computation.
    drop(sym_vac_head);
    drop(sym_vac_data);

    // Phase 3b — generation walkers + causal flux (always MC)
    // We already have Defect CSR in `defect.def_head`/`def_data`.

    let run_gen = |nodes: &[usize], seed_offset: u64| -> (Vec<f64>, Vec<f64>) {
        if nodes.is_empty() {
            return (vec![0.0; steps.len()], vec![0.0; steps.len()]);
        }
        let resolved: Vec<usize> = nodes.iter().map(|&n| defect.merge_into[n]).collect();
        let starts = spectral::distribute_walkers(&resolved, walkers);
        // Use Defect CSR directly; merge_into resolves ghost nodes during walk
        let p = spectral::run_walkers(
            &defect.def_head, &defect.def_data, &starts, steps,
            seed.wrapping_add(seed_offset), Some(&defect.merge_into),
        );
        let ds = spectral::spectral_dimension(steps, &p);
        (p, ds)
    };

    let (p_g1, ds_g1) = run_gen(&defect.gen1_nodes, 100);
    let (p_g2, ds_g2) = run_gen(&defect.gen2_nodes, 200);
    let (p_g3, ds_g3) = run_gen(&defect.gen3_nodes, 300);
    let (p_a1, ds_a1) = run_gen(&defect.anti1_nodes, 400);

    // Causal flux: directed adjacency (successors only, following causal order)
    let (adj_head_dir, adj_data_dir) = {
        let mut rows = Vec::new();
        let mut cols = Vec::new();

        for u in 0..n_points {
            let start = defect.vac_head[u] as usize;
            let end = defect.vac_head[u+1] as usize;
            for &v in &defect.vac_data[start..end] {
                // Recover direction: u -> v if t(u) < t(v)
                if pts[u][0] < pts[v as usize][0] {
                    let ri = defect.merge_into[u] as u32;
                    let ci = defect.merge_into[v as usize] as u32;
                    if ri != ci {
                        rows.push(ri); // forward: cause → effect
                        cols.push(ci);
                    }
                }
            }
        }

        // Manual directed CSR
        let mut head = vec![0u32; n_points + 1];
        for &r in &rows { head[r as usize + 1] += 1; }
        for i in 0..n_points { head[i+1] += head[i]; }
        let mut data = vec![0u32; rows.len()];
        let mut pos = head.clone();
        for (&r, &c) in rows.iter().zip(&cols) {
            data[pos[r as usize] as usize] = c;
            pos[r as usize] += 1;
        }
        (head, data)
    };

    // Free vacuum CSR — no longer needed after directed CSR is built.
    // At N=10M this reclaims ~240 MB.
    drop(std::mem::take(&mut defect.vac_head));
    drop(std::mem::take(&mut defect.vac_data));

    let flux_starts: Vec<usize> = defect.gen1_nodes.iter()
        .map(|&s| defect.merge_into[s])
        .collect();
    let flux_attr_targets: Vec<usize> = defect.anti1_nodes.iter()
        .map(|&s| defect.merge_into[s])
        .collect();
    let flux_repu_targets: Vec<usize> = defect.gen1_nodes.iter()
        .map(|&s| defect.merge_into[s])
        .collect();

    let (flux_attr, flux_repu) = spectral::run_transmission_walkers(
        &adj_head_dir, &adj_data_dir, &flux_starts, &flux_attr_targets, &flux_repu_targets,
        steps, seed.wrapping_add(500), Some(&defect.merge_into),
    );

    // ── Normalized flux (per-charge coupling — Definition, Vol II) ──
    let n_attr = flux_attr_targets.len().max(1) as f64;
    let n_repu = flux_repu_targets.len().max(1) as f64;
    let flux_attr_norm: Vec<f64> = flux_attr.iter().map(|&f| f / n_attr).collect();
    let flux_repu_norm: Vec<f64> = flux_repu.iter().map(|&f| f / n_repu).collect();

    // ── Sterile walkers (Φ = 0: fully phase-cancelled prisms) ────────────────
    let (p_st, ds_st) = if !defect.sterile_nodes.is_empty() {
        let resolved: Vec<usize> = defect.sterile_nodes.iter()
            .map(|&n| defect.merge_into[n]).collect();
        let starts = spectral::distribute_walkers(&resolved, walkers);
        let p = spectral::run_walkers(
            &defect.def_head, &defect.def_data, &starts, steps,
            seed.wrapping_add(600), Some(&defect.merge_into),
        );
        let ds = spectral::spectral_dimension(steps, &p);
        (p, ds)
    } else { (vec![], vec![]) };

    // Populate defect result with generation + flux data
    def.p_gen1 = p_g1;  def.ds_gen1 = ds_g1;
    def.p_gen2 = p_g2;  def.ds_gen2 = ds_g2;
    def.p_gen3 = p_g3;  def.ds_gen3 = ds_g3;
    def.p_anti1 = p_a1; def.ds_anti1 = ds_a1;
    def.flux_attraction = flux_attr;
    def.flux_repulsion = flux_repu;
    def.flux_attr_norm = flux_attr_norm;
    def.flux_repu_norm = flux_repu_norm;
    def.p_sterile = p_st;
    def.ds_sterile = ds_st;
    def.mass_gen1 = defect.mass_gen1;
    def.mass_gen2 = defect.mass_gen2;
    def.mass_gen3 = defect.mass_gen3;
    def.mass_anti1 = defect.mass_anti1;

    let measurement = if measure.any_active() {
        Some(measurement::MeasurementResult {
            traversal: traversal_result,
            half_life: half_life_result,
            modulo: modulo_result,
            vacuum_pol: vacuum_pol_result,
        })
    } else {
        None
    };

    (vac, def, topology, measurement)
}

// ─────────────────────────────────────────────────────────────────────────────
// Ensemble averaging
// ─────────────────────────────────────────────────────────────────────────────

/// Average P(t) across realisations, then recompute d_S from the mean P.
///
/// Physics: d_S is non-linear in P(t), so we average P first across all
/// realisations, then derive d_S from ⟨P⟩. This ensures the ensemble-averaged
/// spectral dimension reflects the mean geometry rather than the mean of
/// individually noisy d_S curves.
///
/// When M > 1, a second pass computes the population standard deviation of
/// each d_S(t) and flux(t) field across realisations for error bars.
fn average_ensemble(
    results: &[(SpectralResult, SpectralResult)],
    steps: &[u32],
) -> (SpectralResult, SpectralResult) {
    let m = results.len() as f64;
    let ns = steps.len();

    // ── Pass 1: accumulate + average a field across all results ──────
    let avg_field = |extract: &dyn Fn(&SpectralResult) -> &[f64], use_def: bool| -> Vec<f64> {
        let first = if use_def { extract(&results[0].1) } else { extract(&results[0].0) };
        if first.is_empty() { return vec![]; }
        let mut acc = vec![0.0; ns];
        for (vac, def) in results {
            let src = if use_def { extract(def) } else { extract(vac) };
            for (i, &v) in src.iter().enumerate().take(ns) {
                acc[i] += v;
            }
        }
        for x in &mut acc { *x /= m; }
        acc
    };

    let vp_g = avg_field(&|r| &r.p_global, false);
    let vp_l = avg_field(&|r| &r.p_local, false);
    let dp_g = avg_field(&|r| &r.p_global, true);
    let dp_l = avg_field(&|r| &r.p_local, true);

    let dp_g1 = avg_field(&|r| &r.p_gen1, true);
    let dp_g2 = avg_field(&|r| &r.p_gen2, true);
    let dp_g3 = avg_field(&|r| &r.p_gen3, true);
    let dp_a1 = avg_field(&|r| &r.p_anti1, true);
    let dp_st = avg_field(&|r| &r.p_sterile, true);
    let df_a  = avg_field(&|r| &r.flux_attraction, true);
    let df_r  = avg_field(&|r| &r.flux_repulsion, true);
    let df_an = avg_field(&|r| &r.flux_attr_norm, true);
    let df_rn = avg_field(&|r| &r.flux_repu_norm, true);

    // Average mass spectrum (scalar fields)
    let avg_mass = |extract: &dyn Fn(&SpectralResult) -> f64| -> f64 {
        results.iter().map(|(_, def)| extract(def)).sum::<f64>() / m
    };
    let mass_gen1  = avg_mass(&|r| r.mass_gen1);
    let mass_gen2  = avg_mass(&|r| r.mass_gen2);
    let mass_gen3  = avg_mass(&|r| r.mass_gen3);
    let mass_anti1 = avg_mass(&|r| r.mass_anti1);

    let ds_or_empty = |p: &[f64]| -> Vec<f64> {
        if p.is_empty() { vec![] } else { spectral::spectral_dimension(steps, p) }
    };

    // Compute derived d_S from ensemble-averaged P
    let ds_vac_g = spectral::spectral_dimension(steps, &vp_g);
    let ds_vac_l = spectral::spectral_dimension(steps, &vp_l);
    let ds_def_g = spectral::spectral_dimension(steps, &dp_g);
    let ds_def_l = spectral::spectral_dimension(steps, &dp_l);
    let ds_g1 = ds_or_empty(&dp_g1);
    let ds_g2 = ds_or_empty(&dp_g2);
    let ds_g3 = ds_or_empty(&dp_g3);
    let ds_a1 = ds_or_empty(&dp_a1);
    let ds_st = ds_or_empty(&dp_st);

    // ── Pass 2: standard deviation across realisations ──────────────
    //
    // For each d_S(t) and flux(t) field, compute the population std_dev
    // from per-realisation values recomputed from individual P(t).
    // When M <= 1 the std fields remain empty (no error bars computable).

    let std_field = |extract: &dyn Fn(&SpectralResult) -> &[f64],
                     mean: &[f64], use_def: bool| -> Vec<f64> {
        if mean.is_empty() || m <= 1.0 { return vec![]; }
        let mut acc = vec![0.0; ns];
        for (vac, def) in results {
            let src = if use_def { extract(def) } else { extract(vac) };
            for (i, &v) in src.iter().enumerate().take(ns) {
                let d = v - mean[i];
                acc[i] += d * d;
            }
        }
        for x in &mut acc { *x = (*x / m).sqrt(); }
        acc
    };

    // Recompute per-realisation d_S from individual P(t) for std computation
    let std_ds_field = |extract_p: &dyn Fn(&SpectralResult) -> &[f64],
                        mean_ds: &[f64], use_def: bool| -> Vec<f64> {
        if mean_ds.is_empty() || m <= 1.0 { return vec![]; }
        let mut acc = vec![0.0; ns];
        for (vac, def) in results {
            let src = if use_def { extract_p(def) } else { extract_p(vac) };
            if src.is_empty() { continue; }
            let ds_i = spectral::spectral_dimension(steps, src);
            for (i, &v) in ds_i.iter().enumerate().take(ns) {
                let d = v - mean_ds[i];
                acc[i] += d * d;
            }
        }
        for x in &mut acc { *x = (*x / m).sqrt(); }
        acc
    };

    // Vacuum std fields
    let vac_ds_global_std = std_ds_field(&|r| &r.p_global, &ds_vac_g, false);
    let vac_ds_local_std = std_ds_field(&|r| &r.p_local, &ds_vac_l, false);

    // Defect std fields
    let def_ds_global_std = std_ds_field(&|r| &r.p_global, &ds_def_g, true);
    let def_ds_local_std = std_ds_field(&|r| &r.p_local, &ds_def_l, true);
    let def_ds_gen1_std = std_ds_field(&|r| &r.p_gen1, &ds_g1, true);
    let def_ds_gen2_std = std_ds_field(&|r| &r.p_gen2, &ds_g2, true);
    let def_ds_gen3_std = std_ds_field(&|r| &r.p_gen3, &ds_g3, true);
    let def_ds_anti1_std = std_ds_field(&|r| &r.p_anti1, &ds_a1, true);
    let def_ds_sterile_std = std_ds_field(&|r| &r.p_sterile, &ds_st, true);
    let def_flux_attr_std = std_field(&|r| &r.flux_attraction, &df_a, true);
    let def_flux_repu_std = std_field(&|r| &r.flux_repulsion, &df_r, true);

    let vac = SpectralResult {
        ds_global: ds_vac_g,
        ds_local: ds_vac_l,
        p_global: vp_g,
        p_local: vp_l,
        p_gen1: vec![], ds_gen1: vec![],
        p_gen2: vec![], ds_gen2: vec![],
        p_gen3: vec![], ds_gen3: vec![],
        p_anti1: vec![], ds_anti1: vec![],
        flux_attraction: vec![], flux_repulsion: vec![],
        flux_attr_norm: vec![], flux_repu_norm: vec![],
        p_sterile: vec![], ds_sterile: vec![],
        mass_gen1: 0.0, mass_gen2: 0.0, mass_gen3: 0.0, mass_anti1: 0.0,
        ds_global_std: vac_ds_global_std, ds_local_std: vac_ds_local_std,
        ds_gen1_std: vec![], ds_gen2_std: vec![], ds_gen3_std: vec![],
        ds_anti1_std: vec![], ds_sterile_std: vec![],
        flux_attraction_std: vec![], flux_repulsion_std: vec![],
    };
    let def = SpectralResult {
        ds_global: ds_def_g,
        ds_local: ds_def_l,
        p_global: dp_g,
        p_local: dp_l,
        ds_gen1: ds_g1, p_gen1: dp_g1,
        ds_gen2: ds_g2, p_gen2: dp_g2,
        ds_gen3: ds_g3, p_gen3: dp_g3,
        ds_anti1: ds_a1, p_anti1: dp_a1,
        flux_attraction: df_a, flux_repulsion: df_r,
        flux_attr_norm: df_an, flux_repu_norm: df_rn,
        p_sterile: dp_st, ds_sterile: ds_st,
        mass_gen1, mass_gen2, mass_gen3, mass_anti1,
        ds_global_std: def_ds_global_std, ds_local_std: def_ds_local_std,
        ds_gen1_std: def_ds_gen1_std, ds_gen2_std: def_ds_gen2_std,
        ds_gen3_std: def_ds_gen3_std, ds_anti1_std: def_ds_anti1_std,
        ds_sterile_std: def_ds_sterile_std,
        flux_attraction_std: def_flux_attr_std,
        flux_repulsion_std: def_flux_repu_std,
    };
    (vac, def)
}

// ─────────────────────────────────────────────────────────────────────────────
// Topology aggregation
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate topology summaries across M realisations.
///
/// Sums prism counts and histogram frequencies, takes max of max_intermediates,
/// and averages generation abundances and masses.
fn aggregate_topology(topos: &[skyrmion::TopologySummary]) -> skyrmion::TopologySummary {
    let m = topos.len();
    if m == 1 {
        return topos[0].clone();
    }
    let mut hist_map: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut total_prisms = 0usize;
    let mut max_inter = 0usize;
    for t in topos {
        total_prisms += t.total_prisms;
        max_inter = max_inter.max(t.max_intermediates);
        for &(n, freq) in &t.prism_histogram {
            *hist_map.entry(n).or_insert(0) += freq;
        }
    }
    let mut hist: Vec<(usize, usize)> = hist_map.into_iter().collect();
    hist.sort_unstable_by_key(|&(n, _)| n);

    // Phase-coherence: SUM across realizations, then recompute ratios
    let vis_total: usize = topos.iter().map(|t| t.visible_mass_total).sum();
    let dark_total: usize = topos.iter().map(|t| t.dark_mass_total).sum();
    let grav_total: usize = topos.iter().map(|t| t.grav_mass_total).sum();
    let psq_total: usize = topos.iter().map(|t| t.phase_sq_total).sum();
    let msq_total: usize = topos.iter().map(|t| t.mass_sq_total).sum();
    let omega = if vis_total > 0 { dark_total as f64 / vis_total as f64 } else { f64::INFINITY };
    let q_topo = if msq_total > 0 { psq_total as f64 / msq_total as f64 } else { 0.0 };
    let alpha = q_topo / (8.0 * std::f64::consts::PI);
    let omega_energy = if q_topo > 0.0 { 1.0 / q_topo - 1.0 } else { f64::INFINITY };

    skyrmion::TopologySummary {
        total_nodes: topos[0].total_nodes,
        total_prisms,
        max_intermediates: max_inter,
        count_gen1: topos.iter().map(|t| t.count_gen1).sum::<usize>() / m,
        count_gen2: topos.iter().map(|t| t.count_gen2).sum::<usize>() / m,
        count_gen3: topos.iter().map(|t| t.count_gen3).sum::<usize>() / m,
        count_antigen1: topos.iter().map(|t| t.count_antigen1).sum::<usize>() / m,
        count_sterile: topos.iter().map(|t| t.count_sterile).sum::<usize>() / m,
        avg_mass_gen1: topos.iter().map(|t| t.avg_mass_gen1).sum::<f64>() / m as f64,
        avg_mass_gen2: topos.iter().map(|t| t.avg_mass_gen2).sum::<f64>() / m as f64,
        avg_mass_gen3: topos.iter().map(|t| t.avg_mass_gen3).sum::<f64>() / m as f64,
        avg_mass_sterile: topos.iter().map(|t| t.avg_mass_sterile).sum::<f64>() / m as f64,
        prism_histogram: hist,
        visible_mass_total: vis_total,
        dark_mass_total: dark_total,
        grav_mass_total: grav_total,
        omega_ratio: omega,
        phase_sq_total: psq_total,
        mass_sq_total: msq_total,
        alpha_em: alpha,
        omega_energy,
        phase_pos_count: topos.iter().map(|t| t.phase_pos_count).sum(),
        phase_zero_count: topos.iter().map(|t| t.phase_zero_count).sum(),
        phase_neg_count: topos.iter().map(|t| t.phase_neg_count).sum(),
        prisms_gen1: topos.iter().map(|t| t.prisms_gen1).sum::<usize>() / m,
        prisms_gen2: topos.iter().map(|t| t.prisms_gen2).sum::<usize>() / m,
        prisms_gen3: topos.iter().map(|t| t.prisms_gen3).sum::<usize>() / m,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Time formatting
// ─────────────────────────────────────────────────────────────────────────────

/// Format a duration in seconds into a human-readable string (e.g. "3m 42s").
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

/// Print CLI usage information and available flags.
fn print_usage() {
    eprintln!("Usage: causal_set_sim [N] [M] [output_dir] [--stream | --inmemory]");
    eprintln!();
    eprintln!("  N           Number of spacetime events (default: 5000)");
    eprintln!("  M           Max ensemble realisations (default: 50, or use --max-ensemble)");
    eprintln!("  output_dir  Directory for results     (default: current dir)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --stream             Force streaming mode (low RAM, writes edges to disk)");
    eprintln!("  --inmemory           Force in-memory mode (fast, needs sufficient RAM)");
    eprintln!("  --epsilon <f64>      Topological error tolerance (default: 0.01)");
    eprintln!("  --tmax <usize>       Maximum diffusion time (default: 15)");
    eprintln!("  --seed <u64>         Base seed (default: system clock for real entropy)");
    eprintln!("  --threads <usize>    Max parallel realisations (default: auto by RAM/cores)");
    eprintln!("  --eigen-cutoff <N>   Eigendecomp threshold (default: 3000)");
    eprintln!("  --batch-size <N>     Max concurrent realizations per batch (default: 4)");
    eprintln!("  --max-ensemble <N>   Safety cap on realisations (default: 50)");
    eprintln!("  --min-ensemble <N>   Min realisations before convergence check (default: 8)");
    eprintln!("  --target-error <F>   Relative SE target for convergence (default: 0.01)");
    eprintln!("  --resume             Resume from last checkpoint (skips completed batches)");
    eprintln!("  --export-slice <PATH>  Export topology slice (single realization, no ensemble)");
    eprintln!("  --measure-mass       Enable traversal mass ratios (M1)");
    eprintln!("  --measure-halflife   Enable half-life census (M2)");
    eprintln!("  --measure-modulo     Enable modulo path integral (M3)");
    eprintln!("  --measure-vacuum     Enable vacuum polarization (M4)");
    eprintln!("  --measure-all        Enable all 4 measurements");
    eprintln!("  --modulo-prime <u64> Prime modulus for M3 (default: 65537)");
    eprintln!("  --modulo-root <u64>  Primitive root for M3 (default: 3)");
    eprintln!("  --help               Show this help message");
    eprintln!();
    eprintln!("Convergence: runs batches until rel. standard error on mass_gen1 drops");
    eprintln!("below --target-error, or M reaches the safety cap (whichever comes first).");
    eprintln!();
    eprintln!("Config: Place a causal_set.toml in the output directory to set RAM limits.");
    eprintln!("  max_ram_gb = 6.0       # hard ceiling in GB (0 = auto-detect)");
    eprintln!("  safety_fraction = 0.70 # fraction of available RAM (when auto)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    // Parse flags (position-independent)
    let force_stream = args.iter().any(|a| a == "--stream");
    let force_inmemory = args.iter().any(|a| a == "--inmemory");
    let resume = args.iter().any(|a| a == "--resume");

    // Parse value flags via windows(2)
    let mut epsilon: f64 = 0.01;
    let mut tmax: usize = 15;
    let mut parsed_seed: Option<u64> = None;
    let mut parsed_threads: Option<usize> = None;
    let mut parsed_eigen_cutoff: Option<usize> = None;
    let mut parsed_batch_size: Option<usize> = None;
    let mut parsed_max_ensemble: Option<usize> = None;
    let mut parsed_min_ensemble: Option<usize> = None;
    let mut parsed_target_error: Option<f64> = None;
    let mut export_slice_path: Option<String> = None;
    let mut parsed_modulo_prime: Option<u64> = None;
    let mut parsed_modulo_root: Option<u64> = None;
    for pair in args.windows(2) {
        if pair[0] == "--epsilon" {
            if let Ok(v) = pair[1].parse::<f64>() { epsilon = v; }
        }
        if pair[0] == "--tmax" {
            if let Ok(v) = pair[1].parse::<usize>() { tmax = v; }
        }
        if pair[0] == "--seed" {
            if let Ok(v) = pair[1].parse::<u64>() { parsed_seed = Some(v); }
        }
        if pair[0] == "--threads" {
            if let Ok(v) = pair[1].parse::<usize>() { parsed_threads = Some(v); }
        }
        if pair[0] == "--eigen-cutoff" {
            if let Ok(v) = pair[1].parse::<usize>() { parsed_eigen_cutoff = Some(v); }
        }
        if pair[0] == "--batch-size" {
            if let Ok(v) = pair[1].parse::<usize>() { parsed_batch_size = Some(v); }
        }
        if pair[0] == "--max-ensemble" {
            if let Ok(v) = pair[1].parse::<usize>() { parsed_max_ensemble = Some(v); }
        }
        if pair[0] == "--min-ensemble" {
            if let Ok(v) = pair[1].parse::<usize>() { parsed_min_ensemble = Some(v); }
        }
        if pair[0] == "--target-error" {
            if let Ok(v) = pair[1].parse::<f64>() { parsed_target_error = Some(v); }
        }
        if pair[0] == "--export-slice" {
            export_slice_path = Some(pair[1].clone());
        }
        if pair[0] == "--modulo-prime" {
            if let Ok(v) = pair[1].parse::<u64>() { parsed_modulo_prime = Some(v); }
        }
        if pair[0] == "--modulo-root" {
            if let Ok(v) = pair[1].parse::<u64>() { parsed_modulo_root = Some(v); }
        }
    }

    // Flags that consume the next argument
    let value_flags: std::collections::HashSet<&str> =
        ["--epsilon", "--tmax", "--seed", "--threads", "--eigen-cutoff", "--batch-size",
         "--max-ensemble", "--min-ensemble", "--target-error", "--export-slice",
         "--modulo-prime", "--modulo-root"]
        .iter().copied().collect();

    // Parse positional args (skip flags and their values)
    let mut positional: Vec<&str> = Vec::new();
    {
        let mut skip_next = false;
        for arg in &args[1..] {
            if skip_next { skip_next = false; continue; }
            if value_flags.contains(arg.as_str()) { skip_next = true; continue; }
            if arg.starts_with("--") { continue; }
            positional.push(arg.as_str());
        }
    }

    let n_points: usize = positional
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);
    let positional_m: usize = positional
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_ENSEMBLE);
    let max_ensemble: usize = parsed_max_ensemble.unwrap_or(positional_m);
    let min_ensemble: usize = parsed_min_ensemble.unwrap_or(DEFAULT_MIN_ENSEMBLE);
    let target_error: f64 = parsed_target_error.unwrap_or(DEFAULT_TARGET_ERROR);
    let cli_output_dir: Option<String> = positional.get(2).map(|s| s.to_string());

    // ── Seed + Eigen cutoff ──────────────────────────────────────────
    let seed_base: u64 = parsed_seed.unwrap_or_else(default_seed);
    let eigen_cutoff: usize = parsed_eigen_cutoff.unwrap_or(DEFAULT_EIGEN_CUTOFF);

    // ── Measurement flags ──────────────────────────────────────────────
    let measure_all = args.iter().any(|a| a == "--measure-all");
    let measure = MeasureFlags {
        mass: measure_all || args.iter().any(|a| a == "--measure-mass"),
        halflife: measure_all || args.iter().any(|a| a == "--measure-halflife"),
        modulo: measure_all || args.iter().any(|a| a == "--measure-modulo"),
        vacuum: measure_all || args.iter().any(|a| a == "--measure-vacuum"),
        modulo_config: measurement::ModuloConfig {
            prime: parsed_modulo_prime.unwrap_or(65537),
            root: parsed_modulo_root.unwrap_or(3),
        },
    };

    if measure.any_active() {
        let active: Vec<&str> = [
            if measure.mass { Some("M1:mass") } else { None },
            if measure.halflife { Some("M2:halflife") } else { None },
            if measure.modulo { Some("M3:modulo") } else { None },
            if measure.vacuum { Some("M4:vacuum") } else { None },
        ].iter().filter_map(|&x| x).collect();
        println!("  measurements: {}", active.join(", "));
        if measure.modulo {
            println!("  modulo config: p={}, g={}", measure.modulo_config.prime, measure.modulo_config.root);
        }
    }

    println!("  seed: {seed_base}{}",
        if parsed_seed.is_some() { " (deterministic)" } else { " (system clock)" });

    // ── Config + Mode selection ────────────────────────────────────────
    // Load config from cwd first; the output_dir inside it is the default.
    // CLI positional arg overrides it.
    memory::ensure_config(".");
    let cfg = memory::load_config(".");

    let output_dir: String = cli_output_dir.unwrap_or_else(|| {
        if cfg.output_dir.is_empty() { ".".to_string() } else { cfg.output_dir.clone() }
    });

    // Create output directory if it doesn't exist
    if output_dir != "." {
        std::fs::create_dir_all(&output_dir).ok();
    }

    let exec_mode = if force_stream {
        println!("  (forced streaming mode via --stream)\n");
        ExecMode::Streaming
    } else if force_inmemory {
        println!("  (forced in-memory mode via --inmemory)\n");
        ExecMode::InMemory
    } else {
        let (recommended, available, estimated) = memory::recommend_mode(n_points, &cfg);
        memory::prompt_mode(recommended, available, estimated, &cfg)
    };

    let tier = if n_points <= eigen_cutoff {
        "eigendecomp"
    } else {
        "mc-walkers"
    };
    println!(
        "Causal Set Spectral Dimension  (N={n_points}, M\u{2265}{min_ensemble}\u{2264}{max_ensemble} (adaptive, target \u{03b5}={target_error}), tier={tier}, mode={exec_mode:?})\n"
    );

    // ── Export-slice early exit (Phases 1-2 only, no spectral walkers) ──
    if let Some(ref slice_path) = export_slice_path {
        use causal_set_sim::anim_export;

        println!("[export-slice] Running single realization (seed={seed_base})...");

        let mut rng = StdRng::seed_from_u64(seed_base);
        let (pts_raw, _big_t) = diamond::sprinkle(n_points, &mut rng);
        let (pts, vac_head, vac_data, momentum) = if n_points <= eigen_cutoff {
            diamond::build_hasse_sparse(&pts_raw)
        } else {
            diamond::build_hasse_direct(&pts_raw)
        };
        drop(pts_raw);

        let (defect, _topo, causal_prisms) =
            skyrmion::apply_defect(n_points, vac_head, vac_data, momentum);

        // Extract directed Hasse edges from vacuum CSR (past→future by time ordering)
        let mut hasse_edges = Vec::new();
        for u in 0..n_points {
            let start = defect.vac_head[u] as usize;
            let end = defect.vac_head[u + 1] as usize;
            for &v in &defect.vac_data[start..end] {
                if pts[u][0] < pts[v as usize][0] {
                    hasse_edges.push((u as u32, v));
                }
            }
        }

        // Convert CausalPrism → PrismDef (usize → u32)
        let prism_defs: Vec<anim_export::PrismDef> = causal_prisms.iter().map(|p| {
            anim_export::PrismDef {
                origin: p.origin as u32,
                destination: p.destination as u32,
                intermediates: p.intermediates.iter().map(|&i| i as u32).collect(),
            }
        }).collect();

        let slice = anim_export::TopologySlice {
            n_total: n_points,
            coordinates: pts,
            hasse_edges,
            prisms: prism_defs,
        };

        anim_export::write_slice(slice_path, &slice)?;

        let file_size = std::fs::metadata(slice_path).map(|m| m.len()).unwrap_or(0);
        println!("[export-slice] Wrote {} ({} bytes)", slice_path, file_size);
        println!("  {} coordinates, {} edges, {} prisms",
            slice.n_total, slice.hasse_edges.len(), slice.prisms.len());
        return Ok(());
    }

    // ── Export-lightcone early exit (Phases 1-2, BFS light cone → CSV) ──
    if let Some(ref lc_path) = export_lightcone_path {
        println!("[export-lightcone] Running single realization (seed={seed_base})...");

        let mut rng = StdRng::seed_from_u64(seed_base);
        let (pts_raw, _big_t) = diamond::sprinkle(n_points, &mut rng);
        let (_pts, vac_head, vac_data, _momentum) = if n_points <= eigen_cutoff {
            diamond::build_hasse_sparse(&pts_raw)
        } else {
            diamond::build_hasse_direct(&pts_raw)
        };
        drop(pts_raw);

        let meta = format!(
            "N: {n_points}  seed: {seed_base}  max_depth: 4\n\
             Directed Hasse DAG — 4-layer BFS light cone for BD operator"
        );

        output::export_lightcone(lc_path, &vac_head, &vac_data, n_points, 4, &meta);
        return Ok(());
    }

    // ── Causal Resolution Theorem: W = ⌈t_max² / ε²⌉ ──────────────
    let walkers: usize = ((tmax as f64 / epsilon).powi(2)).ceil() as usize;
    println!(
        "[Phase 3] Topological Error Tolerance set to \u{03b5} = {epsilon}. At t_max = {tmax},\n\
         Causal Resolution Theorem dictates W = {walkers} spectral walkers required\n\
         to maintain 4D continuous limit.\n"
    );

    // ── Unified Parallel Ensemble ────────────────────────────────────
    //
    // Both modes share:  dense step sampling, parallel Rayon execution,
    //                     ensemble averaging, CSV output, summary.
    //
    // In-memory:  full N-node CSR in RAM per realization.
    // Streaming:  Two-pass sparse scanning (HashMap grid, zero disk I/O).
    //             ~1.5 GB per realization at N=100M.

    // Dense sampling for accurate interpolation at dS = 3.96 transition
    // [1..30] dense, [32..100] coarse = 65 points
    let steps: Vec<u32> = (1..=30).chain((16..=50).map(|i| i * 2)).collect();
    let t0 = Instant::now();

    // Concurrency limits depend on mode:
    //   In-memory:  bounded by RAM (full CSR is huge)
    //   Streaming:  bounded by CPU cores (core CSR is ~1-2 GB)
    let max_concurrent_runs = if let Some(t) = parsed_threads {
        t.min(max_ensemble).max(1)
    } else if exec_mode == ExecMode::Streaming {
        let cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        cpus.min(max_ensemble).min(8)
    } else {
        memory::max_concurrent_runs(n_points)
    };

    // Cap concurrency at batch_size to bound peak memory
    let batch_size = parsed_batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
    let max_concurrent_runs = max_concurrent_runs.min(batch_size);

    let mode_label = if exec_mode == ExecMode::Streaming { "hybrid-parallel" } else { "in-memory" };
    println!("  (parallel ensemble — {mode_label}, concurrency: {max_concurrent_runs}, \
              per-realization checkpoint, adaptive convergence)\n");

    let done = AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_concurrent_runs)
        .build()
        .unwrap();

    let run_one = |i: usize| -> (SpectralResult, SpectralResult, skyrmion::TopologySummary,
                                   Option<measurement::MeasurementResult>) {
        let seed = seed_base + i as u64;

        let result = if exec_mode == ExecMode::Streaming {
            // Sparse scan: no disk I/O, HashMap grid instead of dense arrays
            let chunk_size = 1_000_000;
            let (v, d, t) = skyrmion::process_streaming(n_points, chunk_size, seed, &steps, walkers)
                .expect("Sparse streaming failed");
            (v, d, t, None) // measurements not supported in streaming mode
        } else {
            run_realization(n_points, seed, &steps, walkers, eigen_cutoff, &measure)
        };

        let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed = t0.elapsed().as_secs_f64();
        let remaining = max_ensemble.saturating_sub(completed);

        let msg = if completed == 1 && remaining > 0 {
            let eta = elapsed * remaining as f64;
            format!("  [done {completed}/{max_ensemble}]  first realization took {} — ETA remaining: ~{}",
                    fmt_duration(elapsed), fmt_duration(eta))
        } else if remaining > 0 {
            let rate = elapsed / completed as f64 / max_concurrent_runs as f64;
            let eta = rate * remaining as f64;
             format!("  [done {completed}/{max_ensemble}]  elapsed {} — ETA remaining: ~{}",
                    fmt_duration(elapsed), fmt_duration(eta))
        } else {
             format!("  [done {completed}/{max_ensemble}]  elapsed {}", fmt_duration(elapsed))
        };

        println!("{msg}");
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("simulation.log") {
            use std::io::Write;
            writeln!(file, "{msg}").ok();
        }

        result
    };

    // ── Checkpoint: build parameter fingerprint + try resume ────────────
    let mode_str_ckpt = if exec_mode == ExecMode::Streaming { "streaming" } else { "inmemory" };
    let run_params = checkpoint::make_params(
        n_points, seed_base, epsilon, tmax, eigen_cutoff, mode_str_ckpt,
    );

    let mut spectral_pairs: Vec<(SpectralResult, SpectralResult)> =
        Vec::with_capacity(max_ensemble);
    let mut topo_vec: Vec<skyrmion::TopologySummary> =
        Vec::with_capacity(max_ensemble);
    let mut measure_vec: Vec<measurement::MeasurementResult> =
        Vec::with_capacity(max_ensemble);
    let mut welford_mean: f64 = 0.0;
    let mut welford_m2: f64 = 0.0;

    let start_from = if resume {
        match checkpoint::load(&output_dir, &run_params) {
            Ok(ckpt) => {
                println!("  Resuming: {}/{} realizations completed", ckpt.completed, max_ensemble);
                spectral_pairs = ckpt.spectral_pairs;
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

    // ── Adaptive Batching with Per-Realization Checkpointing ───────────────
    //
    // Process realizations in batches of `max_concurrent_runs`.  Each batch
    // runs in parallel via Rayon; between batches the heavy CSR graphs
    // (~3 GB each at N=10M) are freed, keeping only the lightweight
    // SpectralResult accumulator (~10 KB per realization).
    //
    // After each batch, results are drained one at a time.  Each individual
    // result triggers a Welford update AND a checkpoint write.  This means
    // a crash mid-batch loses the in-flight realizations, but every fully
    // completed realization is persisted immediately.
    //
    // Peak memory: max_concurrent_runs × ~3 GB  (not M × 3 GB).

    let mut current = start_from;
    let mut converged = false;

    while current < max_ensemble && !converged {
        let batch_end = (current + max_concurrent_runs).min(max_ensemble);
        let chunk: Vec<usize> = (current..batch_end).collect();

        println!("  \u{2500}\u{2500} Batch (realizations {}-{}/{}) \u{2500}\u{2500}",
            current + 1, batch_end, max_ensemble);

        let batch_results: Vec<(SpectralResult, SpectralResult, skyrmion::TopologySummary,
                                Option<measurement::MeasurementResult>)> =
            pool.install(|| {
                chunk.par_iter()
                    .with_max_len(1)
                    .map(|&i| run_one(i))
                    .collect()
            });

        // ── Backup checkpoint before combining new batch ──────────────
        if current > start_from {
            let ckpt_path = std::path::Path::new(&output_dir).join(".checkpoint.bin");
            if ckpt_path.exists() {
                let backup = std::path::Path::new(&output_dir)
                    .join(format!(".checkpoint.bin.bak.{current}"));
                if let Err(e) = std::fs::copy(&ckpt_path, &backup) {
                    eprintln!("  [backup] warning: {e}");
                } else {
                    println!("  [backup] .checkpoint.bin → {}", backup.display());
                }
            }
        }

        // Drain batch: Welford update + checkpoint after EACH realization
        for (vac, def, topo, meas) in batch_results {
            let x = def.mass_gen1;
            spectral_pairs.push((vac, def));
            topo_vec.push(topo);
            if let Some(m) = meas {
                measure_vec.push(m);
            }

            // Welford online update
            let count = spectral_pairs.len();
            let delta = x - welford_mean;
            welford_mean += delta / count as f64;
            let delta2 = x - welford_mean;
            welford_m2 += delta * delta2;

            // Checkpoint after EACH realization (atomic write — crash-safe)
            if let Err(e) = checkpoint::save(
                &output_dir, &run_params, &spectral_pairs, &topo_vec,
                welford_mean, welford_m2,
            ) {
                eprintln!("  [checkpoint] warning: {e}");
            }
        }

        // Convergence check (after full batch drained)
        // Require min_ensemble samples before checking — with fewer samples,
        // the Welford variance estimator is unstable (t-distribution with
        // < 7 d.f. underestimates the true SE, causing false convergence).
        let count = spectral_pairs.len();
        if count >= min_ensemble && welford_mean.abs() > 1e-12 {
            let variance = welford_m2 / (count - 1) as f64;  // Bessel correction
            let std_error = (variance / count as f64).sqrt();
            let rel_error = std_error / welford_mean.abs();

            println!("  [convergence] M={count}, mass_gen1={welford_mean:.4}, \
                      SE={std_error:.4}, rel_err={rel_error:.6}");

            if rel_error <= target_error {
                println!("  \u{2713} Statistical convergence reached \
                          (rel_err={rel_error:.6} \u{2264} {target_error})");
                converged = true;
            }
        } else {
            println!("  [checkpoint: {count}/{max_ensemble}]");
        }

        // ── Accumulation metadata log ─────────────────────────────────
        {
            use std::io::Write;
            let log_path = format!("{output_dir}/accumulation.log");
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true).append(true).open(&log_path)
            {
                let count = spectral_pairs.len();
                let variance = if count > 1 { welford_m2 / (count - 1) as f64 } else { 0.0 };
                let std_error = if count > 1 { (variance / count as f64).sqrt() } else { 0.0 };
                let elapsed = t0.elapsed().as_secs();
                let _ = writeln!(f,
                    "batch {}-{}/{} | realizations={} | mass_gen1_mean={:.4} | SE={:.6} | rel_err={:.6} | converged={} | elapsed={}s",
                    current + 1, batch_end, max_ensemble,
                    count, welford_mean, std_error,
                    if welford_mean.abs() > 1e-12 { std_error / welford_mean.abs() } else { f64::NAN },
                    converged, elapsed,
                );
            }
        }

        // ── Intermediate CSV snapshot (generational, non-destructive) ────
        {
            let count = spectral_pairs.len();
            let (snap_vac, snap_def) = average_ensemble(&spectral_pairs, &steps);
            let snap_topo = aggregate_topology(&topo_vec);
            let elapsed = t0.elapsed().as_secs();
            let mode_str = if exec_mode == ExecMode::Streaming { "streaming" } else { "in-memory" };
            let snap_meta = format!(
                "N: {n_points}  M: {count} (intermediate snapshot)  mode: {mode_str}\n\
                 epsilon: {epsilon}  tmax: {tmax}  walkers: {walkers}\n\
                 seed: {seed_base}  elapsed: {elapsed}s"
            );
            output::write_csv(
                &format!("{output_dir}/results_M{count:02}.csv"),
                &steps, &snap_vac, &snap_def, &snap_meta,
            );
            output::export_topology_summary(
                &format!("{output_dir}/topology_summary_M{count:02}.csv"),
                &snap_topo, &snap_meta,
            );
            output::export_mass_spectrum(
                &format!("{output_dir}/mass_spectrum_M{count:02}.csv"),
                &snap_topo.prism_histogram, &snap_meta,
            );
            // Measurement snapshots
            if !measure_vec.is_empty() {
                let meas_agg = measurement::aggregate_measurements(&measure_vec);
                if let Some(ref t) = meas_agg.traversal {
                    output::write_traversal_mass_csv(
                        &format!("{output_dir}/traversal_mass_M{count:02}.csv"), t, &snap_meta);
                }
                if let Some(ref h) = meas_agg.half_life {
                    output::write_half_life_csv(
                        &format!("{output_dir}/half_life_M{count:02}.csv"), h, &snap_meta);
                }
                if let Some(ref m) = meas_agg.modulo {
                    output::write_modulo_interference_csv(
                        &format!("{output_dir}/modulo_interference_M{count:02}.csv"), m, &snap_meta);
                }
                if let Some(ref v) = meas_agg.vacuum_pol {
                    output::write_vacuum_polarization_csv(
                        &format!("{output_dir}/vacuum_polarization_M{count:02}.csv"), v, &snap_meta);
                }
            }
            println!("  [snapshot] wrote *_M{count:02}.csv ({count} realizations, {elapsed}s)");
        }

        current = batch_end;
    }

    if !converged && current >= max_ensemble {
        println!("  [warning] Safety cap M={max_ensemble} reached before convergence");
    }

    // ── Ensemble average from accumulated results ─────────────────────────
    let actual_m = spectral_pairs.len();
    let (vac_avg, def_avg) = average_ensemble(&spectral_pairs, &steps);
    drop(spectral_pairs); // free accumulator before Phase 4 output
    let topo_agg = aggregate_topology(&topo_vec);
    let meas_agg = if !measure_vec.is_empty() {
        Some(measurement::aggregate_measurements(&measure_vec))
    } else {
        None
    };
    drop(measure_vec);

    // ── Phase 4 ─────────────────────────────────────────────────────────
    println!("\n[Phase 4] Output …");
    let csv_path = format!("{output_dir}/results.csv");

    let mode_str = if exec_mode == ExecMode::Streaming { "streaming" } else { "in-memory" };
    let timestamp = {
        use std::time::SystemTime;
        let d = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = d.as_secs();
        let (s, m, h) = (secs % 60, (secs / 60) % 60, (secs / 3600) % 24);
        let days = secs / 86400;
        // Days since 1970-01-01 → year/month/day (good enough for UTC stamp)
        let (y, mo, dy) = {
            let mut y = 1970u64;
            let mut rem = days;
            loop {
                let ylen = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
                if rem < ylen { break; }
                rem -= ylen;
                y += 1;
            }
            let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
            let mdays = [31, if leap {29} else {28}, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let mut mo = 0u64;
            for &ml in &mdays {
                if rem < ml { break; }
                rem -= ml;
                mo += 1;
            }
            (y, mo + 1, rem + 1)
        };
        format!("{y:04}-{mo:02}-{dy:02}T{h:02}:{m:02}:{s:02}Z")
    };
    let dirty_flag = env!("GIT_DIRTY");
    let commit_str = if dirty_flag == "dirty" {
        format!("{} (dirty)", env!("GIT_HASH"))
    } else {
        env!("GIT_HASH").to_string()
    };
    let metadata = format!(
        "commit: {commit_str}\n\
         N: {n_points}  M: {actual_m} (converged={converged}, min={min_ensemble}, max={max_ensemble})  mode: {mode_str}\n\
         epsilon: {epsilon}  tmax: {tmax}  walkers: {walkers}\n\
         algorithm: forward-forward belly (children(u) \u{2229} parents(v))\n\
         timestamp: {timestamp}\n\
         seed: {seed_base}"
    );

    output::write_csv(&csv_path, &steps, &vac_avg, &def_avg, &metadata);
    output::export_topology_summary(
        &format!("{output_dir}/topology_summary.csv"), &topo_agg, &metadata,
    );
    output::export_mass_spectrum(
        &format!("{output_dir}/mass_spectrum.csv"), &topo_agg.prism_histogram, &metadata,
    );

    // ── Measurement CSV output ──────────────────────────────────────────
    if let Some(ref meas) = meas_agg {
        if let Some(ref t) = meas.traversal {
            output::write_traversal_mass_csv(
                &format!("{output_dir}/traversal_mass.csv"), t, &metadata);
        }
        if let Some(ref h) = meas.half_life {
            output::write_half_life_csv(
                &format!("{output_dir}/half_life.csv"), h, &metadata);
        }
        if let Some(ref m) = meas.modulo {
            output::write_modulo_interference_csv(
                &format!("{output_dir}/modulo_interference.csv"), m, &metadata);
        }
        if let Some(ref v) = meas.vacuum_pol {
            output::write_vacuum_polarization_csv(
                &format!("{output_dir}/vacuum_polarization.csv"), v, &metadata);
        }
    }

    // Keep checkpoint for --resume accumulation across runs.
    // CSV and checkpoint coexist; checkpoint is never deleted.

    // ── Summary ─────────────────────────────────────────────────────────
    let mid = steps.len() / 2;
    let last = steps.len() - 1;
    let elapsed = t0.elapsed().as_secs_f64();
    println!("\n── Summary ({actual_m} realisations, converged={converged}) ────────────────────────");
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
    println!("  ── Mass Spectrum (avg N) ──");
    println!(
        "  Gen1 = {:.2}, Gen2 = {:.2}, Gen3 = {:.2}, Anti1 = {:.2}",
        def_avg.mass_gen1, def_avg.mass_gen2, def_avg.mass_gen3, def_avg.mass_anti1
    );
    if let Some(ref meas) = meas_agg {
        if let Some(ref t) = meas.traversal {
            println!("  ── Traversal Mass Ratios ──");
            println!("  Gen1: {:.2} ({} traversals), Gen2: {:.2} ({} traversals), Gen3: {:.2} ({} traversals)",
                t.mean_traversal[0], t.n_traversals[0],
                t.mean_traversal[1], t.n_traversals[1],
                t.mean_traversal[2], t.n_traversals[2]);
            println!("  ratio gen2/gen1 = {:.4}, gen3/gen1 = {:.4}",
                t.ratio_gen2_gen1, t.ratio_gen3_gen1);
        }
        if let Some(ref h) = meas.half_life {
            println!("  ── Half-Life Census ──");
            println!("  Prisms: gen1={}, gen2={}, gen3={}, anti1={}",
                h.gen_counts[0], h.gen_counts[1], h.gen_counts[2], h.gen_counts[3]);
            println!("  stability gen2/gen1 = {:.4}, gen3/gen1 = {:.4}",
                h.stability_ratio_gen2, h.stability_ratio_gen3);
        }
        if let Some(ref m) = meas.modulo {
            println!("  ── Modulo Path Integral (p={}, g={}) ──", m.prime, m.root);
            println!("  mean_intensity = {:.6}, max = {:.6}, constructive = {}, destructive = {}",
                m.mean_intensity, m.max_intensity, m.constructive_count, m.destructive_count);
        }
        if let Some(ref v) = meas.vacuum_pol {
            println!("  ── Vacuum Polarization ──");
            println!("  screening = {:.6}, bare_alpha = {:.8}, screened_alpha = {:.8}",
                v.mean_screening, v.bare_alpha, v.screened_alpha);
            println!("  attempted = {}, rejected = {}, accepted = {}",
                v.total_attempted, v.total_rejected, v.total_accepted);
        }
    }
    println!("  Total time: {}", fmt_duration(elapsed));
    println!("────────────────────────────────────────────────────");
    Ok(())
}
