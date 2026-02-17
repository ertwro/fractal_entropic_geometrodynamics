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

use causal_set_sim::diamond;
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
/// Default number of independent realisations for ensemble averaging.
/// 10 realisations provide ~3% statistical error on P(t) at moderate N.
const DEFAULT_ENSEMBLE: usize = 10;

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
) -> (SpectralResult, SpectralResult, skyrmion::TopologySummary) {
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
    let (mut defect, topology) = skyrmion::apply_defect(n_points, vac_head, vac_data, momentum);

    // Phase 3a — vacuum + defect spectral dimensions
    let (vac, mut def) = if n_points <= eigen_cutoff {
        // Reconstruct Vac Edges for Eigen (only small N)
        let mut vac_rows = Vec::new();
        let mut vac_cols = Vec::new();
        for u in 0..n_points {
            let start = defect.vac_head[u] as usize;
            let end = defect.vac_head[u+1] as usize;
            for &v in &defect.vac_data[start..end] {
                if (u as u32) < v {
                    vac_rows.push(u as u32);
                    vac_cols.push(v);
                }
            }
        }

        // Reconstruct Defect Edges for Eigen (only small N)
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
        // Monte Carlo (Zero-Copy CSR)
        let mut rng_mc = StdRng::seed_from_u64(seed + 2);
        
        // Vacuum (using Phase 1 CSR? No, Phase 1 gave us rows/cols. Skyrmion built CSR.)
        // Skyrmion returns vac_head/vac_data.
        let vac = spectral::compute_monte_carlo_csr(
            n_points,
            &defect.vac_head,
            &defect.vac_data,
            steps,
            &defect.vacuum_core,
            walkers,
            &mut rng_mc
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

    // ── Normalized flux (per-node coupling strength, Conjecture C3) ──
    let n_attr = flux_attr_targets.len().max(1) as f64;
    let n_repu = flux_repu_targets.len().max(1) as f64;
    let flux_attr_norm: Vec<f64> = flux_attr.iter().map(|&f| f / n_attr).collect();
    let flux_repu_norm: Vec<f64> = flux_repu.iter().map(|&f| f / n_repu).collect();

    // ── Sterile walkers (Conjecture C6: N > 5 prisms) ────────────────
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

    (vac, def, topology)
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
    eprintln!("  M           Ensemble realisations     (default: 10)");
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
    eprintln!("  --help               Show this help message");
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

    // Parse value flags via windows(2)
    let mut epsilon: f64 = 0.01;
    let mut tmax: usize = 15;
    let mut parsed_seed: Option<u64> = None;
    let mut parsed_threads: Option<usize> = None;
    let mut parsed_eigen_cutoff: Option<usize> = None;
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
    }

    // Flags that consume the next argument
    let value_flags: std::collections::HashSet<&str> =
        ["--epsilon", "--tmax", "--seed", "--threads", "--eigen-cutoff"]
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
    let m_ensemble: usize = positional
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ENSEMBLE);
    let cli_output_dir: Option<String> = positional.get(2).map(|s| s.to_string());

    // ── Seed + Eigen cutoff ──────────────────────────────────────────
    let seed_base: u64 = parsed_seed.unwrap_or_else(default_seed);
    let eigen_cutoff: usize = parsed_eigen_cutoff.unwrap_or(DEFAULT_EIGEN_CUTOFF);

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
        "Causal Set Spectral Dimension  (N={n_points}, M={m_ensemble}, tier={tier}, mode={exec_mode:?})\n"
    );

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
        t.min(m_ensemble).max(1)
    } else if exec_mode == ExecMode::Streaming {
        let cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        cpus.min(m_ensemble).min(8)
    } else {
        memory::max_concurrent_runs(n_points)
    };

    let mode_label = if exec_mode == ExecMode::Streaming { "hybrid-parallel" } else { "in-memory" };
    println!("  (parallel ensemble — {mode_label}, concurrency: {max_concurrent_runs})\n");

    let done = AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_concurrent_runs)
        .build()
        .unwrap();

    let run_one = |i: usize| -> (SpectralResult, SpectralResult, skyrmion::TopologySummary) {
        let seed = seed_base + i as u64;

        let result = if exec_mode == ExecMode::Streaming {
            // Sparse scan: no disk I/O, HashMap grid instead of dense arrays
            let chunk_size = 1_000_000;
            skyrmion::process_streaming(n_points, chunk_size, seed, &steps, walkers)
                .expect("Sparse streaming failed")
        } else {
            run_realization(n_points, seed, &steps, walkers, eigen_cutoff)
        };

        let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed = t0.elapsed().as_secs_f64();
        let remaining = m_ensemble - completed;

        let msg = if completed == 1 && remaining > 0 {
            let eta = elapsed * remaining as f64;
            format!("  [done {completed}/{m_ensemble}]  first realization took {} — ETA remaining: ~{}",
                    fmt_duration(elapsed), fmt_duration(eta))
        } else if remaining > 0 {
            let rate = elapsed / completed as f64 / max_concurrent_runs as f64;
            let eta = rate * remaining as f64;
             format!("  [done {completed}/{m_ensemble}]  elapsed {} — ETA remaining: ~{}",
                    fmt_duration(elapsed), fmt_duration(eta))
        } else {
             format!("  [done {completed}/{m_ensemble}]  elapsed {}", fmt_duration(elapsed))
        };

        println!("{msg}");
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("simulation.log") {
            use std::io::Write;
            writeln!(file, "{msg}").ok();
        }

        result
    };

    let results: Vec<(SpectralResult, SpectralResult, skyrmion::TopologySummary)> = pool.install(|| {
        (0..m_ensemble)
            .into_par_iter()
            .with_max_len(1)
            .map(run_one)
            .collect()
    });

    // ── Split spectral + topology, then average ───────────────────────────
    let (spectral_pairs, topo_vec): (Vec<_>, Vec<_>) = results.into_iter()
        .map(|(v, d, t)| ((v, d), t))
        .unzip();
    let (vac_avg, def_avg) = average_ensemble(&spectral_pairs, &steps);
    let topo_agg = aggregate_topology(&topo_vec);

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
         N: {n_points}  M: {m_ensemble}  mode: {mode_str}\n\
         epsilon: {epsilon}  tmax: {tmax}  walkers: {walkers}\n\
         algorithm: forward-forward belly (children(u) ∩ parents(v))\n\
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

    // ── Summary ─────────────────────────────────────────────────────────
    let mid = steps.len() / 2;
    let last = steps.len() - 1;
    let elapsed = t0.elapsed().as_secs_f64();
    println!("\n── Summary ({m_ensemble} realisations) ────────────────────────");
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
    println!("  Total time: {}", fmt_duration(elapsed));
    println!("────────────────────────────────────────────────────");
    Ok(())
}
