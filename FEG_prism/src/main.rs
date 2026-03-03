// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Thin CLI for the Kuratowski Calculus engine.
//!
//! Parses command-line arguments and delegates to [`ensemble::runner::run_ensemble`]
//! for the actual simulation, or to [`anim_export`] for topology slice export.

use feg_prism::anim_export;
use feg_prism::config::{self, ExecMode};
use feg_prism::ensemble::runner::{self, EnsembleResult};
use feg_prism::measure::{self, MeasureFlags, ModuloConfig};
use feg_prism::output::{self, Metadata};
use feg_prism::phase1;
use feg_prism::phase2;
use feg_prism::provenance;

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

/// Default eigendecomp cutoff: N <= 3000 uses exact eigendecomp.
const DEFAULT_EIGEN_CUTOFF: usize = 3_000;
const DEFAULT_MAX_ENSEMBLE: usize = 50;
const DEFAULT_TARGET_ERROR: f64 = 0.01;
const DEFAULT_BATCH_SIZE: usize = 4;
const DEFAULT_MIN_ENSEMBLE: usize = 8;

fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

fn print_usage() {
    eprintln!("Usage: feg_prism [N] [M] [output_dir] [--stream | --inmemory]");
    eprintln!();
    eprintln!("  N           Number of spacetime events (default: 5000)");
    eprintln!("  M           Max ensemble realisations (default: 50, or use --max-ensemble)");
    eprintln!("  output_dir  Directory for results     (default: current dir)");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --stream             Force streaming mode (low RAM)");
    eprintln!("  --inmemory           Force in-memory mode (fast)");
    eprintln!("  --epsilon <f64>      Topological error tolerance (default: 0.01)");
    eprintln!("  --tmax <usize>       Maximum diffusion time (default: 15)");
    eprintln!("  --seed <u64>         Base seed (default: system clock)");
    eprintln!("  --threads <usize>    Max parallel realisations (default: auto)");
    eprintln!("  --eigen-cutoff <N>   Eigendecomp threshold (default: 3000)");
    eprintln!("  --batch-size <N>     Max concurrent per batch (default: 4)");
    eprintln!("  --max-ensemble <N>   Safety cap on realisations (default: 50)");
    eprintln!("  --min-ensemble <N>   Min before convergence check (default: 8)");
    eprintln!("  --target-error <F>   Relative SE target (default: 0.01)");
    eprintln!("  --resume             Resume from last checkpoint");
    eprintln!("  --force-all          Run all M realisations (disable adaptive early stop)");
    eprintln!("  --export-slice <PATH>  Export topology slice (single realization)");
    eprintln!("  --measure-mass       M1: Traversal mass ratios");
    eprintln!("  --measure-halflife   M2: Half-life census");
    eprintln!("  --measure-modulo     M3: Modulo path integral");
    eprintln!("  --measure-vacuum     M4: Vacuum polarization");
    eprintln!("  --measure-electroweak  M5: Electroweak sector");
    eprintln!("  --measure-decoherence  M6: Quantum decoherence");
    eprintln!("  --measure-neutrino   M7: Neutrino census");
    eprintln!("  --measure-pmns       M8: PMNS mixing matrix");
    eprintln!("  --measure-higgs      M9: Higgs mechanism");
    eprintln!("  --measure-lagrangian M10: Full SM Lagrangian card");
    eprintln!("  --measure-all        Enable all measurements (M1-M9)");
    eprintln!("  --modulo-prime <u64> Prime modulus for M3 (default: 65537)");
    eprintln!("  --modulo-root <u64>  Primitive root for M3 (default: 3)");
    eprintln!("  --modulo-steps <N>   NTT walk steps for M3/M6 (default: 500)");
    eprintln!("  --decoherence-every <N>  Run M6 every N-th realization (default: 1)");
    eprintln!("  --help               Show this help message");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    // ── Provenance ──────────────────────────────────────────────────────
    provenance::print_provenance();

    // ── Parse flags ─────────────────────────────────────────────────────
    let force_stream = args.iter().any(|a| a == "--stream");
    let force_inmemory = args.iter().any(|a| a == "--inmemory");
    let resume = args.iter().any(|a| a == "--resume");
    let force_all = args.iter().any(|a| a == "--force-all");

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
    let mut parsed_modulo_steps: Option<u32> = None;
    let mut parsed_decoherence_every: Option<usize> = None;

    for pair in args.windows(2) {
        match pair[0].as_str() {
            "--epsilon" => { epsilon = pair[1].parse().unwrap_or(epsilon); }
            "--tmax" => { tmax = pair[1].parse().unwrap_or(tmax); }
            "--seed" => { parsed_seed = pair[1].parse().ok(); }
            "--threads" => { parsed_threads = pair[1].parse().ok(); }
            "--eigen-cutoff" => { parsed_eigen_cutoff = pair[1].parse().ok(); }
            "--batch-size" => { parsed_batch_size = pair[1].parse().ok(); }
            "--max-ensemble" => { parsed_max_ensemble = pair[1].parse().ok(); }
            "--min-ensemble" => { parsed_min_ensemble = pair[1].parse().ok(); }
            "--target-error" => { parsed_target_error = pair[1].parse().ok(); }
            "--export-slice" => { export_slice_path = Some(pair[1].clone()); }
            "--modulo-prime" => { parsed_modulo_prime = pair[1].parse().ok(); }
            "--modulo-root" => { parsed_modulo_root = pair[1].parse().ok(); }
            "--modulo-steps" => { parsed_modulo_steps = pair[1].parse().ok(); }
            "--decoherence-every" => { parsed_decoherence_every = pair[1].parse().ok(); }
            _ => {}
        }
    }

    // ── Positional args ─────────────────────────────────────────────────
    let value_flags: std::collections::HashSet<&str> = [
        "--epsilon", "--tmax", "--seed", "--threads", "--eigen-cutoff",
        "--batch-size", "--max-ensemble", "--min-ensemble", "--target-error",
        "--export-slice", "--modulo-prime", "--modulo-root", "--modulo-steps",
        "--decoherence-every",
    ].iter().copied().collect();

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

    let n_points: usize = positional.first().and_then(|s| s.parse().ok()).unwrap_or(5000);
    let positional_m: usize = positional.get(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MAX_ENSEMBLE);
    let max_ensemble: usize = parsed_max_ensemble.unwrap_or(positional_m);
    let min_ensemble: usize = parsed_min_ensemble.unwrap_or(DEFAULT_MIN_ENSEMBLE);
    let target_error: f64 = parsed_target_error.unwrap_or(DEFAULT_TARGET_ERROR);
    let cli_output_dir: Option<String> = positional.get(2).map(|s| s.to_string());

    let seed_base: u64 = parsed_seed.unwrap_or_else(default_seed);
    let eigen_cutoff: usize = parsed_eigen_cutoff.unwrap_or(DEFAULT_EIGEN_CUTOFF);

    // ── Measurement flags ───────────────────────────────────────────────
    let measure_all = args.iter().any(|a| a == "--measure-all");
    let lagrangian_flag = args.iter().any(|a| a == "--measure-lagrangian");
    let all_m1_m9 = measure_all || lagrangian_flag;
    let pmns_flag = all_m1_m9 || args.iter().any(|a| a == "--measure-pmns");

    let measure = MeasureFlags {
        mass: all_m1_m9 || args.iter().any(|a| a == "--measure-mass"),
        halflife: all_m1_m9 || args.iter().any(|a| a == "--measure-halflife"),
        modulo: all_m1_m9 || args.iter().any(|a| a == "--measure-modulo"),
        vacuum: all_m1_m9 || args.iter().any(|a| a == "--measure-vacuum"),
        electroweak: all_m1_m9 || args.iter().any(|a| a == "--measure-electroweak"),
        decoherence: all_m1_m9 || args.iter().any(|a| a == "--measure-decoherence"),
        neutrino: all_m1_m9 || args.iter().any(|a| a == "--measure-neutrino") || pmns_flag,
        pmns: pmns_flag,
        higgs: all_m1_m9 || args.iter().any(|a| a == "--measure-higgs"),
        lagrangian: lagrangian_flag,
        modulo_config: ModuloConfig {
            prime: parsed_modulo_prime.unwrap_or(65537),
            root: parsed_modulo_root.unwrap_or(3),
            steps: parsed_modulo_steps.unwrap_or(500),
        },
        decoherence_every: parsed_decoherence_every.unwrap_or(1).max(1),
    };

    if measure.any_active() {
        let active: Vec<&str> = [
            if measure.mass { Some("M1:mass") } else { None },
            if measure.halflife { Some("M2:halflife") } else { None },
            if measure.modulo { Some("M3:modulo") } else { None },
            if measure.vacuum { Some("M4:vacuum") } else { None },
            if measure.electroweak { Some("M5:electroweak") } else { None },
            if measure.decoherence { Some("M6:decoherence") } else { None },
            if measure.neutrino { Some("M7:neutrino") } else { None },
            if measure.pmns { Some("M8:pmns") } else { None },
            if measure.higgs { Some("M9:higgs") } else { None },
            if measure.lagrangian { Some("M10:lagrangian") } else { None },
        ].iter().filter_map(|&x| x).collect();
        println!("  measurements: {}", active.join(", "));
        if measure.modulo {
            println!("  modulo config: p={}, g={}", measure.modulo_config.prime, measure.modulo_config.root);
        }
        if measure.decoherence && measure.decoherence_every > 1 {
            println!("  decoherence gating: M6 runs every {} realizations", measure.decoherence_every);
        }
    }

    println!("  seed: {seed_base}{}",
        if parsed_seed.is_some() { " (deterministic)" } else { " (system clock)" });

    // ── Config + Mode selection ─────────────────────────────────────────
    config::ensure_config(".");
    let cfg = config::load_config(".");

    let output_dir: String = cli_output_dir.unwrap_or_else(|| {
        if cfg.output_dir.is_empty() { ".".to_string() } else { cfg.output_dir.clone() }
    });
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
        let (recommended, available, estimated) = config::recommend_mode(n_points, &cfg);
        config::prompt_mode(recommended, available, estimated, &cfg)
    };

    let tier = if n_points <= eigen_cutoff { "eigendecomp" } else { "mc-walkers" };
    println!(
        "FEG Prism  (N={n_points}, M\u{2265}{min_ensemble}\u{2264}{max_ensemble} \
         (adaptive, target \u{03b5}={target_error}), tier={tier}, mode={exec_mode:?})\n"
    );

    // ── Export-slice early exit ──────────────────────────────────────────
    if let Some(ref slice_path) = export_slice_path {
        println!("[export-slice] Running single realization (seed={seed_base})...");

        let mut rng = StdRng::seed_from_u64(seed_base);
        let (pts_raw, _big_t) = phase1::sprinkle(n_points, &mut rng);
        let (pts, vacuum_csr, momentum) = if n_points <= eigen_cutoff {
            phase1::build_hasse_sparse(&pts_raw)
        } else {
            phase1::build_hasse_direct(&pts_raw)
        };
        drop(pts_raw);

        let (defect, _topo, causal_prisms) =
            phase2::apply_defect(n_points, vacuum_csr, momentum);

        // Extract directed edges from vacuum CSR
        let vac_csr = &defect.vacuum_csr;
        let mut hasse_edges = Vec::new();
        for u in 0..n_points {
            for &v in vac_csr.neighbors(u) {
                if pts[u][0] < pts[v as usize][0] {
                    hasse_edges.push((u as u32, v));
                }
            }
        }

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

    // ── Walkers from Causal Resolution Theorem ──────────────────────────
    //
    // W = ⌈(t_max / ε)²⌉ derives from the CLT on the Bernoulli return
    // indicator: each walker contributes X_i ∈ {0,1} with E[X_i] = P(t),
    // giving RSE = 1/√(W · P(t)).  Setting RSE < ε and P(t) ~ t^{−2}
    // (d_s = 4) yields W > t²/ε².
    //
    // t_max = D_max = 15 (MAX_HASSE_DEGREE) because that is the lattice
    // mode decay timescale: after ~D_max lazy-walk steps, the discrete
    // graph artifacts in P(t) have damped by ~1/e and the continuum
    // scaling plateau begins.  Calibrating at the plateau onset is
    // optimal because P(D_max) is the largest return probability in the
    // scaling regime, yielding the smallest required W.
    //
    // For t > D_max, RSE degrades as (t/D_max) · ε.  Auto-convergence
    // (convergence.rs, Welford RSE < 5×10⁻⁴, k=3 consecutive batches)
    // adds walker batches until the observable at the midpoint of the
    // step array (t ≈ 36) stabilizes.  The CRT budget is thus a floor,
    // not a ceiling.
    let walkers: usize = ((tmax as f64 / epsilon).powi(2)).ceil() as usize;
    println!(
        "[Phase 3] Topological Error Tolerance \u{03b5} = {epsilon}. At t_max = {tmax},\n\
         Causal Resolution Theorem: W_base = {walkers} (per-category dilution applied at runtime).\n"
    );

    // Dense step sampling: 1..30 (integer), then 32, 34, ..., 100 (even).
    // Covers short-time lattice transient, scaling plateau, and approach
    // to finite-size saturation P(t) → 1/N.
    let steps: Vec<u32> = (1..=30).chain((16..=50).map(|i| i * 2)).collect();
    let t0 = Instant::now();

    // ── Concurrency ─────────────────────────────────────────────────────
    // --batch-size is the user's explicit concurrency request; trust it.
    // Without it, derive a safe default from available RAM.
    let max_concurrent_runs = if let Some(t) = parsed_threads {
        t.min(max_ensemble).max(1)
    } else if let Some(bs) = parsed_batch_size {
        bs.max(1)
    } else if exec_mode == ExecMode::Streaming {
        let cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        cpus.min(max_ensemble).min(8)
    } else {
        config::max_concurrent_runs(n_points, &cfg)
    };
    let batch_size = parsed_batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
    let max_concurrent_runs = max_concurrent_runs.min(max_ensemble);

    // ── Run ensemble ────────────────────────────────────────────────────
    let EnsembleResult {
        spectral, topology, measurements, actual_m, converged,
    } = runner::run_ensemble(
        n_points, &steps, walkers, eigen_cutoff, &measure, exec_mode,
        seed_base, max_ensemble, min_ensemble, target_error,
        batch_size, max_concurrent_runs, &output_dir, resume, force_all,
        epsilon, tmax,
    );

    // ── M10: Assemble SM Lagrangian card (post-hoc) ─────────────────────
    let mut measurements = measurements;
    if measure.lagrangian {
        if let Some(ref mut meas) = measurements {
            println!("  [M10] Assembling SM Lagrangian card...");
            let card = measure::m10_lagrangian::run(meas, &topology);
            meas.lagrangian = Some(card);
        }
    }

    // ── Phase 4: Output ─────────────────────────────────────────────────
    println!("\n[Phase 4] Output \u{2026}");
    let timestamp = provenance::utc_timestamp();
    let meta = Metadata {
        n_points,
        actual_m,
        converged,
        min_ensemble,
        max_ensemble,
        mode: if exec_mode == ExecMode::Streaming { "streaming".into() } else { "in-memory".into() },
        epsilon,
        tmax,
        walkers,
        seed: seed_base,
        timestamp,
        commit: provenance::commit_string(),
    };

    output::write_spectral_csv(&format!("{output_dir}/results.csv"), &steps, &spectral, &meta);
    output::write_topology_csv(&format!("{output_dir}/topology_summary.csv"), &topology, &meta);
    output::write_mass_spectrum_csv(
        &format!("{output_dir}/mass_spectrum.csv"), &topology.prism_histogram, &meta);

    if let Some(ref meas) = measurements {
        measure::write_all_csv(meas, &output_dir, &meta);
    }

    // ── Summary ─────────────────────────────────────────────────────────
    let elapsed = t0.elapsed().as_secs_f64();
    output::summary::print_summary(
        &spectral, &topology, measurements.as_ref(), &steps,
        actual_m, converged, elapsed,
    );

    Ok(())
}
