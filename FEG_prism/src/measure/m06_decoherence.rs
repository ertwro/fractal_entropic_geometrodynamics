// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M6 — Quantum Measurement Theory (Decoherence + Born Rule)
//!
//! Environment = nodes connected to both poles but not part of the prism.
//! Walkers accumulate modular phase g^S mod p on the symmetric vacuum CSR;
//! coherence is measured as correlation between interior and environment
//! intensities.  Directed path counting through prisms yields Born-rule
//! |psi|^2 predictions verified against walker intensities.
//!
//! Calculo de Kuratowski, Vol I, section 8: modular phase decoherence.

use super::context::MeasureContext;
use crate::output::CsvWriter;
use crate::phase2::defect::CausalPrism;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PrismDecoherence {
    pub prism_idx: usize,
    pub generation: u8,
    pub n_intermediates: usize,
    pub environment_size: usize,
    pub phase_coherence: f64,
    pub n_paths_through: u32,
}

#[derive(Clone, Debug)]
pub struct DecoherenceBin {
    pub env_size_min: usize,
    pub env_size_max: usize,
    pub mean_coherence: f64,
    pub n_prisms: usize,
}

#[derive(Clone, Debug)]
pub struct BornBin {
    pub intensity: f64,
    pub observed_freq: f64,
    pub predicted_freq: f64,
}

#[derive(Clone, Debug)]
pub struct DecoherenceResult {
    pub per_prism: Vec<PrismDecoherence>,
    pub decoherence_curve: Vec<DecoherenceBin>,
    pub born_histogram: Vec<BornBin>,
    pub born_chi_sq: f64,
    pub n_detector_nodes: usize,
    pub mean_env_size: f64,
    pub coherence_decay_r: f64,
    pub prime: u64,
    pub root: u64,
}

// ── Utilities ────────────────────────────────────────────────────────────────

fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 { return 0; }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

#[inline]
fn resolve(node: usize, merge: &[usize]) -> usize {
    let mut cur = node;
    while merge[cur] != cur { cur = merge[cur]; }
    cur
}

fn build_gen_lookup(n: usize, ctx: &MeasureContext) -> Vec<u8> {
    let mut lookup = vec![0u8; n];
    for &node in &ctx.defect.generations.gen1 { if node < n { lookup[node] = 1; } }
    for &node in &ctx.defect.generations.gen2 { if node < n { lookup[node] = 2; } }
    for &node in &ctx.defect.generations.gen3 { if node < n { lookup[node] = 3; } }
    for &node in &ctx.defect.generations.anti1 { if node < n { lookup[node] = 4; } }
    lookup
}

fn classify_prism_generation(prism: &CausalPrism, gen_lookup: &[u8]) -> u8 {
    for &node in &prism.intermediates {
        let g = gen_lookup[node];
        if g >= 1 && g <= 3 { return g; }
    }
    let g = gen_lookup[prism.origin];
    if g >= 1 && g <= 3 { return g; }
    let g = gen_lookup[prism.destination];
    if g >= 1 && g <= 3 { return g; }
    0
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> DecoherenceResult {
    let n = ctx.n_points;
    let p = ctx.modulo_config.prime;
    let g = ctx.modulo_config.root;
    let gen_lookup = build_gen_lookup(n, ctx);
    let merge = &ctx.defect.merge_map;
    let (sym_vac_head, sym_vac_data) = ctx.sym_vacuum.raw();
    let (vac_head, vac_data) = ctx.vacuum_csr.raw();
    let n_walkers = ctx.walkers;

    // Step 1: Build prism membership set for fast exclusion
    let mut node_prism_idx: Vec<Option<usize>> = vec![None; n];
    for (pi, prism) in ctx.prisms.iter().enumerate() {
        if prism.origin < n { node_prism_idx[prism.origin] = Some(pi); }
        if prism.destination < n { node_prism_idx[prism.destination] = Some(pi); }
        for &w in &prism.intermediates {
            if w < n { node_prism_idx[w] = Some(pi); }
        }
    }

    // Step 2: Per-prism environment size via sorted symmetric CSR intersection
    struct PrismEnvInfo {
        prism_idx: usize,
        generation: u8,
        n_inter: usize,
        members: Vec<usize>,
        environment: Vec<usize>,
    }

    let prism_envs: Vec<PrismEnvInfo> = ctx.prisms
        .par_iter()
        .enumerate()
        .map(|(pi, prism)| {
            let gen = classify_prism_generation(prism, &gen_lookup);
            let origin = resolve(prism.origin, merge);
            let dest = resolve(prism.destination, merge);

            // Members of this prism
            let mut members: HashSet<usize> = HashSet::new();
            members.insert(origin);
            members.insert(dest);
            for &w in &prism.intermediates {
                members.insert(resolve(w, merge));
            }

            // 2-hop neighborhood of a node on symmetric CSR
            let nbrs_2hop = |root: usize| -> Vec<usize> {
                let mut set = HashSet::new();
                if root >= n { return vec![]; }
                let s1 = sym_vac_head[root] as usize;
                let e1 = sym_vac_head[root + 1] as usize;
                for &v in &sym_vac_data[s1..e1] {
                    let vi = v as usize;
                    set.insert(vi);
                    if vi < n {
                        let s2 = sym_vac_head[vi] as usize;
                        let e2 = sym_vac_head[vi + 1] as usize;
                        for &w in &sym_vac_data[s2..e2] {
                            set.insert(w as usize);
                        }
                    }
                }
                set.remove(&root);
                let mut v: Vec<usize> = set.into_iter().collect();
                v.sort_unstable();
                v
            };

            let origin_nbrs = nbrs_2hop(origin);
            let dest_nbrs = nbrs_2hop(dest);

            // Intersection: nodes within 2 hops of both poles
            let mut both_poles: Vec<usize> = Vec::new();
            let (mut i, mut j) = (0, 0);
            while i < origin_nbrs.len() && j < dest_nbrs.len() {
                if origin_nbrs[i] == dest_nbrs[j] {
                    both_poles.push(origin_nbrs[i]);
                    i += 1;
                    j += 1;
                } else if origin_nbrs[i] < dest_nbrs[j] {
                    i += 1;
                } else {
                    j += 1;
                }
            }

            // Environment = both_poles MINUS prism_members
            let environment: Vec<usize> = both_poles
                .into_iter()
                .filter(|node| !members.contains(node))
                .collect();

            let members_vec: Vec<usize> = members.into_iter().collect();

            PrismEnvInfo {
                prism_idx: pi,
                generation: gen,
                n_inter: prism.intermediates.len(),
                members: members_vec,
                environment,
            }
        })
        .collect();

    // Step 3: Global modular walkers (reuse M3 pattern)
    let n_steps = ctx.modulo_config.steps;
    let arrivals: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let phase_acc: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();

    (0..n_walkers).into_par_iter().for_each(|wi| {
        let mut rng = StdRng::seed_from_u64(ctx.seed.wrapping_add(wi as u64));
        let mut pos = resolve(rng.gen_range(0..n), merge);
        let mut s: u64 = 0;

        for _t in 0..n_steps {
            let start_idx = sym_vac_head[pos] as usize;
            let end_idx = sym_vac_head[pos + 1] as usize;
            let deg = end_idx - start_idx;

            if deg > 0 && rng.gen_bool(0.5) {
                let next = sym_vac_data[start_idx + rng.gen_range(0..deg)] as usize;
                pos = resolve(next, merge);
                s += 1;
            }

            let phase = pow_mod(g, s, p);
            arrivals[pos].fetch_add(1, Ordering::Relaxed);
            phase_acc[pos].fetch_add(phase, Ordering::Relaxed);
        }
    });

    // Compute per-node modular intensity
    let half_p = p / 2;
    let mut node_intensity = vec![0.0f64; n];
    for i in 0..n {
        let arr = arrivals[i].load(Ordering::Relaxed);
        if arr == 0 { continue; }
        let pacc = phase_acc[i].load(Ordering::Relaxed);
        let centered = pacc % p;
        let sym = if centered > half_p {
            centered as i64 - p as i64
        } else {
            centered as i64
        };
        node_intensity[i] = (sym as f64).powi(2) / (half_p as f64).powi(2);
    }

    // Step 5: Per-prism coherence + directed path counting
    let per_prism: Vec<PrismDecoherence> = prism_envs
        .par_iter()
        .map(|info| {
            // Coherence: variance ratio of interior vs environment intensities
            let interior_ints: Vec<f64> = info.members.iter()
                .filter_map(|&nd| if nd < n { Some(node_intensity[nd]) } else { None })
                .collect();
            let env_ints: Vec<f64> = info.environment.iter()
                .filter_map(|&nd| if nd < n { Some(node_intensity[nd]) } else { None })
                .collect();

            let mean_of = |v: &[f64]| -> f64 {
                if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
            };
            let var_of = |v: &[f64], m: f64| -> f64 {
                if v.len() < 2 { 0.0 } else {
                    v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
                }
            };

            let int_mean = mean_of(&interior_ints);
            let env_mean = mean_of(&env_ints);
            let int_var = var_of(&interior_ints, int_mean);
            let env_var = var_of(&env_ints, env_mean);

            // Coherence = 1 - (env_var / int_var) if both > 0, else use correlation
            let coherence = if int_var > 1e-15 && env_var > 1e-15 {
                1.0 - (env_var / (int_var + env_var))
            } else if !interior_ints.is_empty() && !env_ints.is_empty() {
                let denom = int_mean.abs() + env_mean.abs();
                if denom > 1e-15 { 1.0 - (int_mean - env_mean).abs() / denom } else { 0.0 }
            } else {
                0.0
            };

            // Directed path counting: length-2 paths origin->w->v where v is in environment
            let prism = &ctx.prisms[info.prism_idx];
            let env_set: HashSet<usize> = info.environment.iter().copied().collect();
            let mut path_count: u32 = 0;

            for &w in &prism.intermediates {
                if w >= n { continue; }
                let cs = vac_head[w] as usize;
                let ce = vac_head[w + 1] as usize;
                for &v in &vac_data[cs..ce] {
                    if env_set.contains(&(v as usize)) {
                        path_count += 1;
                    }
                }
            }

            PrismDecoherence {
                prism_idx: info.prism_idx,
                generation: info.generation,
                n_intermediates: info.n_inter,
                environment_size: info.environment.len(),
                phase_coherence: coherence,
                n_paths_through: path_count,
            }
        })
        .collect();

    // Step 6: Born rule verification
    let mut detector_path_count: std::collections::HashMap<usize, u64> =
        std::collections::HashMap::new();
    let mut total_paths: u64 = 0;

    for pd in &per_prism {
        let prism = &ctx.prisms[pd.prism_idx];
        let env = &prism_envs[pd.prism_idx].environment;
        let env_set: HashSet<usize> = env.iter().copied().collect();

        for &w in &prism.intermediates {
            if w >= n { continue; }
            let cs = vac_head[w] as usize;
            let ce = vac_head[w + 1] as usize;
            for &v in &vac_data[cs..ce] {
                let vi = v as usize;
                if env_set.contains(&vi) {
                    *detector_path_count.entry(vi).or_insert(0) += 1;
                    total_paths += 1;
                }
            }
        }
    }

    let n_detector_nodes = detector_path_count.len();

    // Build Born histogram
    let n_born_bins = 10usize;
    let mut born_bins: Vec<(f64, f64, f64)> = Vec::new();

    if total_paths > 0 && n_detector_nodes > 0 {
        let mut det_data: Vec<(f64, f64)> = Vec::new();
        let total_walker_arrivals: u64 = (0..n)
            .map(|i| arrivals[i].load(Ordering::Relaxed))
            .sum();

        let sum_paths_sq: f64 = detector_path_count.values()
            .map(|&pc| (pc as f64).powi(2))
            .sum();

        for (&node, &paths) in &detector_path_count {
            let predicted = if sum_paths_sq > 0.0 {
                (paths as f64).powi(2) / sum_paths_sq
            } else {
                0.0
            };
            let observed = if total_walker_arrivals > 0 {
                arrivals[node].load(Ordering::Relaxed) as f64 / total_walker_arrivals as f64
            } else {
                0.0
            };
            det_data.push((predicted, observed));
        }

        det_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let bin_size = (det_data.len() + n_born_bins - 1) / n_born_bins;
        for chunk in det_data.chunks(bin_size.max(1)) {
            let mean_pred = chunk.iter().map(|d| d.0).sum::<f64>() / chunk.len() as f64;
            let mean_obs = chunk.iter().map(|d| d.1).sum::<f64>() / chunk.len() as f64;
            let intensity = (mean_pred + mean_obs) / 2.0;
            born_bins.push((intensity, mean_obs, mean_pred));
        }
    }

    let born_histogram: Vec<BornBin> = born_bins
        .iter()
        .map(|&(intensity, obs, pred)| BornBin {
            intensity,
            observed_freq: obs,
            predicted_freq: pred,
        })
        .collect();

    // Chi-squared goodness-of-fit
    let born_chi_sq = born_histogram
        .iter()
        .map(|b| {
            if b.predicted_freq > 1e-15 {
                (b.observed_freq - b.predicted_freq).powi(2) / b.predicted_freq
            } else {
                0.0
            }
        })
        .sum::<f64>();

    // Step 7: Decoherence curve -- bin by environment size
    let mut env_bins: std::collections::HashMap<usize, (f64, usize)> =
        std::collections::HashMap::new();

    let bin_width = 3usize;
    for pd in &per_prism {
        let bin_key = pd.environment_size / bin_width;
        let entry = env_bins.entry(bin_key).or_insert((0.0, 0));
        entry.0 += pd.phase_coherence;
        entry.1 += 1;
    }

    let mut decoherence_curve: Vec<DecoherenceBin> = env_bins
        .iter()
        .map(|(&bin_key, &(sum_coh, count))| DecoherenceBin {
            env_size_min: bin_key * bin_width,
            env_size_max: (bin_key + 1) * bin_width - 1,
            mean_coherence: sum_coh / count as f64,
            n_prisms: count,
        })
        .collect();
    decoherence_curve.sort_by_key(|b| b.env_size_min);

    // Mean environment size
    let total_env: usize = per_prism.iter().map(|pd| pd.environment_size).sum();
    let mean_env_size = if !per_prism.is_empty() {
        total_env as f64 / per_prism.len() as f64
    } else {
        0.0
    };

    // Coherence decay correlation: coherence vs 1/sqrt(env_size)
    let coherence_decay_r = {
        let pairs: Vec<(f64, f64)> = per_prism
            .iter()
            .filter(|pd| pd.environment_size > 0)
            .map(|pd| (1.0 / (pd.environment_size as f64).sqrt(), pd.phase_coherence))
            .collect();

        if pairs.len() < 3 {
            0.0
        } else {
            let n_f = pairs.len() as f64;
            let sum_x: f64 = pairs.iter().map(|pp| pp.0).sum();
            let sum_y: f64 = pairs.iter().map(|pp| pp.1).sum();
            let sum_xy: f64 = pairs.iter().map(|pp| pp.0 * pp.1).sum();
            let sum_x2: f64 = pairs.iter().map(|pp| pp.0 * pp.0).sum();
            let sum_y2: f64 = pairs.iter().map(|pp| pp.1 * pp.1).sum();

            let num = n_f * sum_xy - sum_x * sum_y;
            let den = ((n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y)).sqrt();
            if den > 1e-15 { num / den } else { 0.0 }
        }
    };

    DecoherenceResult {
        per_prism,
        decoherence_curve,
        born_histogram,
        born_chi_sq,
        n_detector_nodes,
        mean_env_size,
        coherence_decay_r,
        prime: p,
        root: g,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[DecoherenceResult]) -> DecoherenceResult {
    let m = results.len() as f64;
    // Average born histogram bins element-wise across realizations
    let born_histogram = {
        let histograms: Vec<&Vec<BornBin>> = results.iter()
            .filter(|d| !d.born_histogram.is_empty())
            .map(|d| &d.born_histogram)
            .collect();
        if histograms.is_empty() {
            vec![]
        } else {
            let n_bins = histograms[0].len();
            let mh = histograms.len() as f64;
            (0..n_bins).map(|i| {
                let sum_int: f64 = histograms.iter().map(|h| h[i].intensity).sum();
                let sum_obs: f64 = histograms.iter().map(|h| h[i].observed_freq).sum();
                let sum_pred: f64 = histograms.iter().map(|h| h[i].predicted_freq).sum();
                BornBin {
                    intensity: sum_int / mh,
                    observed_freq: sum_obs / mh,
                    predicted_freq: sum_pred / mh,
                }
            }).collect()
        }
    };
    DecoherenceResult {
        per_prism: vec![],
        decoherence_curve: vec![],
        born_histogram,
        born_chi_sq: results.iter().map(|d| d.born_chi_sq).sum::<f64>() / m,
        n_detector_nodes: results.iter().map(|d| d.n_detector_nodes).sum(),
        mean_env_size: results.iter().map(|d| d.mean_env_size).sum::<f64>() / m,
        coherence_decay_r: results.iter().map(|d| d.coherence_decay_r).sum::<f64>() / m,
        prime: results[0].prime,
        root: results[0].root,
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &DecoherenceResult, w: &mut CsvWriter) {
    w.comment("M6 Quantum Decoherence (per-prism coherence)");
    w.header(&[
        "prism_idx", "generation", "n_intermediates",
        "environment_size", "phase_coherence", "n_paths_through",
    ]);
    for pd in &result.per_prism {
        w.row_fmt(format_args!(
            "{},{},{},{},{:.6},{}",
            pd.prism_idx, pd.generation, pd.n_intermediates,
            pd.environment_size, pd.phase_coherence, pd.n_paths_through
        ));
    }
}

/// Write Born rule verification histogram to a separate CSV.
pub fn write_born_rule_csv(result: &DecoherenceResult, w: &mut CsvWriter) {
    w.comment("M6 Born Rule Verification (binned |psi|^2 vs observed)");
    w.header(&["intensity", "observed_freq", "predicted_freq"]);
    for b in &result.born_histogram {
        w.row_fmt(format_args!("{:.15e},{:.15e},{:.15e}", b.intensity, b.observed_freq, b.predicted_freq));
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &DecoherenceResult) {
    println!("  [M6] Quantum Decoherence:");
    println!("    Detector nodes:  {}", result.n_detector_nodes);
    println!("    Mean env size:   {:.2}", result.mean_env_size);
    println!("    Born chi^2:      {:.6}", result.born_chi_sq);
    println!("    Coherence r:     {:.4}", result.coherence_decay_r);
    println!("    NTT config:      p={}, g={}", result.prime, result.root);
}
