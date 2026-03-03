// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M7 — Neutrino Census (Open Prism Detection — Frustrated K_{2,n})
//!
//! Detect open prisms (neutrinos) -- high-degree core nodes whose forward belly
//! fails to converge onto a single destination pole.
//!
//! Open prisms are the causal-set analogue of neutrinos:
//! - No destination pole -> no cover-time trapping -> near-zero mass
//! - No backflow from destination bounce -> pure left-handed chirality
//! - Unanchored belly size -> generation fluctuation (flavor oscillation)
//!
//! Calculo de Kuratowski, Vol II, section 8: open prisms as massless chiral leptons.

use super::context::MeasureContext;
use crate::convergence::{AutoConverge, ConvergeState};
use crate::output::CsvWriter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeutrinoCandidate {
    pub origin: usize,
    pub belly_size: usize,
    pub max_convergence: usize,
    pub mean_chirality: f64,
    pub escape_time: f64,
    pub confined_cover_time: f64,
    pub confined_cover_std: f64,
    pub effective_gen: u8,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeutrinoResult {
    pub candidates: Vec<NeutrinoCandidate>,
    pub total_count: usize,
    pub mean_belly_size: f64,
    pub mean_chirality: f64,
    pub mean_escape_time: f64,
    pub mean_confined_cover: f64,
    pub mean_confined_std: f64,
    pub gen_counts: [usize; 4],
    pub ratio_to_closed: f64,
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> NeutrinoResult {
    const MIN_PRISM_SHARED: usize = 3;

    let n = ctx.n_points;
    let (vac_head, vac_data) = ctx.vacuum_csr.raw();
    let (sym_vac_head, sym_vac_data) = ctx.sym_vacuum.raw();

    // Step 1: Build is_placed from committed closed prisms
    let mut is_placed = vec![false; n];
    for p in ctx.prisms {
        if p.origin < n { is_placed[p.origin] = true; }
        if p.destination < n { is_placed[p.destination] = true; }
        for &w in &p.intermediates {
            if w < n { is_placed[w] = true; }
        }
    }

    // Step 2: Build reverse CSR from directed vacuum
    let mut rev_deg = vec![0u32; n];
    for u in 0..n {
        let s = vac_head[u] as usize;
        let e = vac_head[u + 1] as usize;
        for &v in &vac_data[s..e] {
            let vi = v as usize;
            if vi < n { rev_deg[vi] += 1; }
        }
    }
    let mut rev_head = vec![0u32; n + 1];
    for i in 0..n {
        rev_head[i + 1] = rev_head[i] + rev_deg[i];
    }
    let total_rev = rev_head[n] as usize;
    let mut rev_data = vec![0u32; total_rev];
    let mut rev_pos = rev_head[..n].to_vec();
    for u in 0..n {
        let s = vac_head[u] as usize;
        let e = vac_head[u + 1] as usize;
        for &v in &vac_data[s..e] {
            let vi = v as usize;
            if vi < n {
                rev_data[rev_pos[vi] as usize] = u as u32;
                rev_pos[vi] += 1;
            }
        }
    }
    for u in 0..n {
        let s = rev_head[u] as usize;
        let e = rev_head[u + 1] as usize;
        rev_data[s..e].sort_unstable();
    }

    // Step 3: Compute undirected degree and core set (top 10%)
    let mut undirected_deg = vec![0u32; n];
    for u in 0..n {
        let out_d = vac_head[u + 1] - vac_head[u];
        let in_d = rev_head[u + 1] - rev_head[u];
        undirected_deg[u] = out_d + in_d;
    }
    let mut sorted_degs: Vec<u32> = undirected_deg.clone();
    sorted_degs.sort_unstable();
    let cutoff_idx = (sorted_degs.len() as f64 * 0.90) as usize;
    let degree_cutoff = sorted_degs.get(cutoff_idx).copied().unwrap_or(0);

    // Step 4: Find neutrino candidates
    let candidates_raw: Vec<(usize, Vec<usize>, usize)> = (0..n)
        .filter(|&u| {
            !is_placed[u]
                && undirected_deg[u] >= degree_cutoff
                && (vac_head[u + 1] - vac_head[u]) as usize >= MIN_PRISM_SHARED
        })
        .filter_map(|u| {
            // Collect unplaced children (belly)
            let s = vac_head[u] as usize;
            let e = vac_head[u + 1] as usize;
            let belly: Vec<usize> = vac_data[s..e]
                .iter()
                .map(|&v| v as usize)
                .filter(|&v| v < n && !is_placed[v])
                .collect();
            if belly.len() < MIN_PRISM_SHARED {
                return None;
            }

            // 2-hop convergence scan: max |belly intersect parents(v)| over 2-hop nodes
            let belly_set: HashSet<usize> = belly.iter().copied().collect();
            let mut max_conv = 0usize;
            for &w in &belly {
                let ws = vac_head[w] as usize;
                let we = vac_head[w + 1] as usize;
                for &v in &vac_data[ws..we] {
                    let vi = v as usize;
                    if vi >= n { continue; }
                    let ps = rev_head[vi] as usize;
                    let pe = rev_head[vi + 1] as usize;
                    let mut shared = 0usize;
                    for &pp in &rev_data[ps..pe] {
                        if belly_set.contains(&(pp as usize)) {
                            shared += 1;
                        }
                    }
                    max_conv = max_conv.max(shared);
                    if max_conv >= MIN_PRISM_SHARED {
                        break;
                    }
                }
                if max_conv >= MIN_PRISM_SHARED {
                    break;
                }
            }

            if max_conv < MIN_PRISM_SHARED {
                Some((u, belly, max_conv))
            } else {
                None
            }
        })
        .collect();

    if candidates_raw.is_empty() {
        return NeutrinoResult {
            candidates: vec![],
            total_count: 0,
            mean_belly_size: 0.0,
            mean_chirality: 0.0,
            mean_escape_time: 0.0,
            mean_confined_cover: 0.0,
            mean_confined_std: 0.0,
            gen_counts: [0; 4],
            ratio_to_closed: 0.0,
        };
    }

    let total_neutrinos = candidates_raw.len();

    // Step 5a: Per-candidate chirality + precomputed belly structures (walker-free)
    struct CandidatePrep {
        origin: usize,
        belly: Vec<usize>,
        belly_set: HashSet<usize>,
        belly_adj: Vec<Vec<usize>>,
        max_conv: usize,
        mean_chirality: f64,
        effective_gen: u8,
    }

    let preps: Vec<CandidatePrep> = candidates_raw
        .par_iter()
        .map(|(origin, belly, max_conv)| {
            let belly_size = belly.len();

            // Chirality: forward-backward asymmetry per belly node
            let mut chi_sum = 0.0f64;
            let mut chi_count = 0usize;
            for &w in belly {
                let out_w = (vac_head[w + 1] - vac_head[w]) as f64;
                let in_w = (rev_head[w + 1] - rev_head[w]) as f64;
                let total = out_w + in_w;
                if total > 0.0 {
                    chi_sum += (out_w - in_w) / total;
                    chi_count += 1;
                }
            }
            let mean_chirality = if chi_count > 0 {
                chi_sum / chi_count as f64
            } else {
                0.0
            };

            let belly_set: HashSet<usize> = belly.iter().copied().collect();
            let belly_idx: std::collections::HashMap<usize, usize> = belly
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i))
                .collect();
            let belly_adj: Vec<Vec<usize>> = belly
                .iter()
                .map(|&w| {
                    let s = sym_vac_head[w] as usize;
                    let e = sym_vac_head[w + 1] as usize;
                    sym_vac_data[s..e]
                        .iter()
                        .filter_map(|&v| belly_idx.get(&(v as usize)).copied())
                        .collect()
                })
                .collect();

            let effective_gen = if belly_size < 3 {
                0
            } else if belly_size <= 4 {
                1
            } else if belly_size <= 6 {
                2
            } else {
                3
            };

            CandidatePrep {
                origin: *origin,
                belly: belly.clone(),
                belly_set,
                belly_adj,
                max_conv: *max_conv,
                mean_chirality,
                effective_gen,
            }
        })
        .collect();

    // Step 5b: Auto-converged escape + cover time measurements
    let ac = AutoConverge::new(ctx.walkers, 2048, ctx.epsilon);
    let mut conv_state = ConvergeState::new();
    let batch_per = (ac.batch_size / total_neutrinos).max(1);

    // Per-candidate accumulators persist across batches
    let mut escape_sums: Vec<f64> = vec![0.0; total_neutrinos];
    let mut escape_counts: Vec<usize> = vec![0; total_neutrinos];
    let mut cover_sums: Vec<f64> = vec![0.0; total_neutrinos];
    // Welford online variance: track mean + M2 (sum of squared deviations
    // from running mean).  Replaces naive E[X^2]-E[X]^2 which suffers
    // catastrophic cancellation for large cover times (~500).
    let mut cover_welford_mean: Vec<f64> = vec![0.0; total_neutrinos];
    let mut cover_welford_m2: Vec<f64> = vec![0.0; total_neutrinos];
    let mut cover_counts: Vec<usize> = vec![0; total_neutrinos];

    loop {
        let wi_offset = conv_state.total_walkers / total_neutrinos;

        // Run batch_per more walkers per candidate (parallel across candidates)
        let batch_results: Vec<(f64, usize, Vec<f64>, usize)> = preps
            .par_iter()
            .map(|prep| {
                let belly_size = prep.belly.len();
                let mut esc_sum = 0.0f64;
                let mut esc_cnt = 0usize;
                let mut cov_vals: Vec<f64> = Vec::with_capacity(batch_per);
                let mut cov_cnt = 0usize;
                let max_cover_steps = (belly_size as u32) * 50 + 100;

                for wi in 0..batch_per {
                    let global_wi = wi_offset + wi;

                    // Escape walk
                    let mut rng = StdRng::seed_from_u64(
                        ctx.seed
                            .wrapping_add(prep.origin as u64)
                            .wrapping_add(global_wi as u64),
                    );
                    let start = prep.belly[rng.gen_range(0..belly_size)];
                    let mut pos = start;
                    let mut escaped = false;
                    for step in 1..=200u32 {
                        let s = sym_vac_head[pos] as usize;
                        let e = sym_vac_head[pos + 1] as usize;
                        let deg = e - s;
                        if deg == 0 { break; }
                        let next = sym_vac_data[s + rng.gen_range(0..deg)] as usize;
                        pos = next;
                        if !prep.belly_set.contains(&pos) {
                            esc_sum += step as f64;
                            esc_cnt += 1;
                            escaped = true;
                            break;
                        }
                    }
                    if !escaped {
                        esc_sum += 200.0;
                        esc_cnt += 1;
                    }

                    // Confined cover walk
                    let mut rng2 = StdRng::seed_from_u64(
                        ctx.seed
                            .wrapping_add(prep.origin as u64)
                            .wrapping_add(global_wi as u64)
                            .wrapping_add(10000),
                    );
                    let start_bi = rng2.gen_range(0..belly_size);
                    let mut pos_bi = start_bi;
                    let mut visited = vec![false; belly_size];
                    visited[pos_bi] = true;
                    let mut n_visited = 1usize;
                    let mut ct = max_cover_steps as f64;

                    for step in 1..=max_cover_steps {
                        let nbrs = &prep.belly_adj[pos_bi];
                        if nbrs.is_empty() { break; }
                        let next_bi = nbrs[rng2.gen_range(0..nbrs.len())];
                        pos_bi = next_bi;
                        if !visited[pos_bi] {
                            visited[pos_bi] = true;
                            n_visited += 1;
                        }
                        if n_visited >= belly_size {
                            ct = step as f64;
                            break;
                        }
                    }
                    cov_vals.push(ct);
                    cov_cnt += 1;
                }

                (esc_sum, esc_cnt, cov_vals, cov_cnt)
            })
            .collect();

        // Batch-only observable for Welford (independent sample)
        let (batch_esc, batch_cnt): (f64, usize) = batch_results.iter()
            .fold((0.0, 0), |(s, c), (es, ec, _, _)| (s + es, c + ec));
        let batch_obs = if batch_cnt > 0 { batch_esc / batch_cnt as f64 } else { 0.0 };

        // Accumulate into per-candidate totals for final output.
        // Cover time variance uses Welford's online algorithm (numerically stable).
        for (ci, (es, ec, cov_vals, _cc)) in batch_results.into_iter().enumerate() {
            escape_sums[ci] += es;
            escape_counts[ci] += ec;
            for ct in cov_vals {
                cover_counts[ci] += 1;
                cover_sums[ci] += ct;
                let delta = ct - cover_welford_mean[ci];
                cover_welford_mean[ci] += delta / cover_counts[ci] as f64;
                let delta2 = ct - cover_welford_mean[ci];
                cover_welford_m2[ci] += delta * delta2;
            }
        }

        // Correct walker accounting: update() adds ac.batch_size, but we
        // dispatched batch_per * total_neutrinos walkers (may differ due to
        // integer division truncation).
        let actual_dispatched = batch_per * total_neutrinos;
        let converged = conv_state.update(batch_obs, &ac);
        conv_state.total_walkers = conv_state.total_walkers - ac.batch_size + actual_dispatched;
        if converged { break; }
        if conv_state.at_limit(&ac) {
            eprintln!("[M7] WARNING: {} walkers without convergence", conv_state.total_walkers);
            break;
        }
    }
    println!("  [M7] converged at {} walkers", conv_state.total_walkers);

    // Assemble final candidates
    let candidates: Vec<NeutrinoCandidate> = preps
        .iter()
        .enumerate()
        .map(|(ci, prep)| {
            let escape_time = if escape_counts[ci] > 0 {
                escape_sums[ci] / escape_counts[ci] as f64
            } else {
                0.0
            };
            let confined_cover_time = if cover_counts[ci] > 0 {
                cover_sums[ci] / cover_counts[ci] as f64
            } else {
                0.0
            };
            let confined_cover_std = if cover_counts[ci] > 1 {
                let var = cover_welford_m2[ci] / (cover_counts[ci] - 1) as f64;
                var.max(0.0).sqrt()
            } else {
                0.0
            };

            NeutrinoCandidate {
                origin: prep.origin,
                belly_size: prep.belly.len(),
                max_convergence: prep.max_conv,
                mean_chirality: prep.mean_chirality,
                escape_time,
                confined_cover_time,
                confined_cover_std,
                effective_gen: prep.effective_gen,
            }
        })
        .collect();

    // Step 6: Aggregate
    let total_count = candidates.len();
    let mean_belly_size =
        candidates.iter().map(|c| c.belly_size as f64).sum::<f64>() / total_count as f64;
    let mean_chirality =
        candidates.iter().map(|c| c.mean_chirality).sum::<f64>() / total_count as f64;
    let mean_escape_time =
        candidates.iter().map(|c| c.escape_time).sum::<f64>() / total_count as f64;
    let mean_confined_cover =
        candidates.iter().map(|c| c.confined_cover_time).sum::<f64>() / total_count as f64;
    let mean_confined_std =
        candidates.iter().map(|c| c.confined_cover_std).sum::<f64>() / total_count as f64;

    let mut gen_counts = [0usize; 4];
    for c in &candidates {
        let gi = c.effective_gen as usize;
        if gi < 4 {
            gen_counts[gi] += 1;
        }
    }

    let ratio_to_closed = if ctx.prisms.is_empty() {
        0.0
    } else {
        total_count as f64 / ctx.prisms.len() as f64
    };

    NeutrinoResult {
        candidates,
        total_count,
        mean_belly_size,
        mean_chirality,
        mean_escape_time,
        mean_confined_cover,
        mean_confined_std,
        gen_counts,
        ratio_to_closed,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[NeutrinoResult]) -> NeutrinoResult {
    let m = results.len() as f64;
    NeutrinoResult {
        candidates: vec![],
        total_count: (results.iter().map(|nr| nr.total_count).sum::<usize>() as f64 / m)
            as usize,
        mean_belly_size: results.iter().map(|nr| nr.mean_belly_size).sum::<f64>() / m,
        mean_chirality: results.iter().map(|nr| nr.mean_chirality).sum::<f64>() / m,
        mean_escape_time: results.iter().map(|nr| nr.mean_escape_time).sum::<f64>() / m,
        mean_confined_cover: results.iter().map(|nr| nr.mean_confined_cover).sum::<f64>() / m,
        mean_confined_std: results.iter().map(|nr| nr.mean_confined_std).sum::<f64>() / m,
        gen_counts: {
            let mut gc = [0usize; 4];
            for nr in results {
                for i in 0..4 {
                    gc[i] += nr.gen_counts[i];
                }
            }
            for i in 0..4 {
                gc[i] = (gc[i] as f64 / m) as usize;
            }
            gc
        },
        ratio_to_closed: results.iter().map(|nr| nr.ratio_to_closed).sum::<f64>() / m,
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &NeutrinoResult, w: &mut CsvWriter) {
    w.comment("M7 Neutrino Census (open prism detection)");
    w.header(&[
        "origin", "belly_size", "max_convergence", "mean_chirality",
        "escape_time", "confined_cover_time", "confined_cover_std", "effective_gen",
    ]);
    for c in &result.candidates {
        w.row_fmt(format_args!(
            "{},{},{},{:.6},{:.4},{:.4},{:.4},{}",
            c.origin, c.belly_size, c.max_convergence, c.mean_chirality,
            c.escape_time, c.confined_cover_time, c.confined_cover_std, c.effective_gen
        ));
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &NeutrinoResult) {
    println!("  [M7] Neutrino Census:");
    println!("    Total neutrinos: {}", result.total_count);
    println!("    Mean belly size: {:.2}", result.mean_belly_size);
    println!("    Mean chirality:  {:.4}", result.mean_chirality);
    println!("    Mean escape:     {:.2}", result.mean_escape_time);
    println!("    Confined cover:  {:.2} +/- {:.2}", result.mean_confined_cover, result.mean_confined_std);
    println!("    Gen counts:      g0={} g1={} g2={} g3={}",
        result.gen_counts[0], result.gen_counts[1], result.gen_counts[2], result.gen_counts[3]);
    println!("    Ratio to closed: {:.4}", result.ratio_to_closed);
}
