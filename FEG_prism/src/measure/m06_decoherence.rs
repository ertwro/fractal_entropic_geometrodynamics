// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M6 — Quantum Measurement Theory (Decoherence + Born Rule)
//!
//! Environment = nodes connected to both poles but not part of the prism.
//!
//! **Propagator**: Causal chain-counting DP on the Hasse DAG with NTT
//! phase rotation.  The retarded wave (fwd) is seeded from prism
//! intermediates and propagated forward; each hop multiplies by
//! W = g^{(p-1)/4} mod p (90° per edge).  The advanced wave (bwd) is
//! seeded from the future boundary (all out-degree-0 nodes) and
//! propagated backward with W⁻¹.  The handshake fwd(v) × bwd(v),
//! centered symmetrically in Z/pZ, gives the Born amplitude at each
//! environment node — Cramer's transactional interpretation.
//!
//! **Observation**: K_{2,2} diamond counts at environment nodes (the
//! geometric measurement events where wavefunction collapse occurs).
//!
//! **Statistic**: Pearson r between the predicted |ψ|² PMF and the
//! observed K_{2,2} PMF — a scale-invariant shape comparison.
//!
//! Calculo de Kuratowski, Vol II, section 9: modular phase decoherence.

use super::context::MeasureContext;
use crate::output::CsvWriter;
use crate::phase2::defect::CausalPrism;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PrismDecoherence {
    pub prism_idx: usize,
    pub generation: u8,
    pub n_intermediates: usize,
    pub environment_size: usize,
    pub phase_coherence: f64,
    pub amplitude_through: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DecoherenceBin {
    pub env_size_min: usize,
    pub env_size_max: usize,
    pub mean_coherence: f64,
    pub n_prisms: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BornBin {
    pub intensity: f64,
    pub observed_freq: f64,
    pub predicted_freq: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DecoherenceResult {
    pub per_prism: Vec<PrismDecoherence>,
    pub decoherence_curve: Vec<DecoherenceBin>,
    pub born_histogram: Vec<BornBin>,
    pub born_r: f64,
    pub born_r_chain: f64,
    pub born_r_null_mean: f64,
    pub born_r_null_std: f64,
    pub born_r_percentile: f64,
    pub born_r_chain_percentile: f64,
    pub n_detector_nodes: usize,
    pub mean_env_size: f64,
    pub coherence_decay_r: f64,
    pub prime: u64,
    pub root: u64,
}

// ── Utilities ────────────────────────────────────────────────────────────────

#[inline]
fn pow_mod(mut base: u64, mut exp: u64, modulo: u64) -> u64 {
    let mut res: u64 = 1;
    base %= modulo;
    while exp > 0 {
        if exp % 2 == 1 { res = ((res as u128 * base as u128) % modulo as u128) as u64; }
        base = ((base as u128 * base as u128) % modulo as u128) as u64;
        exp /= 2;
    }
    res
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

    // ── Phase rotation period ──
    // N = 4: 90° per hop. Matches the K_{2,2} 4-cycle geometry of the
    // observation (diamond defects). High-frequency regime where quantum
    // interference is maximally active.
    let n_period: u64 = 4;
    let w_fwd = pow_mod(g, (p - 1) / n_period, p);
    let w_bwd = pow_mod(w_fwd, p - 2, p);

    let gen_lookup = build_gen_lookup(n, ctx);
    let merge = &ctx.defect.merge_map;
    let sym = match ctx.sym_vacuum {
        Some(s) => s,
        None => return DecoherenceResult {
            per_prism: vec![], decoherence_curve: vec![], born_histogram: vec![],
            born_r: 0.0, born_r_chain: 0.0, born_r_null_mean: 0.0,
            born_r_null_std: 0.0, born_r_percentile: 0.0,
            born_r_chain_percentile: 0.0, n_detector_nodes: 0,
            mean_env_size: 0.0, coherence_decay_r: 0.0, prime: p, root: g,
        },
    };
    let (sym_vac_head, sym_vac_data) = sym.raw();
    let (vac_head, vac_data) = ctx.vacuum_csr.raw();

    // ── Global backward wave (position-independent) ─────────────────────
    // topo_order is scoped so it drops after bwd_global is built (~80 MB freed).
    let bwd_global: Vec<u64> = {
        let mut topo_order: Vec<usize> = (0..n).collect();
        topo_order.sort_by(|&a, &b|
            ctx.sorted_coords[a][0].total_cmp(&ctx.sorted_coords[b][0])
        );

        let mut bwd = vec![0u64; n];
        for v in 0..n {
            let cs = vac_head[v] as usize;
            let ce = vac_head[v + 1] as usize;
            if cs == ce {
                let rv = resolve(v, merge);
                if rv < n { bwd[rv] = 1; }
            }
        }
        for &v in topo_order.iter().rev() {
            let cs = vac_head[v] as usize;
            let ce = vac_head[v + 1] as usize;
            for &c in &vac_data[cs..ce] {
                let rc = resolve(c as usize, merge);
                if rc < n && bwd[rc] != 0 {
                    let flux = (bwd[rc] as u128 * w_bwd as u128 % p as u128) as u64;
                    bwd[v] = ((bwd[v] as u128 + flux as u128) % p as u128) as u64;
                }
            }
        }
        // topo_order dropped here
        bwd
    };

    // ── Fused streaming pass ────────────────────────────────────────────
    //
    // Merges the old Steps 1 (environment), 4 (localized wave + coherence),
    // 5a (K22 diamonds), 5b (handshake amps), 5b' (chain counts) into a
    // single fold/reduce.  Per-prism temporaries (environment vec, BFS
    // reached set, fwd/fwd_count HashMaps) are freed after each prism.
    //
    // Memory: ~100 MB total (down from ~8.5 GB with collected vectors).
    const K_HOPS: usize = 3;

    type FoldState = (
        Vec<PrismDecoherence>,  // per_prism
        HashMap<usize, f64>,    // k22_count
        f64,                    // total_k22
        HashMap<usize, f64>,    // predicted_pairs (handshake amps)
        f64,                    // sum_pairs
        HashMap<usize, f64>,    // chain_pred (simple chain counts)
        f64,                    // sum_chain
    );

    let fold_init = || -> FoldState {
        (Vec::new(), HashMap::new(), 0.0, HashMap::new(), 0.0, HashMap::new(), 0.0)
    };

    let (per_prism, k22_count, total_k22, predicted_pairs, sum_pairs, chain_pred, sum_chain) =
        ctx.prisms.par_iter().enumerate()
            .fold(fold_init, |mut acc, (pi, prism)| {
                let gen = classify_prism_generation(prism, &gen_lookup);
                let origin = resolve(prism.origin, merge);
                let dest = resolve(prism.destination, merge);
                let half_p = p / 2;

                // (a) Compute environment ─────────────────────────────────
                let mut members: HashSet<usize> = HashSet::new();
                members.insert(origin);
                members.insert(dest);
                for &w in &prism.intermediates {
                    members.insert(resolve(w, merge));
                }

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

                let mut both_poles: Vec<usize> = Vec::new();
                {
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
                }

                let environment: Vec<usize> = both_poles
                    .into_iter()
                    .filter(|node| !members.contains(node))
                    .collect();

                // Early exit: no environment → no measurements possible
                if environment.is_empty() {
                    acc.0.push(PrismDecoherence {
                        prism_idx: pi,
                        generation: gen,
                        n_intermediates: prism.intermediates.len(),
                        environment_size: 0,
                        phase_coherence: 0.0,
                        amplitude_through: 0.0,
                    });
                    return acc;
                }

                // (b) BFS forward from intermediates up to K_HOPS ─────────
                // Builds node_to_local mapping for local-indexed arrays
                let mut node_to_local: HashMap<usize, u32> = HashMap::new();
                let mut reached_nodes: Vec<usize> = Vec::new();
                let mut frontier: Vec<usize> = Vec::new();
                for &w in &prism.intermediates {
                    let rw = resolve(w, merge);
                    if rw < n && !node_to_local.contains_key(&rw) {
                        node_to_local.insert(rw, reached_nodes.len() as u32);
                        reached_nodes.push(rw);
                        frontier.push(rw);
                    }
                }
                for _ in 0..K_HOPS {
                    let mut next = Vec::new();
                    for &u in &frontier {
                        let cs = vac_head[u] as usize;
                        let ce = vac_head[u + 1] as usize;
                        for &v in &vac_data[cs..ce] {
                            let rv = resolve(v as usize, merge);
                            if rv < n && !node_to_local.contains_key(&rv) {
                                node_to_local.insert(rv, reached_nodes.len() as u32);
                                reached_nodes.push(rv);
                                next.push(rv);
                            }
                        }
                    }
                    frontier = next;
                }
                let n_reached = reached_nodes.len();

                // Pre-build local adjacency (one-time cost per prism)
                let mut local_adj: Vec<Vec<u32>> = vec![Vec::new(); n_reached];
                for li in 0..n_reached {
                    let u = reached_nodes[li];
                    let cs = vac_head[u] as usize;
                    let ce = vac_head[u + 1] as usize;
                    for &v in &vac_data[cs..ce] {
                        let rv = resolve(v as usize, merge);
                        if let Some(&lv) = node_to_local.get(&rv) {
                            local_adj[li].push(lv);
                        }
                    }
                }

                // (c) Local topo order + localized retarded wave ──────────
                let mut local_order: Vec<u32> = (0..n_reached as u32).collect();
                local_order.sort_unstable_by(|&a, &b|
                    ctx.sorted_coords[reached_nodes[a as usize]][0]
                        .total_cmp(&ctx.sorted_coords[reached_nodes[b as usize]][0])
                );

                let mut fwd = vec![0u64; n_reached];
                for &w in &prism.intermediates {
                    let rw = resolve(w, merge);
                    if let Some(&li) = node_to_local.get(&rw) {
                        fwd[li as usize] = (fwd[li as usize] + 1) % p;
                    }
                }
                for &li in &local_order {
                    let f = fwd[li as usize];
                    if f == 0 { continue; }
                    let flux = (f as u128 * w_fwd as u128 % p as u128) as u64;
                    for &lv in &local_adj[li as usize] {
                        fwd[lv as usize] = ((fwd[lv as usize] as u128 + flux as u128) % p as u128) as u64;
                    }
                }

                // Simple chain counting (no NTT phase) — Born rule test
                let mut fwd_count = vec![0.0f64; n_reached];
                for &w in &prism.intermediates {
                    let rw = resolve(w, merge);
                    if let Some(&li) = node_to_local.get(&rw) {
                        fwd_count[li as usize] += 1.0;
                    }
                }
                for &li in &local_order {
                    let f = fwd_count[li as usize];
                    if f == 0.0 { continue; }
                    for &lv in &local_adj[li as usize] {
                        fwd_count[lv as usize] += f;
                    }
                }

                // (d) Coherence: mean resultant length of fwd phase at env nodes
                let (mut sum_cos, mut sum_sin) = (0.0f64, 0.0f64);
                let mut env_phase_count: usize = 0;
                for &v in &environment {
                    if let Some(&li) = node_to_local.get(&v) {
                        let fv = fwd[li as usize];
                        if fv != 0 {
                            let theta = 2.0 * std::f64::consts::PI * (fv as f64) / (p as f64);
                            sum_cos += theta.cos();
                            sum_sin += theta.sin();
                            env_phase_count += 1;
                        }
                    }
                }
                let coherence = if env_phase_count > 0 {
                    let m = env_phase_count as f64;
                    ((sum_cos / m).powi(2) + (sum_sin / m).powi(2)).sqrt()
                } else {
                    0.0
                };

                // (e) Handshake amplitude + accumulate into predicted_pairs
                let mut amplitude: f64 = 0.0;
                for &v in &environment {
                    if let Some(&li) = node_to_local.get(&v) {
                        let f = fwd[li as usize];
                        let b = bwd_global[v];
                        if f != 0 && b != 0 {
                            let sf = if f > half_p { f as i64 - p as i64 } else { f as i64 };
                            let sb = if b > half_p { b as i64 - p as i64 } else { b as i64 };
                            let handshake = (sf as f64 / half_p as f64) * (sb as f64 / half_p as f64);
                            if handshake != 0.0 {
                                amplitude += handshake.abs();
                                *acc.3.entry(v).or_insert(0.0) += handshake.abs();
                                acc.4 += handshake.abs();
                            }
                        }
                    }
                }

                // Accumulate chain counts into chain_pred
                for &v in &environment {
                    if let Some(&li) = node_to_local.get(&v) {
                        let f = fwd_count[li as usize];
                        if f > 0.0 {
                            *acc.5.entry(v).or_insert(0.0) += f;
                            acc.6 += f;
                        }
                    }
                }

                // (f) K22 diamonds at env nodes ───────────────────────────
                let inter_nbrs: Vec<HashSet<usize>> = prism.intermediates.iter()
                    .map(|&w| {
                        let rw = resolve(w, merge);
                        if rw >= n { return HashSet::new(); }
                        let cs = vac_head[rw] as usize;
                        let ce = vac_head[rw + 1] as usize;
                        vac_data[cs..ce].iter()
                            .map(|&x| resolve(x as usize, merge))
                            .filter(|&x| x < n)
                            .collect::<HashSet<_>>()
                    })
                    .collect();

                for &v in &environment {
                    let pointers: Vec<usize> = (0..inter_nbrs.len())
                        .filter(|&i| inter_nbrs[i].contains(&v))
                        .collect();

                    if pointers.len() < 2 { continue; }

                    let mut diamonds: u64 = 0;
                    for i in 0..pointers.len() {
                        for j in (i+1)..pointers.len() {
                            let common = inter_nbrs[pointers[i]]
                                .intersection(&inter_nbrs[pointers[j]])
                                .filter(|&&w| w != v)
                                .count();
                            diamonds += common as u64;
                        }
                    }

                    if diamonds > 0 {
                        *acc.1.entry(v).or_insert(0.0) += diamonds as f64;
                        acc.2 += diamonds as f64;
                    }
                }

                // Push PrismDecoherence (22 bytes per prism — negligible)
                acc.0.push(PrismDecoherence {
                    prism_idx: pi,
                    generation: gen,
                    n_intermediates: prism.intermediates.len(),
                    environment_size: environment.len(),
                    phase_coherence: coherence,
                    amplitude_through: amplitude,
                });

                // All per-prism temporaries freed here:
                // environment, members, node_to_local, reached_nodes,
                // local_adj, local_order, fwd, fwd_count, inter_nbrs
                acc
            })
            .reduce(fold_init, |mut a, b| {
                a.0.extend(b.0);
                for (k, v) in b.1 { *a.1.entry(k).or_default() += v; }
                a.2 += b.2;
                for (k, v) in b.3 { *a.3.entry(k).or_default() += v; }
                a.4 += b.4;
                for (k, v) in b.5 { *a.5.entry(k).or_default() += v; }
                a.6 += b.6;
                a
            });

    // Step 5: Born rule verification — |ψ|² vs K_{2,2} measurement events

    let n_detector_nodes = predicted_pairs.len();

    // 5c. Born rule: Pearson r between normalized PMFs
    //     Does |ψ|² predict where K_{2,2} defects form?
    //     Both distributions normalized to sum=1 — pure shape comparison.
    let mut born_histogram: Vec<BornBin> = Vec::new();
    let born_r: f64;
    let mut born_r_null_mean: f64 = 0.0;
    let mut born_r_null_std: f64 = 0.0;
    let mut born_r_percentile: f64 = 0.0;

    if !predicted_pairs.is_empty() && sum_pairs > 0.0 && total_k22 > 0.0 {
        let mut det_nodes: Vec<usize> = predicted_pairs.keys()
            .filter(|&&v| k22_count.contains_key(&v))
            .copied()
            .collect();

        if det_nodes.len() >= 3 {
            let nf = det_nodes.len() as f64;

            let mut mean_o = 0.0;
            let mut mean_p = 0.0;
            for &v in &det_nodes {
                mean_o += k22_count[&v] / total_k22;
                mean_p += predicted_pairs[&v] / sum_pairs;
            }
            mean_o /= nf;
            mean_p /= nf;

            let mut num = 0.0;
            let mut den_o = 0.0;
            let mut den_p = 0.0;
            for &v in &det_nodes {
                let d_o = k22_count[&v] / total_k22 - mean_o;
                let d_p = predicted_pairs[&v] / sum_pairs - mean_p;
                num += d_o * d_p;
                den_o += d_o * d_o;
                den_p += d_p * d_p;
            }

            born_r = if den_o > 0.0 && den_p > 0.0 {
                num / (den_o.sqrt() * den_p.sqrt())
            } else {
                0.0
            };

            // Permutation null model: shuffle predicted PMF, recompute Pearson r
            // Invariant: mean-centered arrays have fixed variance under shuffle,
            // so the denominator is precomputed once.  Only the covariance
            // (numerator) changes per permutation.  Parallelised via Rayon.
            if den_o > 0.0 && den_p > 0.0 {
                let obs_c: Vec<f64> = det_nodes.iter()
                    .map(|&v| k22_count[&v] / total_k22 - mean_o)
                    .collect();
                let pred_c: Vec<f64> = det_nodes.iter()
                    .map(|&v| predicted_pairs[&v] / sum_pairs - mean_p)
                    .collect();
                let den = den_o.sqrt() * den_p.sqrt();

                const N_PERM: usize = 200;
                let base_seed = ctx.seed ^ 0xB0_4E;
                let null_rs: Vec<f64> = (0..N_PERM).into_par_iter().map(|k| {
                    let mut rng = StdRng::seed_from_u64(base_seed + k as u64);
                    let mut shuffled = pred_c.clone();
                    for i in (1..shuffled.len()).rev() {
                        let j = rng.gen_range(0..=i);
                        shuffled.swap(i, j);
                    }
                    let cov: f64 = obs_c.iter().zip(shuffled.iter())
                        .map(|(&o, &p)| o * p).sum();
                    cov / den
                }).collect();

                born_r_null_mean = null_rs.iter().sum::<f64>() / N_PERM as f64;
                born_r_null_std = (null_rs.iter()
                    .map(|&r| (r - born_r_null_mean).powi(2))
                    .sum::<f64>() / N_PERM as f64).sqrt();
                born_r_percentile = null_rs.iter()
                    .filter(|&&r| r < born_r).count() as f64 / N_PERM as f64;
            }

            // 5d. Bin detector nodes into ~10 equal-count bins for histogram
            det_nodes.sort_by(|&a, &b| {
                let pa = predicted_pairs[&a] / sum_pairs;
                let pb = predicted_pairs[&b] / sum_pairs;
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let n_target_bins = 10.min(det_nodes.len());
            let bin_size = det_nodes.len() / n_target_bins;
            let remainder = det_nodes.len() % n_target_bins;

            let mut start = 0;
            for i in 0..n_target_bins {
                let this_bin = bin_size + if i < remainder { 1 } else { 0 };
                let end = start + this_bin;
                let slice = &det_nodes[start..end];

                let mut sum_pred = 0.0;
                let mut sum_obs = 0.0;
                let mut sum_int = 0.0;
                for &v in slice {
                    sum_pred += predicted_pairs[&v] / sum_pairs;
                    sum_obs += k22_count[&v] / total_k22;
                    sum_int += predicted_pairs[&v] / sum_pairs;
                }
                let mean_int = sum_int / this_bin as f64;

                born_histogram.push(BornBin {
                    intensity: mean_int,
                    observed_freq: sum_obs,
                    predicted_freq: sum_pred,
                });

                start = end;
            }
        } else {
            born_r = 0.0;
        }
    } else {
        born_r = 0.0;
    }

    // 5c'. Chain-count Born r: does chain_count² predict K₂₂ density?
    let mut born_r_chain_percentile: f64 = 0.0;
    let born_r_chain: f64 = if sum_chain > 0.0 && total_k22 > 0.0 {
        let chain_nodes: Vec<usize> = chain_pred.keys()
            .filter(|&&v| k22_count.contains_key(&v))
            .copied()
            .collect();
        if chain_nodes.len() >= 3 {
            let nf = chain_nodes.len() as f64;
            let sum_pred_sq: f64 = chain_nodes.iter()
                .map(|v| chain_pred[v].powi(2)).sum();
            let (mut mo, mut mp) = (0.0, 0.0);
            for &v in &chain_nodes {
                mo += k22_count[&v] / total_k22;
                mp += chain_pred[&v].powi(2) / sum_pred_sq;
            }
            mo /= nf; mp /= nf;
            let (mut num, mut do2, mut dp2) = (0.0, 0.0, 0.0);
            for &v in &chain_nodes {
                let dv_o = k22_count[&v] / total_k22 - mo;
                let dv_p = chain_pred[&v].powi(2) / sum_pred_sq - mp;
                num += dv_o * dv_p;
                do2 += dv_o * dv_o;
                dp2 += dv_p * dv_p;
            }
            let r = if do2 > 0.0 && dp2 > 0.0 { num / (do2.sqrt() * dp2.sqrt()) } else { 0.0 };

            // Permutation null model for chain prediction (parallelised)
            if do2 > 0.0 && dp2 > 0.0 {
                let obs_c: Vec<f64> = chain_nodes.iter()
                    .map(|&v| k22_count[&v] / total_k22 - mo)
                    .collect();
                let pred_c: Vec<f64> = chain_nodes.iter()
                    .map(|&v| chain_pred[&v].powi(2) / sum_pred_sq - mp)
                    .collect();
                let den = do2.sqrt() * dp2.sqrt();

                const N_PERM: usize = 200;
                let base_seed = ctx.seed ^ 0xC4_41;
                let count_below: usize = (0..N_PERM).into_par_iter().map(|k| {
                    let mut rng = StdRng::seed_from_u64(base_seed + k as u64);
                    let mut shuffled = pred_c.clone();
                    for i in (1..shuffled.len()).rev() {
                        let j = rng.gen_range(0..=i);
                        shuffled.swap(i, j);
                    }
                    let r_null: f64 = obs_c.iter().zip(shuffled.iter())
                        .map(|(&o, &p)| o * p).sum();
                    if r_null / den < r { 1usize } else { 0usize }
                }).sum();
                born_r_chain_percentile = count_below as f64 / N_PERM as f64;
            }

            r
        } else { 0.0 }
    } else { 0.0 };

    // Step 6: Decoherence curve -- bin by environment size
    let mut env_bins: HashMap<usize, (f64, usize)> = HashMap::new();

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
        born_r,
        born_r_chain,
        born_r_null_mean,
        born_r_null_std,
        born_r_percentile,
        born_r_chain_percentile,
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
            let n_bins = histograms.iter().map(|h| h.len()).min().unwrap_or(0);
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
        born_r: results.iter().map(|d| d.born_r).sum::<f64>() / m,
        born_r_chain: results.iter().map(|d| d.born_r_chain).sum::<f64>() / m,
        born_r_null_mean: results.iter().map(|d| d.born_r_null_mean).sum::<f64>() / m,
        born_r_null_std: results.iter().map(|d| d.born_r_null_std).sum::<f64>() / m,
        born_r_percentile: results.iter().map(|d| d.born_r_percentile).sum::<f64>() / m,
        born_r_chain_percentile: results.iter().map(|d| d.born_r_chain_percentile).sum::<f64>() / m,
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
        "environment_size", "phase_coherence", "amplitude_through",
    ]);
    for pd in &result.per_prism {
        w.row_fmt(format_args!(
            "{},{},{},{},{:.6},{:.6}",
            pd.prism_idx, pd.generation, pd.n_intermediates,
            pd.environment_size, pd.phase_coherence, pd.amplitude_through
        ));
    }
}

/// Write Born rule verification histogram to a separate CSV.
pub fn write_born_rule_csv(result: &DecoherenceResult, w: &mut CsvWriter) {
    w.comment("M6 Born Rule Verification (binned |psi|^2 vs observed)");
    w.comment(&format!("born_r={:.6}  n_detector_nodes={}", result.born_r, result.n_detector_nodes));
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
    println!("    Born r(PMF):     {:.6}  (null: {:.3} +/- {:.3}, p={:.4})",
        result.born_r, result.born_r_null_mean, result.born_r_null_std,
        1.0 - result.born_r_percentile);
    println!("    Born r(chain²):  {:.6}  (p={:.4})",
        result.born_r_chain, 1.0 - result.born_r_chain_percentile);
    if !result.per_prism.is_empty() {
        let np = result.per_prism.len() as f64;
        let mean_c: f64 = result.per_prism.iter()
            .map(|pd| pd.phase_coherence).sum::<f64>() / np;
        let var_c: f64 = result.per_prism.iter()
            .map(|pd| (pd.phase_coherence - mean_c).powi(2))
            .sum::<f64>() / np;
        println!("    Coherence var:   {:.6}", var_c);
    }
    println!("    Coherence r:     {:.4}", result.coherence_decay_r);
    println!("    NTT config:      p={}, g={}", result.prime, result.root);
}
