// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18746995

//! Observer modules — physical measurements extracted from the causal set.
//!
//! Four read-only measurement algorithms that extract physics from existing
//! simulation data without modifying the underlying engine.  Each algorithm
//! derives from the Cálculo de Kuratowski (Kuratowski Calculus) of FEG:
//!
//! | Module | Physics | FEG Reference |
//! |--------|---------|---------------|
//! | M1 | Traversal mass ratios (walker traversal time through prisms) | Vol II, Def 3.1 (topological mass = N) |
//! | M2 | Half-life census (cross-ensemble stability statistics) | Vol II, Thm 5.1 (generation persistence) |
//! | M3 | Modulo path integral (NTT-based interference fringes) | Vol I, §6 (modular arithmetic on causal paths) |
//! | M4 | Vacuum polarization (K_{3,3} screening of bare α) | Vol II, Thm 6.3 (Kuratowski K₃,₃ obstruction as charge screening) |
//!
//! Zenodo: <https://doi.org/10.5281/zenodo.18746995>

use crate::skyrmion::{CausalPrism, DefectResult};
use crate::spectral::distribute_walkers;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

// ─────────────────────────────────────────────────────────────────────────────
// Data Structures
// ─────────────────────────────────────────────────────────────────────────────

// ── M1: Traversal Mass Ratios ──

pub struct TraversalRecord {
    pub prism_idx: usize,
    pub generation: u8,
    pub n_belly: usize,
    pub traversal_ticks: u32,
}

pub struct TraversalMassResult {
    pub mean_traversal: [f64; 3],        // per gen 1/2/3
    pub ratio_gen2_gen1: f64,
    pub ratio_gen3_gen1: f64,
    pub n_traversals: [usize; 3],
    pub records: Vec<TraversalRecord>,
}

// ── M2: Half-Life Census ──

pub struct HalfLifeResult {
    pub prism_census: Vec<(usize, u8, usize, i32)>,  // (idx, gen, belly, phase)
    pub gen_counts: [usize; 4],                       // [gen1, gen2, gen3, anti1]
    pub occupancy_by_belly: Vec<(usize, [f64; 3])>,   // (belly_N, [p_gen1, p_gen2, p_gen3])
    pub stability_ratio_gen2: f64,
    pub stability_ratio_gen3: f64,
}

// ── M3: Modulo Path Integral ──
//
// Cálculo de Kuratowski, Vol I §6 (Silva Alvarado): causal path counts are
// reduced modulo a Fermat prime, turning discrete geometry into number-theoretic
// interference.  The NTT root of unity replaces e^{iθ} with a finite-field
// analogue — no floating-point until final normalisation.

pub struct ModuloConfig {
    pub prime: u64,
    pub root: u64,
}

impl Default for ModuloConfig {
    fn default() -> Self {
        // Fermat prime F₄ = 2¹⁶ + 1; primitive root g = 3.
        // Choice dictated by Vol I §6.2: NTT length must divide (p−1) = 2¹⁶,
        // giving power-of-two FFT compatibility on causal path counts.
        Self { prime: 65537, root: 3 }
    }
}

pub struct NodeInterference {
    pub node_id: usize,
    pub n_arrivals: u64,
    pub phase_sum: u64,
    pub intensity: f64,
    pub coords: [f32; 4],
}

pub struct ModuloPathResult {
    pub nodes: Vec<NodeInterference>,
    pub total_walkers: usize,
    pub mean_intensity: f64,
    pub max_intensity: f64,
    pub constructive_count: usize,
    pub destructive_count: usize,
    pub prime: u64,
    pub root: u64,
}

// ── M4: Vacuum Polarization ──

pub struct PrismScreening {
    pub prism_idx: usize,
    pub generation: u8,
    pub n_attempted: usize,
    pub n_rejected_k33: usize,
    pub n_accepted: usize,
    pub local_screening: f64,
}

pub struct VacuumPolResult {
    pub per_prism: Vec<PrismScreening>,
    pub total_attempted: usize,
    pub total_rejected: usize,
    pub total_accepted: usize,
    pub mean_screening: f64,
    pub bare_alpha: f64,
    pub screened_alpha: f64,
}

// ── Aggregate container ──

pub struct MeasurementResult {
    pub traversal: Option<TraversalMassResult>,
    pub half_life: Option<HalfLifeResult>,
    pub modulo: Option<ModuloPathResult>,
    pub vacuum_pol: Option<VacuumPolResult>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Chase the merge-into contraction map until reaching a fixed point.
#[inline]
fn resolve(node: usize, merge: &[usize]) -> usize {
    let mut cur = node;
    while merge[cur] != cur {
        cur = merge[cur];
    }
    cur
}

/// Modular exponentiation via repeated squaring: base^exp mod modulus.
fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
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

/// Build lookup table: node → generation (1/2/3/4=anti1, 0=unclassified).
fn build_gen_lookup(n: usize, defect: &DefectResult) -> Vec<u8> {
    let mut lookup = vec![0u8; n];
    for &node in &defect.gen1_nodes {
        if node < n { lookup[node] = 1; }
    }
    for &node in &defect.gen2_nodes {
        if node < n { lookup[node] = 2; }
    }
    for &node in &defect.gen3_nodes {
        if node < n { lookup[node] = 3; }
    }
    for &node in &defect.anti1_nodes {
        if node < n { lookup[node] = 4; }
    }
    lookup
}

/// Classify a prism's generation by checking which gen list its intermediates belong to.
fn classify_prism_generation(prism: &CausalPrism, gen_lookup: &[u8]) -> u8 {
    for &node in &prism.intermediates {
        let g = gen_lookup[node];
        if g >= 1 && g <= 3 {
            return g;
        }
    }
    let g = gen_lookup[prism.origin];
    if g >= 1 && g <= 3 {
        return g;
    }
    let g = gen_lookup[prism.destination];
    if g >= 1 && g <= 3 {
        return g;
    }
    0
}

/// Check if there is an edge between nodes c and x in either direction
/// using both forward and reverse CSR (sorted, binary-searchable).
fn has_edge_bidirectional(
    c: usize,
    x: usize,
    fwd_head: &[u32],
    fwd_data: &[u32],
    rev_head: &[u32],
    rev_data: &[u32],
) -> bool {
    let x32 = x as u32;
    // Forward: c → x
    let fs = fwd_head[c] as usize;
    let fe = fwd_head[c + 1] as usize;
    if fwd_data[fs..fe].binary_search(&x32).is_ok() {
        return true;
    }
    // Reverse: x → c (c has predecessor x)
    let rs = rev_head[c] as usize;
    let re = rev_head[c + 1] as usize;
    rev_data[rs..re].binary_search(&(x as u32)).is_ok()
}

// ─────────────────────────────────────────────────────────────────────────────
// M1 — Traversal Mass Ratios (Prism-Confined)
// ─────────────────────────────────────────────────────────────────────────────

/// Measure dynamical mass via walker traversal time through prisms.
///
/// Walkers start at prism origins and perform a random walk strictly confined
/// to the prism's internal subgraph (using the defect CSR). Traversal time
/// from the origin pole to the destination pole directly measures the
/// topological mass delay (belly size). Steps into the bulk are rejected
/// (the walker bounces back), forcing it to navigate the K_{2,N} belly.
pub fn measure_traversal_mass(
    n: usize,
    sym_def_head: &[u32],
    sym_def_data: &[u32],
    defect: &DefectResult,
    prisms: &[CausalPrism],
    n_walkers: usize,
    _max_steps: u32, // ignored — cover time sets its own budget
    seed: u64,
) -> TraversalMassResult {
    let merge = &defect.merge_into;
    let gen_lookup = build_gen_lookup(n, defect);

    struct PrismInfo {
        origin: usize,
        destination: usize,
        generation: u8,
        belly: usize,
        intermediates_resolved: Vec<usize>,
    }

    let prism_info: Vec<PrismInfo> = prisms
        .iter()
        .map(|p| {
            let origin = resolve(p.origin, merge);
            let dest = resolve(p.destination, merge);
            let gen = classify_prism_generation(p, &gen_lookup);
            let ints: Vec<usize> = p.intermediates.iter()
                .map(|&i| resolve(i, merge))
                .collect();
            PrismInfo {
                origin,
                destination: dest,
                generation: gen,
                belly: p.intermediates.len(),
                intermediates_resolved: ints,
            }
        })
        .collect();

    // Build node → prism index mapping
    let mut node_to_prism: Vec<Option<usize>> = vec![None; n];
    let mut is_origin: Vec<bool> = vec![false; n];

    for (pi, info) in prism_info.iter().enumerate() {
        node_to_prism[info.origin] = Some(pi);
        is_origin[info.origin] = true;
        node_to_prism[info.destination] = Some(pi);
        for &ri in &info.intermediates_resolved {
            if ri < n {
                node_to_prism[ri] = Some(pi);
            }
        }
    }

    let origins: Vec<usize> = prism_info
        .iter()
        .filter(|p| p.generation >= 1 && p.generation <= 3)
        .map(|p| p.origin)
        .collect();

    if origins.is_empty() {
        return TraversalMassResult {
            mean_traversal: [0.0; 3],
            ratio_gen2_gen1: 0.0,
            ratio_gen3_gen1: 0.0,
            n_traversals: [0; 3],
            records: vec![],
        };
    }

    // Cover time budget: O(N_belly * log(N_belly) * ambient_dilution).
    // With defect degree ~15 and prism degree ~2, dilution factor ~30.
    // For belly=21: 21 * ln(21) * 30 ≈ 1920. Use 5000 for safety.
    let max_cover_steps: u32 = 5000;

    let starts = distribute_walkers(&origins, n_walkers);
    let num_def_nodes = sym_def_head.len() - 1;

    let records: Vec<Vec<TraversalRecord>> = starts
        .par_iter()
        .enumerate()
        .map(|(wi, &start_pos)| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(wi as u64));
            let mut pos = start_pos;
            let mut local_records = Vec::new();

            let mut in_prism = false;
            let mut current_prism_idx = 0usize;
            let mut entry_tick = 0u32;
            let mut visited_belly: HashSet<usize> = HashSet::new();

            if let Some(pi) = node_to_prism[pos] {
                if is_origin[pos] {
                    in_prism = true;
                    current_prism_idx = pi;
                    entry_tick = 0;
                    visited_belly.clear();
                }
            }

            for t in 1..=max_cover_steps {
                let s = if pos < num_def_nodes { sym_def_head[pos] as usize } else { 0 };
                let e = if pos < num_def_nodes { sym_def_head[pos + 1] as usize } else { 0 };
                let deg = e - s;

                // Lazy walk with strict prism confinement
                if deg > 0 && rng.gen_bool(0.5) {
                    let candidate_next = sym_def_data[s + rng.gen_range(0..deg)] as usize;
                    let resolved_next = resolve(candidate_next, merge);

                    if in_prism {
                        // Only step if destination node is in the SAME prism
                        if node_to_prism.get(resolved_next) == Some(&Some(current_prism_idx)) {
                            pos = resolved_next;
                            // Track belly node visits
                            let info = &prism_info[current_prism_idx];
                            if pos != info.origin && pos != info.destination {
                                visited_belly.insert(pos);
                            }
                        }
                        // Otherwise: bounce (stay put)
                    } else {
                        pos = resolved_next;
                    }
                }

                if in_prism {
                    let info = &prism_info[current_prism_idx];
                    let at_dest = pos == info.destination;

                    if at_dest && visited_belly.len() >= info.belly {
                        // Full cover achieved: all belly nodes visited AND at destination
                        local_records.push(TraversalRecord {
                            prism_idx: current_prism_idx,
                            generation: info.generation,
                            n_belly: info.belly,
                            traversal_ticks: t - entry_tick,
                        });
                        in_prism = false;
                        visited_belly.clear();
                    } else if at_dest {
                        // Reached destination before full coverage — reflect back
                        // (causal flux must update entire internal state)
                        pos = info.origin;
                    }
                } else {
                    if let Some(pi) = node_to_prism.get(pos).copied().flatten() {
                        if is_origin[pos] {
                            in_prism = true;
                            current_prism_idx = pi;
                            entry_tick = t;
                            visited_belly.clear();
                        }
                    }
                }
            }

            local_records
        })
        .collect();

    let all_records: Vec<TraversalRecord> = records.into_iter().flatten().collect();

    let mut sum = [0.0f64; 3];
    let mut count = [0usize; 3];

    for r in &all_records {
        let g = r.generation as usize;
        if g >= 1 && g <= 3 {
            sum[g - 1] += r.traversal_ticks as f64;
            count[g - 1] += 1;
        }
    }

    let mean = [
        if count[0] > 0 { sum[0] / count[0] as f64 } else { 0.0 },
        if count[1] > 0 { sum[1] / count[1] as f64 } else { 0.0 },
        if count[2] > 0 { sum[2] / count[2] as f64 } else { 0.0 },
    ];

    TraversalMassResult {
        mean_traversal: mean,
        ratio_gen2_gen1: if mean[0] > 0.0 { mean[1] / mean[0] } else { 0.0 },
        ratio_gen3_gen1: if mean[0] > 0.0 { mean[2] / mean[0] } else { 0.0 },
        n_traversals: count,
        records: all_records,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// M2 — Half-Life Census
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-ensemble stability statistics for Gen2/Gen3 prisms.
///
/// For each prism, records generation, belly size, and net phase.
/// Computes occupancy fractions p(gen_k | N) per belly size
/// and stability ratios τ(gen2)/τ(gen1), τ(gen3)/τ(gen1).
pub fn measure_half_life_census(
    prisms: &[CausalPrism],
    defect: &DefectResult,
    momentum: &[i32],
) -> HalfLifeResult {
    let n = defect.merge_into.len();
    let gen_lookup = build_gen_lookup(n, defect);

    let mut census: Vec<(usize, u8, usize, i32)> = Vec::with_capacity(prisms.len());
    let mut gen_counts = [0usize; 4]; // gen1, gen2, gen3, anti1

    for (pi, prism) in prisms.iter().enumerate() {
        let gen = classify_prism_generation(prism, &gen_lookup);
        let belly = prism.intermediates.len();

        // Net phase: sum of bulk momentum signs for intermediates
        let phase: i32 = prism
            .intermediates
            .iter()
            .filter_map(|&i| momentum.get(i))
            .map(|&m| m.signum())
            .sum();

        census.push((pi, gen, belly, phase));

        match gen {
            1 => gen_counts[0] += 1,
            2 => gen_counts[1] += 1,
            3 => gen_counts[2] += 1,
            4 => gen_counts[3] += 1,
            _ => {}
        }
    }

    // Build occupancy histogram: per belly size, fraction in each generation
    let mut belly_counts: std::collections::HashMap<usize, [usize; 3]> =
        std::collections::HashMap::new();
    for &(_, gen, belly, _) in &census {
        if gen >= 1 && gen <= 3 {
            belly_counts.entry(belly).or_insert([0; 3])[gen as usize - 1] += 1;
        }
    }

    let mut occupancy: Vec<(usize, [f64; 3])> = belly_counts
        .iter()
        .map(|(&belly, counts)| {
            let total = (counts[0] + counts[1] + counts[2]) as f64;
            if total > 0.0 {
                (
                    belly,
                    [
                        counts[0] as f64 / total,
                        counts[1] as f64 / total,
                        counts[2] as f64 / total,
                    ],
                )
            } else {
                (belly, [0.0; 3])
            }
        })
        .collect();
    occupancy.sort_by_key(|&(b, _)| b);

    // Stability ratios: p(gen2)/p(gen1) and p(gen3)/p(gen1) at shared belly sizes
    let (mut p1_sum, mut p2_sum, mut p3_sum) = (0.0f64, 0.0f64, 0.0f64);
    for &(_, probs) in &occupancy {
        if probs[0] > 0.0 {
            p1_sum += probs[0];
            p2_sum += probs[1];
            p3_sum += probs[2];
        }
    }
    let stability_gen2 = if p1_sum > 0.0 { p2_sum / p1_sum } else { 0.0 };
    let stability_gen3 = if p1_sum > 0.0 { p3_sum / p1_sum } else { 0.0 };

    HalfLifeResult {
        prism_census: census,
        gen_counts,
        occupancy_by_belly: occupancy,
        stability_ratio_gen2: stability_gen2,
        stability_ratio_gen3: stability_gen3,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// M3 — Modulo Path Integral
// ─────────────────────────────────────────────────────────────────────────────

/// NTT-based interference fringes from walker phases on the causal set.
///
/// Walkers accumulate a modular phase g^S mod p (where S = cumulative moves)
/// as they traverse the symmetric vacuum graph. Per-node phase coherence
/// reveals constructive/destructive interference patterns.
pub fn measure_modulo_interference(
    n: usize,
    sym_vac_head: &[u32],
    sym_vac_data: &[u32],
    coords: &[[f64; 4]],
    n_walkers: usize,
    n_steps: u32,
    seed: u64,
    merge_into: &[usize],
    config: &ModuloConfig,
) -> ModuloPathResult {
    let p = config.prime;
    let g = config.root;

    let arrivals: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let phase_acc: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();

    (0..n_walkers).into_par_iter().for_each(|wi| {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(wi as u64));
        let mut pos = resolve(rng.gen_range(0..n), merge_into);
        let mut s: u64 = 0;

        for _t in 0..n_steps {
            let start_idx = sym_vac_head[pos] as usize;
            let end_idx = sym_vac_head[pos + 1] as usize;
            let deg = end_idx - start_idx;

            if deg > 0 && rng.gen_bool(0.5) {
                let next = sym_vac_data[start_idx + rng.gen_range(0..deg)] as usize;
                pos = resolve(next, merge_into);
                s += 1;
            }

            let phase = pow_mod(g, s, p);
            arrivals[pos].fetch_add(1, Ordering::Relaxed);
            phase_acc[pos].fetch_add(phase, Ordering::Relaxed);
        }
    });

    // Collect per-node interference results
    let half_p = p / 2;
    let mut nodes = Vec::new();

    for i in 0..n {
        let arr = arrivals[i].load(Ordering::Relaxed);
        if arr == 0 {
            continue;
        }
        let pacc = phase_acc[i].load(Ordering::Relaxed);
        let centered = pacc % p;
        let sym = if centered > half_p {
            centered as i64 - p as i64
        } else {
            centered as i64
        };
        let intensity = (sym as f64).powi(2) / (half_p as f64).powi(2);

        nodes.push(NodeInterference {
            node_id: i,
            n_arrivals: arr,
            phase_sum: pacc,
            intensity,
            coords: [
                coords.get(i).map_or(0.0, |c| c[0] as f32),
                coords.get(i).map_or(0.0, |c| c[1] as f32),
                coords.get(i).map_or(0.0, |c| c[2] as f32),
                coords.get(i).map_or(0.0, |c| c[3] as f32),
            ],
        });
    }

    // Statistics
    let intensities: Vec<f64> = nodes.iter().map(|nd| nd.intensity).collect();
    let mean_int = if !intensities.is_empty() {
        intensities.iter().sum::<f64>() / intensities.len() as f64
    } else {
        0.0
    };
    let max_int = intensities.iter().cloned().fold(0.0f64, f64::max);

    let variance = if intensities.len() > 1 {
        intensities
            .iter()
            .map(|&x| (x - mean_int).powi(2))
            .sum::<f64>()
            / intensities.len() as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    let constructive = intensities
        .iter()
        .filter(|&&x| x > mean_int + 2.0 * std_dev)
        .count();
    let destructive = intensities
        .iter()
        .filter(|&&x| x < (mean_int - 2.0 * std_dev).max(0.0))
        .count();

    ModuloPathResult {
        nodes,
        total_walkers: n_walkers,
        mean_intensity: mean_int,
        max_intensity: max_int,
        constructive_count: constructive,
        destructive_count: destructive,
        prime: p,
        root: g,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// M4 — Vacuum Polarization
// ─────────────────────────────────────────────────────────────────────────────

/// K_{3,3} screening of bare α at Gen1 prism free ports.
///
/// For each Gen1 prism, gathers candidate neighbor nodes and checks whether
/// connecting them would create a K_{3,3} subgraph (forbidden in planar graphs).
/// The screening factor = fraction of candidates NOT blocked by K_{3,3}.
pub fn measure_vacuum_polarization(
    n: usize,
    defect: &DefectResult,
    prisms: &[CausalPrism],
    bare_alpha: f64,
) -> VacuumPolResult {
    let gen_lookup = build_gen_lookup(n, defect);
    let vac_head = &defect.vac_head;
    let vac_data = &defect.vac_data;

    // Build reverse CSR
    let mut rev_deg = vec![0u32; n];
    for u in 0..n {
        let s = vac_head[u] as usize;
        let e = vac_head[u + 1] as usize;
        for &v in &vac_data[s..e] {
            let vi = v as usize;
            if vi < n {
                rev_deg[vi] += 1;
            }
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

    // Sort reverse adjacency lists for binary search
    for u in 0..n {
        let s = rev_head[u] as usize;
        let e = rev_head[u + 1] as usize;
        rev_data[s..e].sort_unstable();
    }

    // Build prism membership set
    let mut is_prism_member = vec![false; n];
    for p in prisms {
        if p.origin < n {
            is_prism_member[p.origin] = true;
        }
        if p.destination < n {
            is_prism_member[p.destination] = true;
        }
        for &i in &p.intermediates {
            if i < n {
                is_prism_member[i] = true;
            }
        }
    }

    // Identify Gen1 prisms with ≥ 3 intermediates
    let gen1_prisms: Vec<(usize, &CausalPrism)> = prisms
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            classify_prism_generation(p, &gen_lookup) == 1 && p.intermediates.len() >= 3
        })
        .collect();

    if gen1_prisms.is_empty() {
        return VacuumPolResult {
            per_prism: vec![],
            total_attempted: 0,
            total_rejected: 0,
            total_accepted: 0,
            mean_screening: 1.0,
            bare_alpha,
            screened_alpha: bare_alpha,
        };
    }

    // Process each Gen1 prism in parallel
    let per_prism: Vec<PrismScreening> = gen1_prisms
        .par_iter()
        .map(|&(pi, prism)| {
            // Gather all prism node indices
            let prism_nodes: Vec<usize> = std::iter::once(prism.origin)
                .chain(std::iter::once(prism.destination))
                .chain(prism.intermediates.iter().copied())
                .collect();

            // Collect candidate neighbors (not prism members)
            let mut candidates: HashSet<usize> = HashSet::new();
            for &pn in &prism_nodes {
                if pn >= n {
                    continue;
                }
                // Forward neighbors
                let fs = vac_head[pn] as usize;
                let fe = vac_head[pn + 1] as usize;
                for &v in &vac_data[fs..fe] {
                    let vi = v as usize;
                    if vi < n && !is_prism_member[vi] {
                        candidates.insert(vi);
                    }
                }
                // Reverse neighbors
                let rs = rev_head[pn] as usize;
                let re = rev_head[pn + 1] as usize;
                for &v in &rev_data[rs..re] {
                    let vi = v as usize;
                    if vi < n && !is_prism_member[vi] {
                        candidates.insert(vi);
                    }
                }
            }

            let n_attempted = candidates.len();
            let mut n_rejected = 0usize;

            let poles = [prism.origin, prism.destination];

            for &c in &candidates {
                // K_{3,3} check: C connects to both poles AND ≥ 3 intermediates
                let connects_pole0 =
                    has_edge_bidirectional(c, poles[0], vac_head, vac_data, &rev_head, &rev_data);
                let connects_pole1 =
                    has_edge_bidirectional(c, poles[1], vac_head, vac_data, &rev_head, &rev_data);

                if connects_pole0 && connects_pole1 {
                    let int_connections = prism
                        .intermediates
                        .iter()
                        .filter(|&&i| {
                            has_edge_bidirectional(
                                c, i, vac_head, vac_data, &rev_head, &rev_data,
                            )
                        })
                        .count();
                    if int_connections >= 3 {
                        n_rejected += 1;
                    }
                }
            }

            let n_accepted = n_attempted - n_rejected;
            PrismScreening {
                prism_idx: pi,
                generation: 1,
                n_attempted,
                n_rejected_k33: n_rejected,
                n_accepted,
                local_screening: if n_attempted > 0 {
                    n_accepted as f64 / n_attempted as f64
                } else {
                    1.0
                },
            }
        })
        .collect();

    let total_attempted: usize = per_prism.iter().map(|p| p.n_attempted).sum();
    let total_rejected: usize = per_prism.iter().map(|p| p.n_rejected_k33).sum();
    let total_accepted: usize = per_prism.iter().map(|p| p.n_accepted).sum();
    let mean_screening = if !per_prism.is_empty() {
        per_prism.iter().map(|p| p.local_screening).sum::<f64>() / per_prism.len() as f64
    } else {
        1.0
    };

    VacuumPolResult {
        per_prism,
        total_attempted,
        total_rejected,
        total_accepted,
        mean_screening,
        bare_alpha,
        screened_alpha: bare_alpha * mean_screening,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ensemble Aggregation
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate measurement results across M realisations.
pub fn aggregate_measurements(results: &[MeasurementResult]) -> MeasurementResult {
    // M1: Weighted average of mean_traversal per gen
    let traversal_agg = {
        let items: Vec<&TraversalMassResult> = results
            .iter()
            .filter_map(|r| r.traversal.as_ref())
            .collect();
        if items.is_empty() {
            None
        } else {
            let mut sum = [0.0f64; 3];
            let mut count = [0usize; 3];
            for t in &items {
                for i in 0..3 {
                    sum[i] += t.mean_traversal[i] * t.n_traversals[i] as f64;
                    count[i] += t.n_traversals[i];
                }
            }
            let mean = [
                if count[0] > 0 { sum[0] / count[0] as f64 } else { 0.0 },
                if count[1] > 0 { sum[1] / count[1] as f64 } else { 0.0 },
                if count[2] > 0 { sum[2] / count[2] as f64 } else { 0.0 },
            ];
            Some(TraversalMassResult {
                mean_traversal: mean,
                ratio_gen2_gen1: if mean[0] > 0.0 { mean[1] / mean[0] } else { 0.0 },
                ratio_gen3_gen1: if mean[0] > 0.0 { mean[2] / mean[0] } else { 0.0 },
                n_traversals: count,
                records: vec![],
            })
        }
    };

    // M2: Merge census, recompute occupancy
    let half_life_agg = {
        let items: Vec<&HalfLifeResult> = results
            .iter()
            .filter_map(|r| r.half_life.as_ref())
            .collect();
        if items.is_empty() {
            None
        } else {
            let mut all_census: Vec<(usize, u8, usize, i32)> = Vec::new();
            let mut gen_counts = [0usize; 4];
            for hl in &items {
                all_census.extend_from_slice(&hl.prism_census);
                for i in 0..4 {
                    gen_counts[i] += hl.gen_counts[i];
                }
            }

            let mut belly_counts: std::collections::HashMap<usize, [usize; 3]> =
                std::collections::HashMap::new();
            for &(_, gen, belly, _) in &all_census {
                if gen >= 1 && gen <= 3 {
                    belly_counts.entry(belly).or_insert([0; 3])[gen as usize - 1] += 1;
                }
            }
            let mut occupancy: Vec<(usize, [f64; 3])> = belly_counts
                .iter()
                .map(|(&belly, counts)| {
                    let total = (counts[0] + counts[1] + counts[2]) as f64;
                    if total > 0.0 {
                        (
                            belly,
                            [
                                counts[0] as f64 / total,
                                counts[1] as f64 / total,
                                counts[2] as f64 / total,
                            ],
                        )
                    } else {
                        (belly, [0.0; 3])
                    }
                })
                .collect();
            occupancy.sort_by_key(|&(b, _)| b);

            let (mut p1_sum, mut p2_sum, mut p3_sum) = (0.0f64, 0.0f64, 0.0f64);
            for &(_, probs) in &occupancy {
                if probs[0] > 0.0 {
                    p1_sum += probs[0];
                    p2_sum += probs[1];
                    p3_sum += probs[2];
                }
            }

            Some(HalfLifeResult {
                prism_census: all_census,
                gen_counts,
                occupancy_by_belly: occupancy,
                stability_ratio_gen2: if p1_sum > 0.0 { p2_sum / p1_sum } else { 0.0 },
                stability_ratio_gen3: if p1_sum > 0.0 { p3_sum / p1_sum } else { 0.0 },
            })
        }
    };

    // M3: Average intensities (per-node data is ensemble-specific, skip)
    let modulo_agg = {
        let items: Vec<&ModuloPathResult> = results
            .iter()
            .filter_map(|r| r.modulo.as_ref())
            .collect();
        if items.is_empty() {
            None
        } else {
            let m = items.len() as f64;
            Some(ModuloPathResult {
                nodes: vec![],
                total_walkers: items.iter().map(|i| i.total_walkers).sum(),
                mean_intensity: items.iter().map(|i| i.mean_intensity).sum::<f64>() / m,
                max_intensity: items
                    .iter()
                    .map(|i| i.max_intensity)
                    .fold(0.0f64, f64::max),
                constructive_count: items.iter().map(|i| i.constructive_count).sum(),
                destructive_count: items.iter().map(|i| i.destructive_count).sum(),
                prime: items[0].prime,
                root: items[0].root,
            })
        }
    };

    // M4: Average screening, sum counts
    let vacuum_agg = {
        let items: Vec<&VacuumPolResult> = results
            .iter()
            .filter_map(|r| r.vacuum_pol.as_ref())
            .collect();
        if items.is_empty() {
            None
        } else {
            let m = items.len() as f64;
            let mean_scr = items.iter().map(|v| v.mean_screening).sum::<f64>() / m;
            let bare_a = items.iter().map(|v| v.bare_alpha).sum::<f64>() / m;
            Some(VacuumPolResult {
                per_prism: vec![],
                total_attempted: items.iter().map(|v| v.total_attempted).sum(),
                total_rejected: items.iter().map(|v| v.total_rejected).sum(),
                total_accepted: items.iter().map(|v| v.total_accepted).sum(),
                mean_screening: mean_scr,
                bare_alpha: bare_a,
                screened_alpha: bare_a * mean_scr,
            })
        }
    };

    MeasurementResult {
        traversal: traversal_agg,
        half_life: half_life_agg,
        modulo: modulo_agg,
        vacuum_pol: vacuum_agg,
    }
}
