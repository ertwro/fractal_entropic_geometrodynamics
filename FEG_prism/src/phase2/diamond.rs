// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Per-prism causal diamond embedding diagnostic.
//!
//! Each `CausalPrism` has pole coordinates `sorted_coords[origin]` and
//! `sorted_coords[destination]`.  The causal diamond \[origin, destination\]
//! contains all elements causally between the two poles.  Its volume,
//! proper-time depth, and chain structure are diffeomorphism-invariant
//! quantities computable with zero free parameters.

use std::collections::VecDeque;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::csr::{CsrGraph, Directed};
use crate::phase2::defect::{CausalPrism, GenerationSets};

/// Per-prism causal diamond embedding statistics (zero free parameters).
pub struct DiamondStats {
    /// Index into the prisms vector.
    pub prism_idx: usize,
    /// Generation label (1, 2, 3; 0 = unclassified).
    pub generation: u8,
    /// Number of intermediate nodes in the prism.
    pub belly_size: usize,
    /// Proper time between poles: √((Δt)² − |Δx⃗|²).
    pub tau: f64,
    /// Number of elements in the causal diamond [origin, destination].
    pub diamond_vol: usize,
    /// Longest directed path from origin to destination.
    pub longest_chain: usize,
    /// Total directed paths origin → destination (saturates at u64::MAX).
    pub chain_count: u64,
    /// diamond_vol / (ρ π² τ⁴ / 24) — deviation from flat-space expectation.
    pub density_ratio: f64,
}

/// Maximum backward BFS expansion to prevent blowup on large graphs.
const MAX_PAST_SET: usize = 50_000;

/// Compute causal diamond stats for a single prism.
///
/// Algorithm:
/// 1. BFS backward from `destination` on `rev_csr` → `past_set` (capped)
/// 2. BFS forward from `origin` on `vacuum_csr`, restricted to `past_set` → `diamond`
/// 3. Sort diamond by time coordinate (topological order)
/// 4. DP forward: longest path + path count from origin to destination
/// 5. τ from pole coordinates, density_ratio from τ and sprinkling density
pub fn compute_diamond(
    prism: &CausalPrism,
    vacuum_csr: &CsrGraph<Directed>,
    rev_csr: &CsrGraph<Directed>,
    sorted_coords: &[[f64; 4]],
    density: f64,
) -> DiamondStats {
    let origin = prism.origin;
    let dest = prism.destination;

    // 1. BFS backward from destination on rev_csr → past_set
    let mut past_set = FxHashSet::default();
    let mut queue = VecDeque::new();
    past_set.insert(dest);
    queue.push_back(dest);
    while let Some(u) = queue.pop_front() {
        if past_set.len() >= MAX_PAST_SET {
            break;
        }
        for &v in rev_csr.neighbors(u) {
            let v = v as usize;
            if past_set.insert(v) {
                queue.push_back(v);
            }
        }
    }

    // 2. BFS forward from origin on vacuum_csr, only visiting past_set → diamond
    let mut diamond = FxHashSet::default();
    let mut queue = VecDeque::new();
    if past_set.contains(&origin) {
        diamond.insert(origin);
        queue.push_back(origin);
        while let Some(u) = queue.pop_front() {
            for &v in vacuum_csr.neighbors(u) {
                let v = v as usize;
                if past_set.contains(&v) && diamond.insert(v) {
                    queue.push_back(v);
                }
            }
        }
    }

    let diamond_vol = diamond.len();

    // 3. Sort diamond nodes by time coordinate (topological order in the DAG)
    let mut diamond_nodes: Vec<usize> = diamond.into_iter().collect();
    diamond_nodes.sort_by(|&a, &b| {
        sorted_coords[a][0]
            .partial_cmp(&sorted_coords[b][0])
            .unwrap()
    });

    // 4. DP forward: longest path and path count from origin to destination
    let mut node_pos = FxHashMap::default();
    for (i, &node) in diamond_nodes.iter().enumerate() {
        node_pos.insert(node, i);
    }

    let k = diamond_nodes.len();
    let mut longest = vec![0usize; k];
    let mut chains = vec![0u64; k];

    if let Some(&origin_pos) = node_pos.get(&origin) {
        chains[origin_pos] = 1;
    }

    for i in 0..k {
        if chains[i] == 0 {
            continue;
        }
        let u = diamond_nodes[i];
        for &v in vacuum_csr.neighbors(u) {
            let v = v as usize;
            if let Some(&j) = node_pos.get(&v) {
                longest[j] = longest[j].max(longest[i] + 1);
                chains[j] = chains[j].saturating_add(chains[i]);
            }
        }
    }

    let (longest_chain, chain_count) = node_pos
        .get(&dest)
        .map(|&p| (longest[p], chains[p]))
        .unwrap_or((0, 0));

    // 5. Proper time from pole coordinates
    let o = &sorted_coords[origin];
    let d = &sorted_coords[dest];
    let dt = d[0] - o[0];
    let dx = d[1] - o[1];
    let dy = d[2] - o[2];
    let dz = d[3] - o[3];
    let interval_sq = dt * dt - (dx * dx + dy * dy + dz * dz);
    let tau = if interval_sq > 0.0 {
        interval_sq.sqrt()
    } else {
        0.0
    };

    // 6. Density ratio: measured / flat-space Alexandrov volume (ρ π² τ⁴ / 24)
    let expected = density * std::f64::consts::PI.powi(2) * tau.powi(4) / 24.0;
    let density_ratio = if expected > 0.0 {
        diamond_vol as f64 / expected
    } else {
        0.0
    };

    DiamondStats {
        prism_idx: 0,
        generation: 0,
        belly_size: prism.intermediates.len(),
        tau,
        diamond_vol,
        longest_chain,
        chain_count,
        density_ratio,
    }
}

/// Compute diamond stats for all prisms with generation labels.
///
/// Builds the reverse CSR once internally and labels each prism's generation
/// from the `GenerationSets` node membership.
pub fn compute_all_diamonds(
    prisms: &[CausalPrism],
    vacuum_csr: &CsrGraph<Directed>,
    sorted_coords: &[[f64; 4]],
    density: f64,
    generations: &GenerationSets,
) -> Vec<DiamondStats> {
    let rev_csr = vacuum_csr.reverse();

    // Build node → generation map
    let n = vacuum_csr.n_nodes();
    let mut gen_map = vec![0u8; n];
    for &node in &generations.gen1 {
        if node < n {
            gen_map[node] = 1;
        }
    }
    for &node in &generations.anti1 {
        if node < n {
            gen_map[node] = 1;
        }
    }
    for &node in &generations.gen2 {
        if node < n {
            gen_map[node] = 2;
        }
    }
    for &node in &generations.gen3 {
        if node < n {
            gen_map[node] = 3;
        }
    }

    prisms
        .iter()
        .enumerate()
        .map(|(idx, prism)| {
            let mut stats =
                compute_diamond(prism, vacuum_csr, &rev_csr, sorted_coords, density);
            stats.prism_idx = idx;
            stats.generation = gen_map[prism.origin];
            stats
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::csr::build_directed_csr;

    /// Triangle DAG: 0→1, 0→2, 1→2
    /// Prism(0, 2, [1]) should give vol=3, longest=2, chains=2.
    #[test]
    fn triangle_diamond() {
        let fwd = build_directed_csr(3, &[0, 0, 1], &[1, 2, 2]);
        let rev = fwd.reverse();

        // Timelike-separated coordinates (dt >> |dx|)
        let coords = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [2.0, 0.1, 0.0, 0.0],
        ];

        let prism = CausalPrism {
            origin: 0,
            destination: 2,
            intermediates: vec![1],
        };

        let stats = compute_diamond(&prism, &fwd, &rev, &coords, 1.0);
        assert_eq!(stats.diamond_vol, 3);
        assert_eq!(stats.longest_chain, 2);
        assert_eq!(stats.chain_count, 2);
    }

    /// All intermediates must lie inside the diamond.
    /// diamond_vol >= belly_size + 2 (origin + destination).
    #[test]
    fn diamond_contains_intermediates() {
        let fwd = build_directed_csr(3, &[0, 0, 1], &[1, 2, 2]);
        let rev = fwd.reverse();

        let coords = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [2.0, 0.1, 0.0, 0.0],
        ];

        let prism = CausalPrism {
            origin: 0,
            destination: 2,
            intermediates: vec![1],
        };

        let stats = compute_diamond(&prism, &fwd, &rev, &coords, 1.0);
        assert!(
            stats.diamond_vol >= stats.belly_size + 2,
            "diamond_vol ({}) must be >= belly_size + 2 ({})",
            stats.diamond_vol,
            stats.belly_size + 2
        );
    }

    /// τ must be positive for timelike-separated poles.
    #[test]
    fn tau_positive_for_causal_pair() {
        let fwd = build_directed_csr(3, &[0, 0, 1], &[1, 2, 2]);
        let rev = fwd.reverse();

        let coords = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [3.0, 0.2, 0.0, 0.0],
        ];

        let prism = CausalPrism {
            origin: 0,
            destination: 2,
            intermediates: vec![1],
        };

        let stats = compute_diamond(&prism, &fwd, &rev, &coords, 1.0);
        assert!(stats.tau > 0.0, "tau must be positive for causal pairs");
    }

    /// Wider DAG: 0→1, 0→2, 0→3, 1→4, 2→4, 3→4
    /// Prism(0, 4, [1,2,3]) — diamond should include all 5 nodes.
    #[test]
    fn wider_diamond() {
        let fwd = build_directed_csr(
            5,
            &[0, 0, 0, 1, 2, 3],
            &[1, 2, 3, 4, 4, 4],
        );
        let rev = fwd.reverse();

        let coords = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [1.0, -0.1, 0.0, 0.0],
            [1.0, 0.0, 0.1, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ];

        let prism = CausalPrism {
            origin: 0,
            destination: 4,
            intermediates: vec![1, 2, 3],
        };

        let stats = compute_diamond(&prism, &fwd, &rev, &coords, 1.0);
        assert_eq!(stats.diamond_vol, 5);
        assert_eq!(stats.longest_chain, 2);
        // 3 independent paths: 0→1→4, 0→2→4, 0→3→4
        assert_eq!(stats.chain_count, 3);
        assert!(stats.diamond_vol >= stats.belly_size + 2);
    }
}
