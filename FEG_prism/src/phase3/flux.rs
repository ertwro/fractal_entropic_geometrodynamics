// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Causal flux — directed transmission walkers for electromagnetic coupling.
//!
//! Walkers follow the arrow of time (cause -> effect) on a directed CSR.
//! Transmission probability from Gen1 sources to AntiGen1 targets measures
//! opposite-charge attraction; Gen1 -> Gen1 measures same-charge repulsion.
//!
//! The directed flux CSR is built from the vacuum undirected CSR by filtering
//! edges to retain only the causal (past -> future) direction and applying
//! the Kuratowski merge-map contraction.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::graph::csr::{CsrGraph, Directed};

// ─── Ghost resolution ────────────────────────────────────────────────────────

/// Resolve a node through the Kuratowski contraction map.
///
/// Chases the `merge_into` chain until a fixed point is reached, ensuring
/// walkers never land on ghost (degree-0) nodes after K_5 absorption.
#[inline]
fn resolve(node: usize, merge: Option<&[usize]>) -> usize {
    match merge {
        None => node,
        Some(m) => {
            let mut cur = node;
            while m[cur] != cur {
                cur = m[cur];
            }
            cur
        }
    }
}

// ─── Directed flux CSR construction ──────────────────────────────────────────

/// Build directed flux CSR from vacuum CSR + merge_map + coordinates.
///
/// Follows causal order: u -> v if t(u) < t(v), applying the Kuratowski
/// merge_map contraction.  Self-loops (merged nodes mapping to the same
/// representative) are dropped.
///
/// This replaces the inline directed CSR construction that was previously
/// embedded in main.rs.
///
/// # Arguments
/// * `vacuum_csr` - The vacuum (undirected or symmetric) CSR graph.
/// * `sorted_coords` - Spacetime coordinates [t, x, y, z] for each sprinkled point.
/// * `merge_map` - Kuratowski contraction map: `merge_map[i]` is the
///   representative node for node `i`. Identity (`merge_map[i] == i`) for
///   non-merged nodes.
/// * `n_nodes` - Total number of nodes (including ghost nodes).
///
/// # Returns
/// A directed CSR where each edge points from past to future through
/// the contracted graph.
pub fn build_flux_csr(
    vacuum_csr: &CsrGraph<Directed>,
    sorted_coords: &[[f64; 4]],
    merge_map: &[usize],
    n_nodes: usize,
) -> CsrGraph<Directed> {
    // Two-pass in-place construction: avoids 2.28 GB scratch (rows+cols)
    // at N=10M by counting degrees first, then filling directly.

    // Pass 1: count out-degrees after merge + causal filter
    let mut deg = vec![0u32; n_nodes];
    for u in 0..n_nodes {
        for &v in vacuum_csr.neighbors(u) {
            if sorted_coords[u][0] < sorted_coords[v as usize][0] {
                let ri = merge_map[u];
                let ci = merge_map[v as usize];
                if ri != ci {
                    deg[ri] += 1;
                }
            }
        }
    }

    // Prefix sum → head
    let mut head = vec![0u32; n_nodes + 1];
    for i in 0..n_nodes {
        head[i + 1] = head[i] + deg[i];
    }
    drop(deg);

    // Pass 2: fill data directly
    let total_edges = head[n_nodes] as usize;
    let mut data = vec![0u32; total_edges];
    let mut pos = head[..n_nodes].to_vec();
    for u in 0..n_nodes {
        for &v in vacuum_csr.neighbors(u) {
            if sorted_coords[u][0] < sorted_coords[v as usize][0] {
                let ri = merge_map[u];
                let ci = merge_map[v as usize] as u32;
                if ri != ci as usize {
                    data[pos[ri] as usize] = ci;
                    pos[ri] += 1;
                }
            }
        }
    }
    drop(pos);

    // Sort each row for deterministic traversal.
    //
    // Path-weighted flux: duplicate merged edges are intentionally retained.
    // When Kuratowski contraction folds multiple micro-paths into a single
    // macro-link (A→B), the multiplicity equals the number of underlying
    // causal histories.  Deduplicating would destroy the causal volume and
    // violate conservation of probability in the sum-over-histories.
    for u in 0..n_nodes {
        let start = head[u] as usize;
        let end = head[u + 1] as usize;
        data[start..end].sort_unstable();
    }

    CsrGraph::new(head, data, n_nodes)
}

// ─── Transmission walkers ────────────────────────────────────────────────────

/// Run directed walkers to measure transmission between signatures.
///
/// Walkers move only past -> future in the directed causal graph (mandatory
/// move at each step, unlike the lazy-walk in `run_walkers`).  At each
/// measurement step, the walker's position is checked against sorted target
/// sets for attraction (Gen1 -> AntiGen1) and repulsion (Gen1 -> Gen1).
///
/// **Strict Finitism**: same `u64` integer accumulation as [`run_walkers`](super::walker::run_walkers).
/// Each walker contributes 0 or 1 per step for attraction/repulsion hits.
/// The single `f64` division occurs only at the final return.
///
/// **Ghost resolution**: when `merge_into` is `Some`, every position is
/// resolved through the contraction map before target lookup.
///
/// # Returns
/// `(attraction, repulsion)` — vectors of transmission probabilities at each step.
pub fn run_transmission_walkers(
    adj_head: &[u32],
    adj_data: &[u32],
    starts: &[usize],
    targets_attraction: &[usize],
    targets_repulsion: &[usize],
    steps: &[u32],
    base_seed: u64,
    merge_into: Option<&[usize]>,
) -> (Vec<f64>, Vec<f64>) {
    let n_w = starts.len();
    if n_w == 0 {
        return (vec![0.0; steps.len()], vec![0.0; steps.len()]);
    }
    let n_s = steps.len();
    let max_t = *steps.last().unwrap_or(&0);

    // Sort targets for fast binary search
    let mut attr = targets_attraction.to_vec();
    attr.sort_unstable();
    let mut repu = targets_repulsion.to_vec();
    repu.sort_unstable();

    let (attr_counts, repu_counts): (Vec<u64>, Vec<u64>) = starts
        .par_iter()
        .enumerate()
        .map(|(wi, &origin)| {
            let mut rng =
                StdRng::seed_from_u64(base_seed.wrapping_add(wi as u64 + 1000));
            let mut pos = resolve(origin, merge_into);
            let mut ca = vec![0u64; n_s];
            let mut cr = vec![0u64; n_s];
            let mut si = 0usize;

            for t in 1..=max_t {
                let start = adj_head[pos] as usize;
                let end = adj_head[pos + 1] as usize;
                let len = end - start;

                if len == 0 {
                    // Walker escapes (ends of causal set)
                    break;
                }
                // Mandatory move in directed graph (causal flux)
                let next = adj_data[start + rng.gen_range(0..len)] as usize;
                pos = resolve(next, merge_into);

                if si < n_s && t == steps[si] {
                    if attr.binary_search(&pos).is_ok() {
                        ca[si] = 1;
                    }
                    if repu.binary_search(&pos).is_ok() {
                        cr[si] = 1;
                    }
                    si += 1;
                }
            }
            (ca, cr)
        })
        .reduce(
            || (vec![0u64; n_s], vec![0u64; n_s]),
            |mut a, b| {
                for i in 0..n_s {
                    a.0[i] += b.0[i];
                    a.1[i] += b.1[i];
                }
                a
            },
        );

    (
        attr_counts
            .iter()
            .map(|&c| c as f64 / n_w as f64)
            .collect(),
        repu_counts
            .iter()
            .map(|&c| c as f64 / n_w as f64)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple directed chain: 0 -> 1 -> 2 -> 3
    fn chain_csr() -> CsrGraph<Directed> {
        // head: [0, 1, 2, 3, 3]  (node 3 has no outgoing edges)
        // data: [1, 2, 3]
        let head = vec![0u32, 1, 2, 3, 3];
        let data = vec![1u32, 2, 3];
        CsrGraph::new(head, data, 4)
    }

    #[test]
    fn build_flux_csr_simple() {
        // 4 points with increasing time: 0, 1, 2, 3
        let coords = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ];
        let vacuum = chain_csr();
        let merge_map = vec![0, 1, 2, 3]; // identity
        let flux = build_flux_csr(&vacuum, &coords, &merge_map, 4);

        // Should produce same chain 0->1, 1->2, 2->3
        assert_eq!(flux.degree(0), 1);
        assert!(flux.has_edge(0, 1));
        assert_eq!(flux.degree(1), 1);
        assert!(flux.has_edge(1, 2));
        assert_eq!(flux.degree(2), 1);
        assert!(flux.has_edge(2, 3));
        assert_eq!(flux.degree(3), 0);
    }

    #[test]
    fn build_flux_csr_with_merge() {
        // Merge node 1 into node 0: merge_map = [0, 0, 2, 3]
        let coords = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ];
        let vacuum = chain_csr();
        let merge_map = vec![0, 0, 2, 3]; // node 1 -> node 0
        let flux = build_flux_csr(&vacuum, &coords, &merge_map, 4);

        // Edge 0->1 becomes 0->0 (self-loop, dropped)
        // Edge 1->2 becomes 0->2 (kept)
        // Edge 2->3 becomes 2->3 (kept)
        assert!(flux.has_edge(0, 2));
        assert!(flux.has_edge(2, 3));
        // No self-loop at 0
        assert!(!flux.has_edge(0, 0));
    }

    #[test]
    fn transmission_walkers_empty_starts() {
        let steps = [1, 2, 4];
        let (attr, repu) = run_transmission_walkers(
            &[0], &[], &[], &[0], &[0], &steps, 0, None,
        );
        assert_eq!(attr, vec![0.0; 3]);
        assert_eq!(repu, vec![0.0; 3]);
    }

    #[test]
    fn transmission_walkers_chain() {
        // Directed chain: 0 -> 1 -> 2 -> 3
        let head = vec![0u32, 1, 2, 3, 3];
        let data = vec![1u32, 2, 3];
        let steps = [1, 2, 3];
        // All walkers start at 0, attraction target = {3}, repulsion target = {1}
        let starts = vec![0; 500];
        let (attr, repu) = run_transmission_walkers(
            &head, &data, &starts, &[3], &[1], &steps, 42, None,
        );
        assert_eq!(attr.len(), 3);
        assert_eq!(repu.len(), 3);
        // At step 1, walker is at node 1 (mandatory move from 0).
        // repulsion target is {1}, so repu[0] should be 1.0
        assert!((repu[0] - 1.0).abs() < 1e-10);
        // At step 3, walker is at node 3 (deterministic chain).
        // attraction target is {3}, so attr[2] should be 1.0
        assert!((attr[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn transmission_walkers_with_merge() {
        // Chain 0 -> 1 -> 2 -> 3, merge node 2 into node 1
        let head = vec![0u32, 1, 2, 3, 3];
        let data = vec![1u32, 2, 3];
        let merge = vec![0, 1, 1, 3]; // node 2 -> node 1
        let steps = [1, 2];
        let starts = vec![0; 100];
        let (attr, _repu) = run_transmission_walkers(
            &head, &data, &starts, &[3], &[1], &steps, 99,
            Some(&merge),
        );
        assert_eq!(attr.len(), 2);
        // All values should be valid probabilities
        for &v in attr.iter() {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }
}
