// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Hasse diagram construction (transitive reduction of the causal order).
//!
//! Two tiers:
//!   - `build_hasse_sparse`: full closure -> sparse A^2 reduction  (N <= 15k)
//!   - `build_hasse_direct`: geometric incremental construction   (N > 15k)
//!
//! Physics invariants preserved:
//!   - Light cone search: spatial radius at time offset dt is dt + 1 (c=1).
//!     No fixed spatial cutoff -- Lorentz invariance is exact.
//!   - Transitive reduction: edge (u,v) is a Hasse link iff no existing
//!     link-child z of u satisfies z < v. This preserves the triangle-free
//!     property required for Causal Prisms (K_{2,n} bipartite structures).

use rayon::prelude::*;
use sprs::TriMat;
use std::sync::atomic::Ordering;

use crate::graph::csr::{CsrGraph, Directed};
use crate::graph::grid::{
    GridDims, is_causal, time_order, quantize_and_sort, build_oci,
    MAX_PROPER_TIME_SQ, MAX_CAUSAL_DEPTH, MAX_HASSE_DEGREE,
};

// ---- Tier 1: sparse A^2 reduction (N <= 15k) --------------------------------

/// Build full causal closure, then remove redundant edges via sparse A^2.
///
/// For a transitively-closed DAG, edge (i,j) is a link iff A^2[i,j] = 0
/// (no 2-hop path exists).
pub fn build_hasse_sparse(pts: &[[f64; 4]]) -> (Vec<[f64; 4]>, CsrGraph<Directed>, Vec<i32>) {
    let n = pts.len();
    let order = time_order(pts);
    let sorted_coords: Vec<[f64; 4]> = order.iter().map(|&i| pts[i]).collect();

    // Build all causal edges (parallel over source nodes) using new indices 0..n
    let edge_groups: Vec<Vec<(u32, u32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let pi = &sorted_coords[i];
            let mut local = Vec::new();
            for j in (i + 1)..n {
                if is_causal(pi, &sorted_coords[j]) {
                    local.push((i as u32, j as u32));
                }
            }
            local
        })
        .collect();
    let edges: Vec<(u32, u32)> = edge_groups.into_iter().flatten().collect();
    println!("    Raw causal edges: {}", edges.len());

    // Sparse adjacency A and A^2
    let mut tri = TriMat::new((n, n));
    for &(r, c) in &edges {
        tri.add_triplet(r as usize, c as usize, 1.0_f64);
    }
    let a = tri.to_csr::<usize>();
    let a2 = &a * &a;

    // Keep only link edges (A^2[i,j] == 0)
    let mut out_deg = vec![0i32; n];
    let mut in_deg = vec![0i32; n];
    let mut node_neighbors = vec![Vec::new(); n];
    for &(r, c) in &edges {
        let val = a2.get(r as usize, c as usize).copied().unwrap_or(0.0);
        if val == 0.0 {
            node_neighbors[r as usize].push(c);
            out_deg[r as usize] += 1;
            in_deg[c as usize] += 1;
        }
    }

    let bulk_momentum: Vec<i32> = out_deg
        .iter()
        .zip(in_deg.iter())
        .map(|(o, i)| o - i)
        .collect();

    // Build CSR
    let mut adj_head = Vec::with_capacity(n + 1);
    let mut total_edges = 0;
    for neighbors in &node_neighbors {
        adj_head.push(total_edges as u32);
        total_edges += neighbors.len();
    }
    adj_head.push(total_edges as u32);

    let mut adj_data = Vec::with_capacity(total_edges);
    for neighbors in node_neighbors {
        adj_data.extend(neighbors);
    }

    // CRITICAL: Sort CSR neighbors for binary_search correctness in Phase 2
    for u in 0..n {
        let start = adj_head[u] as usize;
        let end = adj_head[u + 1] as usize;
        adj_data[start..end].sort_unstable();
    }

    let csr = CsrGraph::<Directed>::new(adj_head, adj_data, n);
    (sorted_coords, csr, bulk_momentum)
}

// ---- Tier 2: direct geometric construction (N > 15k) -------------------------

/// Build Hasse diagram directly without materialising the full closure.
///
/// For each source node i (in time order), iterate over future nodes j.
/// Edge (i,j) is a link iff no existing link-child z of i satisfies z < j.
/// Uses the Occupied Cell Index (OCI) for spatial lookups and adaptive
/// early termination based on the Poisson probability bound.
pub fn build_hasse_direct(pts: &[[f64; 4]]) -> (Vec<[f64; 4]>, CsrGraph<Directed>, Vec<i32>) {
    // 1. Quantize to Integer Lattice & Sort for Data Locality
    let dims = GridDims::from_bounds(pts);
    let (sorted_pts, sorted_coords) = quantize_and_sort(pts, &dims);

    // 3. Build Occupied Cell Index (OCI)
    let occupied = build_oci(&sorted_pts, &dims);

    let t_dim = dims.t_dim;
    let max_dt = (t_dim as i16).min(MAX_CAUSAL_DEPTH);
    {
        let occ_total: usize = occupied.iter().map(|l| l.len()).sum();
        let msg = format!(
            "  [Phase 1] OCI built ({occ_total} occupied cells across {t_dim} layers). \
             Starting parallel search (max_dt={max_dt}, degree_cap={MAX_HASSE_DEGREE}, \
             adaptive termination)..."
        );
        println!("{}", msg);
        use std::fs::OpenOptions;
        use std::io::Write;
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open("simulation.log")
        {
            writeln!(file, "{}", msg).ok();
        }
    }

    let processed = std::sync::atomic::AtomicUsize::new(0);
    let total_nodes = sorted_pts.len();
    let chunk_size = (total_nodes / 10).max(1); // 10%

    let bulk_momentum_atomic: Vec<std::sync::atomic::AtomicI32> = (0..total_nodes)
        .map(|_| std::sync::atomic::AtomicI32::new(0))
        .collect();

    // 4. Parallel Hasse link discovery (OCI + Adaptive Termination)
    let chunk_results: Vec<(Vec<u32>, Vec<u32>)> = sorted_pts
        .par_chunks(2048)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut chunk_degrees = Vec::with_capacity(chunk.len());
            let mut chunk_targets = Vec::with_capacity(chunk.len() * 16);
            let mut children_coords: Vec<[f64; 4]> = Vec::with_capacity(32);
            let mut candidates: Vec<usize> = Vec::with_capacity(128);

            let chunk_offset = chunk_idx * 2048;

            // Progress logging
            let val = processed.fetch_add(chunk.len(), Ordering::Relaxed);
            if val % chunk_size < chunk.len() && val > 0 {
                let pct = (val as f64 / total_nodes as f64) * 100.0;
                let msg = format!("  [Phase 1] Progress: {:.0}% ({val}/{total_nodes})", pct);
                println!("{}", msg);
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("simulation.log")
                {
                    use std::io::Write;
                    writeln!(file, "{}", msg).ok();
                }
            }

            for (i_in_chunk, u_pt) in chunk.iter().enumerate() {
                let u_idx_global = chunk_offset + i_in_chunk;
                let u_orig = u_pt.orig_idx;
                let (qt, qx, qy, qz) = (
                    u_pt.qt as i16,
                    u_pt.qx as i16,
                    u_pt.qy as i16,
                    u_pt.qz as i16,
                );
                children_coords.clear();
                let mut degree = 0;
                let mut consecutive_dry = 0; // ADAPTIVE EARLY TERMINATION

                for dt in 0..max_dt {
                    let target_t = (qt + dt) as usize;
                    if target_t >= t_dim {
                        continue;
                    }
                    let r_max = dt + 1;
                    let r2_limit = ((r_max + 1) * (r_max + 1)) as i32;
                    let dt_f = dt as f64;

                    let layer = &occupied[target_t];
                    let cx_lo = qx - r_max;
                    let cx_hi = qx + r_max;

                    // OCI Binary Search for the active X-band
                    let start_idx = layer.partition_point(|e| e.0 < cx_lo);
                    let end_idx = layer.partition_point(|e| e.0 <= cx_hi);

                    let links_before = children_coords.len();
                    candidates.clear();

                    for &(cx, cy, cz, cell_start, cell_count) in &layer[start_idx..end_idx] {
                        let dx = (cx as i32) - (qx as i32);
                        let dy = (cy as i32) - (qy as i32);
                        let dz = (cz as i32) - (qz as i32);
                        let r2 = dx * dx + dy * dy + dz * dz;

                        if r2 >= r2_limit {
                            continue;
                        }
                        let proper_time_sq = dt_f * dt_f - r2 as f64;
                        if proper_time_sq > MAX_PROPER_TIME_SQ {
                            continue;
                        }

                        for k in 0..cell_count {
                            let v_idx = (cell_start + k) as usize;
                            if u_orig != sorted_pts[v_idx].orig_idx {
                                candidates.push(v_idx);
                            }
                        }
                    }

                    // Process candidates per dt-shell to allow early termination
                    candidates.sort_unstable_by(|&a, &b| {
                        sorted_pts[a].p[0]
                            .partial_cmp(&sorted_pts[b].p[0])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.dedup();

                    let mut causal_checked = 0u32;
                    let mut blocked = 0u32;

                    for &v_idx in &candidates {
                        let v_pt = &sorted_pts[v_idx];
                        if !is_causal(&u_pt.p, &v_pt.p) {
                            continue;
                        }
                        causal_checked += 1;
                        if children_coords.iter().any(|z| is_causal(z, &v_pt.p)) {
                            blocked += 1;
                        } else {
                            children_coords.push(v_pt.p);
                            chunk_targets.push(v_idx as u32);
                            degree += 1;
                            bulk_momentum_atomic[u_idx_global].fetch_add(1, Ordering::Relaxed);
                            bulk_momentum_atomic[v_idx].fetch_sub(1, Ordering::Relaxed);
                        }
                    }

                    // -- Valve 1: Degree cap (Causal Shadow) --
                    // A node with >= MAX_HASSE_DEGREE descendants has a fully
                    // populated causal shadow -- further links are redundant.
                    if children_coords.len() >= MAX_HASSE_DEGREE {
                        break;
                    }

                    // -- Valve 2: Alexandrov volume short-circuit --
                    // If every causal candidate in this dt-shell was
                    // transitively blocked (non-empty Alexandrov set between
                    // u and v), further shells at larger dt have strictly
                    // larger Alexandrov volume and will also be blocked.
                    // One existing link suffices as proof of non-trivial
                    // causal structure below u.
                    if causal_checked > 0 && blocked == causal_checked && degree >= 1 {
                        break;
                    }

                    // -- Valve 3: Adaptive early termination --
                    // Dry shell with no causal candidates at all (empty
                    // lightcone slice). Two consecutive dry shells + >= 2
                    // descendants => break.
                    if children_coords.len() == links_before {
                        consecutive_dry += 1;
                        if consecutive_dry >= 2 && children_coords.len() >= 2 {
                            break;
                        }
                    } else {
                        consecutive_dry = 0;
                    }
                }
                chunk_degrees.push(degree);
            }
            (chunk_degrees, chunk_targets)
        })
        .collect();

    // Free sorted_pts -- only sorted_coords is needed from here on.
    // At N=10M this reclaims ~560 MB before the CSR assembly allocation.
    drop(sorted_pts);

    // 5. Build CSR (head, data)
    let mut adj_head = Vec::with_capacity(total_nodes + 1);
    adj_head.push(0);
    let mut total_edges = 0;

    for (degrees, _) in &chunk_results {
        for &deg in degrees {
            total_edges += deg;
            adj_head.push(total_edges as u32);
        }
    }

    let mut adj_data = Vec::with_capacity(total_edges as usize);
    for (_, targets) in chunk_results {
        adj_data.extend(targets);
    }

    let bulk_momentum: Vec<i32> = bulk_momentum_atomic
        .into_iter()
        .map(|a| a.into_inner())
        .collect();

    // CRITICAL: Sort CSR neighbors to ensure binary_search correctness in Phase 2.
    // Without this, are_connected() and count_slice_intersection() in skyrmion.rs
    // produce false negatives, causing valid Causal Prisms to be missed.
    for u in 0..total_nodes {
        let start = adj_head[u] as usize;
        let end = adj_head[u + 1] as usize;
        adj_data[start..end].sort_unstable();
    }

    let csr = CsrGraph::<Directed>::new(adj_head, adj_data, total_nodes);
    (sorted_coords, csr, bulk_momentum)
}
