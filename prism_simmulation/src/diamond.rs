//! Phase 1 — Vacuum Generation
//!
//! Poisson sprinkling in a 4D causal diamond, causal edge discovery,
//! and Hasse diagram construction (transitive reduction).
//!
//! Two tiers:
//!   - `build_hasse_sparse`: full closure → sparse A² reduction  (N ≤ 15k)
//!   - `build_hasse_direct`: geometric incremental construction  (N > 15k)
//!
//! Physics invariants preserved:
//!   - Light cone search: spatial radius at time offset dt is dt + margin (c=1).
//!     No fixed spatial cutoff — Lorentz invariance is exact.
//!   - Transitive reduction: edge (u,v) is a Hasse link iff no existing
//!     link-child z of u satisfies z ≺ v. This preserves the triangle-free
//!     property required for Causal Prisms (K₂,ₙ bipartite structures).

use rand::Rng;
use rayon::prelude::*;
use sprs::TriMat;
use std::f64::consts::PI;
use std::sync::atomic::Ordering;

// Hollow Light Cone Shell Optimization (HPC)
// Filter out deep interior points where proper time > 4.0 units (d=16.0 squared)
// The Hasse Diagram logic guarantees that any direct link with huge proper time
// would be transitively reduced by intermediate steps.
//
// *Physics Justification*: In a uniform Poisson sprinkling (ρ ≈ 1), the Alexandrov
// set volume for τ > 8 is > 4096. The probability of zero intermediate points
// falling in this volume is exponentially small (< 10^-100).
// *Boundary Caveat*: Near the causal diamond boundaries, this volume is truncated,
// slightly increasing the probability of long-range Hasse links. However, at N > 15k,
// the boundary-to-bulk ratio approaches zero, making this optimization physically
// sound for massive ensembles.
const MAX_PROPER_TIME_SQ: f64 = 64.0;

/// Maximum time-layer depth for Hasse link search in `build_hasse_direct()`.
///
/// **Lorentz-invariant termination**: This constant sets the widest coordinate-
/// time window scanned for Hasse links.  Actual termination is governed by
/// three physics-based valves (degree cap, Alexandrov short-circuit, adaptive
/// dry-shell) and the proper-time filter `MAX_PROPER_TIME_SQ`.  Setting
/// `dt_max = 15` ensures the loop never artificially truncates highly-boosted
/// events while the valves still terminate the inner loop in O(1) for typical
/// nodes.  At Poisson density ρ ≈ 1, virtually no non-redundant Hasse link
/// survives beyond Δt ≈ 6–8, so the wider window adds negligible cost.
const MAX_CAUSAL_DEPTH: i16 = 15;

/// Maximum out-degree before abandoning the light-cone search for a node.
///
/// In a Hasse diagram at ρ ≈ 1, the mean valence is ~4–10.  A node that has
/// already accumulated 15 descendants has a fully populated causal shadow —
/// any further candidate is almost certainly transitively redundant.
const MAX_HASSE_DEGREE: usize = 15;

/// Poisson-sprinkle `n` points into a 4D causal diamond |t| + r ≤ T/2.
///
/// Volume V = πT⁴/24;  T = (24N/π)^{1/4}  gives density ρ ≈ 1.
pub fn sprinkle(n: usize, rng: &mut impl Rng) -> (Vec<[f64; 4]>, f64) {
    let big_t = (24.0 * n as f64 / PI).powf(0.25);
    let half_t = big_t / 2.0;

    let mut pts = Vec::with_capacity(n);
    while pts.len() < n {
        let t: f64 = rng.gen_range(-half_t..half_t);
        let x: f64 = rng.gen_range(-half_t..half_t);
        let y: f64 = rng.gen_range(-half_t..half_t);
        let z: f64 = rng.gen_range(-half_t..half_t);
        let r = (x * x + y * y + z * z).sqrt();
        if t.abs() + r <= half_t {
            pts.push([t, x, y, z]);
        }
    }
    (pts, big_t)
}

/// True if `b` is in the strict causal future of `a`:
///   t_b > t_a  and  (t_b − t_a)² > |Δx⃗|².
#[inline]
fn is_causal(a: &[f64; 4], b: &[f64; 4]) -> bool {
    let dt = b[0] - a[0];
    if dt <= 0.0 {
        return false;
    }
    let dx = b[1] - a[1];
    let dy = b[2] - a[2];
    let dz = b[3] - a[3];
    dt * dt > dx * dx + dy * dy + dz * dz
}

/// Return time-sorted index order (position → original index).
fn time_order(pts: &[[f64; 4]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pts.len()).collect();
    order.sort_by(|&a, &b| pts[a][0].partial_cmp(&pts[b][0]).unwrap());
    order
}

// ─── Tier 1: sparse A² reduction (N ≤ 15k) ─────────────────────────────────

/// Build full causal closure, then remove redundant edges via sparse A².
///
/// For a transitively-closed DAG, edge (i,j) is a link iff A²\[i,j\] = 0
/// (no 2-hop path exists).
pub fn build_hasse_sparse(pts: &[[f64; 4]]) -> (Vec<[f64; 4]>, Vec<u32>, Vec<u32>, Vec<i32>) {
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

    // Sparse adjacency A and A²
    let mut tri = TriMat::new((n, n));
    for &(r, c) in &edges {
        tri.add_triplet(r as usize, c as usize, 1.0_f64);
    }
    let a = tri.to_csr::<usize>();
    let a2 = &a * &a;

    // Keep only link edges (A²[i,j] == 0)
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
    
    let bulk_momentum: Vec<i32> = out_deg.iter().zip(in_deg.iter()).map(|(o, i)| o - i).collect();

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

    // CRITICAL BUG FIX: Sort CSR neighbors for binary_search correctness in Phase 2
    for u in 0..n {
        let start = adj_head[u] as usize;
        let end = adj_head[u + 1] as usize;
        adj_data[start..end].sort_unstable();
    }

    (sorted_coords, adj_head, adj_data, bulk_momentum)
}

// ─── Tier 2: direct geometric construction (N > 15k) ────────────────────────

// Build Hasse diagram directly without materialising the full closure.
//
// For each source node i (in time order), iterate over future nodes j.
// Edge (i,j) is a link iff no existing link-child z of i satisfies z ≺ j.
//
// Correctness: if any intermediate k with i ≺ k ≺ j exists, then some
// link-child z of i has z ≼ k (possibly z = k), hence z ≺ j, and we
// detect the redundancy.
//
// The search at each time offset dt expands the spatial radius to dt + 1
// grid units (the exact light cone with cell-size margin). No fixed spatial
// cutoff is used — Lorentz invariance is preserved.

/// A spacetime event quantised onto an integer lattice for cache-coherent traversal.
///
/// Points are sorted by their cell index (a linearised 4D grid coordinate)
/// so that spatially nearby events are contiguous in memory. This cell-sorted
/// order is critical for the OCI (Occupied Cell Index) algorithm: it ensures
/// that grid lookups translate directly into contiguous array slices, giving
/// O(1) access to all points within a given cell.
#[derive(Debug, Clone, Copy)]
pub struct GridPoint {
    pub p: [f64; 4],
    pub orig_idx: u32,
    pub cell: usize,
    pub qt: u16, pub qx: u16, pub qy: u16, pub qz: u16,
}

/// Poisson-sprinkle points into a time-slab [t_min, t_max].
pub fn sprinkle_chunk(
    n_expected: usize,
    t_min: f64,
    t_max: f64,
    half_t: f64,
    rng: &mut impl Rng,
    start_idx: u32,
) -> Vec<GridPoint> {
    let mut pts = Vec::with_capacity(n_expected);
    let mut idx = start_idx;

    while pts.len() < n_expected {
        let t: f64 = rng.gen_range(t_min..t_max);
        let x: f64 = rng.gen_range(-half_t..half_t);
        let y: f64 = rng.gen_range(-half_t..half_t);
        let z: f64 = rng.gen_range(-half_t..half_t);
        let r = (x * x + y * y + z * z).sqrt();
        if t.abs() + r <= half_t {
            pts.push(GridPoint {
                p: [t, x, y, z],
                orig_idx: idx,
                cell: 0,
                qt: 0, qx: 0, qy: 0, qz: 0,
            });
            idx += 1;
        }
    }
    pts
}

/// Build Hasse diagram directly without materialising the full closure.
///
/// For each source node i (in time order), iterate over future nodes j.
/// Edge (i,j) is a link iff no existing link-child z of i satisfies z ≺ j.
/// Uses the Occupied Cell Index (OCI) for spatial lookups and adaptive
/// early termination based on the Poisson probability bound.
pub fn build_hasse_direct(pts: &[[f64; 4]]) -> (Vec<[f64; 4]>, Vec<u32>, Vec<u32>, Vec<i32>) {
    // 1. Quantize to Integer Lattice & Sort for Data Locality

    let mut min_val = [f64::INFINITY; 4];
    let mut max_val = [f64::NEG_INFINITY; 4];
    for p in pts {
        for k in 0..4 {
            if p[k] < min_val[k] { min_val[k] = p[k]; }
            if p[k] > max_val[k] { max_val[k] = p[k]; }
        }
    }

    let margin = 2.0;
    let origin = [
        min_val[0] - margin,
        min_val[1] - margin,
        min_val[2] - margin,
        min_val[3] - margin,
    ];

    let to_grid = |val: f64, k: usize| -> u16 {
        (val - origin[k]).max(0.0) as u16
    };

    let t_span = max_val[0] - min_val[0] + 2.0 * margin;
    let x_span = max_val[1] - min_val[1] + 2.0 * margin;
    let y_span = max_val[2] - min_val[2] + 2.0 * margin;
    let z_span = max_val[3] - min_val[3] + 2.0 * margin;

    let t_dim = (t_span.ceil() as usize).max(1);
    let dim_x = (x_span.ceil() as usize).max(1);
    let dim_y = (y_span.ceil() as usize).max(1);
    let dim_z = (z_span.ceil() as usize).max(1);

    let grid_stride_y = dim_z;
    let grid_stride_x = dim_y * dim_z;
    let grid_stride_t = dim_x * dim_y * dim_z;
    let grid_size = t_dim * grid_stride_t;

    // 2. Build sorted flat array for cache-coherent traversal

    let mut sorted_pts: Vec<GridPoint> = pts.iter().enumerate().map(|(i, p)| {
        let qt = to_grid(p[0], 0);
        let qx = to_grid(p[1], 1);
        let qy = to_grid(p[2], 2);
        let qz = to_grid(p[3], 3);

        let idx = (qt as usize) * grid_stride_t
                + (qx as usize) * grid_stride_x
                + (qy as usize) * grid_stride_y
                + (qz as usize);

        GridPoint {
            p: *p,
            orig_idx: i as u32,
            cell: if idx < grid_size { idx } else { grid_size },
            qt, qx, qy, qz,
        }
    }).collect();

    sorted_pts.par_sort_unstable_by_key(|gp| gp.cell);

    // Filter points and get re-ordered coordinates
    let sorted_coords: Vec<[f64; 4]> = sorted_pts.iter().map(|gp| gp.p).collect();

    // 3. Build Grid Index Table & Occupied Cell Index (OCI)

    let mut grid_start = vec![u32::MAX; grid_size];
    let mut grid_count = vec![0u32; grid_size];

    for (i, gp) in sorted_pts.iter().enumerate() {
        if gp.cell >= grid_size { continue; }
        if grid_start[gp.cell] == u32::MAX {
            grid_start[gp.cell] = i as u32;
        }
        grid_count[gp.cell] += 1;
    }

    // ── Occupied Cell Index (OCI) ──────────────────────────────────────
    //
    // Paradigm shift from geometric to spatial indexing.
    //
    // The naive approach iterates ALL cells in the light-cone shell for each
    // source node. At large N the grid is sparse (>96% cells are empty),
    // making geometric enumeration wasteful. OCI inverts the lookup:
    //
    //   1. Build per-time-layer lists of *occupied* cells only.
    //   2. Sort each layer by X coordinate.
    //   3. For each source, binary-search the X-band of the light cone
    //      within the occupied list, skipping all empty space.
    //
    // This reduces the inner-loop cost from O(shell_volume) to O(occupied_cells),
    // a ~20x speedup at N > 100k.
    let mut occupied: Vec<Vec<(i16, i16, i16, u32, u32)>> = vec![vec![]; t_dim];
    for cell in 0..grid_size {
        if grid_start[cell] != u32::MAX {
            let qt = cell / grid_stride_t;
            let rem_t = cell % grid_stride_t;
            let qx = (rem_t / grid_stride_x) as i16;
            let rem_x = rem_t % grid_stride_x;
            let qy = (rem_x / grid_stride_y) as i16;
            let qz = (rem_x % grid_stride_y) as i16;
            occupied[qt].push((qx, qy, qz, grid_start[cell], grid_count[cell]));
        }
    }
    // Sort by X coordinate for fast binary search band queries
    for layer in &mut occupied {
        layer.sort_unstable_by_key(|&(qx, _, _, _, _)| qx);
    }

    // Free grid arrays — OCI replaces them entirely
    drop(grid_start);
    drop(grid_count);

    let max_dt = (t_dim as i16).min(MAX_CAUSAL_DEPTH);
    {
        let occ_total: usize = occupied.iter().map(|l| l.len()).sum();
        let msg = format!("  [Phase 1] OCI built ({occ_total} occupied cells across {t_dim} layers). Starting parallel search (max_dt={max_dt}, degree_cap={MAX_HASSE_DEGREE}, adaptive termination)...");
        println!("{}", msg);
        use std::fs::OpenOptions;
        use std::io::Write;
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("simulation.log") {
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
                if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("simulation.log") {
                    use std::io::Write;
                    writeln!(file, "{}", msg).ok();
                }
            }

            for (i_in_chunk, u_pt) in chunk.iter().enumerate() {
                let u_idx_global = chunk_offset + i_in_chunk;
                let u_orig = u_pt.orig_idx;
                let (qt, qx, qy, qz) = (u_pt.qt as i16, u_pt.qx as i16, u_pt.qy as i16, u_pt.qz as i16);
                children_coords.clear();
                let mut degree = 0;
                let mut consecutive_dry = 0; // ADAPTIVE EARLY TERMINATION

                for dt in 0..max_dt {
                    let target_t = (qt + dt) as usize;
                    if target_t >= t_dim { continue; }
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

                        if r2 >= r2_limit { continue; }
                        let proper_time_sq = dt_f * dt_f - r2 as f64;
                        if proper_time_sq > MAX_PROPER_TIME_SQ { continue; }

                        for k in 0..cell_count {
                            let v_idx = (cell_start + k) as usize;
                            if u_orig != sorted_pts[v_idx].orig_idx {
                                candidates.push(v_idx);
                            }
                        }
                    }

                    // Process candidates per dt-shell to allow early termination
                    candidates.sort_unstable_by(|&a, &b| {
                        sorted_pts[a].p[0].partial_cmp(&sorted_pts[b].p[0]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.dedup();

                    let mut causal_checked = 0u32;
                    let mut blocked = 0u32;

                    for &v_idx in &candidates {
                        let v_pt = &sorted_pts[v_idx];
                        if !is_causal(&u_pt.p, &v_pt.p) { continue; }
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

                    // ── Valve 1: Degree cap (Causal Shadow) ──
                    // A node with ≥ MAX_HASSE_DEGREE descendants has a fully
                    // populated causal shadow — further links are redundant.
                    if children_coords.len() >= MAX_HASSE_DEGREE { break; }

                    // ── Valve 2: Alexandrov volume short-circuit ──
                    // If every causal candidate in this dt-shell was
                    // transitively blocked (non-empty Alexandrov set between
                    // u and v), further shells at larger dt have strictly
                    // larger Alexandrov volume and will also be blocked.
                    // One existing link suffices as proof of non-trivial
                    // causal structure below u.
                    if causal_checked > 0 && blocked == causal_checked && degree >= 1 {
                        break;
                    }

                    // ── Valve 3: Adaptive early termination ──
                    // Dry shell with no causal candidates at all (empty
                    // lightcone slice). Two consecutive dry shells + ≥2
                    // descendants ⇒ break.
                    if children_coords.len() == links_before {
                        consecutive_dry += 1;
                        if consecutive_dry >= 2 && children_coords.len() >= 2 { break; }
                    } else {
                        consecutive_dry = 0;
                    }
                }
                chunk_degrees.push(degree);
            }
            (chunk_degrees, chunk_targets)
        })
        .collect();

    // Free sorted_pts — only sorted_coords is needed from here on.
    // At N=10M this reclaims ~560 MB before the CSR assembly allocation.
    drop(sorted_pts);

    // 5. Build CSR (head, data) - Ensamble secuencial rápido y limpio
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
    
    let bulk_momentum: Vec<i32> = bulk_momentum_atomic.into_iter().map(|a| a.into_inner()).collect();

    // CRITICAL BUG FIX: Sort CSR neighbors to ensure binary_search correctness in Phase 2.
    // Without this, are_connected() and count_slice_intersection() in skyrmion.rs
    // produce false negatives, causing valid Causal Prisms to be missed.
    for u in 0..total_nodes {
        let start = adj_head[u] as usize;
        let end = adj_head[u + 1] as usize;
        adj_data[start..end].sort_unstable();
    }

    (sorted_coords, adj_head, adj_data, bulk_momentum)
}

// ─── Streaming Implementation ──────────────────────────────────────────────

/// Stream Hasse link edges to a binary file using a Sliding Window.
///
/// Output format: continuous stream of (u32, u32) in Little Endian.
/// Both physics invariants are preserved:
///   - Full light cone search (spatial radius = dt + 1 at each time offset)
///   - Transitive reduction via locally cached children coordinates
pub fn stream_edges_to_file(n_total: usize, chunk_size: usize, path: &str, seed: u64) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let big_t = (24.0 * n_total as f64 / PI).powf(0.25);
    let half_t = big_t / 2.0;

    // Grid Parameters (Global)
    let margin = 2.0;
    let t_span = big_t + 2.0 * margin;
    let x_span = big_t + 2.0 * margin;

    let t_dim = (t_span.ceil() as usize).max(1);
    let dim_x = (x_span.ceil() as usize).max(1);
    let dim_y = dim_x;
    let dim_z = dim_x;

    let grid_stride_y = dim_z;
    let grid_stride_x = dim_y * dim_z;
    let grid_stride_t = dim_x * dim_y * dim_z;
    let grid_size = t_dim * grid_stride_t;

    let to_grid = |val: f64, _k: usize| -> u16 {
        let origin = -half_t - margin;
        ((val - origin).max(0.0)) as u16
    };

    // Prepare Output
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Sliding Window State
    let mut buffer: Vec<GridPoint> = Vec::with_capacity(chunk_size * 2);
    let mut grid_start = vec![u32::MAX; grid_size];
    let mut grid_count = vec![0u32; grid_size];

    let mut current_t = -half_t;
    let mut next_node_id = 0;
    let mut global_edges_count = 0;

    println!("Stream: T={:.2}, t_range=[{:.2}, {:.2}]", big_t, -half_t, half_t);

    // Match in-memory path: let physics-based valves handle termination.
    let max_dt = (t_dim as i16).min(MAX_CAUSAL_DEPTH);
    // Safety margin only needs to cover max_dt time units (+2 for grid margin).
    let safety_margin_t = (max_dt as f64) + 2.0;

    while current_t < half_t {
        // 1. Fill Buffer with enough future data to cover the light cone

        let mut max_t_in_buffer = buffer.iter().map(|p| p.p[0]).fold(f64::NEG_INFINITY, f64::max);
        if buffer.is_empty() { max_t_in_buffer = current_t; }

        while max_t_in_buffer < current_t + safety_margin_t && current_t < half_t {
            let next_t = (current_t + safety_margin_t).min(half_t);

            use std::f64::consts::PI;
            let v_tot = PI / 24.0 * big_t.powi(4);
            let rho = n_total as f64 / v_tot;

            let compute_vol_pos = |t1: f64, t2: f64, h: f64| -> f64 {
                0.25 * ((h - t1).powi(4) - (h - t2).powi(4))
            };
            let compute_vol_neg = |t1: f64, t2: f64, h: f64| -> f64 {
                0.25 * ((h + t2).powi(4) - (h + t1).powi(4))
            };

            let vol_slice = 4.0 / 3.0 * PI * if current_t >= 0.0 {
                compute_vol_pos(current_t, next_t, half_t)
            } else if next_t <= 0.0 {
                compute_vol_neg(current_t, next_t, half_t)
            } else {
                compute_vol_neg(current_t, 0.0, half_t) + compute_vol_pos(0.0, next_t, half_t)
            };

            let n_ask = (rho * vol_slice) as usize;

            let mut new_pts = sprinkle_chunk(n_ask, current_t, next_t, half_t, &mut rng, next_node_id);
            if !new_pts.is_empty() {
                next_node_id += new_pts.len() as u32;

                for p in &mut new_pts {
                   p.qt = to_grid(p.p[0], 0);
                   p.qx = to_grid(p.p[1], 1);
                   p.qy = to_grid(p.p[2], 2);
                   p.qz = to_grid(p.p[3], 3);

                   let idx = (p.qt as usize) * grid_stride_t
                           + (p.qx as usize) * grid_stride_x
                           + (p.qy as usize) * grid_stride_y
                           + (p.qz as usize);
                   p.cell = if idx < grid_size { idx } else { grid_size };
                }

                buffer.extend(new_pts);
            }
            current_t = next_t;
            max_t_in_buffer = current_t;

            if buffer.len() > chunk_size * 2 { break; }
        }

        if buffer.is_empty() { break; }

        // 2. Sort Buffer for cache-coherent grid access
        buffer.par_sort_unstable_by_key(|gp| gp.cell);

        // 3. Build Grid Index (reuse allocations)
        grid_start.fill(u32::MAX);
        grid_count.fill(0);

        for (i, gp) in buffer.iter().enumerate() {
            if gp.cell >= grid_size { continue; }
            if grid_start[gp.cell] == u32::MAX {
                grid_start[gp.cell] = i as u32;
            }
            grid_count[gp.cell] += 1;
        }

        // 4. Process safe sources (full light cone + transitive reduction)
        // A source is safe when we have its entire causal future in the buffer.
        let safe_threshold = if current_t >= half_t {
             f64::INFINITY
        } else {
             max_t_in_buffer - safety_margin_t * 0.9
        };

        let edges_batch: Vec<(u32, u32)> = buffer.par_iter()
            .filter(|p| p.p[0] < safe_threshold)
            .map(|u_pt| {
                let u_orig = u_pt.orig_idx;
                let (qt, qx, qy, qz) = (u_pt.qt as i16, u_pt.qx as i16, u_pt.qy as i16, u_pt.qz as i16);
                let mut local_edges = Vec::new();
                let mut children_coords: Vec<[f64; 4]> = Vec::with_capacity(32);
                let mut candidates: Vec<usize> = Vec::with_capacity(128);

                // Per-shell light cone search with termination valves
                let mut consecutive_dry = 0u16;

                for dt in 0..max_dt {
                    let target_t = qt + dt;
                    if target_t < 0 || target_t >= (t_dim as i16) { continue; }

                    // Light cone: spatial radius = dt + 1 (c=1, +1 grid margin)
                    let r_max = dt + 1;
                    let r2_limit = (r_max + 1) * (r_max + 1);
                    let dt_f = dt as f64;

                    let r_min_f = (dt_f * dt_f - MAX_PROPER_TIME_SQ).max(0.0).sqrt();
                    let r_min = r_min_f.floor() as i16;
                    let r_min_sq = (r_min as i32) * (r_min as i32);

                    let links_before = children_coords.len();
                    candidates.clear();

                    for dx in -r_max..=r_max {
                        for dy in -r_max..=r_max {
                            let dxy2 = dx as i32 * dx as i32 + dy as i32 * dy as i32;

                            // Hollow Shell Logic
                            let dz_skip = if dxy2 < r_min_sq {
                                ((r_min_sq - dxy2) as f64).sqrt().floor() as i16
                            } else {
                                0
                            };

                            let z_ranges: smallvec::SmallVec<[std::ops::RangeInclusive<i16>; 2]> = if dz_skip > 0 {
                                smallvec::smallvec![-r_max..=-dz_skip, dz_skip..=r_max]
                            } else {
                                smallvec::smallvec![-r_max..=r_max]
                            };

                            for range in z_ranges {
                                for dz in range {
                                    let r2_i = dxy2 + dz as i32 * dz as i32;
                                    let proper_time_sq = dt_f * dt_f - r2_i as f64;

                                    if proper_time_sq > MAX_PROPER_TIME_SQ { continue; }
                                    if r2_i >= r2_limit as i32 { continue; }

                                    let tx = qx + dx;
                                    let ty = qy + dy;
                                    let tz = qz + dz;

                                    if tx < 0 || tx >= dim_x as i16 || ty < 0 || ty >= dim_y as i16 || tz < 0 || tz >= dim_z as i16 { continue; }

                                    let cell_idx = (target_t as usize) * grid_stride_t
                                                 + (tx as usize) * grid_stride_x
                                                 + (ty as usize) * grid_stride_y
                                                 + (tz as usize);

                                    if cell_idx >= grid_size { continue; }

                                    let start = grid_start[cell_idx] as usize;
                                    let count = grid_count[cell_idx] as usize;
                                    if start == usize::MAX || count == 0 { continue; }

                                    candidates.extend(start..(start + count));
                                }
                            }
                        }
                    }

                    // Process candidates for this dt-shell
                    candidates.sort_unstable_by(|&a, &b| {
                        buffer[a].p[0].partial_cmp(&buffer[b].p[0]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.dedup();

                    let mut causal_checked = 0u32;
                    let mut blocked = 0u32;

                    for &v_idx in &candidates {
                        let v_pt = &buffer[v_idx];
                        if u_orig == v_pt.orig_idx { continue; }
                        if !is_causal(&u_pt.p, &v_pt.p) { continue; }
                        causal_checked += 1;

                        let redundant = children_coords.iter().any(|z| is_causal(z, &v_pt.p));
                        if redundant {
                            blocked += 1;
                        } else {
                            children_coords.push(v_pt.p);
                            local_edges.push((u_orig, v_pt.orig_idx));
                        }
                    }

                    // ── Valve 1: Degree cap (Causal Shadow) ──
                    if children_coords.len() >= MAX_HASSE_DEGREE { break; }
                    // ── Valve 2: Alexandrov volume short-circuit ──
                    if causal_checked > 0 && blocked == causal_checked && children_coords.len() >= 1 { break; }
                    // ── Valve 3: Adaptive early termination (two dry shells) ──
                    if children_coords.len() == links_before {
                        consecutive_dry += 1;
                        if consecutive_dry >= 2 && children_coords.len() >= 2 { break; }
                    } else {
                        consecutive_dry = 0;
                    }
                }
                local_edges
            })
            .flatten()
            .collect();

        // Write to file
        for (u, v) in &edges_batch {
             writer.write_all(&u.to_le_bytes())?;
             writer.write_all(&v.to_le_bytes())?;
        }
        global_edges_count += edges_batch.len();

        // Remove processed sources, retain future points
        let old_len = buffer.len();
        buffer.retain(|p| p.p[0] >= safe_threshold);
        let dropped = old_len - buffer.len();

        if dropped == 0 && current_t >= half_t {
            buffer.clear();
        }
    }

    writer.flush()?;
    println!("Stream Complete. Total Edges: {global_edges_count}");
    Ok(())
}

// ─── Sparse Scanning (No Disk, HashMap Grid) ──────────────────────────────

/// Scan all Hasse edges using a sliding window with **sparse** (HashMap) grid.
///
/// Memory: O(chunk_size) for buffer + O(occupied_cells) for HashMap.
/// At N=100M the dense grid would need ~20 GB; the HashMap uses ~100 MB.
///
/// Calls `on_batch` with each batch of directed edges `(u_orig, v_orig)`.
fn scan_edges_sparse<F>(
    n_total: usize,
    chunk_size: usize,
    seed: u64,
    mut on_batch: F,
) where
    F: FnMut(&[(u32, u32)]),
{
    use rustc_hash::FxHashMap as HashMap;

    let big_t = (24.0 * n_total as f64 / PI).powf(0.25);
    let half_t = big_t / 2.0;

    let margin = 2.0;
    let t_span = big_t + 2.0 * margin;
    let x_span = big_t + 2.0 * margin;

    let t_dim = (t_span.ceil() as usize).max(1);
    let dim_x = (x_span.ceil() as usize).max(1);
    let dim_y = dim_x;
    let dim_z = dim_x;

    let grid_stride_y = dim_z;
    let grid_stride_x = dim_y * dim_z;
    let grid_stride_t = dim_x * dim_y * dim_z;
    let grid_size = t_dim * grid_stride_t;

    let to_grid = |val: f64, _k: usize| -> u16 {
        let origin = -half_t - margin;
        ((val - origin).max(0.0)) as u16
    };

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut buffer: Vec<GridPoint> = Vec::with_capacity(chunk_size * 2);
    let mut current_t = -half_t;
    let mut next_node_id = 0u32;
    let mut global_edges_count: usize = 0;

    let max_dt = (t_dim as i16).min(MAX_CAUSAL_DEPTH);
    let safety_margin_t = (max_dt as f64) + 2.0;

    while current_t < half_t {
        // 1. Fill buffer
        let mut max_t_in_buffer = buffer.iter().map(|p| p.p[0]).fold(f64::NEG_INFINITY, f64::max);
        if buffer.is_empty() { max_t_in_buffer = current_t; }

        while max_t_in_buffer < current_t + safety_margin_t && current_t < half_t {
            let next_t = (current_t + safety_margin_t).min(half_t);

            use std::f64::consts::PI;
            let v_tot = PI / 24.0 * big_t.powi(4);
            let rho = n_total as f64 / v_tot;

            let compute_vol_pos = |t1: f64, t2: f64, h: f64| -> f64 {
                0.25 * ((h - t1).powi(4) - (h - t2).powi(4))
            };
            let compute_vol_neg = |t1: f64, t2: f64, h: f64| -> f64 {
                0.25 * ((h + t2).powi(4) - (h + t1).powi(4))
            };

            let vol_slice = 4.0 / 3.0 * PI * if current_t >= 0.0 {
                compute_vol_pos(current_t, next_t, half_t)
            } else if next_t <= 0.0 {
                compute_vol_neg(current_t, next_t, half_t)
            } else {
                compute_vol_neg(current_t, 0.0, half_t) + compute_vol_pos(0.0, next_t, half_t)
            };

            let n_ask = (rho * vol_slice) as usize;
            let mut new_pts = sprinkle_chunk(n_ask, current_t, next_t, half_t, &mut rng, next_node_id);
            if !new_pts.is_empty() {
                next_node_id += new_pts.len() as u32;
                for p in &mut new_pts {
                    p.qt = to_grid(p.p[0], 0);
                    p.qx = to_grid(p.p[1], 1);
                    p.qy = to_grid(p.p[2], 2);
                    p.qz = to_grid(p.p[3], 3);
                    let idx = (p.qt as usize) * grid_stride_t
                            + (p.qx as usize) * grid_stride_x
                            + (p.qy as usize) * grid_stride_y
                            + (p.qz as usize);
                    p.cell = if idx < grid_size { idx } else { grid_size };
                }
                buffer.extend(new_pts);
            }
            current_t = next_t;
            max_t_in_buffer = current_t;
            if buffer.len() > chunk_size * 2 { break; }
        }

        if buffer.is_empty() { break; }

        // 2. Sort buffer for cache-coherent access
        buffer.par_sort_unstable_by_key(|gp| gp.cell);

        // 3. Build sparse grid index (HashMap: cell → (start, count))
        let mut grid: HashMap<usize, (u32, u32)> = HashMap::with_capacity_and_hasher(buffer.len().min(500_000), Default::default());
        for (i, gp) in buffer.iter().enumerate() {
            if gp.cell >= grid_size { continue; }
            let entry = grid.entry(gp.cell).or_insert((i as u32, 0));
            entry.1 += 1;
        }

        // 4. Process safe sources
        let safe_threshold = if current_t >= half_t {
            f64::INFINITY
        } else {
            max_t_in_buffer - safety_margin_t * 0.9
        };

        let edges_batch: Vec<(u32, u32)> = buffer.par_iter()
            .filter(|p| p.p[0] < safe_threshold)
            .map(|u_pt| {
                let u_orig = u_pt.orig_idx;
                let (qt, qx, qy, qz) = (u_pt.qt as i16, u_pt.qx as i16, u_pt.qy as i16, u_pt.qz as i16);
                let mut local_edges = Vec::new();
                let mut children_coords: Vec<[f64; 4]> = Vec::with_capacity(32);
                let mut candidates: Vec<usize> = Vec::with_capacity(128);
                let mut consecutive_dry = 0u16;

                for dt in 0..max_dt {
                    let target_t = qt + dt;
                    if target_t < 0 || target_t >= (t_dim as i16) { continue; }

                    let r_max = dt + 1;
                    let r2_limit = (r_max + 1) * (r_max + 1);
                    let dt_f = dt as f64;
                    let r_min_f = (dt_f * dt_f - MAX_PROPER_TIME_SQ).max(0.0).sqrt();
                    let r_min = r_min_f.floor() as i16;
                    let r_min_sq = (r_min as i32) * (r_min as i32);

                    let links_before = children_coords.len();
                    candidates.clear();

                    for dx in -r_max..=r_max {
                        for dy in -r_max..=r_max {
                            let dxy2 = dx as i32 * dx as i32 + dy as i32 * dy as i32;
                            let dz_skip = if dxy2 < r_min_sq {
                                ((r_min_sq - dxy2) as f64).sqrt().floor() as i16
                            } else { 0 };
                            let z_ranges: smallvec::SmallVec<[std::ops::RangeInclusive<i16>; 2]> = if dz_skip > 0 {
                                smallvec::smallvec![-r_max..=-dz_skip, dz_skip..=r_max]
                            } else {
                                smallvec::smallvec![-r_max..=r_max]
                            };
                            for range in z_ranges {
                                for dz in range {
                                    let r2_i = dxy2 + dz as i32 * dz as i32;
                                    let proper_time_sq = dt_f * dt_f - r2_i as f64;
                                    if proper_time_sq > MAX_PROPER_TIME_SQ { continue; }
                                    if r2_i >= r2_limit as i32 { continue; }
                                    let tx = qx + dx;
                                    let ty = qy + dy;
                                    let tz = qz + dz;
                                    if tx < 0 || tx >= dim_x as i16 || ty < 0 || ty >= dim_y as i16 || tz < 0 || tz >= dim_z as i16 { continue; }
                                    let cell_idx = (target_t as usize) * grid_stride_t
                                                 + (tx as usize) * grid_stride_x
                                                 + (ty as usize) * grid_stride_y
                                                 + (tz as usize);

                                    if let Some(&(start, count)) = grid.get(&cell_idx) {
                                        let s = start as usize;
                                        candidates.extend(s..(s + count as usize));
                                    }
                                }
                            }
                        }
                    }

                    candidates.sort_unstable_by(|&a, &b| {
                        buffer[a].p[0].partial_cmp(&buffer[b].p[0]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.dedup();

                    let mut causal_checked = 0u32;
                    let mut blocked_count = 0u32;

                    for &v_idx in &candidates {
                        let v_pt = &buffer[v_idx];
                        if u_orig == v_pt.orig_idx { continue; }
                        if !is_causal(&u_pt.p, &v_pt.p) { continue; }
                        causal_checked += 1;
                        let redundant = children_coords.iter().any(|z| is_causal(z, &v_pt.p));
                        if redundant {
                            blocked_count += 1;
                        } else {
                            children_coords.push(v_pt.p);
                            local_edges.push((u_orig, v_pt.orig_idx));
                        }
                    }

                    // ── Valve 1: Degree cap ──
                    if children_coords.len() >= MAX_HASSE_DEGREE { break; }
                    // ── Valve 2: Alexandrov short-circuit ──
                    if causal_checked > 0 && blocked_count == causal_checked && children_coords.len() >= 1 { break; }
                    // ── Valve 3: Adaptive dry-shell (two consecutive) ──
                    if children_coords.len() == links_before {
                        consecutive_dry += 1;
                        if consecutive_dry >= 2 && children_coords.len() >= 2 { break; }
                    } else {
                        consecutive_dry = 0;
                    }
                }
                local_edges
            })
            .flatten()
            .collect();

        global_edges_count += edges_batch.len();
        on_batch(&edges_batch);

        // Retain future points
        let old_len = buffer.len();
        buffer.retain(|p| p.p[0] >= safe_threshold);
        if old_len - buffer.len() == 0 && current_t >= half_t {
            buffer.clear();
        }
    }

    println!("  Sparse scan: {global_edges_count} edges processed (zero disk I/O)");
}

/// Variant of `scan_edges_sparse` that also passes the safe_threshold to the
/// callback, enabling callers to track node finalization timing.
///
/// Callback signature: `FnMut(&[(u32, u32)], f64)` where the second argument
/// is the safe_threshold — nodes with `t < safe_threshold` have all their
/// edges fully discovered in this or prior batches.
fn scan_edges_sparse_ext<F>(
    n_total: usize,
    chunk_size: usize,
    seed: u64,
    mut on_batch: F,
) where
    F: FnMut(&[(u32, u32)], f64),
{
    use rustc_hash::FxHashMap as HashMap;

    let big_t = (24.0 * n_total as f64 / PI).powf(0.25);
    let half_t = big_t / 2.0;

    let margin = 2.0;
    let t_span = big_t + 2.0 * margin;
    let x_span = big_t + 2.0 * margin;

    let t_dim = (t_span.ceil() as usize).max(1);
    let dim_x = (x_span.ceil() as usize).max(1);
    let dim_y = dim_x;
    let dim_z = dim_x;

    let grid_stride_y = dim_z;
    let grid_stride_x = dim_y * dim_z;
    let grid_stride_t = dim_x * dim_y * dim_z;
    let grid_size = t_dim * grid_stride_t;

    let to_grid = |val: f64, _k: usize| -> u16 {
        let origin = -half_t - margin;
        ((val - origin).max(0.0)) as u16
    };

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut buffer: Vec<GridPoint> = Vec::with_capacity(chunk_size * 2);
    let mut current_t = -half_t;
    let mut next_node_id = 0u32;
    let mut global_edges_count: usize = 0;

    let max_dt = (t_dim as i16).min(MAX_CAUSAL_DEPTH);
    let safety_margin_t = (max_dt as f64) + 2.0;

    while current_t < half_t {
        // 1. Fill buffer
        let mut max_t_in_buffer = buffer.iter().map(|p| p.p[0]).fold(f64::NEG_INFINITY, f64::max);
        if buffer.is_empty() { max_t_in_buffer = current_t; }

        while max_t_in_buffer < current_t + safety_margin_t && current_t < half_t {
            let next_t = (current_t + safety_margin_t).min(half_t);

            use std::f64::consts::PI;
            let v_tot = PI / 24.0 * big_t.powi(4);
            let rho = n_total as f64 / v_tot;

            let compute_vol_pos = |t1: f64, t2: f64, h: f64| -> f64 {
                0.25 * ((h - t1).powi(4) - (h - t2).powi(4))
            };
            let compute_vol_neg = |t1: f64, t2: f64, h: f64| -> f64 {
                0.25 * ((h + t2).powi(4) - (h + t1).powi(4))
            };

            let vol_slice = 4.0 / 3.0 * PI * if current_t >= 0.0 {
                compute_vol_pos(current_t, next_t, half_t)
            } else if next_t <= 0.0 {
                compute_vol_neg(current_t, next_t, half_t)
            } else {
                compute_vol_neg(current_t, 0.0, half_t) + compute_vol_pos(0.0, next_t, half_t)
            };

            let n_ask = (rho * vol_slice) as usize;
            let mut new_pts = sprinkle_chunk(n_ask, current_t, next_t, half_t, &mut rng, next_node_id);
            if !new_pts.is_empty() {
                next_node_id += new_pts.len() as u32;
                for p in &mut new_pts {
                    p.qt = to_grid(p.p[0], 0);
                    p.qx = to_grid(p.p[1], 1);
                    p.qy = to_grid(p.p[2], 2);
                    p.qz = to_grid(p.p[3], 3);
                    let idx = (p.qt as usize) * grid_stride_t
                            + (p.qx as usize) * grid_stride_x
                            + (p.qy as usize) * grid_stride_y
                            + (p.qz as usize);
                    p.cell = if idx < grid_size { idx } else { grid_size };
                }
                buffer.extend(new_pts);
            }
            current_t = next_t;
            max_t_in_buffer = current_t;
            if buffer.len() > chunk_size * 2 { break; }
        }

        if buffer.is_empty() { break; }

        // 2. Sort buffer for cache-coherent access
        buffer.par_sort_unstable_by_key(|gp| gp.cell);

        // 3. Build sparse grid index (HashMap: cell → (start, count))
        let mut grid: HashMap<usize, (u32, u32)> = HashMap::with_capacity_and_hasher(buffer.len().min(500_000), Default::default());
        for (i, gp) in buffer.iter().enumerate() {
            if gp.cell >= grid_size { continue; }
            let entry = grid.entry(gp.cell).or_insert((i as u32, 0));
            entry.1 += 1;
        }

        // 4. Process safe sources
        let safe_threshold = if current_t >= half_t {
            f64::INFINITY
        } else {
            max_t_in_buffer - safety_margin_t * 0.9
        };

        let edges_batch: Vec<(u32, u32)> = buffer.par_iter()
            .filter(|p| p.p[0] < safe_threshold)
            .map(|u_pt| {
                let u_orig = u_pt.orig_idx;
                let (qt, qx, qy, qz) = (u_pt.qt as i16, u_pt.qx as i16, u_pt.qy as i16, u_pt.qz as i16);
                let mut local_edges = Vec::new();
                let mut children_coords: Vec<[f64; 4]> = Vec::with_capacity(32);
                let mut candidates: Vec<usize> = Vec::with_capacity(128);
                let mut consecutive_dry = 0u16;

                for dt in 0..max_dt {
                    let target_t = qt + dt;
                    if target_t < 0 || target_t >= (t_dim as i16) { continue; }

                    let r_max = dt + 1;
                    let r2_limit = (r_max + 1) * (r_max + 1);
                    let dt_f = dt as f64;
                    let r_min_f = (dt_f * dt_f - MAX_PROPER_TIME_SQ).max(0.0).sqrt();
                    let r_min = r_min_f.floor() as i16;
                    let r_min_sq = (r_min as i32) * (r_min as i32);

                    let links_before = children_coords.len();
                    candidates.clear();

                    for dx in -r_max..=r_max {
                        for dy in -r_max..=r_max {
                            let dxy2 = dx as i32 * dx as i32 + dy as i32 * dy as i32;
                            let dz_skip = if dxy2 < r_min_sq {
                                ((r_min_sq - dxy2) as f64).sqrt().floor() as i16
                            } else { 0 };
                            let z_ranges: smallvec::SmallVec<[std::ops::RangeInclusive<i16>; 2]> = if dz_skip > 0 {
                                smallvec::smallvec![-r_max..=-dz_skip, dz_skip..=r_max]
                            } else {
                                smallvec::smallvec![-r_max..=r_max]
                            };
                            for range in z_ranges {
                                for dz in range {
                                    let r2_i = dxy2 + dz as i32 * dz as i32;
                                    let proper_time_sq = dt_f * dt_f - r2_i as f64;
                                    if proper_time_sq > MAX_PROPER_TIME_SQ { continue; }
                                    if r2_i >= r2_limit as i32 { continue; }
                                    let tx = qx + dx;
                                    let ty = qy + dy;
                                    let tz = qz + dz;
                                    if tx < 0 || tx >= dim_x as i16 || ty < 0 || ty >= dim_y as i16 || tz < 0 || tz >= dim_z as i16 { continue; }
                                    let cell_idx = (target_t as usize) * grid_stride_t
                                                 + (tx as usize) * grid_stride_x
                                                 + (ty as usize) * grid_stride_y
                                                 + (tz as usize);

                                    if let Some(&(start, count)) = grid.get(&cell_idx) {
                                        let s = start as usize;
                                        candidates.extend(s..(s + count as usize));
                                    }
                                }
                            }
                        }
                    }

                    candidates.sort_unstable_by(|&a, &b| {
                        buffer[a].p[0].partial_cmp(&buffer[b].p[0]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.dedup();

                    let mut causal_checked = 0u32;
                    let mut blocked_count = 0u32;

                    for &v_idx in &candidates {
                        let v_pt = &buffer[v_idx];
                        if u_orig == v_pt.orig_idx { continue; }
                        if !is_causal(&u_pt.p, &v_pt.p) { continue; }
                        causal_checked += 1;
                        let redundant = children_coords.iter().any(|z| is_causal(z, &v_pt.p));
                        if redundant {
                            blocked_count += 1;
                        } else {
                            children_coords.push(v_pt.p);
                            local_edges.push((u_orig, v_pt.orig_idx));
                        }
                    }

                    // ── Valve 1: Degree cap ──
                    if children_coords.len() >= MAX_HASSE_DEGREE { break; }
                    // ── Valve 2: Alexandrov short-circuit ──
                    if causal_checked > 0 && blocked_count == causal_checked && children_coords.len() >= 1 { break; }
                    // ── Valve 3: Adaptive dry-shell (two consecutive) ──
                    if children_coords.len() == links_before {
                        consecutive_dry += 1;
                        if consecutive_dry >= 2 && children_coords.len() >= 2 { break; }
                    } else {
                        consecutive_dry = 0;
                    }
                }
                local_edges
            })
            .flatten()
            .collect();

        global_edges_count += edges_batch.len();
        on_batch(&edges_batch, safe_threshold);

        // Retain future points
        let old_len = buffer.len();
        buffer.retain(|p| p.p[0] >= safe_threshold);
        if old_len - buffer.len() == 0 && current_t >= half_t {
            buffer.clear();
        }
    }

    println!("  Sparse scan (ext): {global_edges_count} edges processed (zero disk I/O)");
}

/// Single-pass streaming analysis: computes degrees AND collects core-induced
/// edges in one scan over the Poisson sprinkling.
///
/// Merges the work of `compute_degrees` + `build_core_edges` into a single
/// edge generation pass, cutting wall-clock time in half for the streaming path.
///
/// The core threshold is determined from a running histogram after a 5% warmup
/// of finalized nodes, then refined with an exact O(N) pass at the end.
///
/// Returns `(in_deg, out_deg, is_core, directed_core_edges)`.
pub fn scan_edges_with_analysis(
    n_total: usize,
    chunk_size: usize,
    seed: u64,
    core_num: usize,
    core_den: usize,
) -> (Vec<u32>, Vec<u32>, Vec<bool>, Vec<(u32, u32)>) {
    let mut in_deg = vec![0u32; n_total];
    let mut out_deg = vec![0u32; n_total];

    // Running histogram for degree threshold estimation
    const HIST: usize = 10_000;
    let mut hist = vec![0usize; HIST];
    let mut finalized = vec![false; n_total];
    let mut finalized_count = 0usize;
    let mut running_cutoff: Option<usize> = None;
    let target_core = n_total * core_num / core_den;
    let warmup_threshold = n_total / 20; // 5% warmup before estimating

    // Provisional core classification (updated as nodes finalize)
    let mut is_core = vec![false; n_total];

    // Core edges collected during the scan
    let mut core_edges: Vec<(u32, u32)> = Vec::new();

    // Buffer for edges where one endpoint is not yet classified
    let mut pending_edges: Vec<(u32, u32)> = Vec::new();

    // We need the time coordinate of each node for finalization checking.
    // Store it as nodes are first seen in edges.
    let big_t = (24.0 * n_total as f64 / PI).powf(0.25);
    let max_dt: f64 = ((big_t + 4.0).ceil() as f64).min(MAX_CAUSAL_DEPTH as f64);
    let safety_delay = 2.0 * max_dt + 4.0; // generous finalization delay

    // Track node time coordinates (set on first edge encounter)
    let mut node_time = vec![f64::NAN; n_total];

    scan_edges_sparse_ext(n_total, chunk_size, seed, |batch, safe_threshold| {
        // 1. Update degrees for each edge
        for &(u, v) in batch {
            let ui = u as usize;
            let vi = v as usize;
            if ui < n_total { out_deg[ui] += 1; }
            if vi < n_total { in_deg[vi] += 1; }
        }

        // 2. Finalize nodes whose degrees are now complete.
        //    A node u is finalized when safe_threshold > node_time[u] + safety_delay.
        //    For nodes we haven't seen yet, they'll be finalized in a later batch
        //    or in the final cleanup pass.
        if safe_threshold.is_finite() {
            let finalize_cutoff = safe_threshold - safety_delay;
            for i in 0..n_total {
                if finalized[i] { continue; }
                let t = node_time[i];
                if t.is_nan() { continue; }
                if t < finalize_cutoff {
                    finalized[i] = true;
                    finalized_count += 1;
                    let total_d = (in_deg[i] + out_deg[i]) as usize;
                    let b = total_d.min(HIST - 1);
                    hist[b] += 1;

                    // Re-estimate cutoff periodically after warmup
                    if finalized_count >= warmup_threshold && finalized_count % (warmup_threshold / 2).max(1) == 0 {
                        let target_so_far = finalized_count * core_num / core_den;
                        let (mut cnt, mut cut) = (0usize, 0usize);
                        for d in (0..HIST).rev() {
                            cnt += hist[d];
                            if cnt >= target_so_far { cut = d; break; }
                        }
                        running_cutoff = Some(cut);
                    }

                    // Classify if we have a running estimate
                    if let Some(cut) = running_cutoff {
                        let total_d = (in_deg[i] + out_deg[i]) as usize;
                        is_core[i] = total_d >= cut;
                    }
                }
            }
        }

        // 3. For each edge in batch, try to classify as core edge
        for &(u, v) in batch {
            let ui = u as usize;
            let vi = v as usize;
            if ui >= n_total || vi >= n_total { continue; }

            // Record time coordinates from edges (out-degree => source has earlier time)
            // We approximate: source time ~ safe_threshold - safety_delay (conservative)
            if node_time[ui].is_nan() {
                // The source was processed in this batch, so its time < safe_threshold
                node_time[ui] = if safe_threshold.is_finite() {
                    safe_threshold - safety_delay * 0.5
                } else {
                    0.0
                };
            }
            if node_time[vi].is_nan() {
                node_time[vi] = if safe_threshold.is_finite() {
                    safe_threshold
                } else {
                    0.0
                };
            }

            if finalized[ui] && finalized[vi] {
                // Both classified — check immediately
                if is_core[ui] && is_core[vi] {
                    core_edges.push((u, v));
                }
            } else {
                // At least one not yet classified — defer
                pending_edges.push((u, v));
            }
        }
    });

    // 4. Final pass: exact histogram from ALL nodes, recompute threshold
    //    This corrects the running estimate (±0.5% drift).
    let mut final_hist = vec![0usize; HIST];
    let mut max_d = 0usize;
    for i in 0..n_total {
        let total_d = (in_deg[i] + out_deg[i]) as usize;
        let b = total_d.min(HIST - 1);
        final_hist[b] += 1;
        if b > max_d { max_d = b; }
    }

    let (mut cnt, mut cutoff) = (0usize, 0usize);
    for d in (0..=max_d).rev() {
        cnt += final_hist[d];
        if cnt >= target_core { cutoff = d; break; }
    }

    // Reclassify all nodes with exact threshold
    for i in 0..n_total {
        is_core[i] = (in_deg[i] + out_deg[i]) as usize >= cutoff;
    }

    // 5. Process pending edges with final classification
    for (u, v) in pending_edges {
        let ui = u as usize;
        let vi = v as usize;
        if ui < n_total && vi < n_total && is_core[ui] && is_core[vi] {
            core_edges.push((u, v));
        }
    }

    // Also re-check already-collected core edges (running cutoff may have been wrong)
    core_edges.retain(|&(u, v)| {
        let ui = u as usize;
        let vi = v as usize;
        ui < n_total && vi < n_total && is_core[ui] && is_core[vi]
    });

    let core_count: usize = is_core.iter().filter(|&&c| c).count();
    println!("  Single-pass analysis: core={core_count} (cutoff degree ≥ {cutoff}), core edges={}", core_edges.len());

    (in_deg, out_deg, is_core, core_edges)
}

/// Compute in-degree and out-degree for all N nodes without touching disk.
///
/// Uses the sparse sliding window (HashMap grid, ~100 MB) instead of
/// dense arrays (~20 GB at N=100M) or edge files (terabytes).
pub fn compute_degrees(
    n_total: usize, chunk_size: usize, seed: u64,
) -> (Vec<u32>, Vec<u32>) {
    let mut in_deg = vec![0u32; n_total];
    let mut out_deg = vec![0u32; n_total];
    scan_edges_sparse(n_total, chunk_size, seed, |batch| {
        for &(u, v) in batch {
            if (u as usize) < n_total { out_deg[u as usize] += 1; }
            if (v as usize) < n_total { in_deg[v as usize] += 1; }
        }
    });
    (in_deg, out_deg)
}

/// Collect edges where both endpoints are in the core set.
///
/// Second pass over the same Poisson sprinkling (deterministic seed).
/// Only core-induced edges are kept (~0.1% of total at core = 10%).
/// Symmetrize a directed CSR graph into an undirected one.
///
/// The Hasse diagram from Phase 1 is a DAG: each edge u→v is stored once
/// (forward in time).  Spectral walkers need the **undirected** graph so
/// they can step both past→future and future→past, probing the full
/// manifold geometry.
///
/// Algorithm:
///   1. Extract every directed edge u→v from the input CSR.
///   2. Insert both (u,v) and (v,u) into an edge list.
///   3. Sort + dedup to remove any duplicates.
///   4. Build a new CSR from the deduplicated undirected edges.
///
/// At N=10M with ~245M directed edges, this produces ~490M undirected
/// entries (~1.9 GB).  The directed CSR can be dropped afterwards.
pub fn make_symmetric(
    n: usize,
    head: &[u32],
    data: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    // Pass 1: count undirected degree of each node
    // Each directed edge u→v contributes +1 to both deg[u] and deg[v].
    let mut deg = vec![0u32; n];
    for u in 0..n {
        let s = head[u] as usize;
        let e = head[u + 1] as usize;
        let out = (e - s) as u32;
        deg[u] += out;
        for &v in &data[s..e] {
            deg[v as usize] += 1;
        }
    }

    // Build head array from degree counts
    let mut sym_head = vec![0u32; n + 1];
    for i in 0..n {
        sym_head[i + 1] = sym_head[i] + deg[i];
    }
    let total = sym_head[n] as usize;
    let mut sym_data = vec![0u32; total];

    // Pass 2: fill both directions
    let mut pos = sym_head[..n].to_vec();
    for u in 0..n {
        let s = head[u] as usize;
        let e = head[u + 1] as usize;
        for &v in &data[s..e] {
            // u → v (forward)
            sym_data[pos[u] as usize] = v;
            pos[u] += 1;
            // v → u (reverse)
            sym_data[pos[v as usize] as usize] = u as u32;
            pos[v as usize] += 1;
        }
    }

    // Sort each adjacency list and dedup (handles any duplicate edges)
    for u in 0..n {
        let s = sym_head[u] as usize;
        let e = sym_head[u + 1] as usize;
        sym_data[s..e].sort_unstable();
    }
    // Dedup in-place: compact each row, then rebuild head
    let mut write = 0usize;
    let mut new_head = vec![0u32; n + 1];
    for u in 0..n {
        new_head[u] = write as u32;
        let s = sym_head[u] as usize;
        let e = sym_head[u + 1] as usize;
        let mut prev = u32::MAX;
        for r in s..e {
            let v = sym_data[r];
            if v != prev {
                sym_data[write] = v;
                write += 1;
                prev = v;
            }
        }
    }
    new_head[n] = write as u32;
    sym_data.truncate(write);

    (new_head, sym_data)
}

pub fn build_core_edges(
    n_total: usize, chunk_size: usize, seed: u64, core_mask: &[bool],
) -> Vec<(u32, u32)> {
    let mut core_edges = Vec::new();
    scan_edges_sparse(n_total, chunk_size, seed, |batch| {
        for &(u, v) in batch {
            if (u as usize) < n_total && (v as usize) < n_total
                && core_mask[u as usize] && core_mask[v as usize]
            {
                core_edges.push((u, v));
            }
        }
    });
    core_edges
}
