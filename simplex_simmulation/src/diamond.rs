//! Phase 1 — Vacuum Generation
//!
//! Poisson sprinkling in a 4D causal diamond, causal edge discovery,
//! and Hasse diagram construction (transitive reduction).
//!
//! Two tiers:
//!   - `build_hasse_sparse`: full closure → sparse A² reduction  (N ≤ 15k)
//!   - `build_hasse_direct`: geometric incremental construction  (N > 15k)

use rand::Rng;
use rayon::prelude::*;
use sprs::TriMat;

/// Poisson-sprinkle `n` points into a 4D causal diamond |t| + r ≤ T/2.
///
/// Volume V = T⁴/24;  T = (24N)^{1/4}  gives density ρ ≈ 1.
pub fn sprinkle(n: usize, rng: &mut impl Rng) -> (Vec<[f64; 4]>, f64) {
    let big_t = (24.0 * n as f64).powf(0.25);
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
/// For a transitively-closed DAG, edge (i,j) is a link iff A²[i,j] = 0
/// (no 2-hop path exists).
pub fn build_hasse_sparse(pts: &[[f64; 4]]) -> (Vec<u32>, Vec<u32>) {
    let n = pts.len();
    let order = time_order(pts);

    // Build all causal edges (parallel over source nodes)
    let edge_groups: Vec<Vec<(u32, u32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let pi = &pts[order[i]];
            let mut local = Vec::new();
            for j in (i + 1)..n {
                if is_causal(pi, &pts[order[j]]) {
                    local.push((order[i] as u32, order[j] as u32));
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
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for &(r, c) in &edges {
        let val = a2.get(r as usize, c as usize).copied().unwrap_or(0.0);
        if val == 0.0 {
            rows.push(r);
            cols.push(c);
        }
    }
    (rows, cols)
}

// ─── Tier 2: direct geometric construction (N > 15k) ────────────────────────

/// Build Hasse diagram directly without materialising the full closure.
///
/// For each source node i (in time order), iterate over future nodes j.
/// Edge (i,j) is a link iff no existing link-child z of i satisfies z ≺ j.
///
/// Correctness: if any intermediate k with i ≺ k ≺ j exists, then some
/// link-child z of i has z ≼ k (possibly z = k), hence z ≺ j, and we
/// detect the redundancy.
pub fn build_hasse_direct(pts: &[[f64; 4]]) -> (Vec<u32>, Vec<u32>) {
    let n = pts.len();
    let order = time_order(pts);

    // Parallel over source nodes — each task independently discovers its links
    let edge_groups: Vec<Vec<(u32, u32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let pi = &pts[order[i]];
            // Sorted indices of link-children found so far for this source
            let mut children_sorted: Vec<usize> = Vec::new();
            let mut local_edges = Vec::new();

            for j in (i + 1)..n {
                let pj = &pts[order[j]];
                if !is_causal(pi, pj) {
                    continue;
                }

                // Is j reachable from any existing link-child of i?
                let redundant = children_sorted.iter().any(|&z| {
                    is_causal(&pts[order[z]], pj)
                });

                if !redundant {
                    children_sorted.push(j);
                    local_edges.push((order[i] as u32, order[j] as u32));
                }
            }
            local_edges
        })
        .collect();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for group in edge_groups {
        for (r, c) in group {
            rows.push(r);
            cols.push(c);
        }
    }
    (rows, cols)
}
