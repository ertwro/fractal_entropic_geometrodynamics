//! Benincasa–Dowker operator construction and RMT spacing ratio.
//!
//! Builds the 4D retarded BD matrix from the directed Hasse CSR,
//! extracts eigenvalues of the effective Hamiltonian
//!
//!     H_eff = (B − Bᵀ) / (2i)
//!
//! (Hermitian, real eigenvalues), and computes the spacing ratio ⟨r⟩
//! diagnostic for GUE universality.
//!
//! ## Why Schur, not LAPACK?
//!
//! The matrix D = B − Bᵀ is real skew-symmetric.  Its eigenvalues are
//! purely imaginary: ±iσ_k.  The eigenvalues of H_eff are then ±σ_k/2
//! (real).  We extract these via nalgebra's Schur decomposition, which
//! handles general real matrices and returns complex eigenvalues.  This
//! avoids any new LAPACK/BLAS dependency while reusing the nalgebra
//! crate already linked for the spectral eigendecomposition in Phase 3.
//!
//! For N ≤ 3000 the Schur decomposition completes in O(N³) time,
//! matching the eigendecomp tier already used in `spectral.rs`.

use nalgebra::DMatrix;
use rustc_hash::FxHashMap;

/// BD layer weight for the 4D retracted d'Alembertian (integer-exact).
///
/// depth 0 (direct Hasse link)           = +1
/// depth 1 (shortest 2-hop Hasse path)   = −9
/// depth 2 (shortest 3-hop Hasse path)   = +16
/// depth 3 (shortest 4-hop Hasse path)   = −8
#[inline(always)]
fn bd_weight(depth: u8) -> i64 {
    match depth {
        0 => 1,
        1 => -9,
        2 => 16,
        3 => -8,
        _ => 0,
    }
}

/// Maximum BFS depth (= number of BD layers).
const MAX_DEPTH: usize = 4;

/// Build the N×N Benincasa–Dowker matrix from the directed Hasse CSR.
///
/// Performs a layered BFS from every node through the **directed**
/// Hasse DAG (forward edges only: `v > node`).  Each target is
/// assigned the BD weight corresponding to its shortest forward
/// Hasse-path distance from the source.
///
/// Returns `DMatrix<i64>` — integer-exact; the single f64 conversion
/// is deferred to the Schur eigensolver boundary in
/// [`effective_hamiltonian_eigenvalues`].
///
/// The result is strictly upper triangular (causal order is a strict
/// partial order with time-sorted indices).
pub fn build_bd_matrix(
    adj_head: &[u32],
    adj_data: &[u32],
    n: usize,
) -> DMatrix<i64> {
    let mut b = DMatrix::<i64>::zeros(n, n);
    let mut visited = vec![false; n];

    for src in 0..n {
        // Reset visited for this source
        for v in visited.iter_mut() {
            *v = false;
        }
        visited[src] = true;

        let mut frontier: Vec<usize> = vec![src];

        for depth in 0..MAX_DEPTH {
            let mut next: Vec<usize> = Vec::new();
            for &node in &frontier {
                let lo = adj_head[node] as usize;
                let hi = adj_head[node + 1] as usize;
                for &nbr_u32 in &adj_data[lo..hi] {
                    let nbr = nbr_u32 as usize;
                    // Forward edge: nbr in causal future of node
                    if nbr > node && !visited[nbr] {
                        visited[nbr] = true;
                        next.push(nbr);
                        b[(src, nbr)] = bd_weight(depth as u8);
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
    }
    b
}

/// Compute eigenvalues of H_eff = (B − Bᵀ)/(2i).
///
/// 1. D = B − Bᵀ  (integer antisymmetrization, still `i64`)
/// 2. D → f64      (**measurement boundary**: single exact conversion)
/// 3. Schur(D) → complex eigenvalues λ_k = iσ_k  (purely imaginary)
/// 4. Eigenvalue of H_eff = Im(λ_k) / 2
///
/// Returns a **sorted** vector of real eigenvalues with the near-zero
/// kernel (|h| < 1e-8) removed.  Consumes `b` to save memory.
pub fn effective_hamiltonian_eigenvalues(b: DMatrix<i64>) -> Vec<f64> {
    // Integer antisymmetrization (still i64)
    let n = b.nrows();
    let mut d_int = DMatrix::<i64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            d_int[(i, j)] = b[(i, j)] - b[(j, i)];
        }
    }
    drop(b);

    // ── MEASUREMENT BOUNDARY: i64 → f64 ──
    let d = d_int.map(|x| x as f64);
    drop(d_int);

    let schur = nalgebra::linalg::Schur::new(d);
    let complex_evals = schur.complex_eigenvalues();

    let mut evals: Vec<f64> = complex_evals
        .iter()
        .map(|c| c.im / 2.0)
        .filter(|&v| v.abs() > 1e-8)
        .collect();

    evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    evals
}

/// Compute spacing ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).
///
/// Input: sorted eigenvalues.  Consecutive spacings s_n = E_{n+1} − E_n
/// are computed, degenerate spacings (< 1e-12) discarded, then the
/// ratio of adjacent spacings is formed.
///
/// Returns the vector of individual ratios (empty if fewer than 3
/// non-degenerate spacings).
pub fn spacing_ratios(sorted_evals: &[f64]) -> Vec<f64> {
    let spacings: Vec<f64> = sorted_evals
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&s| s > 1e-12)
        .collect();

    spacings
        .windows(2)
        .map(|w| w[0].min(w[1]) / w[0].max(w[1]))
        .collect()
}

/// Mean and standard error of a sample.
pub fn mean_se(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    if n < 1.0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = data.iter().sum::<f64>() / n;
    if n < 2.0 {
        return (mean, f64::NAN);
    }
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt() / n.sqrt())
}

// ═══════════════════════════════════════════════════════════════════════
// Coarse-graining pipeline: voxelize → collapse → directed BD
// ═══════════════════════════════════════════════════════════════════════

/// Assign each micro-event to a voxel, preserving time ordering.
///
/// Computes a 4D bounding box, derives voxel edge length
/// `ℓ = (volume / n_target)^{1/4}`, bins each point into a
/// `(i32,i32,i32,i32)` voxel key, then sorts voxel keys by
/// time-first tuple order and remaps IDs so that earlier-time
/// voxels get lower IDs.
///
/// Returns `(micro_to_macro, n_macro)`.
pub fn voxelize(pts: &[[f64; 4]], n_target: usize) -> (Vec<usize>, usize) {
    let n = pts.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    // Bounding box
    let mut lo = [f64::INFINITY; 4];
    let mut hi = [f64::NEG_INFINITY; 4];
    for p in pts {
        for d in 0..4 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }

    // Voxel edge length: ℓ = (V / n_target)^{1/4}
    let vol = (0..4).map(|d| (hi[d] - lo[d]).max(1e-15)).product::<f64>();
    let ell = (vol / n_target as f64).powf(0.25).max(1e-15);

    // Bin each point → voxel key, track first-seen assignment
    let mut key_to_temp: FxHashMap<(i32, i32, i32, i32), usize> = FxHashMap::default();
    let mut micro_to_temp = Vec::with_capacity(n);
    let mut next_id: usize = 0;

    for p in pts {
        let key = (
            ((p[0] - lo[0]) / ell).floor() as i32,
            ((p[1] - lo[1]) / ell).floor() as i32,
            ((p[2] - lo[2]) / ell).floor() as i32,
            ((p[3] - lo[3]) / ell).floor() as i32,
        );
        let temp_id = *key_to_temp.entry(key).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        micro_to_temp.push(temp_id);
    }

    let n_macro = next_id;

    // Sort voxel keys by tuple order (time-first) and build remap
    let mut keys_with_id: Vec<((i32, i32, i32, i32), usize)> =
        key_to_temp.into_iter().collect();
    keys_with_id.sort_unstable_by_key(|&(k, _)| k);

    let mut temp_to_final = vec![0usize; n_macro];
    for (final_id, &(_, temp_id)) in keys_with_id.iter().enumerate() {
        temp_to_final[temp_id] = final_id;
    }

    // Remap micro assignments to final (time-sorted) IDs
    let micro_to_macro: Vec<usize> = micro_to_temp
        .iter()
        .map(|&t| temp_to_final[t])
        .collect();

    (micro_to_macro, n_macro)
}

/// Collapse a micro-level Hasse CSR into a macro-level CSR.
///
/// Iterates over all micro edges, maps endpoints through
/// `micro_to_macro`, skips intra-voxel self-loops, deduplicates,
/// and builds a CSR for the macro graph.
///
/// Returns `(macro_head, macro_data)`.
pub fn collapse_hasse_to_macro(
    adj_head: &[u32],
    adj_data: &[u32],
    n_micro: usize,
    micro_to_macro: &[usize],
    n_macro: usize,
) -> (Vec<u32>, Vec<u32>) {
    // Collect macro edges, skip self-loops
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for u in 0..n_micro {
        let mu = micro_to_macro[u] as u32;
        let lo = adj_head[u] as usize;
        let hi = adj_head[u + 1] as usize;
        for &v in &adj_data[lo..hi] {
            let mv = micro_to_macro[v as usize] as u32;
            if mu != mv {
                edges.push((mu, mv));
            }
        }
    }

    // Sort + dedup
    edges.sort_unstable();
    edges.dedup();

    // Build CSR via prefix sum
    let mut macro_head = vec![0u32; n_macro + 1];
    for &(src, _) in &edges {
        macro_head[src as usize + 1] += 1;
    }
    for i in 1..=n_macro {
        macro_head[i] += macro_head[i - 1];
    }
    let macro_data: Vec<u32> = edges.iter().map(|&(_, dst)| dst).collect();

    (macro_head, macro_data)
}

/// Build the N×N BD matrix from a **directed** CSR without index filtering.
///
/// Like [`build_bd_matrix`] but does NOT use `nbr > node` to filter
/// edges — the CSR is assumed to already encode directionality.
/// This is required for macro CSR where voxel IDs are time-sorted
/// but individual edges may point to lower indices within the same
/// time layer.
pub fn build_bd_matrix_directed(
    adj_head: &[u32],
    adj_data: &[u32],
    n: usize,
) -> DMatrix<i64> {
    let mut b = DMatrix::<i64>::zeros(n, n);
    let mut visited = vec![false; n];

    for src in 0..n {
        for v in visited.iter_mut() {
            *v = false;
        }
        visited[src] = true;

        let mut frontier: Vec<usize> = vec![src];

        for depth in 0..MAX_DEPTH {
            let mut next: Vec<usize> = Vec::new();
            for &node in &frontier {
                let lo = adj_head[node] as usize;
                let hi = adj_head[node + 1] as usize;
                for &nbr_u32 in &adj_data[lo..hi] {
                    let nbr = nbr_u32 as usize;
                    if !visited[nbr] {
                        visited[nbr] = true;
                        next.push(nbr);
                        b[(src, nbr)] = bd_weight(depth as u8);
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
    }
    b
}

// ═══════════════════════════════════════════════════════════════════════
// MacroNode infrastructure — integer-exact coarse-graining for RG flow
// ═══════════════════════════════════════════════════════════════════════

/// A coarse-grained "macro" node that bundles a set of microscopic events.
///
/// Used in the RG coarse-graining pipeline: micro-events are grouped into
/// macro-nodes via a `micro_to_macro: &[usize]` map (analogous to the
/// `merge_into` chain in `skyrmion.rs`), and the BD matrix is accumulated
/// at the macro level with integer-exact arithmetic.
pub struct MacroNode {
    /// Macro-node index.
    pub id: usize,
    /// Indices of the micro-events belonging to this macro-node.
    pub micro_nodes: Vec<usize>,
    /// Accumulated integer BD flux for this macro-node.
    pub net_flux: i64,
}

impl MacroNode {
    /// Create a new macro-node with zero flux.
    pub fn new(id: usize) -> Self {
        Self {
            id,
            micro_nodes: Vec::new(),
            net_flux: 0,
        }
    }

    /// Accumulate BD flux from a single micro-event at the given BFS depth.
    ///
    /// Integer-exact: no floating-point involved.
    #[inline]
    pub fn accumulate_flux(&mut self, depth: u8) {
        self.net_flux += bd_weight(depth);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bd_weight_values() {
        assert_eq!(bd_weight(0), 1);
        assert_eq!(bd_weight(1), -9);
        assert_eq!(bd_weight(2), 16);
        assert_eq!(bd_weight(3), -8);
        assert_eq!(bd_weight(4), 0);
        // Out-of-range depths should return 0
        assert_eq!(bd_weight(255), 0);
    }

    /// Build a 5-node linear chain: 0→1→2→3→4
    /// and verify the BD matrix entries are correct integers.
    #[test]
    fn test_integer_bd_matrix_small() {
        // CSR for chain 0→1→2→3→4
        // adj_head[i] = start of neighbors for node i
        // Node 0: neighbors [1]
        // Node 1: neighbors [2]
        // Node 2: neighbors [3]
        // Node 3: neighbors [4]
        // Node 4: no neighbors
        let adj_head: Vec<u32> = vec![0, 1, 2, 3, 4, 4];
        let adj_data: Vec<u32> = vec![1, 2, 3, 4];
        let n = 5;

        let b = build_bd_matrix(&adj_head, &adj_data, n);

        // Direct links (depth 0, weight +1):
        assert_eq!(b[(0, 1)], 1);
        assert_eq!(b[(1, 2)], 1);
        assert_eq!(b[(2, 3)], 1);
        assert_eq!(b[(3, 4)], 1);

        // 2-hop paths (depth 1, weight -9):
        assert_eq!(b[(0, 2)], -9);
        assert_eq!(b[(1, 3)], -9);
        assert_eq!(b[(2, 4)], -9);

        // 3-hop paths (depth 2, weight +16):
        assert_eq!(b[(0, 3)], 16);
        assert_eq!(b[(1, 4)], 16);

        // 4-hop path (depth 3, weight -8):
        assert_eq!(b[(0, 4)], -8);

        // Lower triangle must be zero (forward-only DAG)
        for i in 0..n {
            for j in 0..i {
                assert_eq!(b[(i, j)], 0, "b[({i},{j})] should be 0");
            }
        }
    }

    #[test]
    fn test_macronode_accumulate() {
        let mut mn = MacroNode::new(0);
        assert_eq!(mn.net_flux, 0);

        mn.accumulate_flux(0); // +1
        assert_eq!(mn.net_flux, 1);

        mn.accumulate_flux(1); // -9
        assert_eq!(mn.net_flux, 1 - 9);

        mn.accumulate_flux(2); // +16
        assert_eq!(mn.net_flux, 1 - 9 + 16);

        mn.accumulate_flux(3); // -8
        assert_eq!(mn.net_flux, 1 - 9 + 16 - 8);
        assert_eq!(mn.net_flux, 0); // BD weights sum to zero
    }

    /// Points at earlier times must get lower macro IDs.
    #[test]
    fn test_voxelize_time_ordering() {
        // 4 points spread across time; voxel edge should group none together
        let pts: Vec<[f64; 4]> = vec![
            [3.0, 0.0, 0.0, 0.0], // latest
            [1.0, 0.0, 0.0, 0.0], // earliest
            [2.0, 0.0, 0.0, 0.0], // middle
            [1.0, 1.0, 0.0, 0.0], // same time as 1, different space
        ];
        let (m2m, n_macro) = voxelize(&pts, 4);
        assert_eq!(n_macro, 4);

        // pt[1] (t=1) and pt[3] (t=1) should get lower IDs than pt[2] (t=2)
        assert!(m2m[1] < m2m[2], "t=1 should map before t=2");
        assert!(m2m[3] < m2m[2], "t=1 should map before t=2");
        // pt[2] (t=2) should be before pt[0] (t=3)
        assert!(m2m[2] < m2m[0], "t=2 should map before t=3");
    }

    /// Intra-voxel edges become self-loops and must be removed.
    #[test]
    fn test_collapse_removes_self_loops() {
        // 2 micro nodes in the same voxel, edge 0→1
        let adj_head: Vec<u32> = vec![0, 1, 1];
        let adj_data: Vec<u32> = vec![1];
        let micro_to_macro: Vec<usize> = vec![0, 0]; // same macro

        let (mh, md) = collapse_hasse_to_macro(&adj_head, &adj_data, 2, &micro_to_macro, 1);
        assert_eq!(mh.len(), 2); // n_macro + 1
        assert!(md.is_empty(), "self-loop should be removed");
    }

    /// Multiple micro edges between the same pair of voxels → single macro edge.
    #[test]
    fn test_collapse_deduplicates() {
        // 4 micro nodes: {0,1} → macro 0, {2,3} → macro 1
        // Edges: 0→2, 0→3, 1→2
        let adj_head: Vec<u32> = vec![0, 2, 3, 3, 3];
        let adj_data: Vec<u32> = vec![2, 3, 2];
        let micro_to_macro: Vec<usize> = vec![0, 0, 1, 1];

        let (mh, md) = collapse_hasse_to_macro(&adj_head, &adj_data, 4, &micro_to_macro, 2);
        // Should have exactly one macro edge: 0→1
        assert_eq!(md.len(), 1);
        assert_eq!(md[0], 1);
        assert_eq!(mh[0], 0);
        assert_eq!(mh[1], 1);
        assert_eq!(mh[2], 1);
    }

    /// Edge 2→0 in a directed CSR: build_bd_matrix would miss it (nbr < node),
    /// build_bd_matrix_directed should catch it.
    #[test]
    fn test_bd_directed_no_index_filter() {
        // 3 nodes, edges: 0→1, 2→0 (backward index, forward time)
        let adj_head: Vec<u32> = vec![0, 1, 1, 2];
        let adj_data: Vec<u32> = vec![1, 0];

        let b = build_bd_matrix_directed(&adj_head, &adj_data, 3);
        // 0→1 at depth 0
        assert_eq!(b[(0, 1)], 1);
        // 2→0 at depth 0 — this is the key difference from build_bd_matrix
        assert_eq!(b[(2, 0)], 1);
    }

    /// On a time-sorted linear chain, build_bd_matrix and build_bd_matrix_directed
    /// must produce identical results.
    #[test]
    fn test_bd_directed_matches_time_sorted() {
        let adj_head: Vec<u32> = vec![0, 1, 2, 3, 4, 4];
        let adj_data: Vec<u32> = vec![1, 2, 3, 4];
        let n = 5;

        let b1 = build_bd_matrix(&adj_head, &adj_data, n);
        let b2 = build_bd_matrix_directed(&adj_head, &adj_data, n);

        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    b1[(i, j)],
                    b2[(i, j)],
                    "mismatch at ({i},{j}): build_bd_matrix={}, build_bd_matrix_directed={}",
                    b1[(i, j)],
                    b2[(i, j)]
                );
            }
        }
    }
}
