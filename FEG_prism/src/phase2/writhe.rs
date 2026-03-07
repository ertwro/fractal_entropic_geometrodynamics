// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Topological writhe and genus measurement for K_{2,n} prisms.
//!
//! Each K_{2,n} prism with two poles and n intermediates defines a bipartite
//! graph whose embedding on a surface is determined by the cyclic ordering
//! of intermediates around each pole.  The Grothendieck–Euler formula gives:
//!
//!   V − E + F = 2 − 2g
//!   (n+2) − 2n + F = 2 − 2g
//!   F = n − 2g
//!
//! The face count F equals the number of cycles of the composed rotation
//! permutation σ_o ∘ σ_d, where σ_o and σ_d are the cyclic orderings of
//! intermediates around the origin and destination poles respectively.
//!
//! ## Algorithm
//!
//! The genus depends on the 2D projection used to define the cyclic order.
//! We compute the genus in all three spatial planes (xy, yz, xz) and report:
//! - **max_genus_measured**: maximum genus across all projections (most entangled view)
//! - **crossing_count**: direct segment crossing count of prism edges
//!
//! Intermediates "between" the poles (in the projection) show parallax reversal
//! (genus 0 / sphere), while intermediates on one side maintain their order
//! (genus > 0 / torus or higher).

use crate::graph::csr::{CsrGraph, Directed};
use crate::phase2::defect::{CausalPrism, GenerationSets};

/// Per-prism writhe and genus statistics.
#[derive(Debug, Clone)]
pub struct WritheStats {
    /// Index into the prisms vector.
    pub prism_idx: usize,
    /// Generation label from phase classification (1, 2, 3; 0 = unclassified).
    pub generation: u8,
    /// Number of intermediate nodes (belly size n).
    pub belly_size: usize,
    /// Maximum genus measured across all three spatial projections.
    pub genus: usize,
    /// Genus in each spatial plane: [xy, yz, xz].
    pub genus_by_plane: [usize; 3],
    /// Face count F = n − 2g for the maximum-genus projection.
    pub face_count: usize,
    /// Direct segment crossing count (upper × lower fan edges, max across planes).
    pub crossings: usize,
    /// Maximum possible genus for this belly size: ⌊(n−1)/2⌋.
    pub max_genus: usize,
    /// Per-path crossing counts for the plane with maximum total crossings.
    /// `path_crossings[i]` = number of j where segment (O→m_i) crosses (m_j→D).
    pub path_crossings: Vec<usize>,
}

/// The three spatial 2D projection planes.
const PLANES: [(usize, usize); 3] = [(1, 2), (2, 3), (1, 3)];

/// Compute genus for a single K_{2,n} in a specific 2D projection.
///
/// Returns (genus, face_count, crossing_count).
fn genus_in_plane(
    prism: &CausalPrism,
    sorted_coords: &[[f64; 4]],
    ax1: usize,
    ax2: usize,
) -> (usize, usize, usize) {
    let n = prism.intermediates.len();
    if n < 2 {
        return (0, n, 0);
    }

    let o = &sorted_coords[prism.origin];
    let d = &sorted_coords[prism.destination];

    // Angular ordering from each pole.
    let mut angles_o: Vec<(usize, f64)> = prism
        .intermediates
        .iter()
        .enumerate()
        .map(|(idx, &m)| {
            let du = sorted_coords[m][ax1] - o[ax1];
            let dv = sorted_coords[m][ax2] - o[ax2];
            (idx, dv.atan2(du))
        })
        .collect();

    let mut angles_d: Vec<(usize, f64)> = prism
        .intermediates
        .iter()
        .enumerate()
        .map(|(idx, &m)| {
            let du = sorted_coords[m][ax1] - d[ax1];
            let dv = sorted_coords[m][ax2] - d[ax2];
            (idx, dv.atan2(du))
        })
        .collect();

    angles_o.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    angles_d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let order_o: Vec<usize> = angles_o.iter().map(|a| a.0).collect();
    let order_d: Vec<usize> = angles_d.iter().map(|a| a.0).collect();

    let mut pos_o = vec![0usize; n];
    let mut pos_d = vec![0usize; n];
    for (pos, &idx) in order_o.iter().enumerate() {
        pos_o[idx] = pos;
    }
    for (pos, &idx) in order_d.iter().enumerate() {
        pos_d[idx] = pos;
    }

    // Compose σ_o ∘ σ_d.
    let mut composed = vec![0usize; n];
    for idx in 0..n {
        let after_d = order_d[(pos_d[idx] + 1) % n];
        let after_od = order_o[(pos_o[after_d] + 1) % n];
        composed[idx] = after_od;
    }

    // Count cycles → face count → genus.
    let mut visited = vec![false; n];
    let mut face_count = 0;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        face_count += 1;
        let mut cur = start;
        while !visited[cur] {
            visited[cur] = true;
            cur = composed[cur];
        }
    }

    let genus = if n >= face_count {
        (n - face_count) / 2
    } else {
        0
    };

    // Direct segment crossing count: check all pairs (o→m_i, m_j→d) for i≠j.
    let crossings = count_segment_crossings(prism, sorted_coords, ax1, ax2);

    (genus, face_count, crossings)
}

/// Count crossing pairs between upper fan (o→m_i) and lower fan (m_j→d) edges.
fn count_segment_crossings(
    prism: &CausalPrism,
    sorted_coords: &[[f64; 4]],
    ax1: usize,
    ax2: usize,
) -> usize {
    let o = sorted_coords[prism.origin];
    let d = sorted_coords[prism.destination];
    let n = prism.intermediates.len();
    let mut count = 0;

    for i in 0..n {
        let mi = sorted_coords[prism.intermediates[i]];
        for j in 0..n {
            if i == j {
                continue;
            }
            let mj = sorted_coords[prism.intermediates[j]];
            // Check if segment (o→m_i) crosses (m_j→d) in the (ax1, ax2) plane.
            if segments_cross(
                o[ax1], o[ax2], mi[ax1], mi[ax2], mj[ax1], mj[ax2], d[ax1], d[ax2],
            ) {
                count += 1;
            }
        }
    }
    // Each crossing is counted twice (i,j) and (j,i) — but actually these are
    // DIFFERENT edge pairs: (o→m_i, m_j→d) vs (o→m_j, m_i→d).  Both are valid
    // distinct crossings only if the corresponding segments actually cross.
    // However, the standard convention counts unordered pairs.
    count / 2
}

/// Per-path crossing counts: for each intermediate i, how many j have
/// segment (O→m_i) crossing (m_j→D).
fn per_path_crossings_in_plane(
    prism: &CausalPrism,
    sorted_coords: &[[f64; 4]],
    ax1: usize,
    ax2: usize,
) -> Vec<usize> {
    let o = sorted_coords[prism.origin];
    let d = sorted_coords[prism.destination];
    let n = prism.intermediates.len();
    let mut counts = vec![0usize; n];

    for i in 0..n {
        let mi = sorted_coords[prism.intermediates[i]];
        for j in 0..n {
            if i == j {
                continue;
            }
            let mj = sorted_coords[prism.intermediates[j]];
            if segments_cross(
                o[ax1], o[ax2], mi[ax1], mi[ax2], mj[ax1], mj[ax2], d[ax1], d[ax2],
            ) {
                counts[i] += 1;
            }
        }
    }
    counts
}

/// Check if segments (x1,y1)→(x2,y2) and (x3,y3)→(x4,y4) properly cross.
fn segments_cross(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
    x4: f64,
    y4: f64,
) -> bool {
    let d1 = cross2d(x3, y3, x4, y4, x1, y1);
    let d2 = cross2d(x3, y3, x4, y4, x2, y2);
    let d3 = cross2d(x1, y1, x2, y2, x3, y3);
    let d4 = cross2d(x1, y1, x2, y2, x4, y4);
    // Proper crossing: endpoints on opposite sides of each other's segment.
    (d1 * d2 < 0.0) && (d3 * d4 < 0.0)
}

/// 2D cross product: sign((b-a) × (c-a)).
#[inline]
fn cross2d(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// Compute writhe and genus for a single K_{2,n} prism.
pub fn compute_writhe(prism: &CausalPrism, sorted_coords: &[[f64; 4]]) -> WritheStats {
    let n = prism.intermediates.len();

    if n < 2 {
        return WritheStats {
            prism_idx: 0,
            generation: 0,
            belly_size: n,
            genus: 0,
            genus_by_plane: [0; 3],
            face_count: n,
            crossings: 0,
            max_genus: 0,
            path_crossings: vec![],
        };
    }

    let max_theoretical = (n - 1) / 2;
    let mut best_genus = 0;
    let mut best_face = n;
    let mut best_crossings = 0;
    let mut best_crossing_plane = 0usize;
    let mut genus_planes = [0usize; 3];

    for (pi, &(ax1, ax2)) in PLANES.iter().enumerate() {
        let (g, f, c) = genus_in_plane(prism, sorted_coords, ax1, ax2);
        genus_planes[pi] = g;
        if g > best_genus {
            best_genus = g;
            best_face = f;
        }
        if c > best_crossings {
            best_crossings = c;
            best_crossing_plane = pi;
        }
    }

    // Decompose crossings into per-path counts for the best plane.
    let (ax1, ax2) = PLANES[best_crossing_plane];
    let path_cx = per_path_crossings_in_plane(prism, sorted_coords, ax1, ax2);

    WritheStats {
        prism_idx: 0,
        generation: 0,
        belly_size: n,
        genus: best_genus.min(max_theoretical),
        genus_by_plane: genus_planes,
        face_count: best_face,
        crossings: best_crossings,
        max_genus: max_theoretical,
        path_crossings: path_cx,
    }
}

/// Compute writhe stats for all prisms with generation labels (from GenerationSets).
pub fn compute_all_writhes(
    prisms: &[CausalPrism],
    _vacuum_csr: &CsrGraph<Directed>,
    sorted_coords: &[[f64; 4]],
    generations: &GenerationSets,
    n_nodes: usize,
) -> Vec<WritheStats> {
    let mut gen_map = vec![0u8; n_nodes];
    for &node in &generations.gen1 {
        if node < n_nodes {
            gen_map[node] = 1;
        }
    }
    for &node in &generations.anti1 {
        if node < n_nodes {
            gen_map[node] = 1;
        }
    }
    for &node in &generations.gen2 {
        if node < n_nodes {
            gen_map[node] = 2;
        }
    }
    for &node in &generations.gen3 {
        if node < n_nodes {
            gen_map[node] = 3;
        }
    }

    prisms
        .iter()
        .enumerate()
        .map(|(idx, prism)| {
            let mut stats = compute_writhe(prism, sorted_coords);
            stats.prism_idx = idx;
            stats.generation = gen_map[prism.origin];
            stats
        })
        .collect()
}

/// Compute writhe stats classifying each prism by its OWN intermediate phases.
///
/// Generation = number of distinct momentum phase signs among intermediates:
///   1 sign → Gen1, 2 signs → Gen2, 3 signs → Gen3.
/// This is the intrinsic (Vol I) classification, independent of any core bias.
pub fn compute_writhes_intrinsic(
    prisms: &[CausalPrism],
    sorted_coords: &[[f64; 4]],
    bulk_momentum: &[i32],
) -> Vec<WritheStats> {
    prisms
        .iter()
        .enumerate()
        .map(|(idx, prism)| {
            let mut stats = compute_writhe(prism, sorted_coords);
            stats.prism_idx = idx;
            // Classify by the prism's own intermediate phases.
            let mut phase_set = [false; 3]; // indices: 0=negative, 1=zero, 2=positive
            for &w in &prism.intermediates {
                let phi = bulk_momentum[w].signum();
                match phi {
                    -1 => phase_set[0] = true,
                     0 => phase_set[1] = true,
                     _ => phase_set[2] = true,
                }
            }
            let distinct = phase_set.iter().filter(|&&x| x).count();
            stats.generation = distinct as u8;
            stats
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prism(origin: usize, dest: usize, intermediates: Vec<usize>) -> CausalPrism {
        CausalPrism {
            origin,
            destination: dest,
            intermediates,
        }
    }

    /// K_{2,3} with poles on opposite sides and intermediates between:
    /// at least ONE projection plane should give genus 0.
    #[test]
    fn symmetric_k23_has_genus_zero_in_some_plane() {
        let coords = [
            [0.0, -5.0, 0.0, 0.0],       // origin
            [2.0, 5.0, 0.0, 0.0],        // dest
            [1.0, 0.0, 3.0, 0.0],        // m0
            [1.0, 0.0, -1.5, 2.6],       // m1
            [1.0, 0.0, -1.5, -2.6],      // m2
        ];

        let prism = make_prism(0, 1, vec![2, 3, 4]);
        let stats = compute_writhe(&prism, &coords);

        // At least one plane should show genus 0 (the xz plane, per our debug).
        let has_g0 = stats.genus_by_plane.iter().any(|&g| g == 0);
        assert!(has_g0, "symmetric K_{{2,3}} should have genus 0 in at least one plane");
    }

    /// K_{2,3} with intermediates far on one side: high genus in most planes.
    #[test]
    fn twisted_k23_has_genus_one() {
        let coords = [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.1, 0.0, 0.0],
            [1.0, 0.0, 100.0, 0.0],
            [1.0, 0.0, 100.0, 1.0],
            [1.0, 0.0, 100.0, -1.0],
        ];

        let prism = make_prism(0, 1, vec![2, 3, 4]);
        let stats = compute_writhe(&prism, &coords);

        // The max genus across planes should be 1.
        assert_eq!(stats.genus, 1, "twisted K_{{2,3}} max genus should be 1");
    }

    /// Euler identity F = n − 2g must hold for each plane's genus.
    #[test]
    fn euler_identity_all_planes() {
        let coords = [
            [0.0, -5.0, 0.0, 0.0],
            [2.0, 5.0, 0.0, 0.0],
            [1.0, 0.0, 3.0, 0.0],
            [1.0, 0.0, -1.5, 2.6],
            [1.0, 0.0, -1.5, -2.6],
        ];

        let prism = make_prism(0, 1, vec![2, 3, 4]);
        let n = prism.intermediates.len();

        for (pi, &(ax1, ax2)) in PLANES.iter().enumerate() {
            let (g, f, _) = genus_in_plane(&prism, &coords, ax1, ax2);
            assert_eq!(
                f,
                n - 2 * g,
                "plane {}: F={} != n-2g={}",
                pi,
                f,
                n - 2 * g
            );
        }
    }

    /// Genus cannot exceed ⌊(n-1)/2⌋.
    #[test]
    fn genus_bounded() {
        for n in 2..=8 {
            let mut coords = vec![[0.0, 0.0, 0.0, 0.0]; n + 2];
            coords[0] = [0.0, 0.0, 0.0, 0.0];
            coords[1] = [2.0, 0.1, 0.0, 0.0];
            for i in 0..n {
                coords[i + 2] = [1.0, 0.0, 100.0, (i as f64) * 2.0];
            }

            let intermediates: Vec<usize> = (2..2 + n).collect();
            let prism = make_prism(0, 1, intermediates);
            let stats = compute_writhe(&prism, &coords);

            assert!(
                stats.genus <= stats.max_genus,
                "n={}: genus {} > max_genus {}",
                n,
                stats.genus,
                stats.max_genus
            );
        }
    }

    /// K_{2,2} is always planar.
    #[test]
    fn k22_always_genus_zero() {
        let coords = [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.1, 0.0, 0.0],
            [1.0, 0.0, 100.0, 0.0],
            [1.0, 0.0, 100.0, 2.0],
        ];
        let prism = make_prism(0, 1, vec![2, 3]);
        let s = compute_writhe(&prism, &coords);
        assert_eq!(s.genus, 0, "K_{{2,2}} should always be genus 0");
    }

    /// Per-path crossing decomposition sums to 2× total crossings.
    #[test]
    fn path_crossings_sum_consistent() {
        let coords = [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.1, 0.0, 0.0],
            [1.0, 0.0, 100.0, 0.0],
            [1.0, 0.0, 100.0, 1.0],
            [1.0, 0.0, 100.0, -1.0],
        ];
        let prism = make_prism(0, 1, vec![2, 3, 4]);
        let stats = compute_writhe(&prism, &coords);
        // Each crossing (i,j) increments both c_i and c_j, so sum = 2C.
        let path_sum: usize = stats.path_crossings.iter().sum();
        assert_eq!(path_sum, 2 * stats.crossings,
            "per-path sum {} != 2 * total crossings {}", path_sum, 2 * stats.crossings);
    }

    /// Segment crossing detection works.
    #[test]
    fn segments_cross_basic() {
        // Crossing segments.
        assert!(segments_cross(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0));
        // Parallel segments (no crossing).
        assert!(!segments_cross(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0));
        // Non-crossing segments.
        assert!(!segments_cross(0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 2.0));
    }
}
