// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Grid quantization and Occupied Cell Index (OCI) for spatial lookups.
//!
//! Points are quantized onto an integer lattice and sorted by cell index
//! for cache-coherent traversal. The OCI builds per-time-layer lists of
//! occupied cells, enabling O(occupied) light-cone queries instead of
//! O(shell_volume).

use rayon::prelude::*;

/// Maximum squared proper time for Hasse link consideration.
/// At ρ ≈ 1, the Alexandrov set volume for τ > 8 is > 4096;
/// probability of zero intermediate points is exponentially small.
pub const MAX_PROPER_TIME_SQ: f64 = 64.0;

/// Maximum time-layer depth for Hasse link search.
/// Three physics-based valves handle termination; this is just the outer bound.
pub const MAX_CAUSAL_DEPTH: i16 = 15;

/// Maximum out-degree before abandoning light-cone search for a node.
/// Mean Hasse valence at ρ ≈ 1 is ~4–10; 15 descendants ⟹ saturated causal shadow.
pub const MAX_HASSE_DEGREE: usize = 15;

/// A spacetime event quantized onto an integer lattice for cache-coherent traversal.
#[derive(Debug, Clone, Copy)]
pub struct GridPoint {
    pub p: [f64; 4],
    pub orig_idx: u32,
    pub cell: usize,
    pub qt: u16,
    pub qx: u16,
    pub qy: u16,
    pub qz: u16,
}

/// Grid dimensions and strides for a 4D integer lattice.
pub struct GridDims {
    pub t_dim: usize,
    pub dim_x: usize,
    pub dim_y: usize,
    pub dim_z: usize,
    pub stride_t: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub grid_size: usize,
    pub origin: [f64; 4],
}

impl GridDims {
    /// Compute grid dimensions from point bounding box.
    pub fn from_bounds(pts: &[[f64; 4]]) -> Self {
        let margin = 2.0;
        let mut min_val = [f64::INFINITY; 4];
        let mut max_val = [f64::NEG_INFINITY; 4];
        for p in pts {
            for k in 0..4 {
                if p[k] < min_val[k] { min_val[k] = p[k]; }
                if p[k] > max_val[k] { max_val[k] = p[k]; }
            }
        }

        let origin = [
            min_val[0] - margin,
            min_val[1] - margin,
            min_val[2] - margin,
            min_val[3] - margin,
        ];

        let t_dim = ((max_val[0] - min_val[0] + 2.0 * margin).ceil() as usize).max(1);
        let dim_x = ((max_val[1] - min_val[1] + 2.0 * margin).ceil() as usize).max(1);
        let dim_y = ((max_val[2] - min_val[2] + 2.0 * margin).ceil() as usize).max(1);
        let dim_z = ((max_val[3] - min_val[3] + 2.0 * margin).ceil() as usize).max(1);

        let stride_y = dim_z;
        let stride_x = dim_y * dim_z;
        let stride_t = dim_x * dim_y * dim_z;
        let grid_size = t_dim * stride_t;

        Self {
            t_dim, dim_x, dim_y, dim_z,
            stride_t, stride_x, stride_y,
            grid_size, origin,
        }
    }

    /// Compute grid dimensions from half-T (streaming mode).
    pub fn from_half_t(half_t: f64) -> Self {
        let margin = 2.0;
        let span = 2.0 * half_t + 2.0 * margin;
        let t_dim = (span.ceil() as usize).max(1);
        let dim_x = (span.ceil() as usize).max(1);
        let dim_y = dim_x;
        let dim_z = dim_x;

        let stride_y = dim_z;
        let stride_x = dim_y * dim_z;
        let stride_t = dim_x * dim_y * dim_z;
        let grid_size = t_dim * stride_t;

        let origin_val = -half_t - margin;
        let origin = [origin_val; 4];

        Self {
            t_dim, dim_x, dim_y, dim_z,
            stride_t, stride_x, stride_y,
            grid_size, origin,
        }
    }

    /// Quantize a coordinate to grid units.
    #[inline]
    pub fn to_grid(&self, val: f64, k: usize) -> u16 {
        (val - self.origin[k]).max(0.0) as u16
    }

    /// Linearize grid coordinates to cell index.
    #[inline]
    pub fn cell_index(&self, qt: u16, qx: u16, qy: u16, qz: u16) -> usize {
        (qt as usize) * self.stride_t
            + (qx as usize) * self.stride_x
            + (qy as usize) * self.stride_y
            + (qz as usize)
    }
}

/// Quantize and cell-sort points for OCI lookup.
///
/// Returns `(sorted_points, sorted_coords)` where `sorted_coords[i] = sorted_points[i].p`.
pub fn quantize_and_sort(pts: &[[f64; 4]], dims: &GridDims) -> (Vec<GridPoint>, Vec<[f64; 4]>) {
    let mut sorted_pts: Vec<GridPoint> = pts.iter().enumerate().map(|(i, p)| {
        let qt = dims.to_grid(p[0], 0);
        let qx = dims.to_grid(p[1], 1);
        let qy = dims.to_grid(p[2], 2);
        let qz = dims.to_grid(p[3], 3);
        let idx = dims.cell_index(qt, qx, qy, qz);
        GridPoint {
            p: *p,
            orig_idx: i as u32,
            cell: if idx < dims.grid_size { idx } else { dims.grid_size },
            qt, qx, qy, qz,
        }
    }).collect();

    sorted_pts.par_sort_unstable_by_key(|gp| gp.cell);
    let sorted_coords: Vec<[f64; 4]> = sorted_pts.iter().map(|gp| gp.p).collect();
    (sorted_pts, sorted_coords)
}

/// An entry in an Occupied Cell Index layer: (qx, qy, qz, cell_start, cell_count).
pub type OciEntry = (i16, i16, i16, u32, u32);

/// Build the Occupied Cell Index from cell-sorted points.
///
/// Returns per-time-layer lists of `(qx, qy, qz, start, count)`, sorted by qx
/// within each layer for binary search.
pub fn build_oci(sorted_pts: &[GridPoint], dims: &GridDims) -> Vec<Vec<OciEntry>> {
    // Build grid start/count
    let mut grid_start = vec![u32::MAX; dims.grid_size];
    let mut grid_count = vec![0u32; dims.grid_size];

    for (i, gp) in sorted_pts.iter().enumerate() {
        if gp.cell >= dims.grid_size { continue; }
        if grid_start[gp.cell] == u32::MAX {
            grid_start[gp.cell] = i as u32;
        }
        grid_count[gp.cell] += 1;
    }

    let mut occupied: Vec<Vec<OciEntry>> = vec![vec![]; dims.t_dim];
    for cell in 0..dims.grid_size {
        if grid_start[cell] != u32::MAX {
            let qt = cell / dims.stride_t;
            let rem_t = cell % dims.stride_t;
            let qx = (rem_t / dims.stride_x) as i16;
            let rem_x = rem_t % dims.stride_x;
            let qy = (rem_x / dims.stride_y) as i16;
            let qz = (rem_x % dims.stride_y) as i16;
            occupied[qt].push((qx, qy, qz, grid_start[cell], grid_count[cell]));
        }
    }

    // Sort by X coordinate for binary search band queries
    for layer in &mut occupied {
        layer.sort_unstable_by_key(|&(qx, _, _, _, _)| qx);
    }

    occupied
}

/// True if `b` is in the strict causal future of `a`:
///   t_b > t_a  and  (t_b − t_a)² > |Δx⃗|².
#[inline]
pub fn is_causal(a: &[f64; 4], b: &[f64; 4]) -> bool {
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
pub fn time_order(pts: &[[f64; 4]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pts.len()).collect();
    order.sort_by(|&a, &b| pts[a][0].partial_cmp(&pts[b][0]).unwrap());
    order
}
