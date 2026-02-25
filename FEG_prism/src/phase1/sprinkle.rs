// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Poisson sprinkling into a 4D causal diamond.
//!
//! Two functions:
//!   - `sprinkle`: full diamond sprinkling (returns flat coordinate array)
//!   - `sprinkle_chunk`: time-slab sprinkling for streaming mode (returns GridPoints)

use rand::Rng;
use std::f64::consts::PI;

use crate::graph::grid::GridPoint;

/// Poisson-sprinkle `n` points into a 4D causal diamond |t| + r <= T/2.
///
/// Volume V = piT^4/24;  T = (24N/pi)^{1/4}  gives density rho ~ 1.
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

/// Poisson-sprinkle points into a time-slab [t_min, t_max].
///
/// Used by the streaming/sliding-window path to generate points
/// incrementally. Each point gets a sequential `orig_idx` starting
/// from `start_idx`. Grid quantization fields are left zeroed;
/// the caller is responsible for filling them after choosing a grid.
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
                qt: 0,
                qx: 0,
                qy: 0,
                qz: 0,
            });
            idx += 1;
        }
    }
    pts
}
