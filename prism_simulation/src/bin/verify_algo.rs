// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18746995

//! Verification binary for Hasse diagram construction.
//!
//! Sprinkles N=10,000 points, builds the Hasse diagram via `build_hasse_direct`,
//! and prints edge count + deterministic checksum. Used to verify algorithmic
//! correctness after refactoring (OCI, adaptive termination, CSR sorting).

use causal_set_sim::diamond;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let n = 10000;
    let seed = 12345;
    let mut rng = StdRng::seed_from_u64(seed);

    println!("Generating {n} points...");
    let (pts, _big_t) = diamond::sprinkle(n, &mut rng);

    println!("Running build_hasse_direct (HPC Flat Grid)...");
    let t0 = Instant::now();
    let (_pts_sorted, head, data, _momentum) = diamond::build_hasse_direct(&pts);
    let dur = t0.elapsed();

    println!("Found {} edges in {dur:.2?}", data.len());

    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(data.len());
    for u in 0..(head.len() - 1) {
        let start = head[u] as usize;
        let end = head[u+1] as usize;
        for &v in &data[start..end] {
            edges.push((u as u32, v));
        }
    }
    edges.sort();
    edges.dedup();

    let mut checksum: u64 = 0;
    for (r, c) in &edges {
        checksum = checksum.wrapping_add((*r as u64).wrapping_mul(37).wrapping_add(*c as u64));
    }

    println!("Checksum: {checksum:016X}");
}
