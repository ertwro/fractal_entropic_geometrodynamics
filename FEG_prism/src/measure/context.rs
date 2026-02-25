// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Shared measurement context — immutable borrows into the simulation state.

use crate::graph::csr::{CsrGraph, Directed, Undirected};
use crate::phase2::defect::{DefectOutput, CausalPrism};
use crate::phase2::topology::TopologySummary;
use super::ModuloConfig;

/// Read-only context passed to every M1--M10 measurement function.
///
/// All fields are shared borrows so measurements cannot mutate simulation
/// state.  The context is constructed once per realisation in `run_all`.
pub struct MeasureContext<'a> {
    pub n_points: usize,
    pub pts: &'a [[f64; 4]],
    pub vacuum_csr: &'a CsrGraph<Directed>,
    pub sym_vacuum: &'a CsrGraph<Undirected>,
    pub defect: &'a DefectOutput,
    pub prisms: &'a [CausalPrism],
    pub momentum: &'a [i32],
    pub topology: &'a TopologySummary,
    pub walkers: usize,
    pub seed: u64,
    pub modulo_config: &'a ModuloConfig,
}
