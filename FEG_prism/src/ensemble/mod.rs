// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Ensemble orchestration: adaptive batching, checkpointing, and averaging.
//!
//! Runs M independent Poisson sprinklings through Phases 1-3, ensemble-averages
//! the return probability P(t) before recomputing d_S from the mean, and
//! checkpoints after each realization for crash-safe resumption.

pub mod runner;
pub mod checkpoint;
pub mod averaging;
