// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

pub mod sprinkle;
pub mod hasse;

pub use sprinkle::sprinkle;
pub use hasse::{build_hasse_sparse, build_hasse_direct};
