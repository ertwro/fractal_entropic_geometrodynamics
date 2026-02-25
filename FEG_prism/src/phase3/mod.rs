// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

pub mod spectral;
pub mod walker;
pub mod flux;

pub use spectral::{WalkResult, SpectralOutput, spectral_dimension};
pub use walker::{run_walkers, distribute_walkers, compute_eigen, compute_monte_carlo_csr};
pub use flux::{build_flux_csr, run_transmission_walkers};
