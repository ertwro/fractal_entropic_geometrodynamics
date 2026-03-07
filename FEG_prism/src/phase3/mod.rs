// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

pub mod spectral;
pub mod walker;
pub mod flux;
pub mod laplacian;

pub use spectral::{WalkResult, SpectralOutput, spectral_dimension};
pub use walker::{
    run_walkers, distribute_walkers, distribute_walkers_shuffled,
    run_walkers_unified, count_category_walkers,
    compute_eigen, compute_monte_carlo_csr,
};
pub use flux::{build_flux_csr, run_transmission_walkers};
pub use laplacian::{
    HeatKernelResult, RgFlowResult,
    heat_kernel_exact, heat_kernel_slq, integrate_rg,
    log_spaced_times, weyl_law_check,
};
