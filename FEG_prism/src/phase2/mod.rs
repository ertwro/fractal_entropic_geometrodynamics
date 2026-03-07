// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

pub mod defect;
pub mod diamond;
pub mod topology;
pub mod writhe;

pub use defect::{apply_defect, scan_all_prisms, scan_maximal_prisms, CausalPrism, DefectOutput, GenerationSets, K5Minor};
pub use diamond::{DiamondStats, compute_diamond, compute_all_diamonds};
pub use topology::TopologySummary;
pub use writhe::{WritheStats, compute_writhe, compute_all_writhes, compute_writhes_intrinsic};
