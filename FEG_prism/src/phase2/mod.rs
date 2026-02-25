// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

pub mod defect;
pub mod topology;
pub mod streaming;

pub use defect::{apply_defect, CausalPrism, DefectOutput, GenerationSets};
pub use topology::TopologySummary;
