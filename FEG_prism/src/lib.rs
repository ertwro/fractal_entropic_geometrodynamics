// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! # FEG Prism — Kuratowski Calculus Engine
//!
//! Reference implementation of the Cálculo de Kuratowski (Kuratowski Calculus)
//! from *Fractal Entropic Geometrodynamics* (FEG) by J. P. Silva Alvarado.
//!
//! Zenodo record: <https://doi.org/10.5281/zenodo.18769707>
//!
//! ## Architecture
//!
//! | Phase | Module | Physics |
//! |-------|--------|---------|
//! | 1 | [`phase1`] | Vacuum generation: Poisson sprinkling + Hasse diagram |
//! | 2 | [`phase2`] | Kuratowski contraction: Causal Prism detection + K₅ absorption |
//! | 3 | [`phase3`] | Spectral dimension: random walk return probability |
//! | M1–M10 | [`measure`] | Observable measurements |
//! | — | [`ensemble`] | Adaptive ensemble averaging with checkpointing |
//! | — | [`output`] | CSV serialization with provenance headers |
//!
//! ## Key Types
//!
//! - [`graph::CsrGraph<Directed>`] / [`graph::CsrGraph<Undirected>`] — type-safe CSR
//! - [`phase2::DefectOutput`] — Kuratowski contraction result (owns vacuum CSR)
//! - [`phase3::WalkResult`] / [`phase3::SpectralOutput`] — composed spectral data
//! - [`measure::MeasureResults`] — all M1–M10 measurement outputs

pub mod provenance;
pub mod config;
pub mod graph;
pub mod phase1;
pub mod phase2;
pub mod phase3;
pub mod measure;
pub mod output;
pub mod ensemble;
pub mod anim_export;
