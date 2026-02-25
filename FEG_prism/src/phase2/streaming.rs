// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Phase 2 streaming mode — placeholder for sparse-scan implementation.
//!
//! This module will implement the two-pass sparse scanning pipeline
//! (HashMap grid, zero disk I/O) for N > ~500k when RAM is constrained.
//! Currently, only the in-memory path is implemented.
