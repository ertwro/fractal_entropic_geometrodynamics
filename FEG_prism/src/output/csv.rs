// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Unified CSV writer with provenance header.
//!
//! Every CSV produced by FEG_prism carries a standard provenance block
//! (author, DOI, SHA-256, git commit, timestamp) followed by the run
//! metadata (N, M, mode, epsilon, tmax, walkers, seed).

use std::fs::File;
use std::io::{self, BufWriter, Write};

use super::Metadata;
use crate::provenance;

/// Buffered CSV writer with provenance-stamped header.
pub struct CsvWriter {
    file: BufWriter<File>,
}

impl CsvWriter {
    /// Create a new CSV file at `path` and write the provenance + metadata header.
    pub fn new(path: &str, meta: &Metadata) -> io::Result<Self> {
        let f = File::create(path)?;
        let mut file = BufWriter::new(f);

        // Provenance header (author, DOI, SHA-256, commit, timestamp)
        let header = provenance::file_header(&meta.timestamp);
        for line in header.lines() {
            writeln!(file, "{line}")?;
        }

        // Run metadata
        writeln!(
            file,
            "# N: {}  M: {} (converged={}, min={}, max={})  mode: {}",
            meta.n_points, meta.actual_m, meta.converged,
            meta.min_ensemble, meta.max_ensemble, meta.mode,
        )?;
        writeln!(
            file,
            "# epsilon: {}  tmax: {}  walkers: {}",
            meta.epsilon, meta.tmax, meta.walkers,
        )?;
        writeln!(
            file,
            "# algorithm: forward-forward belly (children(u) \u{2229} parents(v))",
        )?;
        writeln!(file, "# seed: {}", meta.seed)?;

        Ok(Self { file })
    }

    /// Write a comma-separated header row.
    pub fn header(&mut self, cols: &[&str]) {
        let _ = writeln!(self.file, "{}", cols.join(","));
    }

    /// Write a comment line (prefixed with "# ").
    pub fn comment(&mut self, text: &str) {
        let _ = writeln!(self.file, "# {text}");
    }

    /// Write a formatted data row.
    pub fn row_fmt(&mut self, args: std::fmt::Arguments) {
        let _ = writeln!(self.file, "{args}");
    }
}
