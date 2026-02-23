// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18733424

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct PrismDef {
    pub origin: u32,
    pub destination: u32,
    pub intermediates: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
pub struct TopologySlice {
    pub n_total: usize,
    pub coordinates: Vec<[f64; 4]>,
    pub hasse_edges: Vec<(u32, u32)>,
    pub prisms: Vec<PrismDef>,
}

pub fn write_slice(path: &str, slice: &TopologySlice) -> Result<(), String> {
    let bytes = bincode::serialize(slice)
        .map_err(|e| format!("slice serialize failed: {e}"))?;
    std::fs::write(path, &bytes)
        .map_err(|e| format!("slice write failed: {e}"))?;
    Ok(())
}
