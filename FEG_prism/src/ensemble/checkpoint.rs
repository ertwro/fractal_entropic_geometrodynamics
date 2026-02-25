// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Batch-level checkpointing for crash-safe ensemble runs.
//!
//! Saves completed realizations after each batch so that `--resume`
//! can skip already-finished work.  Uses atomic rename to guarantee
//! that a crash during write never corrupts the previous checkpoint.
//!
//! Provenance hash verification on load rejects checkpoints from
//! incompatible forks (different DOI or author).

use crate::phase2::topology::TopologySummary;
use crate::phase3::SpectralOutput;
use crate::provenance;
use serde::{Deserialize, Serialize};
use std::path::Path;

const CHECKPOINT_VERSION: u32 = 3;

/// Parameter fingerprint for validating checkpoint compatibility.
///
/// Stored alongside checkpoint data so that `--resume` can detect
/// when CLI arguments have changed between runs (which would make
/// the saved realizations invalid).
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct RunParams {
    pub n_points: usize,
    pub seed_base: u64,
    pub epsilon_bits: u64,
    pub tmax: usize,
    pub eigen_cutoff: usize,
    pub exec_mode: String,
}

/// Full checkpoint state: parameters + completed results.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    version: u32,
    provenance_hash: [u8; 32],
    params: RunParams,
    pub completed: usize,
    pub spectral_vec: Vec<SpectralOutput>,
    pub topo_vec: Vec<TopologySummary>,
    pub welford_mean: f64,
    pub welford_m2: f64,
}

/// Build a `RunParams` fingerprint from CLI arguments.
pub fn make_params(
    n_points: usize,
    seed_base: u64,
    epsilon: f64,
    tmax: usize,
    eigen_cutoff: usize,
    exec_mode: &str,
) -> RunParams {
    RunParams {
        n_points,
        seed_base,
        epsilon_bits: epsilon.to_bits(),
        tmax,
        eigen_cutoff,
        exec_mode: exec_mode.to_string(),
    }
}

/// Atomically save a checkpoint to `{output_dir}/.checkpoint.bin`.
///
/// Writes to a `.tmp` file first, then renames -- a crash during write
/// never corrupts the previous valid checkpoint.
pub fn save(
    output_dir: &str,
    params: &RunParams,
    spectral_vec: &[SpectralOutput],
    topo_vec: &[TopologySummary],
    welford_mean: f64,
    welford_m2: f64,
) -> Result<(), String> {
    let ckpt = Checkpoint {
        version: CHECKPOINT_VERSION,
        provenance_hash: provenance::PROVENANCE_HASH,
        params: params.clone(),
        completed: spectral_vec.len(),
        spectral_vec: spectral_vec.to_vec(),
        topo_vec: topo_vec.to_vec(),
        welford_mean,
        welford_m2,
    };
    let bytes = bincode::serialize(&ckpt)
        .map_err(|e| format!("checkpoint serialize failed: {e}"))?;

    let dir = Path::new(output_dir);
    let tmp_path = dir.join(".checkpoint.bin.tmp");
    let final_path = dir.join(".checkpoint.bin");

    std::fs::write(&tmp_path, &bytes)
        .map_err(|e| format!("checkpoint write failed: {e}"))?;
    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| format!("checkpoint rename failed: {e}"))?;

    Ok(())
}

/// Load and validate a checkpoint from `{output_dir}/.checkpoint.bin`.
///
/// Checks:
/// 1. Version matches `CHECKPOINT_VERSION`
/// 2. Provenance hash matches (rejects cross-fork checkpoints)
/// 3. `RunParams` matches the current CLI arguments field-by-field
/// 4. Data lengths are consistent
pub fn load(output_dir: &str, expected: &RunParams) -> Result<Checkpoint, String> {
    let path = Path::new(output_dir).join(".checkpoint.bin");
    if !path.exists() {
        return Err("no checkpoint file found, starting fresh".to_string());
    }

    let bytes = std::fs::read(&path)
        .map_err(|e| format!("cannot read checkpoint: {e}"))?;
    let ckpt: Checkpoint = bincode::deserialize(&bytes)
        .map_err(|e| format!("checkpoint deserialize failed (corrupt?): {e}"))?;

    if ckpt.version != CHECKPOINT_VERSION {
        return Err(format!(
            "checkpoint version mismatch (file={}, expected={})",
            ckpt.version, CHECKPOINT_VERSION
        ));
    }

    // Provenance hash verification: reject checkpoints from different forks
    if ckpt.provenance_hash != provenance::PROVENANCE_HASH {
        return Err(
            "provenance hash mismatch: checkpoint was created by a different fork".to_string()
        );
    }

    if &ckpt.params != expected {
        let p = &ckpt.params;
        let e = expected;
        let mut diffs = Vec::new();
        if p.n_points != e.n_points {
            diffs.push(format!("N: {} vs {}", p.n_points, e.n_points));
        }
        if p.seed_base != e.seed_base {
            diffs.push(format!("seed: {} vs {}", p.seed_base, e.seed_base));
        }
        if p.epsilon_bits != e.epsilon_bits {
            diffs.push("epsilon differs".to_string());
        }
        if p.tmax != e.tmax {
            diffs.push(format!("tmax: {} vs {}", p.tmax, e.tmax));
        }
        if p.eigen_cutoff != e.eigen_cutoff {
            diffs.push(format!(
                "eigen-cutoff: {} vs {}", p.eigen_cutoff, e.eigen_cutoff
            ));
        }
        if p.exec_mode != e.exec_mode {
            diffs.push(format!("mode: {} vs {}", p.exec_mode, e.exec_mode));
        }
        return Err(format!("parameter mismatch: {}", diffs.join(", ")));
    }

    if ckpt.spectral_vec.len() != ckpt.completed {
        return Err(format!(
            "data integrity: spectral_vec.len()={} != completed={}",
            ckpt.spectral_vec.len(),
            ckpt.completed
        ));
    }
    if ckpt.topo_vec.len() != ckpt.completed {
        return Err(format!(
            "data integrity: topo_vec.len()={} != completed={}",
            ckpt.topo_vec.len(),
            ckpt.completed
        ));
    }

    Ok(ckpt)
}

/// Remove the checkpoint file (currently unused -- checkpoint persists for --resume).
#[allow(dead_code)]
pub fn delete(output_dir: &str) {
    let path = Path::new(output_dir).join(".checkpoint.bin");
    let _ = std::fs::remove_file(path);
}
