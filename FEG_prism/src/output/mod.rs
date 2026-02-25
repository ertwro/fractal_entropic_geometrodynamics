// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Phase 4 — CSV Serialisation
//!
//! Writes ensemble-averaged observables to CSV for analysis with gnuplot,
//! matplotlib, or any standard plotting tool.
//!
//! ## Column Reference
//!
//! | Column | Physics | Units |
//! |--------|---------|-------|
//! | `step` | Diffusion time t | integer steps |
//! | `P_vac` | Return probability on vacuum graph | probability |
//! | `dS_vac` | Spectral dimension of vacuum | dimensionless |
//! | `P_def` | Return probability on defect graph | probability |
//! | `dS_def` | Spectral dimension with matter | dimensionless |
//! | `P_Gen1..3` | Return probability for generations 1-3 | probability |
//! | `dS_Gen1..3` | Spectral dimension per generation | dimensionless |
//! | `P_Anti1` | Return probability for antimatter | probability |
//! | `dS_Anti1` | Spectral dimension of antimatter | dimensionless |
//! | `Flux_Attr` | Causal flux: Gen1->AntiGen1 (attraction) | probability |
//! | `Flux_Repu` | Causal flux: Gen1->Gen1 (repulsion) | probability |
//! | `Flux_Attr_Norm` | Normalized flux: Flux_Attr / |targets| | per-node prob |
//! | `Flux_Repu_Norm` | Normalized flux: Flux_Repu / |targets| | per-node prob |
//! | `P_Sterile` | Return probability for sterile prism nodes | probability |
//! | `dS_Sterile` | Spectral dimension for sterile prisms | dimensionless |
//! | `Mass_Gen1..3` | Average topological mass per generation | integer (avg) |
//! | `Mass_Anti1` | Average topological mass for antimatter | integer (avg) |
//! | `dS_vac_std` | Std dev of dS_vac across realisations | dimensionless |
//! | `dS_def_std` | Std dev of dS_def across realisations | dimensionless |
//! | `dS_Gen1..3_std` | Std dev of dS per generation | dimensionless |
//! | `dS_Anti1_std` | Std dev of dS antimatter | dimensionless |
//! | `dS_Sterile_std` | Std dev of dS sterile | dimensionless |
//! | `Flux_Attr_std` | Std dev of flux attraction | probability |
//! | `Flux_Repu_std` | Std dev of flux repulsion | probability |

pub mod csv;
pub mod summary;

pub use csv::CsvWriter;

use crate::phase2::topology::TopologySummary;
use crate::phase3::SpectralOutput;

/// Metadata for output file headers.
pub struct Metadata {
    pub n_points: usize,
    pub actual_m: usize,
    pub converged: bool,
    pub min_ensemble: usize,
    pub max_ensemble: usize,
    pub mode: String,
    pub epsilon: f64,
    pub tmax: usize,
    pub walkers: usize,
    pub seed: u64,
    pub timestamp: String,
    pub commit: String,
}

/// Serialise vacuum and defect spectral results to a CSV file.
///
/// Each row corresponds to one diffusion step t. The vacuum result provides
/// control measurements (unmodified Hasse graph); the defect result provides
/// measurements after Kuratowski contraction (Causal Prism + K_5 topology).
pub fn write_spectral_csv(
    path: &str,
    steps: &[u32],
    output: &SpectralOutput,
    meta: &Metadata,
) {
    let mut w = match CsvWriter::new(path, meta) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("  Failed to create {path}: {e}");
            return;
        }
    };

    w.header(&[
        "step", "P_vac", "dS_vac", "P_def", "dS_def",
        "P_Gen1", "dS_Gen1", "P_Gen2", "dS_Gen2", "P_Gen3", "dS_Gen3",
        "P_Anti1", "dS_Anti1",
        "Flux_Attr", "Flux_Repu", "Flux_Attr_Norm", "Flux_Repu_Norm",
        "P_Sterile", "dS_Sterile",
        "Mass_Gen1", "Mass_Gen2", "Mass_Gen3", "Mass_Anti1",
        "dS_vac_std", "dS_def_std",
        "dS_Gen1_std", "dS_Gen2_std", "dS_Gen3_std", "dS_Anti1_std", "dS_Sterile_std",
        "Flux_Attr_std", "Flux_Repu_std",
    ]);

    for (i, &step) in steps.iter().enumerate() {
        w.row_fmt(format_args!(
            "{},{:.15e},{:.4},{:.15e},{:.4},\
             {:.15e},{:.4},{:.15e},{:.4},{:.15e},{:.4},{:.15e},{:.4},\
             {:e},{:e},{:e},{:e},\
             {:.15e},{:.4},\
             {:.2},{:.2},{:.2},{:.2},\
             {:.6},{:.6},\
             {:.6},{:.6},{:.6},{:.6},{:.6},\
             {:e},{:e}",
            step,
            output.vacuum.p.get(i).unwrap_or(&0.0),
            output.vacuum.ds.get(i).unwrap_or(&0.0),
            output.defect.p.get(i).unwrap_or(&0.0),
            output.defect.ds.get(i).unwrap_or(&0.0),
            output.generations[0].p.get(i).unwrap_or(&0.0),
            output.generations[0].ds.get(i).unwrap_or(&0.0),
            output.generations[1].p.get(i).unwrap_or(&0.0),
            output.generations[1].ds.get(i).unwrap_or(&0.0),
            output.generations[2].p.get(i).unwrap_or(&0.0),
            output.generations[2].ds.get(i).unwrap_or(&0.0),
            output.generations[3].p.get(i).unwrap_or(&0.0),
            output.generations[3].ds.get(i).unwrap_or(&0.0),
            output.flux_attr.p.get(i).unwrap_or(&0.0),
            output.flux_repu.p.get(i).unwrap_or(&0.0),
            output.flux_attr_norm.get(i).unwrap_or(&0.0),
            output.flux_repu_norm.get(i).unwrap_or(&0.0),
            output.sterile.p.get(i).unwrap_or(&0.0),
            output.sterile.ds.get(i).unwrap_or(&0.0),
            output.mass[0],
            output.mass[1],
            output.mass[2],
            output.mass[3],
            output.vacuum.ds_std.get(i).unwrap_or(&0.0),
            output.defect.ds_std.get(i).unwrap_or(&0.0),
            output.generations[0].ds_std.get(i).unwrap_or(&0.0),
            output.generations[1].ds_std.get(i).unwrap_or(&0.0),
            output.generations[2].ds_std.get(i).unwrap_or(&0.0),
            output.generations[3].ds_std.get(i).unwrap_or(&0.0),
            output.sterile.ds_std.get(i).unwrap_or(&0.0),
            output.flux_attr.ds_std.get(i).unwrap_or(&0.0),
            output.flux_repu.ds_std.get(i).unwrap_or(&0.0),
        ));
    }
    println!("  Saved {path}");
}

/// Export a single-record topology summary as key-value CSV.
///
/// Contains the structural fingerprint of the generated universe:
/// prism counts, generation abundances, and average topological masses.
pub fn write_topology_csv(
    path: &str,
    topo: &TopologySummary,
    meta: &Metadata,
) {
    let mut w = match CsvWriter::new(path, meta) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("  Failed to create {path}: {e}");
            return;
        }
    };

    w.header(&["key", "value"]);
    w.row_fmt(format_args!("total_nodes,{}", topo.total_nodes));
    w.row_fmt(format_args!("total_prisms,{}", topo.total_prisms));
    w.row_fmt(format_args!("max_intermediates,{}", topo.max_intermediates));
    w.row_fmt(format_args!("count_gen1,{}", topo.count_gen1));
    w.row_fmt(format_args!("count_gen2,{}", topo.count_gen2));
    w.row_fmt(format_args!("count_gen3,{}", topo.count_gen3));
    w.row_fmt(format_args!("count_antigen1,{}", topo.count_antigen1));
    w.row_fmt(format_args!("count_sterile,{}", topo.count_sterile));
    w.row_fmt(format_args!("avg_mass_gen1,{:.4}", topo.avg_mass_gen1));
    w.row_fmt(format_args!("avg_mass_gen2,{:.4}", topo.avg_mass_gen2));
    w.row_fmt(format_args!("avg_mass_gen3,{:.4}", topo.avg_mass_gen3));
    w.row_fmt(format_args!("avg_mass_sterile,{:.4}", topo.avg_mass_sterile));
    w.row_fmt(format_args!("visible_mass_total,{}", topo.visible_mass_total));
    w.row_fmt(format_args!("dark_mass_total,{}", topo.dark_mass_total));
    w.row_fmt(format_args!("grav_mass_total,{}", topo.grav_mass_total));
    w.row_fmt(format_args!("omega_ratio,{:.6}", topo.omega_ratio));
    w.row_fmt(format_args!("phase_sq_total,{}", topo.phase_sq_total));
    w.row_fmt(format_args!("mass_sq_total,{}", topo.mass_sq_total));
    let q_topo = if topo.mass_sq_total > 0 {
        topo.phase_sq_total as f64 / topo.mass_sq_total as f64
    } else {
        0.0
    };
    w.row_fmt(format_args!("q_topo,{:.8}", q_topo));
    w.row_fmt(format_args!("alpha_em,{:.8}", topo.alpha_em));
    w.row_fmt(format_args!("omega_energy,{:.8}", topo.omega_energy));
    w.row_fmt(format_args!("phase_pos_count,{}", topo.phase_pos_count));
    w.row_fmt(format_args!("phase_zero_count,{}", topo.phase_zero_count));
    w.row_fmt(format_args!("phase_neg_count,{}", topo.phase_neg_count));
    w.row_fmt(format_args!("prisms_gen1,{}", topo.prisms_gen1));
    w.row_fmt(format_args!("prisms_gen2,{}", topo.prisms_gen2));
    w.row_fmt(format_args!("prisms_gen3,{}", topo.prisms_gen3));
    println!("  Saved {path}");
}

/// Export the prism mass spectrum histogram as a two-column CSV.
///
/// Each row maps a belly size N (number of intermediates in a K_{2,N} prism)
/// to the frequency of committed prisms with that exact belly size.
pub fn write_mass_spectrum_csv(
    path: &str,
    histogram: &[(usize, usize)],
    meta: &Metadata,
) {
    let mut w = match CsvWriter::new(path, meta) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("  Failed to create {path}: {e}");
            return;
        }
    };

    w.header(&["intermediates_N", "frequency"]);
    for &(n, freq) in histogram {
        w.row_fmt(format_args!("{n},{freq}"));
    }
    println!("  Saved {path}");
}
