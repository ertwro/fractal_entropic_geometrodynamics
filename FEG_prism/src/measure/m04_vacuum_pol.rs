// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M4 — Bifurcated Beta-Functions (Gauge vs Gravity/Confinement)
//!
//! The universe has a bi-metric space: two independent spatial primes
//! (3 and 5) constructing space via two different Kuratowski obstructions.
//!
//! - **K_{3,3} → prime 3** → SU(3)×SU(2) gauge structure → z3 coordinate
//! - **K_5 → prime 5** → gravity/confinement → z5 coordinate
//! - **z2 is temporal** — not a spatial axis
//!
//! Two decoupled scans:
//! - **β_gauge(b)**: scan `z3 mod 3^b` — tracks K_{3,3} prism resolution
//! - **β_grav(c)**: scan `z5 mod 5^c` — tracks K_5 defect resolution

use super::context::MeasureContext;
use crate::output::CsvWriter;

/// Quantize a spatial coordinate to a u64 label for modular binning.
/// Maps the coordinate range [min, max] to [0, SCALE) where SCALE is large
/// enough for meaningful modular arithmetic at multiple resolutions.
const QUANT_SCALE: f64 = 1e15;

#[inline]
fn quantize(val: f64, min_val: f64, span: f64) -> u64 {
    if span <= 0.0 { return 0; }
    (((val - min_val) / span) * QUANT_SCALE) as u64
}

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GaugeBin {
    pub b: u32,           // exponent: resolution = 3^b
    pub n_resolved: usize,
    pub n_unresolved: usize,
    pub phase_sq: usize,
    pub mass_sq: usize,
    pub q: f64,
    pub alpha: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GravBin {
    pub c: u32,           // exponent: resolution = 5^c
    pub n_resolved: usize,
    pub n_unresolved: usize,
    pub phase_sq: usize,
    pub mass_sq: usize,
    pub q: f64,
    pub alpha: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VacuumPolResult {
    pub gauge_bins: Vec<GaugeBin>,
    pub grav_bins: Vec<GravBin>,
    pub global_q: f64,
    pub global_alpha: f64,
    pub mean_local_q: f64,
}

// ── Pre-computed prism info (local to run()) ─────────────────────────────────

struct PrismInfo {
    all_nodes: Vec<usize>,
    phi_abs: usize,
    n_inter: usize,
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> VacuumPolResult {
    let coords = ctx.sorted_coords;

    // Global Q_topo from topology summary
    let global_q = if ctx.topology.mass_sq_total > 0 {
        ctx.topology.phase_sq_total as f64 / ctx.topology.mass_sq_total as f64
    } else {
        0.0
    };
    let global_alpha = global_q / (8.0 * std::f64::consts::PI);

    if ctx.prisms.is_empty() {
        return VacuumPolResult {
            gauge_bins: vec![],
            grav_bins: vec![],
            global_q,
            global_alpha,
            mean_local_q: global_q,
        };
    }

    // ── Pre-compute prism info ───────────────────────────────────────────
    let prism_info: Vec<PrismInfo> = ctx
        .prisms
        .iter()
        .map(|p| {
            let mut all_nodes = Vec::with_capacity(2 + p.intermediates.len());
            all_nodes.push(p.origin);
            all_nodes.push(p.destination);
            all_nodes.extend_from_slice(&p.intermediates);

            let net_phase: i32 = p
                .intermediates
                .iter()
                .map(|&w| ctx.momentum[w].signum())
                .sum();
            PrismInfo {
                all_nodes,
                phi_abs: net_phase.unsigned_abs() as usize,
                n_inter: p.intermediates.len(),
            }
        })
        .collect();

    // ── Pre-compute quantized spatial coords for each prism's nodes ─────
    // Determine coordinate ranges for quantization
    let (mut min_x, mut max_x) = (f64::MAX, f64::MIN);
    let (mut min_y, mut max_y) = (f64::MAX, f64::MIN);
    for info in &prism_info {
        for &nd in &info.all_nodes {
            let c = &coords[nd];
            if c[1] < min_x { min_x = c[1]; }
            if c[1] > max_x { max_x = c[1]; }
            if c[2] < min_y { min_y = c[2]; }
            if c[2] > max_y { max_y = c[2]; }
        }
    }
    let span_x = max_x - min_x;
    let span_y = max_y - min_y;

    let prism_z3: Vec<Vec<u64>> = prism_info
        .iter()
        .map(|info| info.all_nodes.iter().map(|&nd| quantize(coords[nd][1], min_x, span_x)).collect())
        .collect();

    let prism_z5: Vec<Vec<u64>> = prism_info
        .iter()
        .map(|info| info.all_nodes.iter().map(|&nd| quantize(coords[nd][2], min_y, span_y)).collect())
        .collect();

    // ── Gauge scan: z3 mod 3^b ───────────────────────────────────────────
    let mut gauge_bins: Vec<GaugeBin> = Vec::new();
    for b in 0u32.. {
        let mod3 = match 3u64.checked_pow(b) {
            Some(v) => v,
            None => break,
        };

        let mut n_resolved: usize = 0;
        let mut n_unresolved: usize = 0;
        let mut phase_sq: usize = 0;
        let mut mass_sq: usize = 0;

        for (pi, info) in prism_info.iter().enumerate() {
            let z3s = &prism_z3[pi];
            let cell3 = z3s[0] % mod3;
            let resolved = z3s[1..].iter().all(|&z3| z3 % mod3 == cell3);

            if resolved {
                n_resolved += 1;
                phase_sq += info.phi_abs * info.phi_abs;
                mass_sq += info.n_inter * info.n_inter;
            } else {
                n_unresolved += 1;
            }
        }

        if n_resolved == 0 {
            break;
        }

        let q = if mass_sq > 0 {
            phase_sq as f64 / mass_sq as f64
        } else {
            0.0
        };
        let alpha = q / (8.0 * std::f64::consts::PI);

        gauge_bins.push(GaugeBin {
            b,
            n_resolved,
            n_unresolved,
            phase_sq,
            mass_sq,
            q,
            alpha,
        });
    }

    // ── Gravity scan: z5 mod 5^c ─────────────────────────────────────────
    let mut grav_bins: Vec<GravBin> = Vec::new();
    for c in 0u32.. {
        let mod5 = match 5u64.checked_pow(c) {
            Some(v) => v,
            None => break,
        };

        let mut n_resolved: usize = 0;
        let mut n_unresolved: usize = 0;
        let mut phase_sq: usize = 0;
        let mut mass_sq: usize = 0;

        for (pi, info) in prism_info.iter().enumerate() {
            let z5s = &prism_z5[pi];
            let cell5 = z5s[0] % mod5;
            let resolved = z5s[1..].iter().all(|&z5| z5 % mod5 == cell5);

            if resolved {
                n_resolved += 1;
                phase_sq += info.phi_abs * info.phi_abs;
                mass_sq += info.n_inter * info.n_inter;
            } else {
                n_unresolved += 1;
            }
        }

        if n_resolved == 0 {
            break;
        }

        let q = if mass_sq > 0 {
            phase_sq as f64 / mass_sq as f64
        } else {
            0.0
        };
        let alpha = q / (8.0 * std::f64::consts::PI);

        grav_bins.push(GravBin {
            c,
            n_resolved,
            n_unresolved,
            phase_sq,
            mass_sq,
            q,
            alpha,
        });
    }

    VacuumPolResult {
        gauge_bins,
        grav_bins,
        global_q,
        global_alpha,
        mean_local_q: global_q,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[VacuumPolResult]) -> VacuumPolResult {
    let global_q = results.iter().map(|r| r.global_q).sum::<f64>() / results.len() as f64;
    let global_alpha = global_q / (8.0 * std::f64::consts::PI);

    // Aggregate gauge bins: join by b
    let max_b = results.iter().flat_map(|r| r.gauge_bins.iter().map(|g| g.b)).max().unwrap_or(0);
    let mut gauge_bins: Vec<GaugeBin> = Vec::new();
    for b in 0..=max_b {
        let mut total_phase_sq: usize = 0;
        let mut total_mass_sq: usize = 0;
        let mut total_resolved: usize = 0;
        let mut total_unresolved: usize = 0;

        for r in results {
            if let Some(bin) = r.gauge_bins.iter().find(|x| x.b == b) {
                total_phase_sq += bin.phase_sq;
                total_mass_sq += bin.mass_sq;
                total_resolved += bin.n_resolved;
                total_unresolved += bin.n_unresolved;
            }
        }

        if total_resolved == 0 { continue; }

        let q = if total_mass_sq > 0 {
            total_phase_sq as f64 / total_mass_sq as f64
        } else {
            0.0
        };
        let alpha = q / (8.0 * std::f64::consts::PI);

        gauge_bins.push(GaugeBin {
            b,
            n_resolved: total_resolved,
            n_unresolved: total_unresolved,
            phase_sq: total_phase_sq,
            mass_sq: total_mass_sq,
            q,
            alpha,
        });
    }

    // Aggregate grav bins: join by c
    let max_c = results.iter().flat_map(|r| r.grav_bins.iter().map(|g| g.c)).max().unwrap_or(0);
    let mut grav_bins: Vec<GravBin> = Vec::new();
    for c in 0..=max_c {
        let mut total_phase_sq: usize = 0;
        let mut total_mass_sq: usize = 0;
        let mut total_resolved: usize = 0;
        let mut total_unresolved: usize = 0;

        for r in results {
            if let Some(bin) = r.grav_bins.iter().find(|x| x.c == c) {
                total_phase_sq += bin.phase_sq;
                total_mass_sq += bin.mass_sq;
                total_resolved += bin.n_resolved;
                total_unresolved += bin.n_unresolved;
            }
        }

        if total_resolved == 0 { continue; }

        let q = if total_mass_sq > 0 {
            total_phase_sq as f64 / total_mass_sq as f64
        } else {
            0.0
        };
        let alpha = q / (8.0 * std::f64::consts::PI);

        grav_bins.push(GravBin {
            c,
            n_resolved: total_resolved,
            n_unresolved: total_unresolved,
            phase_sq: total_phase_sq,
            mass_sq: total_mass_sq,
            q,
            alpha,
        });
    }

    VacuumPolResult {
        gauge_bins,
        grav_bins,
        global_q,
        global_alpha,
        mean_local_q: global_q,
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &VacuumPolResult, w: &mut CsvWriter) {
    w.comment("M4 Gauge Beta-Function (z3 mod 3^b)");
    w.header(&[
        "b",
        "n_resolved",
        "n_unresolved",
        "phase_sq",
        "mass_sq",
        "q",
        "alpha",
    ]);
    for g in &result.gauge_bins {
        w.row_fmt(format_args!(
            "{},{},{},{},{},{:.6},{:.8}",
            g.b, g.n_resolved, g.n_unresolved, g.phase_sq, g.mass_sq, g.q, g.alpha
        ));
    }

    w.comment("M4 Gravity Beta-Function (z5 mod 5^c)");
    w.header(&[
        "c",
        "n_resolved",
        "n_unresolved",
        "phase_sq",
        "mass_sq",
        "q",
        "alpha",
    ]);
    for g in &result.grav_bins {
        w.row_fmt(format_args!(
            "{},{},{},{},{},{:.6},{:.8}",
            g.c, g.n_resolved, g.n_unresolved, g.phase_sq, g.mass_sq, g.q, g.alpha
        ));
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &VacuumPolResult) {
    println!("  [M4] Bifurcated Beta-Functions:");
    println!(
        "    Global Q_topo:   {:.4}  (1/alpha = {:.1})",
        result.global_q,
        if result.global_alpha > 0.0 {
            1.0 / result.global_alpha
        } else {
            0.0
        }
    );

    // Gauge track
    if !result.gauge_bins.is_empty() {
        let total = result.gauge_bins[0].n_resolved + result.gauge_bins[0].n_unresolved;
        let active = result.gauge_bins.iter().filter(|g| g.n_resolved > 0).count();
        println!("    β_gauge (z3 mod 3^b): {} active levels ({} prisms)", active, total);
        for g in result.gauge_bins.iter().filter(|g| g.n_resolved > 0) {
            let inv_alpha = if g.alpha > 0.0 { 1.0 / g.alpha } else { 0.0 };
            println!(
                "      b={}: Q={:.4}, 1/α={:.1}  (resolved={}/{})",
                g.b, g.q, inv_alpha,
                g.n_resolved, g.n_resolved + g.n_unresolved
            );
        }
    }

    // Grav track
    if !result.grav_bins.is_empty() {
        let total = result.grav_bins[0].n_resolved + result.grav_bins[0].n_unresolved;
        let active = result.grav_bins.iter().filter(|g| g.n_resolved > 0).count();
        println!("    β_grav (z5 mod 5^c): {} active levels ({} prisms)", active, total);
        for g in result.grav_bins.iter().filter(|g| g.n_resolved > 0) {
            let inv_alpha = if g.alpha > 0.0 { 1.0 / g.alpha } else { 0.0 };
            println!(
                "      c={}: Q={:.4}, 1/α={:.1}  (resolved={}/{})",
                g.c, g.q, inv_alpha,
                g.n_resolved, g.n_resolved + g.n_unresolved
            );
        }
    }
}
