// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Terminal summary output.
//!
//! Prints the same summary block that the old main.rs produced after Phase 4,
//! including spectral dimensions, mass spectrum, P saturation, topology
//! summary, and all measurement sub-summaries (via `measure::print_all_summaries`).

use crate::measure::MeasureResults;
use crate::phase2::topology::TopologySummary;
use crate::phase3::SpectralOutput;

/// Format a duration in seconds into a human-readable string (e.g. "3m 42s").
fn fmt_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{secs:.0}s")
    } else if secs < 3600.0 {
        format!("{}m {:02}s", secs as u64 / 60, secs as u64 % 60)
    } else {
        let h = secs as u64 / 3600;
        let m = (secs as u64 % 3600) / 60;
        format!("{h}h {m:02}m")
    }
}

/// Print the full terminal summary for an ensemble run.
///
/// Mirrors the summary block from the old monolithic main.rs (lines 1427-1611).
/// Measurement sub-summaries are delegated to `crate::measure::print_all_summaries`.
pub fn print_summary(
    output: &SpectralOutput,
    topo: &TopologySummary,
    meas: Option<&MeasureResults>,
    steps: &[u32],
    actual_m: usize,
    converged: bool,
    elapsed: f64,
) {
    let mid = steps.len() / 2;
    let last = steps.len() - 1;
    let walks_empty = output.vacuum.p.is_empty();

    println!(
        "\n\u{2500}\u{2500} Summary ({actual_m} realisations, converged={converged}) \
         \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}"
    );

    if walks_empty {
        println!("  d_S: skipped (topology-only)");
        println!("  P saturation: skipped (topology-only)");
    } else {
        // Spectral dimensions at midpoint
        println!(
            "  d_S vacuum global  (t={}): {:.2}",
            steps[mid],
            output.vacuum.ds.get(mid).unwrap_or(&0.0),
        );
        println!(
            "  d_S defect global  (t={}): {:.2}",
            steps[mid],
            output.defect.ds.get(mid).unwrap_or(&0.0),
        );
        println!(
            "  d_S core on vacuum (t={}): {:.2}",
            steps[mid],
            output.vac_core.ds.get(mid).unwrap_or(&0.0),
        );
        println!(
            "  d_S core on defect (t={}): {:.2}",
            steps[mid],
            output.def_core.ds.get(mid).unwrap_or(&0.0),
        );

        // P saturation at last step
        println!("  \u{2500}\u{2500} P saturation (t={}) \u{2500}\u{2500}", steps[last]);
        let p_loc_vac = output.vac_core.p.get(last).copied().unwrap_or(0.0);
        let p_loc_def = output.def_core.p.get(last).copied().unwrap_or(0.0);
        let ratio = if p_loc_vac > 0.0 { p_loc_def / p_loc_vac } else { 0.0 };
        println!(
            "  P_loc_vac = {:.6e}  P_loc_def = {:.6e}  ratio = {:.4}",
            p_loc_vac, p_loc_def, ratio,
        );
    }

    // Mass spectrum
    println!("  \u{2500}\u{2500} Mass Spectrum (avg N) \u{2500}\u{2500}");
    println!(
        "  Gen1 = {:.2}, Gen2 = {:.2}, Gen3 = {:.2}, Anti1 = {:.2}",
        output.mass[0], output.mass[1], output.mass[2], output.mass[3],
    );

    // Topology summary
    println!("  \u{2500}\u{2500} Topology \u{2500}\u{2500}");
    println!(
        "  Prisms: {}, max_belly: {}, nodes: {}",
        topo.total_prisms, topo.max_intermediates, topo.total_nodes,
    );
    println!(
        "  Generations: gen1={}, gen2={}, gen3={}, anti1={}, sterile={}",
        topo.count_gen1, topo.count_gen2, topo.count_gen3,
        topo.count_antigen1, topo.count_sterile,
    );
    println!(
        "  Dark/Visible = {:.2}, alpha_em = {:.8}, Omega_energy = {:.2}",
        topo.omega_ratio, topo.alpha_em, topo.omega_energy,
    );

    // Measurement sub-summaries
    if let Some(meas) = meas {
        crate::measure::print_all_summaries(meas);
    }

    println!("  Total time: {}", fmt_duration(elapsed));
    println!(
        "\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}"
    );
}
