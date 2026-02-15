//! Phase 4 — CSV Output
//!
//! Writes results to a CSV file plottable with gnuplot or matplotlib.

use crate::spectral::SpectralResult;
use std::io::Write;

pub fn write_csv(
    path: &str,
    steps: &[u32],
    vacuum: &SpectralResult,
    defect: &SpectralResult,
) {
    let file = std::fs::File::create(path);
    let mut file = match file {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  Failed to create {}: {}", path, e);
            return;
        }
    };

    let _ = writeln!(
        file,
        "step,P_vac,dS_vac,P_def,dS_def,P_loc_vac,dS_loc_vac,P_loc_def,dS_loc_def"
    );
    for i in 0..steps.len() {
        let _ = writeln!(
            file,
            "{},{:.6e},{:.4},{:.6e},{:.4},{:.6e},{:.4},{:.6e},{:.4}",
            steps[i],
            vacuum.p_global[i],
            vacuum.ds_global[i],
            defect.p_global[i],
            defect.ds_global[i],
            vacuum.p_local[i],
            vacuum.ds_local[i],
            defect.p_local[i],
            defect.ds_local[i],
        );
    }
    println!("  Saved {}", path);
}
