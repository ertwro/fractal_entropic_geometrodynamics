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
//! | `P_Gen1..3` | Return probability for generations 1–3 | probability |
//! | `dS_Gen1..3` | Spectral dimension per generation | dimensionless |
//! | `P_Anti1` | Return probability for antimatter | probability |
//! | `dS_Anti1` | Spectral dimension of antimatter | dimensionless |
//! | `Flux_Attr` | Causal flux: Gen1→AntiGen1 (attraction) | probability |
//! | `Flux_Repu` | Causal flux: Gen1→Gen1 (repulsion) | probability |
//! | `Flux_Attr_Norm` | Normalized flux: Flux_Attr / |targets| (per-charge def.) | per-node prob |
//! | `Flux_Repu_Norm` | Normalized flux: Flux_Repu / |targets| (per-charge def.) | per-node prob |
//! | `P_Sterile` | Return probability for sterile prism nodes (Φ=0) | probability |
//! | `dS_Sterile` | Spectral dimension for sterile prisms (C6) | dimensionless |
//! | `Mass_Gen1..3` | Average topological mass (N) per generation | integer (avg) |
//! | `Mass_Anti1` | Average topological mass for antimatter | integer (avg) |
//! | `dS_vac_std` | Std dev of dS_vac across realisations | dimensionless |
//! | `dS_def_std` | Std dev of dS_def across realisations | dimensionless |
//! | `dS_Gen1..3_std` | Std dev of dS per generation | dimensionless |
//! | `dS_Anti1_std` | Std dev of dS antimatter | dimensionless |
//! | `dS_Sterile_std` | Std dev of dS sterile | dimensionless |
//! | `Flux_Attr_std` | Std dev of flux attraction | probability |
//! | `Flux_Repu_std` | Std dev of flux repulsion | probability |

use crate::skyrmion::TopologySummary;
use crate::spectral::SpectralResult;
use std::io::Write;

/// Serialise vacuum and defect spectral results to a CSV file.
///
/// Each row corresponds to one diffusion step t. The vacuum result provides
/// control measurements (unmodified Hasse graph); the defect result provides
/// measurements after Kuratowski contraction (Causal Prism + K₅ topology).
pub fn write_csv(
    path: &str,
    steps: &[u32],
    vacuum: &SpectralResult,
    defect: &SpectralResult,
    metadata: &str,
) {
    let file = std::fs::File::create(path);
    let mut file = match file {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  Failed to create {path}: {e}");
            return;
        }
    };

    for line in metadata.lines() {
        let _ = writeln!(file, "# {line}");
    }

    let _ = writeln!(
        file,
        "step,P_vac,dS_vac,P_def,dS_def,\
         P_Gen1,dS_Gen1,P_Gen2,dS_Gen2,P_Gen3,dS_Gen3,P_Anti1,dS_Anti1,\
         Flux_Attr,Flux_Repu,Flux_Attr_Norm,Flux_Repu_Norm,\
         P_Sterile,dS_Sterile,\
         Mass_Gen1,Mass_Gen2,Mass_Gen3,Mass_Anti1,\
         dS_vac_std,dS_def_std,\
         dS_Gen1_std,dS_Gen2_std,dS_Gen3_std,dS_Anti1_std,dS_Sterile_std,\
         Flux_Attr_std,Flux_Repu_std"
    );
    for (i, &step) in steps.iter().enumerate() {
        let _ = writeln!(
            file,
            "{},{:.15e},{:.4},{:.15e},{:.4},\
             {:.15e},{:.4},{:.15e},{:.4},{:.15e},{:.4},{:.15e},{:.4},\
             {:e},{:e},{:e},{:e},\
             {:.15e},{:.4},\
             {:.2},{:.2},{:.2},{:.2},\
             {:.6},{:.6},\
             {:.6},{:.6},{:.6},{:.6},{:.6},\
             {:e},{:e}",
            step,
            vacuum.p_global.get(i).unwrap_or(&0.0),
            vacuum.ds_global.get(i).unwrap_or(&0.0),
            defect.p_global.get(i).unwrap_or(&0.0),
            defect.ds_global.get(i).unwrap_or(&0.0),
            defect.p_gen1.get(i).unwrap_or(&0.0),
            defect.ds_gen1.get(i).unwrap_or(&0.0),
            defect.p_gen2.get(i).unwrap_or(&0.0),
            defect.ds_gen2.get(i).unwrap_or(&0.0),
            defect.p_gen3.get(i).unwrap_or(&0.0),
            defect.ds_gen3.get(i).unwrap_or(&0.0),
            defect.p_anti1.get(i).unwrap_or(&0.0),
            defect.ds_anti1.get(i).unwrap_or(&0.0),
            defect.flux_attraction.get(i).unwrap_or(&0.0),
            defect.flux_repulsion.get(i).unwrap_or(&0.0),
            defect.flux_attr_norm.get(i).unwrap_or(&0.0),
            defect.flux_repu_norm.get(i).unwrap_or(&0.0),
            defect.p_sterile.get(i).unwrap_or(&0.0),
            defect.ds_sterile.get(i).unwrap_or(&0.0),
            defect.mass_gen1,
            defect.mass_gen2,
            defect.mass_gen3,
            defect.mass_anti1,
            vacuum.ds_global_std.get(i).unwrap_or(&0.0),
            defect.ds_global_std.get(i).unwrap_or(&0.0),
            defect.ds_gen1_std.get(i).unwrap_or(&0.0),
            defect.ds_gen2_std.get(i).unwrap_or(&0.0),
            defect.ds_gen3_std.get(i).unwrap_or(&0.0),
            defect.ds_anti1_std.get(i).unwrap_or(&0.0),
            defect.ds_sterile_std.get(i).unwrap_or(&0.0),
            defect.flux_attraction_std.get(i).unwrap_or(&0.0),
            defect.flux_repulsion_std.get(i).unwrap_or(&0.0),
        );
    }
    println!("  Saved {path}");
}

/// Export a single-record topology summary as key-value CSV.
///
/// Contains the structural fingerprint of the generated universe:
/// prism counts, generation abundances, and average topological masses.
pub fn export_topology_summary(
    path: &str,
    topo: &TopologySummary,
    metadata: &str,
) {
    let file = std::fs::File::create(path);
    let mut file = match file {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  Failed to create {path}: {e}");
            return;
        }
    };

    for line in metadata.lines() {
        let _ = writeln!(file, "# {line}");
    }

    let _ = writeln!(file, "key,value");
    let _ = writeln!(file, "total_nodes,{}", topo.total_nodes);
    let _ = writeln!(file, "total_prisms,{}", topo.total_prisms);
    let _ = writeln!(file, "max_intermediates,{}", topo.max_intermediates);
    let _ = writeln!(file, "count_gen1,{}", topo.count_gen1);
    let _ = writeln!(file, "count_gen2,{}", topo.count_gen2);
    let _ = writeln!(file, "count_gen3,{}", topo.count_gen3);
    let _ = writeln!(file, "count_antigen1,{}", topo.count_antigen1);
    let _ = writeln!(file, "count_sterile,{}", topo.count_sterile);
    let _ = writeln!(file, "avg_mass_gen1,{:.4}", topo.avg_mass_gen1);
    let _ = writeln!(file, "avg_mass_gen2,{:.4}", topo.avg_mass_gen2);
    let _ = writeln!(file, "avg_mass_gen3,{:.4}", topo.avg_mass_gen3);
    let _ = writeln!(file, "avg_mass_sterile,{:.4}", topo.avg_mass_sterile);
    let _ = writeln!(file, "visible_mass_total,{}", topo.visible_mass_total);
    let _ = writeln!(file, "dark_mass_total,{}", topo.dark_mass_total);
    let _ = writeln!(file, "grav_mass_total,{}", topo.grav_mass_total);
    let _ = writeln!(file, "omega_ratio,{:.6}", topo.omega_ratio);
    let _ = writeln!(file, "phase_sq_total,{}", topo.phase_sq_total);
    let _ = writeln!(file, "mass_sq_total,{}", topo.mass_sq_total);
    let q_topo = if topo.mass_sq_total > 0 {
        topo.phase_sq_total as f64 / topo.mass_sq_total as f64
    } else { 0.0 };
    let _ = writeln!(file, "q_topo,{:.8}", q_topo);
    let _ = writeln!(file, "alpha_em,{:.8}", topo.alpha_em);
    let _ = writeln!(file, "omega_energy,{:.8}", topo.omega_energy);
    let _ = writeln!(file, "phase_pos_count,{}", topo.phase_pos_count);
    let _ = writeln!(file, "phase_zero_count,{}", topo.phase_zero_count);
    let _ = writeln!(file, "phase_neg_count,{}", topo.phase_neg_count);
    let _ = writeln!(file, "prisms_gen1,{}", topo.prisms_gen1);
    let _ = writeln!(file, "prisms_gen2,{}", topo.prisms_gen2);
    let _ = writeln!(file, "prisms_gen3,{}", topo.prisms_gen3);
    println!("  Saved {path}");
}

/// Export the prism mass spectrum histogram as a two-column CSV.
///
/// Each row maps a belly size N (number of intermediates in a K_{2,N} prism)
/// to the frequency of committed prisms with that exact belly size.
pub fn export_mass_spectrum(
    path: &str,
    histogram: &[(usize, usize)],
    metadata: &str,
) {
    let file = std::fs::File::create(path);
    let mut file = match file {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  Failed to create {path}: {e}");
            return;
        }
    };

    for line in metadata.lines() {
        let _ = writeln!(file, "# {line}");
    }

    let _ = writeln!(file, "intermediates_N,frequency");
    for &(n, freq) in histogram {
        let _ = writeln!(file, "{n},{freq}");
    }
    println!("  Saved {path}");
}

/// Export the directed 4-layer future light cone of every node as CSV.
///
/// For each node `src`, performs a layered BFS through the **directed**
/// Hasse DAG (following only forward edges `v > src`, i.e. causal future).
/// Writes `(source, target, layer)` triples for layers 1..`max_depth`.
///
/// The vacuum CSR from Phase 1 is directed: `adj[u]` lists the children
/// (causal future) of `u`.  If a symmetric CSR is passed instead, the
/// `v > node` filter recovers the forward direction.
///
/// Intended for N ≤ 3 000 exact-diag graphs.  Used by `gue_correlation_bd.py`
/// to construct the Benincasa–Dowker matrix with weights `[+1, -9, +16, -8]`.
pub fn export_lightcone(
    path: &str,
    adj_head: &[u32],
    adj_data: &[u32],
    n: usize,
    max_depth: usize,
    metadata: &str,
) {
    let file = std::fs::File::create(path);
    let file = match file {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  Failed to create {path}: {e}");
            return;
        }
    };
    let mut w = std::io::BufWriter::with_capacity(1 << 20, file);

    for line in metadata.lines() {
        let _ = writeln!(w, "# {line}");
    }
    let _ = writeln!(w, "source,target,layer");

    let mut visited = vec![false; n];
    let mut count: usize = 0;

    for src in 0..n {
        // Reset visited for this source
        for v in visited.iter_mut() {
            *v = false;
        }
        visited[src] = true;

        let mut frontier: Vec<usize> = vec![src];

        for depth in 1..=max_depth {
            let mut next: Vec<usize> = Vec::new();
            for &node in &frontier {
                let lo = adj_head[node] as usize;
                let hi = adj_head[node + 1] as usize;
                for &nbr_u32 in &adj_data[lo..hi] {
                    let nbr = nbr_u32 as usize;
                    // Forward edge: nbr is in the causal future of node
                    if nbr > node && !visited[nbr] {
                        visited[nbr] = true;
                        next.push(nbr);
                        let _ = writeln!(w, "{src},{nbr},{depth}");
                        count += 1;
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
    }

    let _ = w.flush();
    println!("  Lightcone: {count} edges → {path}");
}
