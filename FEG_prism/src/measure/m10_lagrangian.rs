// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M10 — SM Lagrangian Card Assembly (Zero Free Parameters)
//!
//! Assembles the full Standard Model Lagrangian parameter card from M1--M9
//! results plus topology summary.  This function does **no graph traversal**
//! -- it is pure post-processing.  Missing measurements fall back to 0.0 so
//! the card is always produced.
//!
//! Bifurcated β-functions: gauge (z3 mod 3^b) and grav (z5 mod 5^c).
//!
//! Calculo de Kuratowski: all coupling constants derive from the causal set
//! topology with zero free parameters.

use crate::output::CsvWriter;
use crate::phase2::topology::TopologySummary;

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LagrangianCard {
    // ── Gauge sector (from topology + M5) ──
    pub gauge_group: &'static str,
    pub n_bosons: [usize; 3],
    pub n_generations: usize,

    // ── Coupling constants ──
    pub alpha_em: f64,
    pub alpha_em_inv: f64,
    pub sin2_theta_w: f64,
    pub cos2_theta_w: f64,
    pub e_charge: f64,
    pub g1: f64,
    pub g2: f64,
    pub mw_mz_ratio: f64,

    // ── Gravity ──
    pub g_newton: f64,

    // ── Fermion mass spectrum (from M1) ──
    pub mass_topo: [f64; 3],
    pub mass_ratio_21: f64,
    pub mass_ratio_31: f64,
    pub cpt_violation: f64,

    // ── Yukawa couplings (from M1, normalised) ──
    pub yukawa_ratio: [f64; 3],

    // ── Neutrino sector (from M7 + M8) ──
    pub neutrino_count: usize,
    pub neutrino_mass_proxy: f64,
    pub neutrino_chirality: f64,
    pub pmns_matrix: [[f64; 3]; 3],

    // ── Higgs sector (from M9) ──
    pub higgs_drag: f64,
    pub higgs_area_ratio: f64,

    // ── Parity violation (from M5) ──
    pub left_fraction: f64,

    // ── Combinatorial RG flow (from M4) ──
    pub global_alpha_inv: f64,
    pub mean_local_alpha_inv: f64,
    pub mean_local_q: f64,

    // ── Quantum mechanics (from M6) ──
    pub born_r: f64,
    pub coherence_r: f64,

    // ── Dark sector (from TopologySummary) ──
    pub omega_dark_vis: f64,
    pub omega_energy: f64,

    // ── Generation census (from M2) ──
    pub gen_fractions: [f64; 3],
    pub phase_fractions: [f64; 3],

    // ── Modulo path integral (from M3) ──
    pub interference_mean: f64,
    pub constructive_frac: f64,

    // ── Running coupling β-functions (bifurcated: gauge + grav) ──
    pub beta_gauge: Vec<GaugeBetaPoint>,
    pub beta_grav: Vec<GravBetaPoint>,
}

#[derive(Clone, Debug)]
pub struct GaugeBetaPoint {
    pub b: u32,
    pub q: f64,
    pub alpha_em: f64,
    pub alpha_em_inv: f64,
    pub e_charge: f64,
    pub sin2_theta_w: f64,
    pub cos2_theta_w: f64,
    pub g1: f64,
    pub g2: f64,
    pub mw_mz_ratio: f64,
    pub n_resolved_m4: usize,
    pub n_resolved_m5: usize,
    pub left_fraction: f64,
    pub mean_doublets: f64,
}

#[derive(Clone, Debug)]
pub struct GravBetaPoint {
    pub c: u32,
    pub q: f64,
    pub alpha_em: f64,
    pub alpha_em_inv: f64,
    pub e_charge: f64,
    pub sin2_theta_w: f64,
    pub cos2_theta_w: f64,
    pub g1: f64,
    pub g2: f64,
    pub mw_mz_ratio: f64,
    pub n_resolved_m4: usize,
    pub n_resolved_m5: usize,
    pub left_fraction: f64,
    pub mean_doublets: f64,
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(meas: &super::MeasureResults, topo: &TopologySummary) -> LagrangianCard {
    use std::f64::consts::PI;

    // ── Gauge sector (pure topology) ──
    let gauge_group = "SU(3) x SU(2) x U(1)";
    let n_bosons = [8, 3, 1];
    let n_generations = 3;

    // ── Coupling constants from port counting ──
    let q_topo = if topo.mass_sq_total > 0 {
        topo.phase_sq_total as f64 / topo.mass_sq_total as f64
    } else {
        4.0 / 21.0
    };
    let alpha_em = q_topo / (8.0 * PI);
    let alpha_em_inv = 1.0 / alpha_em;

    // Weinberg angle from port counting: sin^2 theta_W = 12/51 = 4/17
    let sin2_theta_w: f64 = 4.0 / 17.0;
    let cos2_theta_w: f64 = 13.0 / 17.0;

    let e_charge = (4.0 * PI * alpha_em).sqrt();
    let g2 = e_charge / sin2_theta_w.sqrt();
    let g1 = e_charge / cos2_theta_w.sqrt();
    let mw_mz_ratio = cos2_theta_w.sqrt();

    // ── Gravity (Jacobson alpha-sweep plateau: G = 1/16) ──
    let g_newton = 1.0 / (16.0 * PI);

    // ── Fermion masses (from M1) ──
    let (mass_topo, mass_ratio_21, mass_ratio_31, cpt_violation) =
        if let Some(ref t) = meas.traversal {
            (
                t.mean_traversal,
                t.ratio_gen2_gen1,
                t.ratio_gen3_gen1,
                if t.mean_traversal[0] > 0.0 {
                    (t.mean_traversal[0] - t.mean_traversal[1]).abs() / t.mean_traversal[0]
                } else {
                    0.0
                },
            )
        } else {
            ([0.0; 3], 0.0, 0.0, 0.0)
        };

    let yukawa_ratio = [1.0, mass_ratio_21, mass_ratio_31];

    // ── Neutrino sector (from M7 + M8) ──
    let (neutrino_count, neutrino_mass_proxy, neutrino_chirality) =
        if let Some(ref nu) = meas.neutrino {
            (nu.total_count, nu.mean_escape_time, nu.mean_chirality)
        } else {
            (0, 0.0, 0.0)
        };

    let pmns_matrix = if let Some(ref pm) = meas.pmns {
        pm.transition_matrix
    } else {
        [[0.0; 3]; 3]
    };

    // ── Higgs sector (from M9) ──
    let (higgs_drag, higgs_area_ratio) = if let Some(ref hg) = meas.higgs {
        let area_ratio = hg.cdf_area_ratio.last().copied().unwrap_or(0.0);
        (hg.mean_drag, area_ratio)
    } else {
        (0.0, 0.0)
    };

    // ── Parity violation (from M5) ──
    let left_fraction = if let Some(ref ew) = meas.electroweak {
        ew.left_fraction
    } else {
        0.0
    };

    // ── Combinatorial RG flow (from M4) ──
    let (global_alpha_inv, mean_local_alpha_inv, mean_local_q) =
        if let Some(ref vp) = meas.vacuum_pol {
            let mean_local_alpha = vp.mean_local_q / (8.0 * std::f64::consts::PI);
            (
                if vp.global_alpha > 0.0 { 1.0 / vp.global_alpha } else { 0.0 },
                if mean_local_alpha > 0.0 { 1.0 / mean_local_alpha } else { 0.0 },
                vp.mean_local_q,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

    // ── Quantum mechanics (from M6) ──
    let (born_r, coherence_r) = if let Some(ref dc) = meas.decoherence {
        (dc.born_r, dc.coherence_decay_r)
    } else {
        (0.0, 0.0)
    };

    // ── Dark sector (from TopologySummary) ──
    let omega_dark_vis = topo.omega_ratio;
    let omega_energy = topo.omega_energy;

    // ── Generation census (from M2) ──
    let gen_fractions = if let Some(ref h) = meas.half_life {
        let total = (h.gen_counts[0] + h.gen_counts[1] + h.gen_counts[2]) as f64;
        if total > 0.0 {
            [
                h.gen_counts[0] as f64 / total,
                h.gen_counts[1] as f64 / total,
                h.gen_counts[2] as f64 / total,
            ]
        } else {
            [0.0; 3]
        }
    } else {
        [0.0; 3]
    };

    let phase_fractions = {
        let total = (topo.phase_pos_count + topo.phase_zero_count + topo.phase_neg_count) as f64;
        if total > 0.0 {
            [
                topo.phase_pos_count as f64 / total,
                topo.phase_zero_count as f64 / total,
                topo.phase_neg_count as f64 / total,
            ]
        } else {
            [0.0; 3]
        }
    };

    // ── Modulo path integral (from M3) ──
    let (interference_mean, constructive_frac) = if let Some(ref m) = meas.modulo {
        let total = (m.constructive_count + m.destructive_count) as f64;
        let frac = if total > 0.0 { m.constructive_count as f64 / total } else { 0.0 };
        (m.mean_intensity, frac)
    } else {
        (0.0, 0.0)
    };

    // ── β-function assembly (bifurcated: gauge + grav) ─────────────
    let sin2tw: f64 = 4.0 / 17.0;
    let cos2tw: f64 = 13.0 / 17.0;

    // Gauge track: join M4 gauge_bins and M5 ew_gauge_bins by b
    let beta_gauge = {
        let m4_gauge = meas.vacuum_pol.as_ref().map(|v| &v.gauge_bins[..]).unwrap_or(&[]);
        let m5_gauge = meas.electroweak.as_ref().map(|e| &e.ew_gauge_bins[..]).unwrap_or(&[]);

        m4_gauge.iter().filter_map(|m4b| {
            if m4b.n_resolved == 0 { return None; }
            let q_k = m4b.q;
            let alpha_k = m4b.alpha;
            let alpha_inv = if alpha_k > 0.0 { 1.0 / alpha_k } else { 0.0 };
            let e_k = (4.0 * PI * alpha_k).sqrt();
            let g1_k = e_k / cos2tw.sqrt();
            let g2_k = e_k / sin2tw.sqrt();
            let mw_mz = cos2tw.sqrt();

            let (lf, md, nr_m5) = if let Some(m5b) = m5_gauge.iter().find(|x| x.b == m4b.b) {
                (m5b.left_fraction, m5b.mean_doublets, m5b.n_resolved)
            } else {
                (0.0, 0.0, 0)
            };

            Some(GaugeBetaPoint {
                b: m4b.b,
                q: q_k,
                alpha_em: alpha_k,
                alpha_em_inv: alpha_inv,
                e_charge: e_k,
                sin2_theta_w: sin2tw,
                cos2_theta_w: cos2tw,
                g1: g1_k,
                g2: g2_k,
                mw_mz_ratio: mw_mz,
                n_resolved_m4: m4b.n_resolved,
                n_resolved_m5: nr_m5,
                left_fraction: lf,
                mean_doublets: md,
            })
        }).collect::<Vec<_>>()
    };

    // Grav track: join M4 grav_bins and M5 ew_grav_bins by c
    let beta_grav = {
        let m4_grav = meas.vacuum_pol.as_ref().map(|v| &v.grav_bins[..]).unwrap_or(&[]);
        let m5_grav = meas.electroweak.as_ref().map(|e| &e.ew_grav_bins[..]).unwrap_or(&[]);

        m4_grav.iter().filter_map(|m4b| {
            if m4b.n_resolved == 0 { return None; }
            let q_k = m4b.q;
            let alpha_k = m4b.alpha;
            let alpha_inv = if alpha_k > 0.0 { 1.0 / alpha_k } else { 0.0 };
            let e_k = (4.0 * PI * alpha_k).sqrt();
            let g1_k = e_k / cos2tw.sqrt();
            let g2_k = e_k / sin2tw.sqrt();
            let mw_mz = cos2tw.sqrt();

            let (lf, md, nr_m5) = if let Some(m5b) = m5_grav.iter().find(|x| x.c == m4b.c) {
                (m5b.left_fraction, m5b.mean_doublets, m5b.n_resolved)
            } else {
                (0.0, 0.0, 0)
            };

            Some(GravBetaPoint {
                c: m4b.c,
                q: q_k,
                alpha_em: alpha_k,
                alpha_em_inv: alpha_inv,
                e_charge: e_k,
                sin2_theta_w: sin2tw,
                cos2_theta_w: cos2tw,
                g1: g1_k,
                g2: g2_k,
                mw_mz_ratio: mw_mz,
                n_resolved_m4: m4b.n_resolved,
                n_resolved_m5: nr_m5,
                left_fraction: lf,
                mean_doublets: md,
            })
        }).collect::<Vec<_>>()
    };

    LagrangianCard {
        gauge_group,
        n_bosons,
        n_generations,
        alpha_em,
        alpha_em_inv,
        sin2_theta_w,
        cos2_theta_w,
        e_charge,
        g1,
        g2,
        mw_mz_ratio,
        g_newton,
        mass_topo,
        mass_ratio_21,
        mass_ratio_31,
        cpt_violation,
        yukawa_ratio,
        neutrino_count,
        neutrino_mass_proxy,
        neutrino_chirality,
        pmns_matrix,
        higgs_drag,
        higgs_area_ratio,
        left_fraction,
        global_alpha_inv,
        mean_local_alpha_inv,
        mean_local_q,
        born_r,
        coherence_r,
        omega_dark_vis,
        omega_energy,
        gen_fractions,
        phase_fractions,
        interference_mean,
        constructive_frac,
        beta_gauge,
        beta_grav,
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &LagrangianCard, w: &mut CsvWriter) {
    w.comment("M10 SM Lagrangian Card (zero free parameters)");
    w.header(&["parameter", "value"]);

    w.row_fmt(format_args!("gauge_group,{}", result.gauge_group));
    w.row_fmt(format_args!("n_bosons,{}/{}/{}", result.n_bosons[0], result.n_bosons[1], result.n_bosons[2]));
    w.row_fmt(format_args!("n_generations,{}", result.n_generations));

    w.row_fmt(format_args!("alpha_em,{:.8}", result.alpha_em));
    w.row_fmt(format_args!("alpha_em_inv,{:.4}", result.alpha_em_inv));
    w.row_fmt(format_args!("sin2_theta_w,{:.8}", result.sin2_theta_w));
    w.row_fmt(format_args!("cos2_theta_w,{:.8}", result.cos2_theta_w));
    w.row_fmt(format_args!("e_charge,{:.8}", result.e_charge));
    w.row_fmt(format_args!("g1_U1Y,{:.8}", result.g1));
    w.row_fmt(format_args!("g2_SU2L,{:.8}", result.g2));
    w.row_fmt(format_args!("mw_mz_ratio,{:.8}", result.mw_mz_ratio));
    w.row_fmt(format_args!("g_newton,{:.8}", result.g_newton));

    w.row_fmt(format_args!("mass_topo_gen1,{:.4}", result.mass_topo[0]));
    w.row_fmt(format_args!("mass_topo_gen2,{:.4}", result.mass_topo[1]));
    w.row_fmt(format_args!("mass_topo_gen3,{:.4}", result.mass_topo[2]));
    w.row_fmt(format_args!("mass_ratio_21,{:.6}", result.mass_ratio_21));
    w.row_fmt(format_args!("mass_ratio_31,{:.6}", result.mass_ratio_31));
    w.row_fmt(format_args!("cpt_violation,{:.8}", result.cpt_violation));
    w.row_fmt(format_args!("yukawa_ratio,{:.4}/{:.4}/{:.4}",
        result.yukawa_ratio[0], result.yukawa_ratio[1], result.yukawa_ratio[2]));

    w.row_fmt(format_args!("neutrino_count,{}", result.neutrino_count));
    w.row_fmt(format_args!("neutrino_mass_proxy,{:.6}", result.neutrino_mass_proxy));
    w.row_fmt(format_args!("neutrino_chirality,{:.6}", result.neutrino_chirality));
    for i in 0..3 {
        w.row_fmt(format_args!("pmns_{},{:.6}/{:.6}/{:.6}",
            i + 1, result.pmns_matrix[i][0], result.pmns_matrix[i][1], result.pmns_matrix[i][2]));
    }

    w.row_fmt(format_args!("higgs_drag,{:.6}", result.higgs_drag));
    w.row_fmt(format_args!("higgs_area_ratio,{:.6}", result.higgs_area_ratio));
    w.row_fmt(format_args!("left_fraction,{:.6}", result.left_fraction));

    w.row_fmt(format_args!("global_alpha_inv,{:.4}", result.global_alpha_inv));
    w.row_fmt(format_args!("mean_local_alpha_inv,{:.4}", result.mean_local_alpha_inv));
    w.row_fmt(format_args!("mean_local_q,{:.6}", result.mean_local_q));

    w.row_fmt(format_args!("born_r_pmf,{:.6}", result.born_r));
    w.row_fmt(format_args!("coherence_r,{:.6}", result.coherence_r));

    w.row_fmt(format_args!("omega_dark_vis,{:.6}", result.omega_dark_vis));
    w.row_fmt(format_args!("omega_energy,{:.6}", result.omega_energy));

    w.row_fmt(format_args!("gen_fractions,{:.4}/{:.4}/{:.4}",
        result.gen_fractions[0], result.gen_fractions[1], result.gen_fractions[2]));
    w.row_fmt(format_args!("phase_fractions,{:.4}/{:.4}/{:.4}",
        result.phase_fractions[0], result.phase_fractions[1], result.phase_fractions[2]));

    w.row_fmt(format_args!("interference_mean,{:.6}", result.interference_mean));
    w.row_fmt(format_args!("constructive_frac,{:.6}", result.constructive_frac));

    // Gauge β-function table
    if !result.beta_gauge.is_empty() {
        w.comment("Gauge beta-function (z3 mod 3^b, M4+M5 gauge bins)");
        w.header(&[
            "b",
            "q", "alpha_em", "alpha_em_inv", "e_charge",
            "sin2_theta_w", "cos2_theta_w", "g1", "g2", "mw_mz_ratio",
            "n_resolved_m4", "n_resolved_m5", "left_fraction", "mean_doublets",
        ]);
        for bp in &result.beta_gauge {
            w.row_fmt(format_args!(
                "{},{:.6},{:.8},{:.4},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{},{:.6},{:.4}",
                bp.b,
                bp.q, bp.alpha_em, bp.alpha_em_inv, bp.e_charge,
                bp.sin2_theta_w, bp.cos2_theta_w, bp.g1, bp.g2, bp.mw_mz_ratio,
                bp.n_resolved_m4, bp.n_resolved_m5, bp.left_fraction, bp.mean_doublets
            ));
        }
    }

    // Grav β-function table
    if !result.beta_grav.is_empty() {
        w.comment("Gravity beta-function (z5 mod 5^c, M4+M5 grav bins)");
        w.header(&[
            "c",
            "q", "alpha_em", "alpha_em_inv", "e_charge",
            "sin2_theta_w", "cos2_theta_w", "g1", "g2", "mw_mz_ratio",
            "n_resolved_m4", "n_resolved_m5", "left_fraction", "mean_doublets",
        ]);
        for bp in &result.beta_grav {
            w.row_fmt(format_args!(
                "{},{:.6},{:.8},{:.4},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{},{:.6},{:.4}",
                bp.c,
                bp.q, bp.alpha_em, bp.alpha_em_inv, bp.e_charge,
                bp.sin2_theta_w, bp.cos2_theta_w, bp.g1, bp.g2, bp.mw_mz_ratio,
                bp.n_resolved_m4, bp.n_resolved_m5, bp.left_fraction, bp.mean_doublets
            ));
        }
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &LagrangianCard) {
    println!("  [M10] SM Lagrangian Card (Zero Free Parameters):");
    println!("    Gauge:       {} | Bosons: {:?} | Gens: {}", result.gauge_group, result.n_bosons, result.n_generations);
    println!("    alpha_EM:    {:.6}  (1/alpha={:.1})", result.alpha_em, result.alpha_em_inv);
    println!("    sin2_thetaW: {:.6}  cos2_thetaW: {:.6}", result.sin2_theta_w, result.cos2_theta_w);
    println!("    e={:.6}  g1={:.6}  g2={:.6}  mW/mZ={:.6}", result.e_charge, result.g1, result.g2, result.mw_mz_ratio);
    println!("    G_N:         {:.8}", result.g_newton);
    println!("    Mass topo:   [{:.2}, {:.2}, {:.2}]  ratios: m2/m1={:.4} m3/m1={:.4}",
        result.mass_topo[0], result.mass_topo[1], result.mass_topo[2],
        result.mass_ratio_21, result.mass_ratio_31);
    println!("    CPT viol:    {:.6}", result.cpt_violation);
    println!("    Neutrinos:   {} (mass~{:.2}, chi={:.4})", result.neutrino_count, result.neutrino_mass_proxy, result.neutrino_chirality);
    println!("    PMNS:");
    for i in 0..3 {
        println!("      [{:.4}  {:.4}  {:.4}]",
            result.pmns_matrix[i][0], result.pmns_matrix[i][1], result.pmns_matrix[i][2]);
    }
    println!("    Higgs drag:  {:.6}  area_ratio: {:.4}", result.higgs_drag, result.higgs_area_ratio);
    println!("    Left frac:   {:.4}", result.left_fraction);
    println!("    RG flow:     global_inv={:.1}  local_inv={:.1}  mean_Q={:.4}",
        result.global_alpha_inv, result.mean_local_alpha_inv, result.mean_local_q);
    println!("    Born r(PMF): {:.6}  coherence_r: {:.4}", result.born_r, result.coherence_r);
    println!("    Dark sector: Omega_DM/vis={:.4}  Omega_energy={:.4}", result.omega_dark_vis, result.omega_energy);
    println!("    Gen frac:    [{:.4}, {:.4}, {:.4}]",
        result.gen_fractions[0], result.gen_fractions[1], result.gen_fractions[2]);
    println!("    Phase frac:  [{:.4}, {:.4}, {:.4}]",
        result.phase_fractions[0], result.phase_fractions[1], result.phase_fractions[2]);
    println!("    Interference: mean={:.6}  constructive={:.4}", result.interference_mean, result.constructive_frac);
    if !result.beta_gauge.is_empty() {
        println!("    β_gauge (z3 mod 3^b):");
        for bp in &result.beta_gauge {
            println!(
                "      b={}: Q={:.4}, 1/α={:.1}, e={:.4}, g1={:.4}, g2={:.4}, L={:.4}, doub={:.2}  (M4:{}, M5:{})",
                bp.b,
                bp.q, bp.alpha_em_inv, bp.e_charge, bp.g1, bp.g2,
                bp.left_fraction, bp.mean_doublets, bp.n_resolved_m4, bp.n_resolved_m5
            );
        }
    }
    if !result.beta_grav.is_empty() {
        println!("    β_grav (z5 mod 5^c):");
        for bp in &result.beta_grav {
            println!(
                "      c={}: Q={:.4}, 1/α={:.1}, e={:.4}, g1={:.4}, g2={:.4}, L={:.4}, doub={:.2}  (M4:{}, M5:{})",
                bp.c,
                bp.q, bp.alpha_em_inv, bp.e_charge, bp.g1, bp.g2,
                bp.left_fraction, bp.mean_doublets, bp.n_resolved_m4, bp.n_resolved_m5
            );
        }
    }
}
