// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! M5 — Electroweak Sector (SU(2) Chirality + U(1) Port Counting)
//!
//! Per-intermediate chirality chi = (out_d - in_d) / (out_d + in_d) classifies
//! handedness.  SU(2) doublets = min(n_left, n_right).  Port counting on
//! poles vs intermediates yields a topological charge Q_topo that should
//! converge to 1/4 (topological collider, exact locked asymptote).
//!
//! Bifurcated congruence bins: gauge (z3 mod 3^b) and grav (z5 mod 5^c).
//!
//! Calculo de Kuratowski, Vol II, section 7: parity violation from causal arrow.

use super::context::MeasureContext;
use crate::output::CsvWriter;
use crate::phase2::defect::CausalPrism;
use rayon::prelude::*;

// ── Data Structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PrismElectroweak {
    pub prism_idx: usize,
    pub generation: u8,
    pub n_intermediates: usize,
    pub n_left: usize,
    pub n_right: usize,
    pub n_balanced: usize,
    pub su2_doublets: usize,
    pub chirality_sum: f64,
    pub mean_abs_chirality: f64,
    pub free_ports_origin: u32,
    pub free_ports_dest: u32,
    pub free_ports_intermediates: u32,
    pub weak_filtered_ports: u32,
    pub causal_filtered_ports: u32,
    pub local_q_topo: f64,
    pub transparency: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EwGaugeBin {
    pub b: u32,
    pub n_resolved: usize,
    pub n_unresolved: usize,
    pub n_left: usize,
    pub n_right: usize,
    pub sum_doublets: usize,
    pub total_free_ports: u64,
    pub total_causal_ports: u64,
    pub left_fraction: f64,
    pub mean_doublets: f64,
    pub q_topo_port: f64,
    pub alpha_port: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EwGravBin {
    pub c: u32,
    pub n_resolved: usize,
    pub n_unresolved: usize,
    pub n_left: usize,
    pub n_right: usize,
    pub sum_doublets: usize,
    pub total_free_ports: u64,
    pub total_causal_ports: u64,
    pub left_fraction: f64,
    pub mean_doublets: f64,
    pub q_topo_port: f64,
    pub alpha_port: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ElectroweakResult {
    pub per_prism: Vec<PrismElectroweak>,
    pub ew_gauge_bins: Vec<EwGaugeBin>,
    pub ew_grav_bins: Vec<EwGravBin>,
    pub mean_chirality_imbalance: f64,
    pub left_fraction: f64,
    pub mean_doublets: f64,
    pub q_topo_port: f64,
    pub alpha_port: f64,
    pub total_free_ports: u64,
    pub total_weak_filtered: u64,
    pub total_causal_filtered: u64,
    pub gen_left_fraction: [f64; 3],
    pub gen_mean_chirality: [f64; 3],
    pub gen_prism_count: [usize; 3],
}

// ── Utilities ────────────────────────────────────────────────────────────────

fn build_gen_lookup(n: usize, ctx: &MeasureContext) -> Vec<u8> {
    let mut lookup = vec![0u8; n];
    for &node in &ctx.defect.generations.gen1 {
        if node < n { lookup[node] = 1; }
    }
    for &node in &ctx.defect.generations.gen2 {
        if node < n { lookup[node] = 2; }
    }
    for &node in &ctx.defect.generations.gen3 {
        if node < n { lookup[node] = 3; }
    }
    for &node in &ctx.defect.generations.anti1 {
        if node < n { lookup[node] = 4; }
    }
    lookup
}

fn classify_prism_generation(prism: &CausalPrism, gen_lookup: &[u8]) -> u8 {
    for &node in &prism.intermediates {
        let g = gen_lookup[node];
        if g >= 1 && g <= 3 { return g; }
    }
    let g = gen_lookup[prism.origin];
    if g >= 1 && g <= 3 { return g; }
    let g = gen_lookup[prism.destination];
    if g >= 1 && g <= 3 { return g; }
    0
}

// ── Measurement ──────────────────────────────────────────────────────────────

pub fn run(ctx: &MeasureContext) -> ElectroweakResult {
    const MAX_HASSE_DEGREE: u32 = 15;

    let n = ctx.n_points;
    let gen_lookup = build_gen_lookup(n, ctx);
    let (vac_head, _vac_data) = ctx.vacuum_csr.raw();

    // Build reverse CSR
    let mut rev_deg = vec![0u32; n];
    for u in 0..n {
        let s = vac_head[u] as usize;
        let e = vac_head[u + 1] as usize;
        let vac_data = ctx.vacuum_csr.data();
        for &v in &vac_data[s..e] {
            let vi = v as usize;
            if vi < n {
                rev_deg[vi] += 1;
            }
        }
    }
    let mut rev_head = vec![0u32; n + 1];
    for i in 0..n {
        rev_head[i + 1] = rev_head[i] + rev_deg[i];
    }

    // Process each prism in parallel
    let per_prism: Vec<PrismElectroweak> = ctx.prisms
        .par_iter()
        .enumerate()
        .map(|(pi, prism)| {
            let gen = classify_prism_generation(prism, &gen_lookup);
            let n_inter = prism.intermediates.len();

            // 1. Per-intermediate chirality
            let mut n_left = 0usize;
            let mut n_right = 0usize;
            let mut n_balanced = 0usize;
            let mut chi_sum = 0.0f64;
            let mut abs_chi_sum = 0.0f64;

            for &w in &prism.intermediates {
                if w >= n { continue; }
                let out_d = (vac_head[w + 1] - vac_head[w]) as f64;
                let in_d = (rev_head[w + 1] - rev_head[w]) as f64;
                let total = out_d + in_d;
                if total > 0.0 {
                    let chi = (out_d - in_d) / total;
                    chi_sum += chi;
                    abs_chi_sum += chi.abs();
                    if chi > 0.0 {
                        n_left += 1;
                    } else if chi < 0.0 {
                        n_right += 1;
                    } else {
                        n_balanced += 1;
                    }
                } else {
                    n_balanced += 1;
                }
            }

            // 2. SU(2) doublets = min(n_left, n_right)
            let su2_doublets = n_left.min(n_right);

            let mean_abs_chi = if n_inter > 0 {
                abs_chi_sum / n_inter as f64
            } else {
                0.0
            };

            // 3. Port counting (U(1) hypercharge)
            let degree_of = |node: usize| -> u32 {
                if node >= n { return 0; }
                vac_head[node + 1] - vac_head[node]
            };

            let free_origin = MAX_HASSE_DEGREE.saturating_sub(degree_of(prism.origin));
            let free_dest = MAX_HASSE_DEGREE.saturating_sub(degree_of(prism.destination));
            let free_inter: u32 = prism
                .intermediates
                .iter()
                .map(|&w| MAX_HASSE_DEGREE.saturating_sub(degree_of(w)))
                .sum();

            let total_free = free_origin + free_dest + free_inter;
            // Weak filtered: poles only (intermediate ports forbidden)
            let weak_filtered = free_origin + free_dest;
            // Causal filtered: one pole only (time-ordering)
            let causal_filtered = free_origin;

            let local_q = if total_free > 0 {
                causal_filtered as f64 / total_free as f64
            } else {
                0.0
            };
            let transparency = if total_free > 0 {
                weak_filtered as f64 / total_free as f64
            } else {
                0.0
            };

            PrismElectroweak {
                prism_idx: pi,
                generation: gen,
                n_intermediates: n_inter,
                n_left,
                n_right,
                n_balanced,
                su2_doublets,
                chirality_sum: chi_sum,
                mean_abs_chirality: mean_abs_chi,
                free_ports_origin: free_origin,
                free_ports_dest: free_dest,
                free_ports_intermediates: free_inter,
                weak_filtered_ports: weak_filtered,
                causal_filtered_ports: causal_filtered,
                local_q_topo: local_q,
                transparency,
            }
        })
        .collect();

    // Aggregation
    let total_prisms = per_prism.len();
    let mut total_left = 0usize;
    let mut total_right = 0usize;
    let mut total_doublets = 0usize;
    let mut sum_chi_imbalance = 0.0f64;
    let mut total_free: u64 = 0;
    let mut total_weak: u64 = 0;
    let mut total_causal: u64 = 0;

    let mut gen_left_count = [0usize; 3];
    let mut gen_total_inter = [0usize; 3];
    let mut gen_chi_sum = [0.0f64; 3];
    let mut gen_chi_count = [0usize; 3];
    let mut gen_prism_count = [0usize; 3];

    for p in &per_prism {
        total_left += p.n_left;
        total_right += p.n_right;
        total_doublets += p.su2_doublets;
        sum_chi_imbalance += (p.n_left as f64 - p.n_right as f64).abs();
        total_free += (p.free_ports_origin + p.free_ports_dest + p.free_ports_intermediates) as u64;
        total_weak += p.weak_filtered_ports as u64;
        total_causal += p.causal_filtered_ports as u64;

        let gi = p.generation as usize;
        if gi >= 1 && gi <= 3 {
            let idx = gi - 1;
            gen_left_count[idx] += p.n_left;
            gen_total_inter[idx] += p.n_intermediates;
            gen_chi_sum[idx] += p.chirality_sum;
            gen_chi_count[idx] += p.n_intermediates;
            gen_prism_count[idx] += 1;
        }
    }

    let total_lr = (total_left + total_right) as f64;
    let left_fraction = if total_lr > 0.0 { total_left as f64 / total_lr } else { 0.5 };
    let mean_chi_imbalance = if total_prisms > 0 {
        sum_chi_imbalance / total_prisms as f64
    } else {
        0.0
    };
    let mean_doublets = if total_prisms > 0 {
        total_doublets as f64 / total_prisms as f64
    } else {
        0.0
    };
    let q_topo_port = if total_free > 0 {
        total_causal as f64 / total_free as f64
    } else {
        0.0
    };
    let alpha_port = q_topo_port / (8.0 * std::f64::consts::PI);

    let gen_left_fraction = [
        if gen_total_inter[0] > 0 { gen_left_count[0] as f64 / gen_total_inter[0] as f64 } else { 0.0 },
        if gen_total_inter[1] > 0 { gen_left_count[1] as f64 / gen_total_inter[1] as f64 } else { 0.0 },
        if gen_total_inter[2] > 0 { gen_left_count[2] as f64 / gen_total_inter[2] as f64 } else { 0.0 },
    ];
    let gen_mean_chirality = [
        if gen_chi_count[0] > 0 { gen_chi_sum[0] / gen_chi_count[0] as f64 } else { 0.0 },
        if gen_chi_count[1] > 0 { gen_chi_sum[1] / gen_chi_count[1] as f64 } else { 0.0 },
        if gen_chi_count[2] > 0 { gen_chi_sum[2] / gen_chi_count[2] as f64 } else { 0.0 },
    ];

    // ── Bifurcated congruence binning ────────────────────────────────────
    let (ew_gauge_bins, ew_grav_bins) = {
        let coords = ctx.sorted_coords;

        // Determine coordinate ranges for quantization
        let (mut min_x, mut max_x) = (f64::MAX, f64::MIN);
        let (mut min_y, mut max_y) = (f64::MAX, f64::MIN);
        for p in ctx.prisms {
            for &nd in std::iter::once(&p.origin)
                .chain(std::iter::once(&p.destination))
                .chain(p.intermediates.iter())
            {
                let c = &coords[nd];
                if c[1] < min_x { min_x = c[1]; }
                if c[1] > max_x { max_x = c[1]; }
                if c[2] < min_y { min_y = c[2]; }
                if c[2] > max_y { max_y = c[2]; }
            }
        }
        let span_x = max_x - min_x;
        let span_y = max_y - min_y;
        let quantize = |val: f64, min_val: f64, span: f64| -> u64 {
            if span <= 0.0 { return 0; }
            (((val - min_val) / span) * 1e15) as u64
        };

        // Pre-compute quantized spatial coords per prism's nodes
        let prism_z3: Vec<Vec<u64>> = ctx.prisms.iter().map(|p| {
            let mut nodes = Vec::with_capacity(2 + p.intermediates.len());
            nodes.push(p.origin);
            nodes.push(p.destination);
            nodes.extend_from_slice(&p.intermediates);
            nodes.iter().map(|&nd| quantize(coords[nd][1], min_x, span_x)).collect()
        }).collect();

        let prism_z5: Vec<Vec<u64>> = ctx.prisms.iter().map(|p| {
            let mut nodes = Vec::with_capacity(2 + p.intermediates.len());
            nodes.push(p.origin);
            nodes.push(p.destination);
            nodes.extend_from_slice(&p.intermediates);
            nodes.iter().map(|&nd| quantize(coords[nd][2], min_y, span_y)).collect()
        }).collect();

        // Gauge scan: z3 mod 3^b
        let mut gauge_bins: Vec<EwGaugeBin> = Vec::new();
        for b in 0u32.. {
            let mod3 = match 3u64.checked_pow(b) {
                Some(v) => v,
                None => break,
            };

            let mut n_resolved: usize = 0;
            let mut n_unresolved: usize = 0;
            let mut bin_left: usize = 0;
            let mut bin_right: usize = 0;
            let mut bin_doublets: usize = 0;
            let mut bin_free: u64 = 0;
            let mut bin_causal: u64 = 0;

            for (pi, z3s) in prism_z3.iter().enumerate() {
                let cell3 = z3s[0] % mod3;
                let resolved = z3s[1..].iter().all(|&z3| z3 % mod3 == cell3);
                if resolved {
                    n_resolved += 1;
                    let p = &per_prism[pi];
                    bin_left += p.n_left;
                    bin_right += p.n_right;
                    bin_doublets += p.su2_doublets;
                    bin_free += (p.free_ports_origin + p.free_ports_dest + p.free_ports_intermediates) as u64;
                    bin_causal += p.causal_filtered_ports as u64;
                } else {
                    n_unresolved += 1;
                }
            }

            if n_resolved == 0 { break; }

            let total_lr = (bin_left + bin_right) as f64;
            let left_frac = if total_lr > 0.0 { bin_left as f64 / total_lr } else { 0.0 };
            let mean_doub = if n_resolved > 0 { bin_doublets as f64 / n_resolved as f64 } else { 0.0 };
            let q_port = if bin_free > 0 { bin_causal as f64 / bin_free as f64 } else { 0.0 };
            let a_port = q_port / (8.0 * std::f64::consts::PI);

            gauge_bins.push(EwGaugeBin {
                b,
                n_resolved,
                n_unresolved,
                n_left: bin_left,
                n_right: bin_right,
                sum_doublets: bin_doublets,
                total_free_ports: bin_free,
                total_causal_ports: bin_causal,
                left_fraction: left_frac,
                mean_doublets: mean_doub,
                q_topo_port: q_port,
                alpha_port: a_port,
            });
        }

        // Grav scan: z5 mod 5^c
        let mut grav_bins: Vec<EwGravBin> = Vec::new();
        for c in 0u32.. {
            let mod5 = match 5u64.checked_pow(c) {
                Some(v) => v,
                None => break,
            };

            let mut n_resolved: usize = 0;
            let mut n_unresolved: usize = 0;
            let mut bin_left: usize = 0;
            let mut bin_right: usize = 0;
            let mut bin_doublets: usize = 0;
            let mut bin_free: u64 = 0;
            let mut bin_causal: u64 = 0;

            for (pi, z5s) in prism_z5.iter().enumerate() {
                let cell5 = z5s[0] % mod5;
                let resolved = z5s[1..].iter().all(|&z5| z5 % mod5 == cell5);
                if resolved {
                    n_resolved += 1;
                    let p = &per_prism[pi];
                    bin_left += p.n_left;
                    bin_right += p.n_right;
                    bin_doublets += p.su2_doublets;
                    bin_free += (p.free_ports_origin + p.free_ports_dest + p.free_ports_intermediates) as u64;
                    bin_causal += p.causal_filtered_ports as u64;
                } else {
                    n_unresolved += 1;
                }
            }

            if n_resolved == 0 { break; }

            let total_lr = (bin_left + bin_right) as f64;
            let left_frac = if total_lr > 0.0 { bin_left as f64 / total_lr } else { 0.0 };
            let mean_doub = if n_resolved > 0 { bin_doublets as f64 / n_resolved as f64 } else { 0.0 };
            let q_port = if bin_free > 0 { bin_causal as f64 / bin_free as f64 } else { 0.0 };
            let a_port = q_port / (8.0 * std::f64::consts::PI);

            grav_bins.push(EwGravBin {
                c,
                n_resolved,
                n_unresolved,
                n_left: bin_left,
                n_right: bin_right,
                sum_doublets: bin_doublets,
                total_free_ports: bin_free,
                total_causal_ports: bin_causal,
                left_fraction: left_frac,
                mean_doublets: mean_doub,
                q_topo_port: q_port,
                alpha_port: a_port,
            });
        }

        (gauge_bins, grav_bins)
    };

    ElectroweakResult {
        per_prism,
        ew_gauge_bins,
        ew_grav_bins,
        mean_chirality_imbalance: mean_chi_imbalance,
        left_fraction,
        mean_doublets,
        q_topo_port,
        alpha_port,
        total_free_ports: total_free,
        total_weak_filtered: total_weak,
        total_causal_filtered: total_causal,
        gen_left_fraction,
        gen_mean_chirality,
        gen_prism_count,
    }
}

// ── Ensemble Aggregation ─────────────────────────────────────────────────────

pub fn aggregate(results: &[ElectroweakResult]) -> ElectroweakResult {
    let m = results.len() as f64;
    let mut gen_left = [0.0f64; 3];
    let mut gen_chi = [0.0f64; 3];
    let mut gen_count = [0usize; 3];
    for ew in results {
        for i in 0..3 {
            gen_left[i] += ew.gen_left_fraction[i];
            gen_chi[i] += ew.gen_mean_chirality[i];
            gen_count[i] += ew.gen_prism_count[i];
        }
    }
    for i in 0..3 { gen_left[i] /= m; gen_chi[i] /= m; }
    let total_free: u64 = results.iter().map(|e| e.total_free_ports).sum();
    let total_weak: u64 = results.iter().map(|e| e.total_weak_filtered).sum();
    let total_causal: u64 = results.iter().map(|e| e.total_causal_filtered).sum();
    let q_topo = if total_free > 0 { total_causal as f64 / total_free as f64 } else { 0.0 };

    // Aggregate EW gauge bins: join by b
    let max_b = results.iter().flat_map(|r| r.ew_gauge_bins.iter().map(|g| g.b)).max().unwrap_or(0);
    let mut ew_gauge_bins: Vec<EwGaugeBin> = Vec::new();
    for b in 0..=max_b {
        let mut n_res: usize = 0;
        let mut n_unres: usize = 0;
        let mut tot_left: usize = 0;
        let mut tot_right: usize = 0;
        let mut tot_doub: usize = 0;
        let mut tot_fp: u64 = 0;
        let mut tot_cp: u64 = 0;
        for r in results {
            if let Some(bin) = r.ew_gauge_bins.iter().find(|x| x.b == b) {
                n_res += bin.n_resolved;
                n_unres += bin.n_unresolved;
                tot_left += bin.n_left;
                tot_right += bin.n_right;
                tot_doub += bin.sum_doublets;
                tot_fp += bin.total_free_ports;
                tot_cp += bin.total_causal_ports;
            }
        }
        if n_res == 0 { continue; }
        let total_lr = (tot_left + tot_right) as f64;
        let left_frac = if total_lr > 0.0 { tot_left as f64 / total_lr } else { 0.0 };
        let mean_doub = if n_res > 0 { tot_doub as f64 / n_res as f64 } else { 0.0 };
        let q_port = if tot_fp > 0 { tot_cp as f64 / tot_fp as f64 } else { 0.0 };
        let a_port = q_port / (8.0 * std::f64::consts::PI);
        ew_gauge_bins.push(EwGaugeBin {
            b,
            n_resolved: n_res, n_unresolved: n_unres,
            n_left: tot_left, n_right: tot_right, sum_doublets: tot_doub,
            total_free_ports: tot_fp, total_causal_ports: tot_cp,
            left_fraction: left_frac, mean_doublets: mean_doub,
            q_topo_port: q_port, alpha_port: a_port,
        });
    }

    // Aggregate EW grav bins: join by c
    let max_c = results.iter().flat_map(|r| r.ew_grav_bins.iter().map(|g| g.c)).max().unwrap_or(0);
    let mut ew_grav_bins: Vec<EwGravBin> = Vec::new();
    for c in 0..=max_c {
        let mut n_res: usize = 0;
        let mut n_unres: usize = 0;
        let mut tot_left: usize = 0;
        let mut tot_right: usize = 0;
        let mut tot_doub: usize = 0;
        let mut tot_fp: u64 = 0;
        let mut tot_cp: u64 = 0;
        for r in results {
            if let Some(bin) = r.ew_grav_bins.iter().find(|x| x.c == c) {
                n_res += bin.n_resolved;
                n_unres += bin.n_unresolved;
                tot_left += bin.n_left;
                tot_right += bin.n_right;
                tot_doub += bin.sum_doublets;
                tot_fp += bin.total_free_ports;
                tot_cp += bin.total_causal_ports;
            }
        }
        if n_res == 0 { continue; }
        let total_lr = (tot_left + tot_right) as f64;
        let left_frac = if total_lr > 0.0 { tot_left as f64 / total_lr } else { 0.0 };
        let mean_doub = if n_res > 0 { tot_doub as f64 / n_res as f64 } else { 0.0 };
        let q_port = if tot_fp > 0 { tot_cp as f64 / tot_fp as f64 } else { 0.0 };
        let a_port = q_port / (8.0 * std::f64::consts::PI);
        ew_grav_bins.push(EwGravBin {
            c,
            n_resolved: n_res, n_unresolved: n_unres,
            n_left: tot_left, n_right: tot_right, sum_doublets: tot_doub,
            total_free_ports: tot_fp, total_causal_ports: tot_cp,
            left_fraction: left_frac, mean_doublets: mean_doub,
            q_topo_port: q_port, alpha_port: a_port,
        });
    }

    ElectroweakResult {
        per_prism: vec![],
        ew_gauge_bins,
        ew_grav_bins,
        mean_chirality_imbalance: results.iter().map(|e| e.mean_chirality_imbalance).sum::<f64>() / m,
        left_fraction: results.iter().map(|e| e.left_fraction).sum::<f64>() / m,
        mean_doublets: results.iter().map(|e| e.mean_doublets).sum::<f64>() / m,
        q_topo_port: q_topo,
        alpha_port: q_topo / (8.0 * std::f64::consts::PI),
        total_free_ports: total_free,
        total_weak_filtered: total_weak,
        total_causal_filtered: total_causal,
        gen_left_fraction: gen_left,
        gen_mean_chirality: gen_chi,
        gen_prism_count: gen_count,
    }
}

// ── CSV Output ───────────────────────────────────────────────────────────────

pub fn write_csv(result: &ElectroweakResult, w: &mut CsvWriter) {
    w.comment("M5 Electroweak Sector (SU(2) chirality + U(1) port counting)");
    w.header(&[
        "prism_idx", "generation", "n_intermediates",
        "n_left", "n_right", "n_balanced", "su2_doublets",
        "chirality_sum", "mean_abs_chirality",
        "free_ports_origin", "free_ports_dest", "free_ports_inter",
        "weak_filtered", "causal_filtered", "local_q_topo", "transparency",
    ]);
    for p in &result.per_prism {
        w.row_fmt(format_args!(
            "{},{},{},{},{},{},{},{:.6},{:.6},{},{},{},{},{},{:.6},{:.6}",
            p.prism_idx, p.generation, p.n_intermediates,
            p.n_left, p.n_right, p.n_balanced, p.su2_doublets,
            p.chirality_sum, p.mean_abs_chirality,
            p.free_ports_origin, p.free_ports_dest, p.free_ports_intermediates,
            p.weak_filtered_ports, p.causal_filtered_ports,
            p.local_q_topo, p.transparency
        ));
    }
    // EW gauge bins
    if !result.ew_gauge_bins.is_empty() {
        w.comment("M5 EW Gauge Beta-Function (z3 mod 3^b)");
        w.header(&[
            "b",
            "n_resolved", "n_unresolved",
            "n_left", "n_right", "sum_doublets",
            "total_free_ports", "total_causal_ports",
            "left_fraction", "mean_doublets", "q_topo_port", "alpha_port",
        ]);
        for g in &result.ew_gauge_bins {
            w.row_fmt(format_args!(
                "{},{},{},{},{},{},{},{},{:.6},{:.4},{:.6},{:.8}",
                g.b,
                g.n_resolved, g.n_unresolved,
                g.n_left, g.n_right, g.sum_doublets,
                g.total_free_ports, g.total_causal_ports,
                g.left_fraction, g.mean_doublets, g.q_topo_port, g.alpha_port
            ));
        }
    }
    // EW grav bins
    if !result.ew_grav_bins.is_empty() {
        w.comment("M5 EW Gravity Beta-Function (z5 mod 5^c)");
        w.header(&[
            "c",
            "n_resolved", "n_unresolved",
            "n_left", "n_right", "sum_doublets",
            "total_free_ports", "total_causal_ports",
            "left_fraction", "mean_doublets", "q_topo_port", "alpha_port",
        ]);
        for g in &result.ew_grav_bins {
            w.row_fmt(format_args!(
                "{},{},{},{},{},{},{},{},{:.6},{:.4},{:.6},{:.8}",
                g.c,
                g.n_resolved, g.n_unresolved,
                g.n_left, g.n_right, g.sum_doublets,
                g.total_free_ports, g.total_causal_ports,
                g.left_fraction, g.mean_doublets, g.q_topo_port, g.alpha_port
            ));
        }
    }
}

// ── Terminal Summary ─────────────────────────────────────────────────────────

pub fn print_summary(result: &ElectroweakResult) {
    println!("  [M5] Electroweak Sector:");
    println!("    Left fraction:   {:.4}", result.left_fraction);
    println!("    Mean doublets:   {:.2}", result.mean_doublets);
    println!("    Q_topo (port):   {:.6}", result.q_topo_port);
    println!("    alpha (port):    {:.6}  (1/alpha={:.1})",
        result.alpha_port, if result.alpha_port > 0.0 { 1.0 / result.alpha_port } else { 0.0 });
    println!("    Gen left frac:   [{:.4}, {:.4}, {:.4}]",
        result.gen_left_fraction[0], result.gen_left_fraction[1], result.gen_left_fraction[2]);
    println!("    Gen mean chi:    [{:.4}, {:.4}, {:.4}]",
        result.gen_mean_chirality[0], result.gen_mean_chirality[1], result.gen_mean_chirality[2]);
    if !result.ew_gauge_bins.is_empty() {
        let active = result.ew_gauge_bins.iter().filter(|g| g.n_resolved > 0).count();
        println!("    EW β_gauge (z3 mod 3^b): {} active levels:", active);
        for g in result.ew_gauge_bins.iter().filter(|g| g.n_resolved > 0) {
            let inv_alpha = if g.alpha_port > 0.0 { 1.0 / g.alpha_port } else { 0.0 };
            println!(
                "      b={}: Q_ew={:.4}, 1/α_ew={:.1}, L={:.4}, doub={:.2}  (resolved={}/{})",
                g.b,
                g.q_topo_port, inv_alpha, g.left_fraction, g.mean_doublets,
                g.n_resolved, g.n_resolved + g.n_unresolved
            );
        }
    }
    if !result.ew_grav_bins.is_empty() {
        let active = result.ew_grav_bins.iter().filter(|g| g.n_resolved > 0).count();
        println!("    EW β_grav (z5 mod 5^c): {} active levels:", active);
        for g in result.ew_grav_bins.iter().filter(|g| g.n_resolved > 0) {
            let inv_alpha = if g.alpha_port > 0.0 { 1.0 / g.alpha_port } else { 0.0 };
            println!(
                "      c={}: Q_ew={:.4}, 1/α_ew={:.1}, L={:.4}, doub={:.2}  (resolved={}/{})",
                g.c,
                g.q_topo_port, inv_alpha, g.left_fraction, g.mean_doublets,
                g.n_resolved, g.n_resolved + g.n_unresolved
            );
        }
    }
}
