// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Diagnostic: Heat kernel spectral flow → RG integration → M_EW/M_Pl
//!
//! ```text
//! Usage: cargo run --release --example heat_kernel_rg [N] [seed]
//! Default: N=3000, seed=42
//! ```
//!
//! Pipeline:
//! 1. Phase 1: sprinkle + Hasse diagram
//! 2. Symmetrize vacuum CSR → undirected graph
//! 3. Auto-select Tier 1 (N ≤ 5000) or Tier 2 (SLQ)
//! 4. Heat kernel K(T), spectral dimension D_s(T), RG integration

use feg_prism::{phase1, phase3};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(42);

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Heat Kernel Spectral Flow & RG Integration Diagnostic      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!("N = {n}, seed = {seed}");
    println!();

    // ── Phase 1: Vacuum generation ──────────────────────────────────────────
    println!("[Phase 1] Sprinkling + Hasse diagram...");
    let mut rng = StdRng::seed_from_u64(seed);
    let (pts_raw, _half_t) = phase1::sprinkle(n, &mut rng);

    let (_euclidean, vacuum_csr, _momentum) = if n <= 5000 {
        phase1::build_hasse_sparse(&pts_raw)
    } else {
        phase1::build_hasse_direct(&pts_raw)
    };
    drop(pts_raw);

    // ── Symmetrize ──────────────────────────────────────────────────────────
    println!("[Phase 1] Symmetrizing CSR ({} nodes, {} directed edges)...",
        vacuum_csr.n_nodes(), vacuum_csr.n_edge_slots());
    let sym_csr = vacuum_csr.symmetrize();
    drop(vacuum_csr);
    println!("  Undirected: {} edge slots\n", sym_csr.n_edge_slots());

    // ── Heat kernel ─────────────────────────────────────────────────────────
    let times = phase3::log_spaced_times(0.01, 1e6, 120);

    let use_exact = n <= 5000;
    let hk = if use_exact {
        println!("[Tier 1] Exact eigendecomposition ({n}×{n})...");
        phase3::heat_kernel_exact(&sym_csr, &times)
    } else {
        let n_probes = 30;
        let k_lanczos = 200;
        println!("[Tier 2] Stochastic Lanczos Quadrature (m={n_probes}, k={k_lanczos})...");
        phase3::heat_kernel_slq(&sym_csr, &times, n_probes, k_lanczos, seed)
    };
    drop(sym_csr);

    // ── 1. Eigenvalue spectrum ──────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  1. EIGENVALUE SPECTRUM");
    println!("══════════════════════════════════════════════════════════════");
    println!("  Spectral gap (λ_min > 0):  {:.6}", hk.spectral_gap);

    if !hk.eigenvalues.is_empty() {
        println!("  λ_min:  {:.6}", hk.eigenvalues.first().unwrap());
        println!("  λ_max:  {:.6}", hk.eigenvalues.last().unwrap());

        let (slope, r_sq) = phase3::weyl_law_check(&hk.eigenvalues);
        println!("  Weyl law: slope = {slope:.4} (expect ~2.0 for 4D), R² = {r_sq:.6}");

        // Zero eigenvalue count
        let n_zero = hk.eigenvalues.iter().filter(|&&v| v < 1e-10).count();
        println!("  Zero eigenvalues: {n_zero} (= connected components)");
    } else {
        println!("  (Eigenvalues not available in SLQ mode)");
    }

    // ── 2. Heat kernel trace ────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  2. HEAT KERNEL TRACE K(T)");
    println!("══════════════════════════════════════════════════════════════");
    println!("  {:>12}  {:>14}  {:>10}", "T", "K(T)", "D_s(T)");
    println!("  {:>12}  {:>14}  {:>10}", "────", "──────", "──────");

    // Print at ~15 representative points
    let step = (hk.diffusion_times.len() / 15).max(1);
    for i in (0..hk.diffusion_times.len()).step_by(step) {
        println!("  {:>12.4e}  {:>14.4}  {:>10.4}",
            hk.diffusion_times[i], hk.heat_trace[i], hk.spectral_dim[i]);
    }
    // Always show last
    let last = hk.diffusion_times.len() - 1;
    if last % step != 0 {
        println!("  {:>12.4e}  {:>14.4}  {:>10.4}",
            hk.diffusion_times[last], hk.heat_trace[last], hk.spectral_dim[last]);
    }

    println!("\n  K(T_min) = {:.4} (expect ≈ {n})", hk.heat_trace[0]);
    println!("  K(T_max) = {:.4} (expect → 1)", hk.heat_trace[last]);

    // ── 3. Spectral dimension flow ──────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  3. SPECTRAL DIMENSION FLOW D_s(T)");
    println!("══════════════════════════════════════════════════════════════");
    println!("  {:>12}  {:>10}  {:>14}", "T", "D_s(T)", "μ/M_Pl");
    println!("  {:>12}  {:>10}  {:>14}", "────", "──────", "──────");
    for i in (0..hk.diffusion_times.len()).step_by(step) {
        let t = hk.diffusion_times[i];
        let mu_over_mpl = 1.0 / t.sqrt();
        println!("  {:>12.4e}  {:>10.4}  {:>14.4e}",
            t, hk.spectral_dim[i], mu_over_mpl);
    }

    // ── 4. RG integration ───────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  4. RG FLOW");
    println!("══════════════════════════════════════════════════════════════");

    // α₀ = Q_topo / (8π) ≈ (1/4) / (8π) ≈ 0.00995
    let alpha_0 = 0.25 / (8.0 * std::f64::consts::PI);
    let n_f = 6_u32;
    let b0 = (33.0 - 2.0 * n_f as f64) / (12.0 * std::f64::consts::PI);

    println!("  α₀ = Q_topo/(8π) = {alpha_0:.6}");
    println!("  n_f = {n_f} (3 generations × 2 flavors)");
    println!("  b₀ = (33 − 2n_f)/(12π) = {b0:.6}");

    let rg = phase3::integrate_rg(&hk, alpha_0, n_f, -50.0, 20000);

    println!("\n  {:>14}  {:>10}  {:>10}  {:>10}", "ln(μ/M_Pl)", "α(μ)", "1/α(μ)", "D_s(μ)");
    println!("  {:>14}  {:>10}  {:>10}  {:>10}", "──────────", "────", "──────", "──────");
    let rg_step = (rg.ln_mu.len() / 20).max(1);
    for i in (0..rg.ln_mu.len()).step_by(rg_step) {
        println!("  {:>14.4}  {:>10.6}  {:>10.2}  {:>10.4}",
            rg.ln_mu[i], rg.alpha[i], rg.inv_alpha[i], rg.ds_at_mu[i]);
    }

    // ── 5. Mass predictions ─────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  5. MASS SCALE PREDICTIONS");
    println!("══════════════════════════════════════════════════════════════");

    let sm_ew_over_pl = 246.0 / 1.22e19;  // ≈ 2.0×10⁻¹⁷
    println!("  SM reference:  M_EW/M_Pl = {sm_ew_over_pl:.2e}");
    println!();

    if rg.m_ew_over_m_pl > 0.0 {
        println!("  Predicted (α ∼ 1):  M_EW/M_Pl = {:.2e}", rg.m_ew_over_m_pl);
        let ratio = (rg.m_ew_over_m_pl / sm_ew_over_pl).log10();
        println!("  log₁₀(predicted/SM) = {ratio:.2} decades");
    } else {
        println!("  α never reached 1.0 — coupling remains perturbative");
        println!("  (try larger N or different D_s flow)");
    }

    if rg.m_qcd_over_m_pl > 0.0 {
        let sm_qcd_over_pl = 0.2 / 1.22e19;  // Λ_QCD ≈ 200 MeV
        println!("\n  Predicted (α ∼ 0.12):  M_QCD/M_Pl = {:.2e}", rg.m_qcd_over_m_pl);
        let ratio = (rg.m_qcd_over_m_pl / sm_qcd_over_pl).log10();
        println!("  log₁₀(predicted/SM) = {ratio:.2} decades");
    }

    // Final coupling value
    let alpha_final = *rg.alpha.last().unwrap();
    println!("\n  Final α at ln μ = {:.1}: {alpha_final:.6}", rg.ln_mu.last().unwrap());
    println!("  Final 1/α = {:.2}", rg.inv_alpha.last().unwrap());

    println!("\n[Done]");
}
