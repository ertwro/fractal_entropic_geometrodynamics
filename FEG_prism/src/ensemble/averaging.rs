// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Ensemble averaging of spectral observables.
//!
//! Physics: d_S is non-linear in P(t), so we average P first across all
//! realisations, then derive d_S from <P>.  This ensures the ensemble-averaged
//! spectral dimension reflects the mean geometry rather than the mean of
//! individually noisy d_S curves.
//!
//! When M > 1, a second pass computes the sample standard deviation of
//! each d_S(t) and flux(t) field across realisations for error bars
//! (Bessel-corrected: divisor M-1).

use crate::phase3::spectral::{spectral_dimension, SpectralOutput, WalkResult};

/// Average P(t) across realisations, then recompute d_S from the mean P.
///
/// For each WalkResult field (vacuum, defect, generations, sterile, flux),
/// averages P(t) element-wise across all M results, then recomputes d_S
/// from the averaged P.  When M > 1, also computes per-step standard
/// deviation of d_S across realisations for error bars.
pub fn average_ensemble(
    results: &[SpectralOutput],
    steps: &[u32],
) -> SpectralOutput {
    let m = results.len();
    if m == 0 {
        return SpectralOutput::default();
    }
    if m == 1 {
        return results[0].clone();
    }

    let mf = m as f64;
    let ns = steps.len();

    // ── Helper: average a P(t) field across all results ──────────────
    let avg_p = |extract: &dyn Fn(&SpectralOutput) -> &[f64]| -> Vec<f64> {
        let first = extract(&results[0]);
        if first.is_empty() {
            return vec![];
        }
        let mut acc = vec![0.0; ns];
        for r in results {
            let src = extract(r);
            for (i, &v) in src.iter().enumerate().take(ns) {
                acc[i] += v;
            }
        }
        for x in &mut acc {
            *x /= mf;
        }
        acc
    };

    // ── Helper: compute d_S from averaged P, or empty ────────────────
    let ds_or_empty = |p: &[f64]| -> Vec<f64> {
        if p.is_empty() {
            vec![]
        } else {
            spectral_dimension(steps, p)
        }
    };

    // ── Helper: sample std_dev of per-realisation d_S around ensemble mean ──
    // Recomputes d_S from each individual P(t) and measures variance.
    // Uses Bessel's correction (mf - 1.0) for unbiased sample std_dev.
    let std_ds_field =
        |extract_p: &dyn Fn(&SpectralOutput) -> &[f64], mean_ds: &[f64]| -> Vec<f64> {
            if mean_ds.is_empty() || mf <= 1.0 {
                return vec![];
            }
            let mut acc = vec![0.0; ns];
            for r in results {
                let src = extract_p(r);
                if src.is_empty() {
                    continue;
                }
                let ds_i = spectral_dimension(steps, src);
                for (i, &v) in ds_i.iter().enumerate().take(ns) {
                    let d = v - mean_ds[i];
                    acc[i] += d * d;
                }
            }
            for x in &mut acc {
                *x = (*x / (mf - 1.0)).sqrt();
            }
            acc
        };

    // ── Helper: sample std_dev of a raw field (flux values, not d_S) ──
    // Uses Bessel's correction (mf - 1.0) for unbiased sample std_dev.
    let std_raw_field =
        |extract: &dyn Fn(&SpectralOutput) -> &[f64], mean: &[f64]| -> Vec<f64> {
            if mean.is_empty() || mf <= 1.0 {
                return vec![];
            }
            let mut acc = vec![0.0; ns];
            for r in results {
                let src = extract(r);
                for (i, &v) in src.iter().enumerate().take(ns) {
                    let d = v - mean[i];
                    acc[i] += d * d;
                }
            }
            for x in &mut acc {
                *x = (*x / (mf - 1.0)).sqrt();
            }
            acc
        };

    // ── Helper: build a WalkResult from averaged P + std_dev ─────────
    let make_walk = |extract_p: &dyn Fn(&SpectralOutput) -> &[f64]| -> WalkResult {
        let p = avg_p(extract_p);
        let ds = ds_or_empty(&p);
        let ds_std = std_ds_field(extract_p, &ds);
        WalkResult { p, ds, ds_std }
    };

    // ── Average all WalkResult fields ────────────────────────────────
    let vacuum = make_walk(&|r| &r.vacuum.p);
    let vac_core = make_walk(&|r| &r.vac_core.p);
    let defect = make_walk(&|r| &r.defect.p);
    let def_core = make_walk(&|r| &r.def_core.p);

    let gen0 = make_walk(&|r| &r.generations[0].p);
    let gen1 = make_walk(&|r| &r.generations[1].p);
    let gen2 = make_walk(&|r| &r.generations[2].p);
    let gen3 = make_walk(&|r| &r.generations[3].p);
    let sterile = make_walk(&|r| &r.sterile.p);

    // Flux: average P + compute std_dev of P directly (flux_attr/repu
    // are transmission probabilities, not return probabilities).
    let flux_attr_p = avg_p(&|r| &r.flux_attr.p);
    let flux_attr_ds = ds_or_empty(&flux_attr_p);
    let flux_attr_ds_std = std_raw_field(&|r| &r.flux_attr.p, &flux_attr_p);
    let flux_attr = WalkResult {
        p: flux_attr_p,
        ds: flux_attr_ds,
        ds_std: flux_attr_ds_std,
    };

    let flux_repu_p = avg_p(&|r| &r.flux_repu.p);
    let flux_repu_ds = ds_or_empty(&flux_repu_p);
    let flux_repu_ds_std = std_raw_field(&|r| &r.flux_repu.p, &flux_repu_p);
    let flux_repu = WalkResult {
        p: flux_repu_p,
        ds: flux_repu_ds,
        ds_std: flux_repu_ds_std,
    };

    // Normalized flux (scalar per step)
    let flux_attr_norm = avg_p(&|r| &r.flux_attr_norm);
    let flux_repu_norm = avg_p(&|r| &r.flux_repu_norm);

    // Average mass spectrum (scalar fields)
    let mass = [
        results.iter().map(|r| r.mass[0]).sum::<f64>() / mf,
        results.iter().map(|r| r.mass[1]).sum::<f64>() / mf,
        results.iter().map(|r| r.mass[2]).sum::<f64>() / mf,
        results.iter().map(|r| r.mass[3]).sum::<f64>() / mf,
    ];

    SpectralOutput {
        vacuum,
        vac_core,
        defect,
        def_core,
        generations: [gen0, gen1, gen2, gen3],
        sterile,
        flux_attr,
        flux_repu,
        flux_attr_norm,
        flux_repu_norm,
        mass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_output(p_val: f64, mass_val: f64) -> SpectralOutput {
        let steps = [1, 2, 4];
        let p = vec![p_val; 3];
        let ds = spectral_dimension(&steps, &p);
        let wr = WalkResult {
            p: p.clone(),
            ds: ds.clone(),
            ds_std: vec![],
        };
        SpectralOutput {
            vacuum: wr.clone(),
            vac_core: wr.clone(),
            defect: wr.clone(),
            def_core: wr.clone(),
            generations: [wr.clone(), wr.clone(), wr.clone(), wr.clone()],
            sterile: wr.clone(),
            flux_attr: wr.clone(),
            flux_repu: wr,
            flux_attr_norm: p.clone(),
            flux_repu_norm: p,
            mass: [mass_val; 4],
        }
    }

    #[test]
    fn single_result_passthrough() {
        let r = make_output(0.5, 3.0);
        let avg = average_ensemble(&[r.clone()], &[1, 2, 4]);
        assert_eq!(avg.mass, [3.0; 4]);
        assert_eq!(avg.vacuum.p, r.vacuum.p);
    }

    #[test]
    fn two_results_average_mass() {
        let r1 = make_output(0.4, 3.0);
        let r2 = make_output(0.6, 5.0);
        let avg = average_ensemble(&[r1, r2], &[1, 2, 4]);
        assert!((avg.mass[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn two_results_average_p() {
        let r1 = make_output(0.3, 3.0);
        let r2 = make_output(0.5, 3.0);
        let avg = average_ensemble(&[r1, r2], &[1, 2, 4]);
        assert!((avg.vacuum.p[0] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn empty_input() {
        let avg = average_ensemble(&[], &[1, 2, 4]);
        assert!(avg.vacuum.p.is_empty());
    }

    #[test]
    fn std_populated_when_m_gt_1() {
        let r1 = make_output(0.3, 3.0);
        let r2 = make_output(0.5, 5.0);
        let avg = average_ensemble(&[r1, r2], &[1, 2, 4]);
        // With M=2, std fields should be populated (non-empty)
        assert_eq!(avg.vacuum.ds_std.len(), 3);
    }
}
