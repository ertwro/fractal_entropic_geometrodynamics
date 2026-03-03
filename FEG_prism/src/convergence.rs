// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Dynamic auto-convergence for Monte Carlo observables.
//!
//! Each measurement module runs walkers in batches and stops when the
//! Relative Standard Error (RSE) of the batch-mean observable drops
//! below `epsilon` for `k_required` consecutive batches.
//!
//! Uses Welford's online algorithm for numerically stable running
//! variance — zero hardcoded physics parameters.
//!
//! # Dimensional flow and the parameter-free walker budget
//!
//! In quantum gravity (Causal Dynamical Triangulations, Asymptotic Safety,
//! Causal Sets), spacetime does not have a fixed dimension.  It experiences
//! **dimensional flow**: the spectral dimension d_S(t) changes with the
//! diffusion scale t.  It typically flows from something highly fractal or
//! low-dimensional at the Planck scale (small t) to exactly 4 at
//! macroscopic scales (large t).
//!
//! Because the dimension is flowing, the rate at which information dilutes
//! is also flowing.  The return probability is governed by the spectral
//! dimension:
//!
//! ```text
//!   P(t) ~ t^{-d_S/2}
//! ```
//!
//! When d_S is not constant, P(t) is the integral of the dimensional flow
//! over that scale.  The Causal Resolution Theorem (CRT) derives the
//! exact walker budget from the CLT on the Bernoulli return indicator:
//! each walker contributes X_i in {0,1} with E\[X_i\] = P(t), giving
//! RSE = 1/sqrt(W * P(t)).  Setting RSE <= epsilon yields:
//!
//! ```text
//!   W >= 1 / (epsilon^2 * P(t_max))
//! ```
//!
//! This is the **Diffusion Limit** — the number of walkers needed so that
//! the signal-to-noise ratio at the worst (latest) diffusion time meets
//! the precision target.  There is also a **Space-Filling Limit**: you
//! need at least |S| walkers so every origin in a sub-manifold S fires
//! at least once.
//!
//! The unified, parameter-free formula for any sub-manifold S is:
//!
//! ```text
//!   W_S = max( |S|,  ceil(1 / (epsilon^2 * P_S(t_max))) )
//! ```
//!
//! where
//!
//! ```text
//!   P_bulk(t_max) = t_max^{-2}           (d_S = 4 at macroscopic scales)
//!   P_S(t_max)    = P_bulk * |S| / N     (geometric dilution into bulk)
//! ```
//!
//! Zero free parameters.  The formula is computed from (epsilon, t_max,
//! |S|, N) alone — see `run_converge_loop` in `ensemble/runner.rs`.
//!
//! # Why converge on P(t_max), not d_S
//!
//! The CRT bounds the variance of the primitive observable P(t).  The
//! spectral dimension d_S is a logarithmic derivative:
//!
//! ```text
//!   d_S(t) ~ -2 * d(ln P) / d(ln t)
//! ```
//!
//! By error propagation, Var(ln P) ~ RSE_P^2 = epsilon^2.  Because the
//! derivative subtracts two adjacent noisy measurements, the noise adds
//! but the signal difference is small:
//!
//! ```text
//!   Var(d_S) ~ 2 * epsilon^2 / (Delta ln t)^2
//! ```
//!
//! Demanding RSE(d_S) <= epsilon inadvertently requires an internal P(t)
//! precision of epsilon/(Delta ln t) ~ epsilon/10, which needs ~100x
//! more walkers than the CRT allocates.
//!
//! The fix: track P(t_max) as the convergence observable.  Since P(t) is
//! monotonically decreasing, t_max has the smallest signal and highest
//! variance.  If RSE(P(t_max)) <= epsilon, every earlier point is
//! strictly better — the entire curve is validated in one shot.
//!
//! The spectral dimension d_S is computed only once from the converged
//! cumulative P(t) curve, exactly as averaging.rs computes d_S from the
//! ensemble-mean P(t) rather than averaging per-realization d_S values.
//!
//! # Why t_max = D_max = 15 (lattice mode decay argument)
//!
//! The lazy random walk has transition matrix T = 1/2(I + D^{-1}A), whose
//! eigenvalues lie in \[0, 1\].  The return probability decomposes as:
//!
//! ```text
//!   P(t) = (1/N) sum_k  lambda_k^t
//! ```
//!
//! Three bands contribute:
//!
//! 1. **Long-wavelength modes** (lambda_k ~ 1): continuum Laplacian
//!    eigenmodes.  Their contribution follows P_cont(t) ~ t^{-d_S/2}.
//!    This is the signal.
//!
//! 2. **Lattice modes** (lambda_k intermediate): arise from discrete graph
//!    structure — degree fluctuations, parity effects, local motifs.
//!    Eigenvalues cluster near lambda ~ 1 - O(1/D_max), so their decay
//!    timescale is t_lattice = 1/|ln lambda| ~ D_max.
//!
//! 3. **High-frequency modes** (lambda_k ~ 0): decay in O(1) steps,
//!    irrelevant for t >= 2.
//!
//! After t ~ D_max = 15 steps, lattice modes have decayed by a factor
//! (1 - 1/D)^D ~ 1/e, and P(t) enters the continuum scaling plateau
//! where d_S(t) is physically meaningful.
//!
//! Calibrating at t_max = D_max is optimal because P(D_max) is the
//! largest return probability in the scaling regime, yielding the smallest
//! required W.  The same epsilon that the CRT uses to derive the walker
//! ceiling is passed to AutoConverge — one precision target, zero free
//! parameters.
//!
//! **Note:** D_max = 15 is NOT a mixing time in the coupon-collector
//! sense (that would be O(D ln D) ~ 41).  It is the lattice mode decay
//! time — the timescale on which discrete graph artifacts in P(t) damp
//! below the continuum signal.

/// Static configuration for a convergence loop.
pub struct AutoConverge {
    pub batch_size: usize,
    pub epsilon: f64,
    pub k_required: usize,
    pub max_walkers: usize,
}

impl AutoConverge {
    pub fn new(max_walkers: usize, batch_size: usize, epsilon: f64) -> Self {
        Self {
            batch_size,
            epsilon,
            k_required: 3,
            max_walkers,
        }
    }
}

/// Mutable state tracking convergence via Welford's online algorithm.
pub struct ConvergeState {
    pub total_walkers: usize,
    batch_count: usize,
    mean: f64,
    m2: f64,
    consecutive: usize,
}

impl ConvergeState {
    pub fn new() -> Self {
        Self {
            total_walkers: 0,
            batch_count: 0,
            mean: 0.0,
            m2: 0.0,
            consecutive: 0,
        }
    }

    /// Feed one batch's observable mean.  Returns `true` when the
    /// Relative Standard Error has been below `epsilon` for
    /// `k_required` consecutive batches.
    pub fn update(&mut self, batch_mean: f64, ac: &AutoConverge) -> bool {
        self.total_walkers += ac.batch_size;
        self.batch_count += 1;

        // Welford's algorithm: running mean and sum-of-squares
        let delta = batch_mean - self.mean;
        self.mean += delta / self.batch_count as f64;
        let delta2 = batch_mean - self.mean;
        self.m2 += delta * delta2;

        // If the batch observable is exactly zero, we are resolving noise —
        // the sub-population has too few returns to measure d_S.
        // Do not count this toward convergence.
        if batch_mean == 0.0 {
            self.consecutive = 0;
            return false;
        }

        // Need at least 2 batches to estimate variance
        if self.batch_count < 2 {
            return false;
        }

        let variance = self.m2 / (self.batch_count - 1) as f64;
        let standard_error = (variance / self.batch_count as f64).sqrt();

        // Relative SE (absolute fallback when mean ≈ 0)
        let relative_se = if self.mean.abs() > 1e-8 {
            standard_error / self.mean.abs()
        } else {
            standard_error
        };

        if relative_se < ac.epsilon {
            self.consecutive += 1;
        } else {
            self.consecutive = 0;
        }

        self.consecutive >= ac.k_required
    }

    /// Returns `true` when total walkers have reached the safety cap.
    pub fn at_limit(&self, ac: &AutoConverge) -> bool {
        self.total_walkers >= ac.max_walkers
    }
}
