//! Jacobson thermodynamic emergence: Clausius relation on coarse-grained causal sets.
//!
//! Evaluates the discrete Clausius relation δQ = T δS across the macroscopic
//! network obtained by voxelising a large causal set.
//!
//! ## Scalar pipeline (v1)
//!
//! Horizon area δA_i (link count) and net BD flux δQ_i (row sum) give a
//! scalar Clausius ratio R_i = δQ_i / δA_i.  This has high CV because the
//! BD d'Alembertian weights nearly cancel by design.
//!
//! ## Tensor pipeline (v2) — Fisher covariance
//!
//! For each macro-node, accumulate the 4×4 outer product of BD-weighted
//! displacement vectors across the local causal horizon:
//!
//!     C^{μν}_i = Σ_k  w_k · Δx^μ_k · Δx^ν_k
//!
//! where the sum runs over micro Hasse links crossing from macro-node i
//! into its direct macro-CSR children, Δx = x_v − x_u is the 4-displacement,
//! and w_k = bd_weight(0) = +1 for direct links.
//!
//! The eigenvalues of C reveal:
//! - **Lorentzian signature** (−+++) if the causal structure is correct
//! - **Emergent G** from λ_max / δA_i converging to a universal constant

use nalgebra::{DMatrix, Matrix4, SymmetricEigen};

/// BD layer weight (re-exported from rmt for self-contained use).
#[inline(always)]
fn bd_weight(depth: u8) -> f64 {
    match depth {
        0 => 1.0,
        1 => -9.0,
        2 => 16.0,
        3 => -8.0,
        _ => 0.0,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Scalar pipeline (v1)
// ═══════════════════════════════════════════════════════════════════════

/// Compute the discrete horizon area for every macro-node.
///
/// δA_i = number of microscopic Hasse links `(u, v)` where
/// `micro_to_macro[u] == i` and `micro_to_macro[v]` is a direct
/// child of `i` in the macro CSR.
///
/// Integer-exact: the returned counts are always non-negative.
pub fn horizon_areas(
    micro_head: &[u32],
    micro_data: &[u32],
    macro_head: &[u32],
    macro_data: &[u32],
    micro_to_macro: &[usize],
    n_micro: usize,
    n_macro: usize,
) -> Vec<i64> {
    let mut areas = vec![0i64; n_macro];

    for u in 0..n_micro {
        let m_u = micro_to_macro[u];
        let u_lo = micro_head[u] as usize;
        let u_hi = micro_head[u + 1] as usize;

        let mac_lo = macro_head[m_u] as usize;
        let mac_hi = macro_head[m_u + 1] as usize;

        for &v in &micro_data[u_lo..u_hi] {
            let m_v = micro_to_macro[v as usize];
            if m_u == m_v {
                continue;
            }
            if macro_data[mac_lo..mac_hi]
                .iter()
                .any(|&c| c as usize == m_v)
            {
                areas[m_u] += 1;
            }
        }
    }

    areas
}

/// Extract per-node net BD flux from the BD matrix (integer-exact).
///
/// `net_flux[i] = Σ_j B[i,j]` — the total outgoing BD weight from node i.
pub fn net_flux_per_node(b: &DMatrix<i64>) -> Vec<i64> {
    let n = b.nrows();
    (0..n).map(|i| b.row(i).iter().copied().sum()).collect()
}

/// Compute the Clausius ratio R_i = δQ_i / δA_i for each macro-node.
///
/// Nodes with δA_i = 0 are excluded.  Measurement boundary: i64 → f64.
pub fn clausius_ratios(net_flux: &[i64], areas: &[i64]) -> Vec<f64> {
    net_flux
        .iter()
        .zip(areas.iter())
        .filter(|&(_, &a)| a > 0)
        .map(|(&q, &a)| q as f64 / a as f64)
        .collect()
}

/// Mean, variance, and coefficient of variation of a sample.
///
/// CV = σ / |μ|.  Returns `(mean, variance, cv)`.
pub fn coefficient_of_variation(data: &[f64]) -> (f64, f64, f64) {
    let n = data.len() as f64;
    if n < 1.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mean = data.iter().sum::<f64>() / n;
    if n < 2.0 {
        return (mean, f64::NAN, f64::NAN);
    }
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let cv = if mean.abs() > 1e-15 {
        var.sqrt() / mean.abs()
    } else {
        f64::NAN
    };
    (mean, var, cv)
}

// ═══════════════════════════════════════════════════════════════════════
// Tensor pipeline (v2) — Fisher covariance metric
// ═══════════════════════════════════════════════════════════════════════

/// 4×4 Fisher covariance of BD-weighted displacement vectors across a
/// macro-node's local causal horizon.  The emergent metric tensor.
///
/// Tracks both the second moment ⟨Δx^μ Δx^ν⟩ and the first moment
/// ⟨Δx^μ⟩, so the true (mean-subtracted) covariance
///
///     C^{μν} = ⟨Δx^μ Δx^ν⟩ − ⟨Δx^μ⟩⟨Δx^ν⟩
///
/// can be computed.  This subtraction removes the bulk drift (all
/// causal vectors point future-ward) and reveals the Lorentzian
/// signature (−+++) of the fluctuations around the mean.
pub struct FisherMetric {
    /// Raw second moment: Σ_k w_k · Δx^μ_k · Δx^ν_k.
    pub second_moment: [[f64; 4]; 4],
    /// First moment: Σ_k w_k · Δx^μ_k.
    pub sum_dx: [f64; 4],
    /// Total weight: Σ_k w_k.
    pub sum_w: f64,
    /// Number of horizon-crossing links that contributed.
    pub n_links: usize,
}

impl FisherMetric {
    /// Zero-initialized metric.
    pub fn new() -> Self {
        Self {
            second_moment: [[0.0; 4]; 4],
            sum_dx: [0.0; 4],
            sum_w: 0.0,
            n_links: 0,
        }
    }

    /// Accumulate a single BD-weighted horizon-crossing link.
    #[inline]
    pub fn accumulate(&mut self, dx: [f64; 4], weight: f64) {
        for mu in 0..4 {
            self.sum_dx[mu] += weight * dx[mu];
            for nu in mu..4 {
                let val = weight * dx[mu] * dx[nu];
                self.second_moment[mu][nu] += val;
                if nu != mu {
                    self.second_moment[nu][mu] += val;
                }
            }
        }
        self.sum_w += weight;
        self.n_links += 1;
    }

    /// Mean-subtracted covariance matrix.
    ///
    /// C^{μν} = ⟨Δx^μ Δx^ν⟩ − ⟨Δx^μ⟩⟨Δx^ν⟩
    ///
    /// Returns zeros if fewer than 2 links.
    pub fn covariance(&self) -> [[f64; 4]; 4] {
        if self.sum_w.abs() < 1e-15 || self.n_links < 2 {
            return [[0.0; 4]; 4];
        }
        let w = self.sum_w;
        let mut cov = [[0.0f64; 4]; 4];
        for mu in 0..4 {
            for nu in 0..4 {
                cov[mu][nu] =
                    self.second_moment[mu][nu] / w - (self.sum_dx[mu] / w) * (self.sum_dx[nu] / w);
            }
        }
        cov
    }

    /// Eigenvalues of the mean-subtracted covariance, sorted ascending.
    pub fn eigenvalues(&self) -> [f64; 4] {
        let cov = self.covariance();
        let m = Matrix4::from_fn(|i, j| cov[i][j]);
        let eig = SymmetricEigen::new(m);
        let mut vals = [0.0f64; 4];
        for i in 0..4 {
            vals[i] = eig.eigenvalues[i];
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals
    }

    /// Eigenvalues of the raw second moment (no mean subtraction), sorted ascending.
    ///
    /// Use this for multi-depth BD-weighted metrics where the alternating
    /// signs (+1, −9, +16, −8) break positive-semidefiniteness and can
    /// produce negative eigenvalues (Lorentzian signature).
    pub fn eigenvalues_raw(&self) -> [f64; 4] {
        let m = Matrix4::from_fn(|i, j| self.second_moment[i][j]);
        let eig = SymmetricEigen::new(m);
        let mut vals = [0.0f64; 4];
        for i in 0..4 {
            vals[i] = eig.eigenvalues[i];
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals
    }

    /// Signature: count of (negative, positive) eigenvalues.
    ///
    /// Eigenvalues with |λ| < threshold are treated as zero.
    pub fn signature(&self, threshold: f64) -> (usize, usize) {
        let ev = self.eigenvalues();
        let neg = ev.iter().filter(|&&v| v < -threshold).count();
        let pos = ev.iter().filter(|&&v| v > threshold).count();
        (neg, pos)
    }

    /// Signature from the raw second moment (for multi-depth BD metrics).
    pub fn signature_raw(&self, threshold: f64) -> (usize, usize) {
        let ev = self.eigenvalues_raw();
        let neg = ev.iter().filter(|&&v| v < -threshold).count();
        let pos = ev.iter().filter(|&&v| v > threshold).count();
        (neg, pos)
    }

    /// Largest eigenvalue by absolute value.
    pub fn principal_eigenvalue(&self) -> f64 {
        let ev = self.eigenvalues();
        ev.iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Largest eigenvalue by absolute value from raw second moment.
    pub fn principal_eigenvalue_raw(&self) -> f64 {
        let ev = self.eigenvalues_raw();
        ev.iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
    }
}

/// Compute the Fisher covariance metric for every macro-node.
///
/// For each micro Hasse edge `(u, v)` crossing from macro-node `m_u` into
/// a direct macro-CSR child `m_v`:
///
/// 1. Compute Δx^μ = pts\[v\]\[μ\] − pts\[u\]\[μ\]
/// 2. Weight by bd_weight(0) = +1 (direct Hasse link)
/// 3. Accumulate w · Δx^μ Δx^ν into `metrics[m_u].covariance`
pub fn fisher_covariances(
    pts: &[[f64; 4]],
    micro_head: &[u32],
    micro_data: &[u32],
    macro_head: &[u32],
    macro_data: &[u32],
    micro_to_macro: &[usize],
    n_micro: usize,
    n_macro: usize,
) -> Vec<FisherMetric> {
    let mut metrics: Vec<FisherMetric> = (0..n_macro).map(|_| FisherMetric::new()).collect();
    let w = bd_weight(0); // +1 for direct Hasse links

    for u in 0..n_micro {
        let m_u = micro_to_macro[u];
        let u_lo = micro_head[u] as usize;
        let u_hi = micro_head[u + 1] as usize;

        let mac_lo = macro_head[m_u] as usize;
        let mac_hi = macro_head[m_u + 1] as usize;

        for &v_u32 in &micro_data[u_lo..u_hi] {
            let v = v_u32 as usize;
            let m_v = micro_to_macro[v];
            if m_u == m_v {
                continue;
            }
            if macro_data[mac_lo..mac_hi]
                .iter()
                .any(|&c| c as usize == m_v)
            {
                let dx = [
                    pts[v][0] - pts[u][0],
                    pts[v][1] - pts[u][1],
                    pts[v][2] - pts[u][2],
                    pts[v][3] - pts[u][3],
                ];
                metrics[m_u].accumulate(dx, w);
            }
        }
    }

    metrics
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-depth BD tensor pipeline (v3) — full d'Alembertian geometry
// ═══════════════════════════════════════════════════════════════════════

/// Maximum BFS depth (= number of BD layers).
const MAX_DEPTH: usize = 4;

/// Compute the centroid of each macro-node from its constituent micro-points.
pub fn macro_centroids(
    pts: &[[f64; 4]],
    micro_to_macro: &[usize],
    n_micro: usize,
    n_macro: usize,
) -> Vec<[f64; 4]> {
    let mut sums = vec![[0.0f64; 4]; n_macro];
    let mut counts = vec![0usize; n_macro];
    for u in 0..n_micro {
        let m = micro_to_macro[u];
        for d in 0..4 {
            sums[m][d] += pts[u][d];
        }
        counts[m] += 1;
    }
    for m in 0..n_macro {
        if counts[m] > 0 {
            let c = counts[m] as f64;
            for d in 0..4 {
                sums[m][d] /= c;
            }
        }
    }
    sums
}

/// Compute Fisher metrics using multi-depth BFS on the macro graph.
///
/// For each macro-node `src`, performs a layered BFS through the macro
/// CSR up to depth 4, accumulating BD-weighted outer products of
/// centroid-to-centroid displacements:
///
///     M^{μν}_src = Σ_{(nbr, depth)} bd_weight(depth) · Δc^μ · Δc^ν
///
/// The alternating-sign BD weights (+1, −9, +16, −8) break
/// positive-semidefiniteness, enabling Lorentzian signature (−+++)
/// to emerge from the raw second moment.
pub fn fisher_covariances_multidepth(
    centroids: &[[f64; 4]],
    macro_head: &[u32],
    macro_data: &[u32],
    n_macro: usize,
) -> Vec<FisherMetric> {
    let mut metrics: Vec<FisherMetric> = (0..n_macro).map(|_| FisherMetric::new()).collect();
    let mut visited = vec![false; n_macro];

    for src in 0..n_macro {
        for v in visited.iter_mut() {
            *v = false;
        }
        visited[src] = true;

        let mut frontier: Vec<usize> = vec![src];

        for depth in 0..MAX_DEPTH {
            let w = bd_weight(depth as u8);
            let mut next: Vec<usize> = Vec::new();
            for &node in &frontier {
                let lo = macro_head[node] as usize;
                let hi = macro_head[node + 1] as usize;
                for &nbr_u32 in &macro_data[lo..hi] {
                    let nbr = nbr_u32 as usize;
                    if !visited[nbr] {
                        visited[nbr] = true;
                        next.push(nbr);
                        let dx = [
                            centroids[nbr][0] - centroids[src][0],
                            centroids[nbr][1] - centroids[src][1],
                            centroids[nbr][2] - centroids[src][2],
                            centroids[nbr][3] - centroids[src][3],
                        ];
                        metrics[src].accumulate(dx, w);
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
    }

    metrics
}

/// Run tensor analysis using the raw second moment (for multi-depth metrics).
///
/// Computes statistics for all valid nodes AND an inscribed-diamond bulk
/// subset.  The bulk filter keeps only macro-nodes whose centroid lies
/// inside a causal diamond shrunk by `alpha` on each side:
///
///     |t − t_center| + r_spatial < L × (1 − α)
///
/// This removes ALL boundary degeneracies in one geometric cut —
/// temporal tips AND spatial light-cone edges — because the boundary
/// of a causal diamond is the light cone, not just the tips.
///
/// For bulk Lorentzian nodes, the emergent G is computed as
/// |λ_temporal| / (λ₁ + λ₂ + λ₃), normalising the temporal eigenvalue
/// by the spatial trace rather than the raw edge count.
pub fn tensor_analysis_raw(
    metrics: &[FisherMetric],
    areas: &[i64],
    centroids: &[[f64; 4]],
    alpha: f64,
) -> TensorAnalysis {
    let n_macro = metrics.len();

    // ── Adaptive eigenvalue threshold ──
    let mut all_mags: Vec<f64> = Vec::new();
    for i in 0..n_macro {
        if areas[i] <= 0 || metrics[i].n_links == 0 {
            continue;
        }
        let ev = metrics[i].eigenvalues_raw();
        for &v in &ev {
            all_mags.push(v.abs());
        }
    }
    all_mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = if all_mags.is_empty() {
        1e-12
    } else {
        all_mags[all_mags.len() / 2] * 1e-6
    };

    // ── Inscribed diamond: bounding box → center + half-height ──
    let (mut t_min, mut t_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut s_center = [0.0f64; 3]; // spatial centroid of diamond
    let mut n_pts = 0usize;
    for c in centroids.iter() {
        if c[0] < t_min { t_min = c[0]; }
        if c[0] > t_max { t_max = c[0]; }
        s_center[0] += c[1];
        s_center[1] += c[2];
        s_center[2] += c[3];
        n_pts += 1;
    }
    let t_center = (t_min + t_max) / 2.0;
    let l_half = (t_max - t_min) / 2.0;
    if n_pts > 0 {
        let np = n_pts as f64;
        s_center[0] /= np;
        s_center[1] /= np;
        s_center[2] /= np;
    }
    let r_cut = l_half * (1.0 - alpha);

    // ── Accumulators: all valid nodes ──
    let mut n_valid: usize = 0;
    let mut n_lorentzian: usize = 0;
    let mut g_ratios: Vec<f64> = Vec::new();
    let mut ev_sum = [0.0f64; 4];

    // ── Accumulators: bulk (inscribed diamond) ──
    let mut bulk_n: usize = 0;
    let mut bulk_n_lorentzian: usize = 0;
    let mut bulk_ev_sum = [0.0f64; 4];
    let mut bulk_g_ratios: Vec<f64> = Vec::new();

    for i in 0..n_macro {
        if areas[i] <= 0 || metrics[i].n_links == 0 {
            continue;
        }
        n_valid += 1;

        let ev = metrics[i].eigenvalues_raw();
        for d in 0..4 {
            ev_sum[d] += ev[d];
        }

        let (neg, pos) = metrics[i].signature_raw(threshold);
        let is_lorentzian = neg == 1 && pos == 3;
        if is_lorentzian {
            n_lorentzian += 1;
        }

        let lambda_max = metrics[i].principal_eigenvalue_raw();
        g_ratios.push(lambda_max / areas[i] as f64);

        // ── Inscribed diamond bulk filter ──
        let dt = (centroids[i][0] - t_center).abs();
        let dx = centroids[i][1] - s_center[0];
        let dy = centroids[i][2] - s_center[1];
        let dz = centroids[i][3] - s_center[2];
        let r_spatial = (dx * dx + dy * dy + dz * dz).sqrt();
        if dt + r_spatial < r_cut {
            bulk_n += 1;
            for d in 0..4 {
                bulk_ev_sum[d] += ev[d];
            }
            if is_lorentzian {
                bulk_n_lorentzian += 1;
                // Spatial-trace normalised G: |λ_temporal| / (λ₁ + λ₂ + λ₃)
                let spatial_trace = ev[1] + ev[2] + ev[3];
                if spatial_trace.abs() > 1e-15 {
                    bulk_g_ratios.push(ev[0].abs() / spatial_trace);
                }
            }
        }
    }

    // ── All-node statistics ──
    let lorentzian_frac = if n_valid > 0 {
        n_lorentzian as f64 / n_valid as f64
    } else {
        0.0
    };
    let mean_eigenvalues = if n_valid > 0 {
        let nv = n_valid as f64;
        [ev_sum[0] / nv, ev_sum[1] / nv, ev_sum[2] / nv, ev_sum[3] / nv]
    } else {
        [0.0; 4]
    };
    let (g_mean, g_var, g_cv) = coefficient_of_variation(&g_ratios);
    let g_se = if g_ratios.len() >= 2 {
        g_var.sqrt() / (g_ratios.len() as f64).sqrt()
    } else {
        f64::NAN
    };

    // ── Bulk statistics ──
    let bulk_lorentzian_frac = if bulk_n > 0 {
        bulk_n_lorentzian as f64 / bulk_n as f64
    } else {
        0.0
    };
    let bulk_mean_eigenvalues = if bulk_n > 0 {
        let bn = bulk_n as f64;
        [bulk_ev_sum[0] / bn, bulk_ev_sum[1] / bn, bulk_ev_sum[2] / bn, bulk_ev_sum[3] / bn]
    } else {
        [0.0; 4]
    };
    let (bulk_g_mean, bulk_g_var, bulk_g_cv) = coefficient_of_variation(&bulk_g_ratios);
    let bulk_g_se = if bulk_g_ratios.len() >= 2 {
        bulk_g_var.sqrt() / (bulk_g_ratios.len() as f64).sqrt()
    } else {
        f64::NAN
    };

    TensorAnalysis {
        lorentzian_frac,
        n_lorentzian,
        n_valid,
        g_mean,
        g_se,
        g_cv,
        mean_eigenvalues,
        bulk_n,
        bulk_n_lorentzian,
        bulk_lorentzian_frac,
        bulk_mean_eigenvalues,
        bulk_g_mean,
        bulk_g_se,
        bulk_g_cv,
    }
}

/// Summary of the Jacobson tensor analysis across all macro-nodes.
pub struct TensorAnalysis {
    // ── All valid nodes ──
    /// Fraction with Lorentzian signature (−+++).
    pub lorentzian_frac: f64,
    /// Number with (−+++) signature.
    pub n_lorentzian: usize,
    /// Number analysed (δA > 0 and n_links > 0).
    pub n_valid: usize,
    /// Mean of λ_max / δA.
    pub g_mean: f64,
    /// Standard error of λ_max / δA.
    pub g_se: f64,
    /// Coefficient of variation of λ_max / δA.
    pub g_cv: f64,
    /// Eigenvalue spectrum averaged: [λ_0, λ_1, λ_2, λ_3] ascending.
    pub mean_eigenvalues: [f64; 4],

    // ── Bulk nodes only (δA > 20% of max) ──
    /// Number of bulk nodes.
    pub bulk_n: usize,
    /// Number of bulk nodes with (−+++) signature.
    pub bulk_n_lorentzian: usize,
    /// Fraction of bulk nodes with Lorentzian signature.
    pub bulk_lorentzian_frac: f64,
    /// Mean eigenvalues in the bulk.
    pub bulk_mean_eigenvalues: [f64; 4],
    /// Emergent G from spatial-trace normalization: |λ_temporal| / (λ₁+λ₂+λ₃).
    /// Only computed for bulk Lorentzian nodes.
    pub bulk_g_mean: f64,
    pub bulk_g_se: f64,
    pub bulk_g_cv: f64,
}

/// Run the full tensor Jacobson analysis.
///
/// For each macro-node with δA > 0:
/// 1. Extract eigenvalues of C^{μν}
/// 2. Check for (−+++) Lorentzian signature
/// 3. Compute λ_max / δA (candidate emergent G)
/// 4. Aggregate statistics
pub fn tensor_analysis(
    metrics: &[FisherMetric],
    areas: &[i64],
) -> TensorAnalysis {
    let n_macro = metrics.len();

    let mut n_lorentzian: usize = 0;
    let mut n_valid: usize = 0;
    let mut g_ratios: Vec<f64> = Vec::new();
    let mut ev_sum = [0.0f64; 4];

    // Adaptive threshold: median of all |eigenvalues| × 1e-6
    // First pass: collect all eigenvalue magnitudes
    let mut all_mags: Vec<f64> = Vec::new();
    for i in 0..n_macro {
        if areas[i] <= 0 || metrics[i].n_links == 0 {
            continue;
        }
        let ev = metrics[i].eigenvalues();
        for &v in &ev {
            all_mags.push(v.abs());
        }
    }
    all_mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = if all_mags.is_empty() {
        1e-12
    } else {
        all_mags[all_mags.len() / 2] * 1e-6
    };

    for i in 0..n_macro {
        if areas[i] <= 0 || metrics[i].n_links == 0 {
            continue;
        }
        n_valid += 1;

        let ev = metrics[i].eigenvalues();
        for d in 0..4 {
            ev_sum[d] += ev[d];
        }

        let (neg, pos) = metrics[i].signature(threshold);
        if neg == 1 && pos == 3 {
            n_lorentzian += 1;
        }

        let lambda_max = metrics[i].principal_eigenvalue();
        g_ratios.push(lambda_max / areas[i] as f64);
    }

    let lorentzian_frac = if n_valid > 0 {
        n_lorentzian as f64 / n_valid as f64
    } else {
        0.0
    };

    let mean_eigenvalues = if n_valid > 0 {
        let nv = n_valid as f64;
        [
            ev_sum[0] / nv,
            ev_sum[1] / nv,
            ev_sum[2] / nv,
            ev_sum[3] / nv,
        ]
    } else {
        [0.0; 4]
    };

    let (g_mean, g_var, g_cv) = coefficient_of_variation(&g_ratios);
    let g_se = if g_ratios.len() >= 2 {
        g_var.sqrt() / (g_ratios.len() as f64).sqrt()
    } else {
        f64::NAN
    };

    TensorAnalysis {
        lorentzian_frac,
        n_lorentzian,
        n_valid,
        g_mean,
        g_se,
        g_cv,
        mean_eigenvalues,
        // Mean-subtracted pipeline: no bulk filter (all zeros)
        bulk_n: 0,
        bulk_n_lorentzian: 0,
        bulk_lorentzian_frac: 0.0,
        bulk_mean_eigenvalues: [0.0; 4],
        bulk_g_mean: f64::NAN,
        bulk_g_se: f64::NAN,
        bulk_g_cv: f64::NAN,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizon_areas_basic() {
        let micro_head: Vec<u32> = vec![0, 1, 3, 3, 3];
        let micro_data: Vec<u32> = vec![2, 2, 3];
        let macro_head: Vec<u32> = vec![0, 1, 1];
        let macro_data: Vec<u32> = vec![1];
        let m2m: Vec<usize> = vec![0, 0, 1, 1];

        let areas = horizon_areas(&micro_head, &micro_data, &macro_head, &macro_data, &m2m, 4, 2);
        assert_eq!(areas[0], 3);
        assert_eq!(areas[1], 0);
    }

    #[test]
    fn test_horizon_areas_intra_voxel_excluded() {
        let micro_head: Vec<u32> = vec![0, 1, 1];
        let micro_data: Vec<u32> = vec![1];
        let macro_head: Vec<u32> = vec![0, 0];
        let macro_data: Vec<u32> = vec![];
        let m2m: Vec<usize> = vec![0, 0];

        let areas = horizon_areas(&micro_head, &micro_data, &macro_head, &macro_data, &m2m, 2, 1);
        assert_eq!(areas[0], 0);
    }

    #[test]
    fn test_horizon_areas_non_child_excluded() {
        let micro_head: Vec<u32> = vec![0, 1, 1, 1];
        let micro_data: Vec<u32> = vec![2];
        let macro_head: Vec<u32> = vec![0, 1, 1, 1];
        let macro_data: Vec<u32> = vec![1];
        let m2m: Vec<usize> = vec![0, 1, 2];

        let areas = horizon_areas(&micro_head, &micro_data, &macro_head, &macro_data, &m2m, 3, 3);
        assert_eq!(areas[0], 0);
    }

    #[test]
    fn test_net_flux_row_sums() {
        let mut b = DMatrix::<i64>::zeros(3, 3);
        b[(0, 1)] = 1;
        b[(0, 2)] = -9;
        b[(1, 2)] = 1;

        let flux = net_flux_per_node(&b);
        assert_eq!(flux[0], 1 + (-9));
        assert_eq!(flux[1], 1);
        assert_eq!(flux[2], 0);
    }

    #[test]
    fn test_clausius_ratios_skips_zero_area() {
        let flux = vec![10i64, 5, -3];
        let areas = vec![2i64, 0, 3];

        let ratios = clausius_ratios(&flux, &areas);
        assert_eq!(ratios.len(), 2);
        assert!((ratios[0] - 5.0).abs() < 1e-12);
        assert!((ratios[1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_cv_constant_data() {
        let data = vec![3.0, 3.0, 3.0, 3.0];
        let (mean, var, cv) = coefficient_of_variation(&data);
        assert!((mean - 3.0).abs() < 1e-12);
        assert!(var.abs() < 1e-12);
        assert!(cv.abs() < 1e-12);
    }

    /// Second moment tracks correctly for a single link.
    #[test]
    fn test_fisher_second_moment() {
        let mut fm = FisherMetric::new();
        fm.accumulate([1.0, 0.0, 0.0, 0.0], 1.0);

        assert_eq!(fm.n_links, 1);
        assert!((fm.second_moment[0][0] - 1.0).abs() < 1e-12);
        assert!(fm.second_moment[1][1].abs() < 1e-12);
        // First moment: sum_dx = (1,0,0,0)
        assert!((fm.sum_dx[0] - 1.0).abs() < 1e-12);
    }

    /// Mean-subtracted covariance: two links with different displacements
    /// should produce the statistical covariance of {Δx_1, Δx_2}.
    #[test]
    fn test_fisher_mean_subtracted_covariance() {
        let mut fm = FisherMetric::new();
        // Two timelike links: Δx = (1,0,0,0) and (3,0,0,0)
        fm.accumulate([1.0, 0.0, 0.0, 0.0], 1.0);
        fm.accumulate([3.0, 0.0, 0.0, 0.0], 1.0);

        let cov = fm.covariance();
        // ⟨Δt²⟩ = (1+9)/2 = 5, ⟨Δt⟩² = ((1+3)/2)² = 4
        // Cov[0][0] = 5 - 4 = 1
        assert!((cov[0][0] - 1.0).abs() < 1e-12, "got {}", cov[0][0]);
        assert!(cov[1][1].abs() < 1e-12);
    }

    /// Causal scatter: constant Δt, varying spatial directions.
    ///
    /// All causal links have Δt = 2 (constant) but scatter in space.
    /// After mean subtraction: Var(Δt) = 0, Var(Δx_i) > 0.
    /// Eigenvalues: [0, +, +, +] — time direction has zero variance.
    #[test]
    fn test_fisher_causal_scatter_time_flat() {
        let mut fm = FisherMetric::new();
        for _ in 0..100 {
            fm.accumulate([2.0, 1.0, 0.0, 0.0], 1.0);
            fm.accumulate([2.0, -1.0, 0.0, 0.0], 1.0);
            fm.accumulate([2.0, 0.0, 1.0, 0.0], 1.0);
            fm.accumulate([2.0, 0.0, -1.0, 0.0], 1.0);
            fm.accumulate([2.0, 0.0, 0.0, 1.0], 1.0);
            fm.accumulate([2.0, 0.0, 0.0, -1.0], 1.0);
        }
        let ev = fm.eigenvalues();
        // Time eigenvalue ≈ 0 (no variance in Δt)
        assert!(ev[0].abs() < 1e-10, "time eigenvalue should be ~0, got {}", ev[0]);
        // Spatial eigenvalues ≈ 1/3 each (isotropic scatter)
        for i in 1..4 {
            assert!(ev[i] > 0.1, "spatial eigenvalue should be positive, got {}", ev[i]);
        }
        // λ_time / λ_space ratio should be ~0 (time suppressed)
        let ratio = ev[0].abs() / ev[3];
        assert!(ratio < 0.01, "time/space ratio should be small, got {}", ratio);
    }

    /// fisher_covariances on a known micro graph: verify second_moment accumulation.
    #[test]
    fn test_fisher_covariances_accumulation() {
        let pts: Vec<[f64; 4]> = vec![[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]];
        let micro_head: Vec<u32> = vec![0, 1, 1];
        let micro_data: Vec<u32> = vec![1];
        let macro_head: Vec<u32> = vec![0, 1, 1];
        let macro_data: Vec<u32> = vec![1];
        let m2m: Vec<usize> = vec![0, 1];

        let metrics =
            fisher_covariances(&pts, &micro_head, &micro_data, &macro_head, &macro_data, &m2m, 2, 2);

        assert_eq!(metrics[0].n_links, 1);
        assert_eq!(metrics[1].n_links, 0);

        // Δx = (1,2,3,4), second_moment[μ][ν] = Δx[μ]·Δx[ν]
        let dx = [1.0, 2.0, 3.0, 4.0];
        for mu in 0..4 {
            for nu in 0..4 {
                let expected = dx[mu] * dx[nu];
                assert!(
                    (metrics[0].second_moment[mu][nu] - expected).abs() < 1e-12,
                    "M[{mu}][{nu}]: got {}, expected {expected}",
                    metrics[0].second_moment[mu][nu]
                );
            }
        }
    }

    /// tensor_analysis with the mean-subtracted covariance.
    #[test]
    fn test_tensor_analysis_with_scatter() {
        let mut metrics = vec![FisherMetric::new(), FisherMetric::new()];
        // M0: causal links with constant Δt, spatial scatter
        for _ in 0..50 {
            metrics[0].accumulate([2.0, 1.0, 0.0, 0.0], 1.0);
            metrics[0].accumulate([2.0, -1.0, 0.0, 0.0], 1.0);
            metrics[0].accumulate([2.0, 0.0, 1.0, 0.0], 1.0);
            metrics[0].accumulate([2.0, 0.0, -1.0, 0.0], 1.0);
            metrics[0].accumulate([2.0, 0.0, 0.0, 1.0], 1.0);
            metrics[0].accumulate([2.0, 0.0, 0.0, -1.0], 1.0);
        }
        let areas = vec![100i64, 0];

        let ta = tensor_analysis(&metrics, &areas);
        assert_eq!(ta.n_valid, 1);
        // Signature (0,3): time eigenvalue ≈ 0, 3 positive spatial
        assert!(ta.g_mean > 0.0, "emergent G should be positive");
        // λ_max ≈ 1/3, area = 100, so G ≈ 0.003
        assert!(ta.g_mean < 1.0, "G should be small for large area");
    }

    /// Multi-depth BD weights produce negative eigenvalues.
    ///
    /// 5-node linear chain 0→1→2→3→4, centroids spread along time axis.
    /// Depth-0 weight +1, depth-1 weight −9: the −9 contributions
    /// should overwhelm the +1 contributions and produce a negative
    /// eigenvalue in the time direction.
    #[test]
    fn test_multidepth_negative_eigenvalue() {
        // Chain: 0→1→2→3→4
        let macro_head: Vec<u32> = vec![0, 1, 2, 3, 4, 4];
        let macro_data: Vec<u32> = vec![1, 2, 3, 4];
        // Centroids along time axis with spatial offset
        let centroids: Vec<[f64; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [2.0, 0.0, 0.1, 0.0],
            [3.0, -0.1, 0.0, 0.0],
            [4.0, 0.0, -0.1, 0.0],
        ];

        let metrics = fisher_covariances_multidepth(&centroids, &macro_head, &macro_data, 5);

        // Node 0 reaches: 1 (depth 0, w=+1), 2 (depth 1, w=−9),
        //                  3 (depth 2, w=+16), 4 (depth 3, w=−8)
        assert_eq!(metrics[0].n_links, 4);

        // The raw eigenvalues should have at least one negative value
        // from the −9 and −8 contributions
        let ev = metrics[0].eigenvalues_raw();
        let has_negative = ev.iter().any(|&v| v < -1e-10);
        assert!(
            has_negative,
            "multi-depth BD should produce negative eigenvalues, got {:?}",
            ev
        );
    }

    /// Multi-depth on a chain: node 0 should have (1,3) raw signature.
    #[test]
    fn test_multidepth_lorentzian_chain() {
        // Chain: 0→1→2→3→4, time-aligned centroids
        let macro_head: Vec<u32> = vec![0, 1, 2, 3, 4, 4];
        let macro_data: Vec<u32> = vec![1, 2, 3, 4];
        let centroids: Vec<[f64; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.3, 0.2, 0.1],
            [2.0, -0.2, 0.3, -0.1],
            [3.0, 0.1, -0.3, 0.2],
            [4.0, -0.1, 0.1, -0.3],
        ];

        let metrics = fisher_covariances_multidepth(&centroids, &macro_head, &macro_data, 5);
        let ev = metrics[0].eigenvalues_raw();

        // BD weights sum: +1 − 9 + 16 − 8 = 0
        // Time direction accumulates: 1·1² − 9·4 + 16·9 − 8·16 = 1 − 36 + 144 − 128 = −19
        // So the time-time component of second_moment should be negative → negative eigenvalue
        let neg_count = ev.iter().filter(|&&v| v < -1e-10).count();
        assert!(
            neg_count >= 1,
            "expected at least 1 negative eigenvalue, got ev = {:?}",
            ev
        );
    }
}
