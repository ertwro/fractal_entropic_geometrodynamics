// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18746995

//! Phase 2 — Causal Prism Topological Defect (Pure Integer Bipartite Calculus)
//!
//! Implements the Cálculo de Kuratowski: matter emerges as K₂,ₙ bipartite
//! obstructions (Causal Prisms) in a triangle-free Hasse DAG, with the strong
//! force realised as K₅ threat contraction preserving planarity.  This is the
//! core of *Fractal Entropic Geometrodynamics* (FEG) by J. P. Silva Alvarado.
//!
//! Zenodo: <https://doi.org/10.5281/zenodo.18746995>
//!
//! Key theorems implemented below:
//!   - Vol II, Thm 4.2: Uniqueness of Bifurcation-Convergence (MIN_PRISM_SHARED ≥ 3)
//!   - Vol II, Sole Principle of Interaction: K₅ minor ↔ confinement
//!   - Vol II, Def 3.1: Topological mass M = |belly|
//!   - Vol II, §5: Generation classification by bulk-momentum signature
//!
//! THE THREE GEOMETRIES — implemented in code:
//!
//!   I.  GEOMETRY OF ORDER (Phase 1 → this module's input):
//!       Transitive Reduction produces a triangle-free DAG.  The "empty"
//!       structure is not empty — it is a 4D vacuum whose Lorentz invariance
//!       is emergent from redundancy removal on a Poisson process.
//!
//!  II.  GEOMETRY OF MATTER (this module — Prism detection):
//!       The Causal Prism K_{2,N} is the unique topological particle permitted
//!       by the triangle-free constraint.  Mass is not a property added to a
//!       field — it is the integer count N of intermediate causal paths.
//!       The mass hierarchy (electron/muon/tau) reduces to N ∈ {3, 4, 5}.
//!
//! III.  GEOMETRY OF LOGIC (this module — O(N) algorithm):
//!       The 2-hop detection algorithm runs in O(N) because the PHYSICS is
//!       local.  Hasse links have bounded proper time (τ ≤ 8); therefore
//!       K_{2,N} poles must be within bounded causal distance.  Any valid
//!       prism partner v of node u is reachable in exactly 2 hops through a
//!       shared intermediate w.  This is not merely an optimisation — it is
//!       a computational proof that PHYSICAL LOCALITY IS AN INFORMATION
//!       PRINCIPLE.  The reason the universe can exist at 10^{80} particles
//!       without "crashing" is the same reason this code processes 10^8 nodes
//!       in minutes: reality only cares about its immediate neighbours.
//!
//!       O(N) scaling IS the physics.  An O(N²) algorithm would imply
//!       non-local interactions — a universe that could not scale.
//!
//! ─── Technical Details ───────────────────────────────────────────────────
//!
//! The transitive reduction in Phase 1 produces a triangle-free DAG by
//! mathematical necessity. Any pair of nodes (u, w, v) with u→w, w→v, u→v
//! would make u→v redundant and be removed. Therefore K₄ cliques (which
//! require triangles) CANNOT exist — zero K4 is not a bug, it is a theorem.
//!
//! The correct fundamental topological particle is the **Causal Prism**
//! (Prisma Causal): a bipartite K_{2,N} subgraph where:
//!   - Two pole nodes (Origin, Destination) at graph-distance ≥ 2
//!   - N ≥ MIN_PRISM_SHARED intermediate nodes shared by both poles
//!   - Intermediates are mutually disconnected (guaranteed by triangle-free DAG)
//!
//! Detection — 2-hop forward-forward traversal (the key topological insight):
//!   For u ∈ core_nodes:
//!     For w ∈ children(u):             [1-hop: belly candidate]
//!       For v ∈ children(w), v ∈ core: [2-hop: future pole candidate]
//!         belly = children(u) ∩ parents(v)
//!         If |belly| ≥ MIN_PRISM_SHARED: Prism(u, v, belly)
//!
//!   Completeness proof: if u and v are poles of a timelike K_{2,N}, then
//!   every belly node w_i satisfies u → w_i → v.  The path u → w_i → v
//!   guarantees v is discovered as a 2-hop candidate via w_i.
//!   Therefore the forward-forward search finds ALL valid partners — it is exact.
//!
//! Complexity: O(|Core| × D²) candidate collection + O(cands × D) verification
//!   D ≈ 4–15 for Hasse diagrams → effectively O(N) in practice.
//!   At N=100M, core=10M: ~10¹⁰ operations ≈ minutes (vs O(core²) ≈ days).
//!
//! Zero floating-point until Phase 3:
//!   - Core identification:  integer degree ranking
//!   - Prism detection:      2-hop traversal + sorted-CSR intersection O(D)
//!   - Threat contraction:   node connecting to both poles + ≥ PRISM_THREAT intermediates
//!   - Signature encoding:   bit-packed quartile bulk-momentum over all prism components

use rayon::prelude::*;
use serde::{Serialize, Deserialize};

/// Fraction of nodes selected as topological core (numerator / denominator).
const CORE_NUM: usize = 1;
const CORE_DEN: usize = 10;

/// Minimum shared intermediates to qualify as a Causal Prism (K_{2,N}, N ≥ 3).
///
/// Cálculo de Kuratowski derivation (Silva Alvarado, FEG Vol II, Thm 4.2):
/// Uniqueness of Bifurcation-Convergence requires ≥ 3 independent length-2 paths
/// in a K₃-free Hasse graph.  K_{2,2} yields only S₂ → U(1); the SU(3) gauge
/// structure demands S₃, hence N ≥ 3 (Vol II, Sole Principle of Interaction).
///
/// DOI: 10.5281/zenodo.18746995
const MIN_PRISM_SHARED: usize = 3;

/// K₅ threat threshold: external node connected to both poles AND ≥ 2
/// intermediates → 5 mutually reachable vertices → K₅ minor.
///
/// Cálculo de Kuratowski derivation (Silva Alvarado, FEG Vol II, Axiom §1.4):
/// Kuratowski's Theorem guarantees that a finite graph is planar iff it contains
/// no K₅ or K₃,₃ minor.  Absorption into the nearest pole is the minimal
/// planarity-preserving contraction — the discrete analogue of colour confinement.
///
/// DOI: 10.5281/zenodo.18746995
const PRISM_THREAT: usize = 2;

// ─── Data Structures ─────────────────────────────────────────────────────────

/// A Causal Prism defect: irreducible bipartite K_{2,N} structure.
///
/// Origin and Destination are the two poles (at graph-distance ≥ 2 in the
/// undirected Hasse).  Intermediates are their N ≥ 3 shared neighbours —
/// mutually disconnected by the triangle-free property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPrism {
    pub origin: usize,
    pub destination: usize,
    pub intermediates: Vec<usize>,
}

/// Result of the Causal Prism contraction with particle classification.
pub struct DefectResult {
    /// Vacuum Hasse CSR row pointers.
    pub vac_head: Vec<u32>,
    /// Vacuum Hasse CSR column indices (sorted neighbours).
    pub vac_data: Vec<u32>,
    /// Defect CSR row pointers (after threat contraction + prism pole-completion).
    pub def_head: Vec<u32>,
    /// Defect CSR column indices.
    pub def_data: Vec<u32>,
    /// Core node indices (top 10% by degree) in the vacuum graph.
    pub vacuum_core: Vec<usize>,
    /// Core nodes surviving threat contraction.
    pub defect_core: Vec<usize>,
    /// Generation 1 prism node indices — most abundant signature (electron-like).
    pub gen1_nodes: Vec<usize>,
    /// Generation 2 prism node indices — second most abundant (muon-like).
    pub gen2_nodes: Vec<usize>,
    /// Generation 3 prism node indices — third most abundant (tau-like).
    pub gen3_nodes: Vec<usize>,
    /// Anti-Generation 1 prism node indices — CPT conjugate (positron-like).
    pub anti1_nodes: Vec<usize>,
    /// Sterile prism node indices — prisms with Φ = 0 (fully phase-cancelled, dark matter).
    pub sterile_nodes: Vec<usize>,
    /// Merge map: `merge_into[i]` = canonical node.  Identity for non-merged nodes.
    pub merge_into: Vec<usize>,
    /// Average topological mass (N = number of intermediates) for Generation 1.
    pub mass_gen1: f64,
    /// Average topological mass for Generation 2.
    pub mass_gen2: f64,
    /// Average topological mass for Generation 3.
    pub mass_gen3: f64,
    /// Average topological mass for Anti-Generation 1.
    pub mass_anti1: f64,
}

/// Aggregate topology data from Phase 2 (prism detection + classification).
///
/// Exported alongside spectral results so that `output.rs` can write
/// `topology_summary.csv` and `mass_spectrum.csv` without reaching back
/// into the defect graph.
#[derive(Clone, Serialize, Deserialize)]
pub struct TopologySummary {
    pub total_nodes: usize,
    pub total_prisms: usize,
    pub max_intermediates: usize,
    pub count_gen1: usize,
    pub count_gen2: usize,
    pub count_gen3: usize,
    pub count_antigen1: usize,
    /// Number of sterile prism nodes (Φ = 0, fully phase-cancelled).
    pub count_sterile: usize,
    pub avg_mass_gen1: f64,
    pub avg_mass_gen2: f64,
    pub avg_mass_gen3: f64,
    /// Average topological mass for sterile prisms.
    pub avg_mass_sterile: f64,
    /// Histogram of committed prisms by belly size: (N_intermediates, frequency).
    pub prism_histogram: Vec<(usize, usize)>,
    /// Phase-coherence mass decomposition (Theorem: zero free parameters).
    /// Σ|Φ(P)| — total visible (EM) mass across all prisms.
    pub visible_mass_total: usize,
    /// Σ(N - |Φ(P)|) — total dark mass across all prisms.
    pub dark_mass_total: usize,
    /// ΣN — total gravitational mass across all prisms.
    pub grav_mass_total: usize,
    /// Ω_dark / Ω_vis = Σ(N - |Φ|) / Σ|Φ| — linear mass ratio.
    pub omega_ratio: f64,
    /// Σ|Φ(P)|² — total EM self-energy (numerator of α).
    pub phase_sq_total: usize,
    /// ΣN² — total gravitational self-energy (denominator of α).
    pub mass_sq_total: usize,
    /// α = Q_topo / (8π) — emergent fine structure constant.
    pub alpha_em: f64,
    /// Ω_energy = 1/Q_topo − 1 = (ΣN² − Σ|Φ|²) / Σ|Φ|² — self-energy dark matter ratio.
    /// Satisfies the exact identity α(1 + Ω_energy) = 1/(8π).
    pub omega_energy: f64,
    /// Intermediate phase census: count of intermediate nodes with φ = +1 (source-like).
    pub phase_pos_count: usize,
    /// Intermediate phase census: count of intermediate nodes with φ = 0 (balanced).
    pub phase_zero_count: usize,
    /// Intermediate phase census: count of intermediate nodes with φ = -1 (sink-like).
    pub phase_neg_count: usize,
    /// Number of prisms per generation (prism count, not node count).
    pub prisms_gen1: usize,
    pub prisms_gen2: usize,
    pub prisms_gen3: usize,
}

/// Build a frequency histogram of committed prism belly sizes.
fn build_prism_histogram(prisms: &[CausalPrism]) -> Vec<(usize, usize)> {
    let mut hist: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for p in prisms {
        *hist.entry(p.intermediates.len()).or_insert(0) += 1;
    }
    let mut pairs: Vec<(usize, usize)> = hist.into_iter().collect();
    pairs.sort_unstable_by_key(|&(n, _)| n);
    pairs
}

// ─── Signature helpers ───────────────────────────────────────────────────────

/// Encode the topological signature of a Prism as a packed u32.
///
/// Collects bulk_momentum from all components, sorts, takes four quartile
/// values [q0, q1, q2, q3], normalises to [-10, +10], offsets by +16 (fits
/// in 5 bits), and packs into bits [0..4], [5..9], [10..14], [15..19].
///
/// The anti-signature (CPT conjugate) reverses and negates: ā_i = –c_{3–i} + 16.
#[inline]
fn prism_signature(prism: &CausalPrism, bulk_momentum: &[i32]) -> u32 {
    let mut mv: Vec<i32> = Vec::with_capacity(2 + prism.intermediates.len());
    mv.push(bulk_momentum[prism.origin]);
    mv.push(bulk_momentum[prism.destination]);
    for &w in &prism.intermediates {
        mv.push(bulk_momentum[w]);
    }
    mv.sort_unstable();
    let n = mv.len();
    let q = [
        mv[0],
        mv[n / 4],
        mv[n / 2],
        mv[3 * n / 4],
    ];
    let max_m = q.iter().map(|x| x.abs()).max().unwrap_or(0);
    if max_m == 0 {
        return 0;
    }
    ((q[0] * 10 / max_m + 16) as u32)
        | (((q[1] * 10 / max_m + 16) as u32) << 5)
        | (((q[2] * 10 / max_m + 16) as u32) << 10)
        | (((q[3] * 10 / max_m + 16) as u32) << 15)
}

#[inline]
fn anti_signature(sig: u32) -> u32 {
    let c0 = (sig & 0x1F) as i32 - 16;
    let c1 = ((sig >> 5) & 0x1F) as i32 - 16;
    let c2 = ((sig >> 10) & 0x1F) as i32 - 16;
    let c3 = ((sig >> 15) & 0x1F) as i32 - 16;
    ((-c3 + 16) as u32)
        | (((-c2 + 16) as u32) << 5)
        | (((-c1 + 16) as u32) << 10)
        | (((-c0 + 16) as u32) << 15)
}

// ─── Main function ────────────────────────────────────────────────────────────

/// Apply the Causal Prism contraction with particle classification.
///
/// Searches for K_{2,N} bipartite structures via 2-hop traversal on the
/// triangle-free Hasse CSR, classifies them by topological signature into
/// generations, and performs threat contraction.
pub fn apply_defect(
    n: usize,
    adj_head_vac: Vec<u32>,
    adj_data_vac: Vec<u32>,
    bulk_momentum: Vec<i32>,
) -> (DefectResult, TopologySummary, Vec<CausalPrism>) {

    // ════════════════════════════════════════════════════════════════
    //  1. CSR helpers (Zero-Copy, sorted adjacency guaranteed by Phase 1)
    // ════════════════════════════════════════════════════════════════

    // Forward (directed) CSR — successors of u.
    let get_nb = |u: usize| -> &[u32] {
        let s = adj_head_vac[u] as usize;
        let e = adj_head_vac[u + 1] as usize;
        if s <= e && e <= adj_data_vac.len() { &adj_data_vac[s..e] } else { &[] }
    };

    // ── Build reverse (incoming) CSR: predecessors of each node ──
    //
    // The directed Hasse CSR stores u→v (forward only).  For the belly
    // intersection we need parents(v) = {w : w→v}.  The forward-forward
    // candidate search finds v via u→w→v, then the belly is computed as
    // children(u) ∩ parents(v) using this reverse CSR.
    //
    // Memory: same size as forward CSR.
    let mut rev_deg = vec![0u32; n];
    for u in 0..n {
        for &v in get_nb(u) {
            rev_deg[v as usize] += 1;
        }
    }
    let mut rev_head = Vec::with_capacity(n + 1);
    rev_head.push(0u32);
    for &d in &rev_deg {
        let prev = *rev_head.last().unwrap();
        rev_head.push(prev + d);
    }
    let total_rev = *rev_head.last().unwrap() as usize;
    let mut rev_data = vec![0u32; total_rev];
    let mut rev_pos = vec![0u32; n];
    for u in 0..n {
        for &v in get_nb(u) {
            let vi = v as usize;
            let pos = (rev_head[vi] + rev_pos[vi]) as usize;
            rev_data[pos] = u as u32;
            rev_pos[vi] += 1;
        }
    }
    drop(rev_pos);
    // Sort each reverse adjacency list for consistency
    for u in 0..n {
        let s = rev_head[u] as usize;
        let e = rev_head[u + 1] as usize;
        rev_data[s..e].sort_unstable();
    }

    let get_nb_rev = |u: usize| -> &[u32] {
        let s = rev_head[u] as usize;
        let e = rev_head[u + 1] as usize;
        if s <= e && e <= rev_data.len() { &rev_data[s..e] } else { &[] }
    };

    // Count intersection of two SORTED slices.  O(|a| + |b|).
    let count_isect = |a: &[u32], b: &[u32]| -> usize {
        let (mut i, mut j, mut c) = (0, 0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less    => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal   => { c += 1; i += 1; j += 1; }
            }
        }
        c
    };

    // Collect intersection of two SORTED slices into a Vec<usize>.
    let extract_isect = |a: &[u32], b: &[u32]| -> Vec<usize> {
        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less    => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal   => {
                    out.push(a[i] as usize); i += 1; j += 1;
                }
            }
        }
        out
    };

    // Binary-search connectivity check.
    let connected = |u: usize, v: u32| -> bool {
        get_nb(u).binary_search(&v).is_ok()
    };

    // ════════════════════════════════════════════════════════════════
    //  2. Core = top CORE_NUM/CORE_DEN nodes by degree
    // ════════════════════════════════════════════════════════════════
    let target_core = (n * CORE_NUM / CORE_DEN).max(1);

    // Use undirected degree (out + in) for a balanced core at the diamond's centre.
    let mut by_deg: Vec<(usize, usize)> = (0..n)
        .map(|i| {
            let out_d = (adj_head_vac[i + 1] - adj_head_vac[i]) as usize;
            let in_d  = (rev_head[i + 1] - rev_head[i]) as usize;
            (out_d + in_d, i)
        })
        .collect();
    by_deg.par_sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let cutoff = by_deg.get(target_core.saturating_sub(1)).map(|&(d, _)| d).unwrap_or(0);
    let mut is_core = vec![false; n];
    let mut core_count = 0usize;
    for &(deg, node) in &by_deg {
        if core_count >= target_core && deg < cutoff { break; }
        if deg >= cutoff { is_core[node] = true; core_count += 1; }
    }
    let core_nodes: Vec<usize> = (0..n).filter(|&i| is_core[i]).collect();

    {
        let msg = format!(
            "[Phase 2] Causal Prism search (K_{{2,N}} bipartite, 2-hop traversal)\n  \
             Core: {core_count} nodes (top {CORE_NUM}/{CORE_DEN}, cutoff degree ≥ {cutoff})"
        );
        println!("{}", msg);
        use std::fs::OpenOptions; use std::io::Write;
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("simulation.log") {
            writeln!(f, "{}", msg).ok();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  3. Prism detection (Timelike K_{2,N} — Forward-Forward)
    //
    //  The Causal Prism is a TIMELIKE structure: Past Pole → Belly → Future Pole.
    //  K_{2,N} means two poles (u = past, v = future) with belly nodes w_i:
    //    u → w_i  and  w_i → v  for all i ∈ {1...N}
    //  i.e., the belly = children(u) ∩ parents(v).
    //
    //  Algorithm:
    //    For each u ∈ core:
    //      For w ∈ children(u):                  [1-hop: belly candidate]
    //        For v ∈ children(w), v ∈ core:      [2-hop: future pole candidate]
    //          belly = children(u) ∩ parents(v)
    //          if |belly| ≥ MIN_PRISM_SHARED: Prism(u, v, belly)
    //    Select best partner per u (greedy maximum belly size).
    // ════════════════════════════════════════════════════════════════
    println!("  Scanning for Causal Prisms (forward-forward, belly intersection)...");

    let mut prisms: Vec<CausalPrism> = Vec::new();
    let mut is_placed = vec![false; n];
    let total = core_nodes.len();
    let report_step = (total / 10).max(1);

    // Diagnostic: track intersection count distribution
    let mut max_isect_global = 0usize;
    let mut isect_histogram = vec![0usize; 20];  // counts for isect 0..19

    // Reusable candidate buffer — avoids per-iteration allocation.
    let mut cands: Vec<u32> = Vec::with_capacity(512);

    for (idx, &u) in core_nodes.iter().enumerate() {
        if idx % report_step == 0 && idx > 0 {
            let pct = idx as f64 / total as f64 * 100.0;
            let msg = format!(
                "  [Phase 2] {pct:.0}% ({idx}/{total}) | Prisms: {}",
                prisms.len()
            );
            println!("{}", msg);
            use std::fs::OpenOptions; use std::io::Write;
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("simulation.log") {
                writeln!(f, "{}", msg).ok();
            }
        }

        if is_placed[u] { continue; }

        let nb_u = get_nb(u);
        // ── Minimum valence pruning: u needs ≥ MIN_PRISM_SHARED children
        // to form a K_{2,N} belly. Skip early to avoid the 2-hop traversal.
        if nb_u.len() < MIN_PRISM_SHARED { continue; }

        // ── 2-hop candidate collection via u→w→v (O(D²) per node) ─────
        //
        // Path: u → w → v  ("forward-forward")
        //   For each child w of u (u→w), find all children v of w (w→v).
        //   Any such v is a future pole candidate reachable in 2 forward hops.
        //   Completeness: if u and v are poles of a timelike K_{2,N}, then
        //   every belly node w_i satisfies u→w_i→v, so v is found through w_i.
        cands.clear();
        for &w in nb_u {
            let w = w as usize;
            for &v in get_nb(w) {
                let v = v as usize;
                if v != u && is_core[v] && !is_placed[v] {
                    cands.push(v as u32);
                }
            }
        }
        cands.sort_unstable();
        cands.dedup();

        // ── Find best Prism partner among local candidates ─────
        let mut best_v           = usize::MAX;
        let mut best_count       = 0usize;
        let mut best_intermediates: Vec<usize> = Vec::new();

        for &v_u32 in &cands {
            let v = v_u32 as usize;
            if is_placed[v] { continue; }

            let pv     = get_nb_rev(v);
            if pv.len() < MIN_PRISM_SHARED { continue; }
            let shared = count_isect(nb_u, pv);

            // Diagnostic tracking
            max_isect_global = max_isect_global.max(shared);
            if shared < 20 { isect_histogram[shared] += 1; }

            if shared >= MIN_PRISM_SHARED && shared > best_count {
                let intermediates = extract_isect(nb_u, pv);
                // Verify every intermediate is still available.
                if intermediates.iter().all(|&w| !is_placed[w]) {
                    best_count       = shared;
                    best_v           = v;
                    best_intermediates = intermediates;
                }
            }
        }

        if best_v == usize::MAX { continue; }

        // ── Commit Prism ────────────────────────────────────────────
        prisms.push(CausalPrism {
            origin:        u,
            destination:   best_v,
            intermediates: best_intermediates.clone(),
        });
        is_placed[u]       = true;
        is_placed[best_v]  = true;
        for &w in &best_intermediates { is_placed[w] = true; }
    }

    {
        let msg = format!("  Causal Prisms found: {} (K_{{2,N}}, N ≥ {})", prisms.len(), MIN_PRISM_SHARED);
        println!("{}", msg);
        use std::fs::OpenOptions; use std::io::Write;
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("simulation.log") {
            writeln!(f, "{}", msg).ok();
        }
    }

    // Diagnostic output
    println!("  [Diagnostic] Max intersection count seen: {}", max_isect_global);
    println!("  [Diagnostic] Intersection histogram (count | frequency):");
    for (isect_size, &freq) in isect_histogram.iter().enumerate() {
        if freq > 0 {
            println!("    {} shared intermediates: {} candidate pairs", isect_size, freq);
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  4. Topological signature (diagnostic) + Vol I phase classification
    //
    //  φ(w) = sign(bulk_momentum[w]) ∈ {-1, 0, +1}  — causal phase
    //  g(P) = |{φ(w_i) : w_i ∈ intermediates}|       — generation (1,2,3)
    //  Φ(P) = Σ φ(w_i)                               — net phase
    //  Matter: Φ > 0, Antimatter: Φ < 0
    // ════════════════════════════════════════════════════════════════

    // Diagnostic: keep quartile signature for logging
    let mut sig_map: std::collections::HashMap<u32, Vec<usize>> =
        std::collections::HashMap::new();
    for prism in &prisms {
        let sig = prism_signature(prism, &bulk_momentum);
        let mut nodes = vec![prism.origin, prism.destination];
        nodes.extend_from_slice(&prism.intermediates);
        sig_map.entry(sig).or_default().extend(nodes);
    }

    // Vol I phase classification
    struct PrismClass {
        generation: usize,   // g(P) = 1, 2, or 3
        net_phase: i32,      // Φ(P) = Σ φ(w_i)
        nodes: Vec<usize>,   // all component nodes
        n_inter: usize,      // mass = number of intermediates
    }

    let classify_prism = |prism: &CausalPrism| -> PrismClass {
        let mut phase_set = std::collections::HashSet::new();
        let mut net_phase: i32 = 0;
        for &w in &prism.intermediates {
            let phi = bulk_momentum[w].signum();
            phase_set.insert(phi);
            net_phase += phi;
        }
        let generation = phase_set.len(); // 1, 2, or 3
        let mut nodes = vec![prism.origin, prism.destination];
        nodes.extend_from_slice(&prism.intermediates);
        PrismClass { generation, net_phase, nodes, n_inter: prism.intermediates.len() }
    };

    let classifications: Vec<PrismClass> = prisms.iter().map(|p| classify_prism(p)).collect();

    // ════════════════════════════════════════════════════════════════
    //  5. Threat detection & contraction
    //
    //  Threat criterion: external node t connected to BOTH poles AND
    //  ≥ PRISM_THREAT intermediates.  Absorb into the higher-degree pole.
    // ════════════════════════════════════════════════════════════════
    let mut merge_into: Vec<usize> = (0..n).collect();
    let mut is_merged  = vec![false; n];
    let mut merge_count = 0usize;

    // Undirected adjacency check: t↔v exists if t→v OR v→t in the directed Hasse.
    // The K₅ threat criterion is topological (undirected), not causal (directed).
    let connected_undirected = |u: usize, v: usize| -> bool {
        connected(u, v as u32) || connected(v, u as u32)
    };

    for prism in &prisms {
        let deg_o = (adj_head_vac[prism.origin + 1] - adj_head_vac[prism.origin]) as usize;
        let deg_d = (adj_head_vac[prism.destination + 1] - adj_head_vac[prism.destination]) as usize;
        let absorber = if deg_o >= deg_d { prism.origin } else { prism.destination };

        // Gather all external neighbours of the prism components (BOTH directions).
        // Forward (children) + reverse (parents) — CPT-symmetric threat scan.
        let mut threats: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &v in get_nb(prism.origin)       { threats.insert(v as usize); }
        for &v in get_nb(prism.destination)  { threats.insert(v as usize); }
        for &w in &prism.intermediates {
            for &v in get_nb(w) { threats.insert(v as usize); }
        }
        for &v in get_nb_rev(prism.origin)       { threats.insert(v as usize); }
        for &v in get_nb_rev(prism.destination)  { threats.insert(v as usize); }
        for &w in &prism.intermediates {
            for &v in get_nb_rev(w) { threats.insert(v as usize); }
        }

        // Remove prism members from threat candidates
        threats.remove(&prism.origin);
        threats.remove(&prism.destination);
        for &w in &prism.intermediates { threats.remove(&w); }

        for t in threats {
            if is_placed[t] || is_merged[t] { continue; }
            let to_origin = connected_undirected(t, prism.origin);
            let to_dest   = connected_undirected(t, prism.destination);
            let to_inter: usize = prism.intermediates.iter()
                .filter(|&&w| connected_undirected(t, w))
                .count();
            if (to_origin && to_dest) && to_inter >= PRISM_THREAT {
                merge_into[t] = absorber;
                is_merged[t]  = true;
                merge_count  += 1;
            }
        }
    }

    {
        let msg = format!("  Threat contractions: {merge_count}");
        println!("{}", msg);
        use std::fs::OpenOptions; use std::io::Write;
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("simulation.log") {
            writeln!(f, "{}", msg).ok();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  6. Resolve transitive merge chains  (integer pointer chase)
    // ════════════════════════════════════════════════════════════════
    for i in 0..n {
        let mut t = merge_into[i];
        while merge_into[t] != t { t = merge_into[t]; }
        merge_into[i] = t;
    }

    // ════════════════════════════════════════════════════════════════
    //  7. Build Defect CSR  (Zero-Copy)
    //
    //  Vacuum edges with merge_into applied, plus one completing edge
    //  per Prism: origin ↔ destination.  This edge is always NEW because
    //  in a triangle-free graph the two poles are never directly connected.
    //  We do NOT add intermediate–intermediate edges (preserves bipartite).
    // ════════════════════════════════════════════════════════════════
    let mut def_edges: Vec<(u32, u32)> = Vec::new();

    // The vacuum CSR is DIRECTED (forward-only: u→v where u ≺ v causally).
    // Each edge appears exactly once.  Do NOT filter by index ordering
    // (u < v) — in the cell-sorted build_hasse_direct path, index order
    // does not match time order within the same quantised time layer.
    for u in 0..n {
        let start = adj_head_vac[u] as usize;
        let end   = adj_head_vac[u + 1] as usize;
        for &v_u32 in &adj_data_vac[start..end] {
            let v = v_u32 as usize;
            let ri = merge_into[u] as u32;
            let ci = merge_into[v] as u32;
            if ri != ci {
                def_edges.push((ri, ci));
                def_edges.push((ci, ri));
            }
        }
    }

    // Pole-completion edges
    for prism in &prisms {
        let oc = merge_into[prism.origin]      as u32;
        let dc = merge_into[prism.destination] as u32;
        if oc != dc {
            let (lo, hi) = if oc < dc { (oc, dc) } else { (dc, oc) };
            def_edges.push((lo, hi));
            def_edges.push((hi, lo));
        }
    }

    def_edges.par_sort_unstable();
    def_edges.dedup();

    let mut adj_head_def = vec![0u32; n + 1];
    let mut adj_data_def = vec![0u32; def_edges.len()];
    let mut cur = 0usize;
    for (i, &(u, v)) in def_edges.iter().enumerate() {
        let ui = u as usize;
        while cur < ui { cur += 1; adj_head_def[cur] = i as u32; }
        adj_data_def[i] = v;
    }
    while cur < n { cur += 1; adj_head_def[cur] = def_edges.len() as u32; }
    drop(def_edges);

    // ════════════════════════════════════════════════════════════════
    //  8. Core index vectors
    // ════════════════════════════════════════════════════════════════
    let vacuum_core = core_nodes.clone();
    let defect_core: Vec<usize> = core_nodes.iter().filter(|&&i| !is_merged[i]).cloned().collect();

    // ════════════════════════════════════════════════════════════════
    //  9. Generation classification (Vol I: phase class counting)
    //
    //  Gen g = prisms with g(P) = g distinct phase classes.
    //  Gen1 matter: g=1, Φ>0.  Anti1: g=1, Φ<0.
    // ════════════════════════════════════════════════════════════════
    let mut gen1_nodes: Vec<usize> = Vec::new();
    let mut gen2_nodes: Vec<usize> = Vec::new();
    let mut gen3_nodes: Vec<usize> = Vec::new();
    let mut anti1_nodes: Vec<usize> = Vec::new();

    for c in &classifications {
        match c.generation {
            1 => {
                if c.net_phase > 0 {
                    gen1_nodes.extend_from_slice(&c.nodes);
                } else if c.net_phase < 0 {
                    anti1_nodes.extend_from_slice(&c.nodes);
                } else {
                    // Φ = 0 within g=1 means all intermediates have bm = 0.
                    // Assign to gen1 (matter-neutral, still generation 1).
                    gen1_nodes.extend_from_slice(&c.nodes);
                }
            }
            2 => gen2_nodes.extend_from_slice(&c.nodes),
            3 => gen3_nodes.extend_from_slice(&c.nodes),
            _ => {} // g > 3 impossible with φ ∈ {-1, 0, +1}
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  10. Mass Spectrum (Topological Inertia)
    //
    //  Mass = N (number of intermediates), averaged per generation.
    // ════════════════════════════════════════════════════════════════
    let avg_mass = |gen: usize, matter: Option<bool>| -> f64 {
        let matching: Vec<&PrismClass> = classifications.iter()
            .filter(|c| c.generation == gen && match matter {
                Some(true) => c.net_phase > 0 || (c.net_phase == 0 && gen == 1),
                Some(false) => c.net_phase < 0,
                None => true,
            })
            .collect();
        if matching.is_empty() { return 0.0; }
        let total: usize = matching.iter().map(|c| c.n_inter).sum();
        total as f64 / matching.len() as f64
    };

    let mass_gen1  = avg_mass(1, Some(true));
    let mass_gen2  = avg_mass(2, None);
    let mass_gen3  = avg_mass(3, None);
    let mass_anti1 = avg_mass(1, Some(false));

    // Diagnostic: quartile signature summary (for backwards compatibility in logs)
    let mut sig_counts: Vec<(u32, usize)> = sig_map.iter()
        .map(|(&s, v)| (s, v.len())).collect();
    sig_counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    // ── Diagnostics ─────────────────────────────────────────────────
    {
        let vac_e = adj_data_vac.len();       // directed: each edge stored once
        let def_e = adj_data_def.len() / 2;   // undirected: each edge stored twice
        let mut msg = format!(
            "  Edges: Vac={vac_e} / Def={def_e} (Zero-Copy CSR)\n  \
             Active core: {} / {}", defect_core.len(), vacuum_core.len()
        );
        if !sig_counts.is_empty() {
            msg.push_str("\n  Top signatures:");
            for (i, &(sig, cnt)) in sig_counts.iter().enumerate().take(5) {
                let s = [
                    (sig & 0x1F) as i32 - 16,
                    ((sig >> 5)  & 0x1F) as i32 - 16,
                    ((sig >> 10) & 0x1F) as i32 - 16,
                    ((sig >> 15) & 0x1F) as i32 - 16,
                ];
                msg.push_str(&format!("\n    Rank {}: {s:?}  (nodes={cnt})", i + 1));
            }
        }
        // Vol I phase classification summary
        let gen_dist: Vec<usize> = (1..=3).map(|g| {
            classifications.iter().filter(|c| c.generation == g).count()
        }).collect();
        msg.push_str(&format!(
            "\n  Phase classification (Vol I): g=1:{} g=2:{} g=3:{} prisms",
            gen_dist[0], gen_dist[1], gen_dist[2]
        ));
        msg.push_str(&format!(
            "\n  Classified: Gen1={}, Gen2={}, Gen3={}, AntiGen1={}",
            gen1_nodes.len(), gen2_nodes.len(), gen3_nodes.len(), anti1_nodes.len()
        ));
        println!("{}", msg);
        use std::fs::OpenOptions; use std::io::Write;
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("simulation.log") {
            writeln!(f, "{}", msg).ok();
        }
    }

    let prism_histogram = build_prism_histogram(&prisms);
    let max_intermediates = prisms.iter()
        .map(|p| p.intermediates.len()).max().unwrap_or(0);

    // ── Intermediate phase census (occupancy model diagnostic) ──
    let mut phase_pos_count: usize = 0;
    let mut phase_zero_count: usize = 0;
    let mut phase_neg_count: usize = 0;
    for prism in &prisms {
        for &w in &prism.intermediates {
            match bulk_momentum[w].signum() {
                1  => phase_pos_count += 1,
                0  => phase_zero_count += 1,
                -1 => phase_neg_count += 1,
                _  => {}
            }
        }
    }
    let phase_total = (phase_pos_count + phase_zero_count + phase_neg_count) as f64;
    if phase_total > 0.0 {
        println!("  Intermediate phase census: φ=+1: {} ({:.4})  φ=0: {} ({:.4})  φ=-1: {} ({:.4})",
            phase_pos_count, phase_pos_count as f64 / phase_total,
            phase_zero_count, phase_zero_count as f64 / phase_total,
            phase_neg_count, phase_neg_count as f64 / phase_total);
    }

    // ── Per-generation prism counts ──
    let prisms_gen1 = classifications.iter().filter(|c| c.generation == 1).count();
    let prisms_gen2 = classifications.iter().filter(|c| c.generation == 2).count();
    let prisms_gen3 = classifications.iter().filter(|c| c.generation == 3).count();
    println!("  Prism generation census: Gen1={prisms_gen1}  Gen2={prisms_gen2}  Gen3={prisms_gen3}  (total={})",
        prisms_gen1 + prisms_gen2 + prisms_gen3);

    // ── Phase-coherence mass decomposition (Theorem: zero free parameters) ──
    //
    //  M_grav(P) = N,  M_vis(P) = |Φ(P)|,  M_dark(P) = N - |Φ(P)|
    //  Ω_dark/Ω_vis = Σ(N - |Φ|) / Σ|Φ|
    //  Q_topo = Σ|Φ|² / ΣN²   (topological charge ratio, intrinsic)
    //  α_EM  = Q_topo / (8π)   (observed coupling, geometric synthesis)
    let mut visible_mass_total: usize = 0;
    let mut dark_mass_total: usize = 0;
    let mut grav_mass_total: usize = 0;
    let mut phase_sq_total: usize = 0;
    let mut mass_sq_total: usize = 0;
    for c in &classifications {
        let phi_abs = c.net_phase.unsigned_abs() as usize;
        visible_mass_total += phi_abs;
        dark_mass_total += c.n_inter - phi_abs;
        grav_mass_total += c.n_inter;
        phase_sq_total += phi_abs * phi_abs;
        mass_sq_total += c.n_inter * c.n_inter;
    }
    let omega_ratio = if visible_mass_total > 0 {
        dark_mass_total as f64 / visible_mass_total as f64
    } else { f64::INFINITY };
    let q_topo = if mass_sq_total > 0 {
        phase_sq_total as f64 / mass_sq_total as f64
    } else { 0.0 };
    let alpha_em = q_topo / (8.0 * std::f64::consts::PI);
    let omega_energy = if q_topo > 0.0 { 1.0 / q_topo - 1.0 } else { f64::INFINITY };

    {
        let msg = format!(
            "  Phase-coherence: Σ|Φ|={visible_mass_total}  Σ(N-|Φ|)={dark_mass_total}  ΣN={grav_mass_total}\n  \
             Q_topo = Σ|Φ|²/ΣN² = {q_topo:.6}  |  α = Q/(8π) = {alpha_em:.6}  (1/α = {:.1})\n  \
             Ω_linear = {omega_ratio:.4}  |  Ω_energy = 1/Q-1 = {omega_energy:.4}  [α(1+Ω)={:.6}  vs 1/(8π)={:.6}]",
            1.0 / alpha_em,
            alpha_em * (1.0 + omega_energy),
            1.0 / (8.0 * std::f64::consts::PI)
        );
        println!("{}", msg);
        use std::fs::OpenOptions; use std::io::Write;
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("simulation.log") {
            writeln!(f, "{}", msg).ok();
        }
    }

    // ── Sterile prism nodes (Φ = 0: fully phase-cancelled → truly dark) ──
    let sterile_nodes: Vec<usize> = classifications.iter()
        .filter(|c| c.net_phase == 0)
        .flat_map(|c| c.nodes.iter().cloned())
        .collect();

    let sterile_prisms_count = classifications.iter()
        .filter(|c| c.net_phase == 0).count();
    let avg_mass_sterile = if sterile_prisms_count == 0 { 0.0 } else {
        let total: usize = classifications.iter()
            .filter(|c| c.net_phase == 0)
            .map(|c| c.n_inter).sum();
        total as f64 / sterile_prisms_count as f64
    };

    let topology = TopologySummary {
        total_nodes: n,
        total_prisms: prisms.len(),
        max_intermediates,
        count_gen1: gen1_nodes.len(),
        count_gen2: gen2_nodes.len(),
        count_gen3: gen3_nodes.len(),
        count_antigen1: anti1_nodes.len(),
        count_sterile: sterile_nodes.len(),
        avg_mass_gen1: mass_gen1,
        avg_mass_gen2: mass_gen2,
        avg_mass_gen3: mass_gen3,
        avg_mass_sterile,
        prism_histogram,
        visible_mass_total,
        dark_mass_total,
        grav_mass_total,
        omega_ratio,
        phase_sq_total,
        mass_sq_total,
        alpha_em,
        omega_energy,
        phase_pos_count,
        phase_zero_count,
        phase_neg_count,
        prisms_gen1,
        prisms_gen2,
        prisms_gen3,
    };

    (DefectResult {
        vac_head: adj_head_vac,
        vac_data: adj_data_vac,
        def_head: adj_head_def,
        def_data: adj_data_def,
        vacuum_core,
        defect_core,
        gen1_nodes,
        gen2_nodes,
        gen3_nodes,
        anti1_nodes,
        sterile_nodes,
        merge_into,
        mass_gen1,
        mass_gen2,
        mass_gen3,
        mass_anti1,
    }, topology, prisms)
}

// ─── Streaming Implementation ─────────────────────────────────────────────────
use crate::spectral::SpectralResult;

/// Diskless Causal Prism Analysis (Sparse Scanning Mode).
///
/// Single-pass strategy using HashMap-grid sparse scanner (zero disk I/O):
///   scan_edges_with_analysis — counts degrees, classifies core, and collects
///   core-induced edges all in one scan over the Poisson sprinkling.
///
/// Memory: O(N) for degree arrays + O(core CSR) + O(chunk) for sliding window.
/// At N=100M: ~1.3 GB per realization vs 20+ GB for dense grid.
///
/// Returns `(vacuum, defect)` SpectralResult tuple, compatible with
/// the in-memory path's `average_ensemble()`.
pub fn process_streaming(
    n_total: usize,
    chunk_size: usize,
    seed: u64,
    steps: &[u32],
    walkers: usize,
) -> std::io::Result<(SpectralResult, SpectralResult, TopologySummary)> {
    use crate::diamond;

    println!("Phase 2: Causal Prism search (sparse scan, zero disk I/O)...");

    // ── Single-pass: degrees + core classification + core edges ─────────
    let (in_deg, out_deg, is_core, directed_core) =
        diamond::scan_edges_with_analysis(n_total, chunk_size, seed, CORE_NUM, CORE_DEN);

    let bulk_momentum: Vec<i32> = in_deg.iter().zip(&out_deg)
        .map(|(&i, &o)| o as i32 - i as i32).collect();
    drop(in_deg);
    drop(out_deg);

    let target_core = n_total * CORE_NUM / CORE_DEN;
    let mut core_map = vec![u32::MAX; n_total];
    let mut core_global: Vec<usize> = Vec::with_capacity(target_core);
    let mut lc = 0u32;
    for i in 0..n_total {
        if is_core[i] { core_map[i] = lc; core_global.push(i); lc += 1; }
    }
    let num_core = lc as usize;
    drop(is_core);
    println!("  Core size: {num_core}");
    println!("  Core directed edges: {}", directed_core.len());

    // Remap to local core indices + symmetrize for undirected CSR
    let mut core_edges: Vec<(u32, u32)> = Vec::with_capacity(directed_core.len() * 2);
    let mut directed_local: Vec<(u32, u32)> = Vec::with_capacity(directed_core.len());
    for &(u, v) in &directed_core {
        let ul = core_map[u as usize]; let vl = core_map[v as usize];
        if ul != u32::MAX && vl != u32::MAX {
            core_edges.push((ul, vl));
            core_edges.push((vl, ul));
            directed_local.push((ul, vl));
        }
    }
    drop(directed_core);
    drop(core_map);
    println!("  Core edges (undirected): {}", core_edges.len());

    core_edges.par_sort_unstable();
    core_edges.dedup();

    let mut adj_head = vec![0u32; num_core + 1];
    let mut adj_data = vec![0u32; core_edges.len()];
    {
        let mut cur = 0usize;
        for (i, &(u, v)) in core_edges.iter().enumerate() {
            let ui = u as usize;
            while cur < ui { cur += 1; adj_head[cur] = i as u32; }
            adj_data[i] = v;
        }
        while cur < num_core { cur += 1; adj_head[cur] = core_edges.len() as u32; }
    }
    drop(core_edges);

    // ── Build directed-only CSR: children(u) ──────────────────────
    //
    // Forward directed CSR from directed_local (u→v edges).
    // Used for forward-forward candidate search: u→w→v.
    directed_local.sort_unstable();
    directed_local.dedup();
    let mut dir_head = vec![0u32; num_core + 1];
    for &(u, _) in &directed_local { dir_head[u as usize + 1] += 1; }
    for i in 0..num_core { dir_head[i + 1] += dir_head[i]; }
    let mut dir_data = vec![0u32; directed_local.len()];
    {
        let mut dir_pos = vec![0u32; num_core];
        for &(u, v) in &directed_local {
            let ui = u as usize;
            let pos = (dir_head[ui] + dir_pos[ui]) as usize;
            dir_data[pos] = v;
            dir_pos[ui] += 1;
        }
    }
    for u in 0..num_core {
        let s = dir_head[u] as usize;
        let e = dir_head[u + 1] as usize;
        dir_data[s..e].sort_unstable();
    }

    // ── Build reverse-directed CSR: parents(v) ──────────────────────
    //
    // For the belly intersection we need parents(v) = {w : w→v}.
    // Built from directed_local by inverting source/target.
    let mut rev_dir_head = vec![0u32; num_core + 1];
    for &(_, v) in &directed_local { rev_dir_head[v as usize + 1] += 1; }
    for i in 0..num_core { rev_dir_head[i + 1] += rev_dir_head[i]; }
    let mut rev_dir_data = vec![0u32; directed_local.len()];
    {
        let mut rev_pos = vec![0u32; num_core];
        for &(u, v) in &directed_local {
            let vi = v as usize;
            let pos = (rev_dir_head[vi] + rev_pos[vi]) as usize;
            rev_dir_data[pos] = u;
            rev_pos[vi] += 1;
        }
    }
    for u in 0..num_core {
        let s = rev_dir_head[u] as usize;
        let e = rev_dir_head[u + 1] as usize;
        rev_dir_data[s..e].sort_unstable();
    }

    // ── 2-hop Prism detection on core subgraph ─────────────────────
    //
    // get_nb_dir:     directed successors = children(u)
    // get_nb_dir_rev: directed predecessors = parents(v)
    let get_nb_dir = |u: usize| -> &[u32] {
        let s = dir_head[u] as usize; let e = dir_head[u+1] as usize;
        if s <= e && e <= dir_data.len() { &dir_data[s..e] } else { &[] }
    };
    let get_nb_dir_rev = |u: usize| -> &[u32] {
        let s = rev_dir_head[u] as usize; let e = rev_dir_head[u+1] as usize;
        if s <= e && e <= rev_dir_data.len() { &rev_dir_data[s..e] } else { &[] }
    };

    let count_isect = |a: &[u32], b: &[u32]| -> usize {
        let (mut i, mut j, mut c) = (0, 0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less    => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal   => { c += 1; i += 1; j += 1; }
            }
        }
        c
    };

    let extract_isect = |a: &[u32], b: &[u32]| -> Vec<usize> {
        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less    => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal   => { out.push(a[i] as usize); i += 1; j += 1; }
            }
        }
        out
    };

    let mut prisms: Vec<CausalPrism> = Vec::new();
    let mut is_placed = vec![false; num_core];
    let mut cands: Vec<u32> = Vec::with_capacity(512);

    // Diagnostic: track intersection count distribution
    let mut max_isect_global = 0usize;
    let mut isect_histogram = vec![0usize; 20];

    println!("  Scanning for Causal Prisms (forward-forward, belly intersection, O(N))...");
    for u in 0..num_core {
        if is_placed[u] { continue; }
        let nb_u_dir = get_nb_dir(u);
        if nb_u_dir.len() < MIN_PRISM_SHARED { continue; }

        // ── 2-hop candidate collection via u→w→v (forward-forward) ──
        //
        // For each child w of u, find all children v of w.
        // Any such v is a future pole candidate reachable in 2 forward hops.
        cands.clear();
        for &w in nb_u_dir {
            let w = w as usize;
            if w >= num_core { continue; }
            for &v in get_nb_dir(w) {
                let v = v as usize;
                if v != u && !is_placed[v] { cands.push(v as u32); }
            }
        }
        cands.sort_unstable();
        cands.dedup();

        // Belly intersection: children(u) ∩ parents(v)
        let mut best_v = usize::MAX;
        let mut best_n = 0usize;
        let mut best_inter: Vec<usize> = Vec::new();

        for &v_u32 in &cands {
            let v = v_u32 as usize;
            if is_placed[v] { continue; }
            let pv_dir = get_nb_dir_rev(v);
            if pv_dir.len() < MIN_PRISM_SHARED { continue; }
            let shared = count_isect(nb_u_dir, pv_dir);
            max_isect_global = max_isect_global.max(shared);
            if shared < 20 { isect_histogram[shared] += 1; }
            if shared >= MIN_PRISM_SHARED && shared > best_n {
                let inter = extract_isect(nb_u_dir, pv_dir);
                if inter.iter().all(|&w| !is_placed[w]) {
                    best_n = shared; best_v = v; best_inter = inter;
                }
            }
        }
        if best_v == usize::MAX { continue; }

        // Commit prism
        prisms.push(CausalPrism { origin: u, destination: best_v, intermediates: best_inter.clone() });
        is_placed[u] = true;
        is_placed[best_v] = true;
        for &w in &best_inter { is_placed[w] = true; }
    }

    println!("  [Streaming] Causal Prisms: {}", prisms.len());

    // Diagnostic output
    println!("  [Diagnostic] Max intersection count seen: {}", max_isect_global);
    println!("  [Diagnostic] Intersection histogram (count | frequency):");
    for (isect_size, &freq) in isect_histogram.iter().enumerate() {
        if freq > 0 {
            println!("    {} shared intermediates: {} candidate pairs", isect_size, freq);
        }
    }

    // ── Vol I phase classification (streaming) ──────────────────────
    //
    //  φ(w) = sign(bulk_momentum[w]) ∈ {-1, 0, +1}
    //  g(P) = |{φ(w_i)}|  — generation (1, 2, or 3)
    //  Φ(P) = Σ φ(w_i)    — matter (Φ>0) vs antimatter (Φ<0)
    //
    //  Note: prism indices are LOCAL (core subgraph). Remap to global
    //  for bulk_momentum lookup.

    let mut starts_gen1:  Vec<usize> = Vec::new();
    let mut starts_gen2:  Vec<usize> = Vec::new();
    let mut starts_gen3:  Vec<usize> = Vec::new();
    let mut starts_anti1: Vec<usize> = Vec::new();

    // For mass computation: (generation, is_matter, n_intermediates)
    struct StreamPrismClass { generation: usize, net_phase: i32, n_inter: usize }
    let mut stream_classes: Vec<StreamPrismClass> = Vec::with_capacity(prisms.len());

    for prism in &prisms {
        let mut phase_set = std::collections::HashSet::new();
        let mut net_phase: i32 = 0;
        for &w_local in &prism.intermediates {
            let w_global = core_global[w_local];
            let phi = bulk_momentum[w_global].signum();
            phase_set.insert(phi);
            net_phase += phi;
        }
        let generation = phase_set.len();
        let mut nodes = vec![prism.origin, prism.destination];
        nodes.extend_from_slice(&prism.intermediates);

        stream_classes.push(StreamPrismClass {
            generation, net_phase, n_inter: prism.intermediates.len(),
        });

        match generation {
            1 => {
                if net_phase > 0 || (net_phase == 0) {
                    starts_gen1.extend_from_slice(&nodes);
                }
                if net_phase < 0 {
                    starts_anti1.extend_from_slice(&nodes);
                }
            }
            2 => starts_gen2.extend_from_slice(&nodes),
            3 => starts_gen3.extend_from_slice(&nodes),
            _ => {}
        }
    }

    // Mass spectrum (Vol I)
    let avg_mass_stream = |gen: usize, matter: Option<bool>| -> f64 {
        let matching: Vec<&StreamPrismClass> = stream_classes.iter()
            .filter(|c| c.generation == gen && match matter {
                Some(true) => c.net_phase > 0 || (c.net_phase == 0 && gen == 1),
                Some(false) => c.net_phase < 0,
                None => true,
            })
            .collect();
        if matching.is_empty() { return 0.0; }
        let total: usize = matching.iter().map(|c| c.n_inter).sum();
        total as f64 / matching.len() as f64
    };

    let mass_gen1  = avg_mass_stream(1, Some(true));
    let mass_gen2  = avg_mass_stream(2, None);
    let mass_gen3  = avg_mass_stream(3, None);
    let mass_anti1 = avg_mass_stream(1, Some(false));

    // Diagnostic: quartile signature summary (backwards-compatible logging)
    let mut sig_map_diag: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    for prism in &prisms {
        let local_prism = CausalPrism {
            origin:        core_global[prism.origin],
            destination:   core_global[prism.destination],
            intermediates: prism.intermediates.iter().map(|&w| core_global[w]).collect(),
        };
        let sig = prism_signature(&local_prism, &bulk_momentum);
        *sig_map_diag.entry(sig).or_insert(0) += 1;
    }
    let mut sig_counts: Vec<(u32, usize)> = sig_map_diag.into_iter().collect();
    sig_counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    println!("  Top topological signatures:");
    for (i, &(sig, cnt)) in sig_counts.iter().enumerate().take(5) {
        let s = [
            (sig & 0x1F) as i32 - 16, ((sig >> 5)  & 0x1F) as i32 - 16,
            ((sig >> 10) & 0x1F) as i32 - 16, ((sig >> 15) & 0x1F) as i32 - 16,
        ];
        println!("    Rank {}: {s:?}  (prisms={cnt})", i + 1);
    }

    let gen_dist: Vec<usize> = (1..=3).map(|g| {
        stream_classes.iter().filter(|c| c.generation == g).count()
    }).collect();
    println!("  Phase classification (Vol I): g=1:{} g=2:{} g=3:{} prisms",
        gen_dist[0], gen_dist[1], gen_dist[2]);
    println!(
        "  Classified: Gen1={}, Gen2={}, Gen3={}, AntiGen1={}",
        starts_gen1.len(), starts_gen2.len(), starts_gen3.len(), starts_anti1.len()
    );

    // ── Phase 3: spectral dimension on core graph ──────────────────
    use crate::spectral;

    let vac_origins: Vec<usize> = (0..num_core).collect();
    let vac_starts = spectral::distribute_walkers(&vac_origins, walkers);
    println!("  Vacuum walkers: {walkers}");
    let p_vac  = spectral::run_walkers(&adj_head, &adj_data, &vac_starts, &steps, seed, None);
    let ds_vac = spectral::spectral_dimension(&steps, &p_vac);

    // Defect: run walkers from prism nodes only
    let prism_starts: Vec<usize> = {
        let ps: Vec<usize> = prisms.iter()
            .flat_map(|pr| std::iter::once(pr.origin)
                .chain(std::iter::once(pr.destination))
                .chain(pr.intermediates.iter().cloned()))
            .collect();
        if ps.is_empty() {
            spectral::distribute_walkers(&vac_origins, walkers)
        } else {
            spectral::distribute_walkers(&ps, walkers)
        }
    };
    println!("  Defect walkers: {}", prism_starts.len());
    let p_def  = spectral::run_walkers(&adj_head, &adj_data, &prism_starts, &steps, seed.wrapping_add(1), None);
    let ds_def = spectral::spectral_dimension(&steps, &p_def);

    // Generation-specific walkers
    let run_gen = |nodes: &[usize], label: &str, s_add: u64| -> (Vec<f64>, Vec<f64>) {
        if nodes.is_empty() {
            return (vec![0.0; steps.len()], vec![0.0; steps.len()]);
        }
        let starts = spectral::distribute_walkers(nodes, walkers);
        println!("  [Gen {label}] walkers: {walkers}");
        let p  = spectral::run_walkers(&adj_head, &adj_data, &starts, &steps, seed.wrapping_add(s_add), None);
        let ds = spectral::spectral_dimension(&steps, &p);
        (p, ds)
    };

    let (p_g1, ds_g1) = run_gen(&starts_gen1,  "Gen1",  2);
    let (p_g2, ds_g2) = run_gen(&starts_gen2,  "Gen2",  3);
    let (p_g3, ds_g3) = run_gen(&starts_gen3,  "Gen3",  4);
    let (p_a1, ds_a1) = run_gen(&starts_anti1, "Anti1", 5);

    // ── Directed CSR for causal flux (reuses Pass 2 edges) ──────────
    // No third scan needed: directed_local already has directed core edges.
    let (p_attr, p_repu) = if !starts_gen1.is_empty() && !starts_anti1.is_empty() {
        let (adj_head_dir, adj_data_dir) = {
            let mut head = vec![0u32; num_core + 1];
            for &(r, _) in &directed_local { head[r as usize + 1] += 1; }
            for i in 0..num_core { head[i + 1] += head[i]; }
            let mut data = vec![0u32; directed_local.len()];
            let mut pos = head.clone();
            for &(r, c) in &directed_local {
                data[pos[r as usize] as usize] = c;
                pos[r as usize] += 1;
            }
            (head, data)
        };

        let (fa, fr) = spectral::run_transmission_walkers(
            &adj_head_dir, &adj_data_dir,
            &starts_gen1, &starts_anti1, &starts_gen1,
            &steps, seed.wrapping_add(100), None,
        );
        (fa, fr)
    } else {
        (vec![0.0; steps.len()], vec![0.0; steps.len()])
    };

    // ── Prism histogram + max intermediates ─────────────────────────
    let prism_histogram = build_prism_histogram(&prisms);
    let max_intermediates = prisms.iter()
        .map(|p| p.intermediates.len()).max().unwrap_or(0);

    // ── Intermediate phase census (occupancy model diagnostic) ──
    let mut phase_pos_count: usize = 0;
    let mut phase_zero_count: usize = 0;
    let mut phase_neg_count: usize = 0;
    for prism in &prisms {
        for &w_local in &prism.intermediates {
            let w_global = core_global[w_local];
            match bulk_momentum[w_global].signum() {
                1  => phase_pos_count += 1,
                0  => phase_zero_count += 1,
                -1 => phase_neg_count += 1,
                _  => {}
            }
        }
    }
    let phase_total = (phase_pos_count + phase_zero_count + phase_neg_count) as f64;
    if phase_total > 0.0 {
        println!("  Intermediate phase census: φ=+1: {} ({:.4})  φ=0: {} ({:.4})  φ=-1: {} ({:.4})",
            phase_pos_count, phase_pos_count as f64 / phase_total,
            phase_zero_count, phase_zero_count as f64 / phase_total,
            phase_neg_count, phase_neg_count as f64 / phase_total);
    }

    // ── Per-generation prism counts ──
    let prisms_gen1 = stream_classes.iter().filter(|c| c.generation == 1).count();
    let prisms_gen2 = stream_classes.iter().filter(|c| c.generation == 2).count();
    let prisms_gen3 = stream_classes.iter().filter(|c| c.generation == 3).count();
    println!("  Prism generation census: Gen1={prisms_gen1}  Gen2={prisms_gen2}  Gen3={prisms_gen3}  (total={})",
        prisms_gen1 + prisms_gen2 + prisms_gen3);

    // ── Phase-coherence mass decomposition (Theorem: zero free parameters) ──
    let mut visible_mass_total: usize = 0;
    let mut dark_mass_total: usize = 0;
    let mut grav_mass_total: usize = 0;
    let mut phase_sq_total: usize = 0;
    let mut mass_sq_total: usize = 0;
    for c in &stream_classes {
        let phi_abs = c.net_phase.unsigned_abs() as usize;
        visible_mass_total += phi_abs;
        dark_mass_total += c.n_inter - phi_abs;
        grav_mass_total += c.n_inter;
        phase_sq_total += phi_abs * phi_abs;
        mass_sq_total += c.n_inter * c.n_inter;
    }
    let omega_ratio = if visible_mass_total > 0 {
        dark_mass_total as f64 / visible_mass_total as f64
    } else { f64::INFINITY };
    let q_topo = if mass_sq_total > 0 {
        phase_sq_total as f64 / mass_sq_total as f64
    } else { 0.0 };
    let alpha_em = q_topo / (8.0 * std::f64::consts::PI);
    let omega_energy = if q_topo > 0.0 { 1.0 / q_topo - 1.0 } else { f64::INFINITY };

    println!("  Phase-coherence: Σ|Φ|={visible_mass_total}  Σ(N-|Φ|)={dark_mass_total}  ΣN={grav_mass_total}");
    println!("  Q_topo = Σ|Φ|²/ΣN² = {q_topo:.6}  |  α = Q/(8π) = {alpha_em:.6}  (1/α = {:.1})", 1.0 / alpha_em);
    println!("  Ω_linear = {omega_ratio:.4}  |  Ω_energy = 1/Q-1 = {omega_energy:.4}  [α(1+Ω)={:.6}  vs 1/(8π)={:.6}]",
        alpha_em * (1.0 + omega_energy), 1.0 / (8.0 * std::f64::consts::PI));

    // ── Sterile prism nodes (Φ = 0: fully phase-cancelled → truly dark) ──
    let sterile_nodes: Vec<usize> = stream_classes.iter()
        .enumerate()
        .filter(|(_, c)| c.net_phase == 0)
        .flat_map(|(idx, _)| {
            let p = &prisms[idx];
            std::iter::once(p.origin)
                .chain(std::iter::once(p.destination))
                .chain(p.intermediates.iter().cloned())
        })
        .collect();

    let sterile_prisms_count = stream_classes.iter()
        .filter(|c| c.net_phase == 0).count();
    let avg_mass_sterile = if sterile_prisms_count == 0 { 0.0 } else {
        let total: usize = stream_classes.iter()
            .filter(|c| c.net_phase == 0)
            .map(|c| c.n_inter).sum();
        total as f64 / sterile_prisms_count as f64
    };

    // ── Sterile walkers ──────────────────────────────────────────────
    let (p_st, ds_st) = if !sterile_nodes.is_empty() {
        let starts_st = spectral::distribute_walkers(&sterile_nodes, walkers);
        println!("  [Sterile] walkers: {walkers} (sterile nodes: {})", sterile_nodes.len());
        let p = spectral::run_walkers(&adj_head, &adj_data, &starts_st, &steps, seed.wrapping_add(6), None);
        let ds = spectral::spectral_dimension(&steps, &p);
        (p, ds)
    } else { (vec![], vec![]) };

    // ── Normalized flux (per-node coupling strength) ─────────────────
    let n_attr = starts_anti1.len().max(1) as f64;
    let n_repu = starts_gen1.len().max(1) as f64;
    let flux_attr_norm: Vec<f64> = p_attr.iter().map(|&f| f / n_attr).collect();
    let flux_repu_norm: Vec<f64> = p_repu.iter().map(|&f| f / n_repu).collect();

    // ── Topology summary ──────────────────────────────────────────
    let topology = TopologySummary {
        total_nodes: n_total,
        total_prisms: prisms.len(),
        max_intermediates,
        count_gen1: starts_gen1.len(),
        count_gen2: starts_gen2.len(),
        count_gen3: starts_gen3.len(),
        count_antigen1: starts_anti1.len(),
        count_sterile: sterile_nodes.len(),
        avg_mass_gen1: mass_gen1,
        avg_mass_gen2: mass_gen2,
        avg_mass_gen3: mass_gen3,
        avg_mass_sterile,
        prism_histogram,
        visible_mass_total,
        dark_mass_total,
        grav_mass_total,
        omega_ratio,
        phase_sq_total,
        mass_sq_total,
        alpha_em,
        omega_energy,
        phase_pos_count,
        phase_zero_count,
        phase_neg_count,
        prisms_gen1,
        prisms_gen2,
        prisms_gen3,
    };

    // Split into (vacuum, defect) tuple matching in-memory format
    let vac = SpectralResult {
        p_global: p_vac.clone(), ds_global: ds_vac.clone(),
        p_local: p_vac, ds_local: ds_vac,
        p_gen1: vec![], ds_gen1: vec![],
        p_gen2: vec![], ds_gen2: vec![],
        p_gen3: vec![], ds_gen3: vec![],
        p_anti1: vec![], ds_anti1: vec![],
        flux_attraction: vec![], flux_repulsion: vec![],
        flux_attr_norm: vec![], flux_repu_norm: vec![],
        p_sterile: vec![], ds_sterile: vec![],
        mass_gen1: 0.0, mass_gen2: 0.0, mass_gen3: 0.0, mass_anti1: 0.0,
        ds_global_std: vec![], ds_local_std: vec![],
        ds_gen1_std: vec![], ds_gen2_std: vec![], ds_gen3_std: vec![],
        ds_anti1_std: vec![], ds_sterile_std: vec![],
        flux_attraction_std: vec![], flux_repulsion_std: vec![],
    };
    let def = SpectralResult {
        p_global: p_def.clone(), ds_global: ds_def.clone(),
        p_local: p_def, ds_local: ds_def,
        p_gen1: p_g1, ds_gen1: ds_g1,
        p_gen2: p_g2, ds_gen2: ds_g2,
        p_gen3: p_g3, ds_gen3: ds_g3,
        p_anti1: p_a1, ds_anti1: ds_a1,
        flux_attraction: p_attr,
        flux_repulsion:  p_repu,
        flux_attr_norm, flux_repu_norm,
        p_sterile: p_st, ds_sterile: ds_st,
        mass_gen1, mass_gen2, mass_gen3, mass_anti1,
        ds_global_std: vec![], ds_local_std: vec![],
        ds_gen1_std: vec![], ds_gen2_std: vec![], ds_gen3_std: vec![],
        ds_anti1_std: vec![], ds_sterile_std: vec![],
        flux_attraction_std: vec![], flux_repulsion_std: vec![],
    };
    Ok((vac, def, topology))
}
