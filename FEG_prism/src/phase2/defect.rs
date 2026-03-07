// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Phase 2 — Causal Prism Topological Defect (Pure Integer Bipartite Calculus)
//!
//! Implements the Calculo de Kuratowski: matter emerges as K_{2,n} bipartite
//! obstructions (Causal Prisms) in a triangle-free Hasse DAG, with the strong
//! force realised as K_5 threat contraction preserving planarity.  This is the
//! core of *Fractal Entropic Geometrodynamics* (FEG) by J. P. Silva Alvarado.
//!
//! Zenodo: <https://doi.org/10.5281/zenodo.18769707>
//!
//! Key theorems implemented below:
//!   - Vol II, Thm 4.2: Uniqueness of Bifurcation-Convergence (MIN_PRISM_SHARED >= 3)
//!   - Vol II, Sole Principle of Interaction: K_5 minor <-> confinement
//!   - Vol II, Def 3.1: Topological mass M = |belly|
//!   - Vol II, section 5: Generation classification by bulk-momentum signature
//!
//! THE THREE GEOMETRIES — implemented in code:
//!
//!   I.  GEOMETRY OF ORDER (Phase 1 -> this module's input):
//!       Transitive Reduction produces a triangle-free DAG.  The "empty"
//!       structure is not empty — it is a 4D vacuum whose Lorentz invariance
//!       is emergent from redundancy removal on a Poisson process.
//!
//!  II.  GEOMETRY OF MATTER (this module — Prism detection):
//!       The Causal Prism K_{2,N} is the unique topological particle permitted
//!       by the triangle-free constraint.  Mass is not a property added to a
//!       field — it is the integer count N of intermediate causal paths.
//!       The mass hierarchy (electron/muon/tau) reduces to N in {3, 4, 5}.
//!
//! III.  GEOMETRY OF LOGIC (this module — O(N) algorithm):
//!       The 2-hop detection algorithm runs in O(N) because the PHYSICS is
//!       local.  Hasse links have bounded proper time (tau <= 8); therefore
//!       K_{2,N} poles must be within bounded causal distance.  Any valid
//!       prism partner v of node u is reachable in exactly 2 hops through a
//!       shared intermediate w.  This is not merely an optimisation — it is
//!       a computational proof that PHYSICAL LOCALITY IS AN INFORMATION
//!       PRINCIPLE.
//!
//! --- Technical Details ---
//!
//! The transitive reduction in Phase 1 produces a triangle-free DAG by
//! mathematical necessity. Any pair of nodes (u, w, v) with u->w, w->v, u->v
//! would make u->v redundant and be removed. Therefore K_4 cliques (which
//! require triangles) CANNOT exist — zero K4 is not a bug, it is a theorem.
//!
//! The correct fundamental topological particle is the **Causal Prism**
//! (Prisma Causal): a bipartite K_{2,N} subgraph where:
//!   - Two pole nodes (Origin, Destination) at graph-distance >= 2
//!   - N >= MIN_PRISM_SHARED intermediate nodes shared by both poles
//!   - Intermediates are mutually disconnected (guaranteed by triangle-free DAG)
//!
//! Detection — 2-hop forward-forward traversal (the key topological insight):
//!   For u in core_nodes:
//!     For w in children(u):             [1-hop: belly candidate]
//!       For v in children(w), v in core: [2-hop: future pole candidate]
//!         belly = children(u) intersect parents(v)
//!         If |belly| >= MIN_PRISM_SHARED: Prism(u, v, belly)

use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::graph::csr::{CsrGraph, Directed, Undirected};
use crate::phase2::topology::TopologySummary;

/// Fraction of nodes selected as topological core (numerator / denominator).
const CORE_NUM: usize = 1;
const CORE_DEN: usize = 10;

/// Minimum shared intermediates to qualify as a Causal Prism (K_{2,N}, N >= 3).
///
/// Calculo de Kuratowski derivation (Silva Alvarado, FEG Vol II, Thm 4.2):
/// Uniqueness of Bifurcation-Convergence requires >= 3 independent length-2 paths
/// in a K_3-free Hasse graph.  K_{2,2} yields only S_2 -> U(1); the SU(3) gauge
/// structure demands S_3, hence N >= 3 (Vol II, Sole Principle of Interaction).
///
/// DOI: 10.5281/zenodo.18769707
const MIN_PRISM_SHARED: usize = 3;

/// K_5 threat threshold: external node connected to both poles AND >= 2
/// intermediates -> 5 mutually reachable vertices -> K_5 minor.
///
/// Calculo de Kuratowski derivation (Silva Alvarado, FEG Vol II, Axiom 1.4):
/// Kuratowski's Theorem guarantees that a finite graph is planar iff it contains
/// no K_5 or K_{3,3} minor.  Absorption into the nearest pole is the minimal
/// planarity-preserving contraction — the discrete analogue of colour confinement.
///
/// DOI: 10.5281/zenodo.18769707
const PRISM_THREAT: usize = 2;

// --- Data Structures --------------------------------------------------------

/// A Causal Prism defect: irreducible bipartite K_{2,N} structure.
///
/// Origin and Destination are the two poles (at graph-distance >= 2 in the
/// undirected Hasse).  Intermediates are their N >= 3 shared neighbours —
/// mutually disconnected by the triangle-free property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPrism {
    pub origin: usize,
    pub destination: usize,
    pub intermediates: Vec<usize>,
}

/// A detected K_5 minor from threat contraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K5Minor {
    /// The 5 vertices forming the K_5 minor.
    pub vertices: [usize; 5],
    /// Index of the source prism that spawned this K_5 threat.
    pub source_prism_idx: usize,
    /// Classification by prime-5 tree level of the threat node.
    pub z5_level: u16,
}

/// Result of the Causal Prism contraction with particle classification.
pub struct DefectOutput {
    /// Original directed Hasse (moved in — ownership transfer).
    pub vacuum_csr: CsrGraph<Directed>,
    /// Post-contraction defect graph (symmetric, undirected).
    pub defect_csr: CsrGraph<Undirected>,
    /// Merge map: node -> canonical representative.  Identity for non-merged nodes.
    pub merge_map: Vec<usize>,
    /// Core node indices (top 10% by degree) in the vacuum graph.
    pub vacuum_core: Vec<usize>,
    /// Core nodes surviving threat contraction.
    pub defect_core: Vec<usize>,
    /// Generation classification sets and mass spectrum.
    pub generations: GenerationSets,
    /// Detected K_5 minors from threat contraction.
    pub k5_minors: Vec<K5Minor>,
}

/// Generation classification by bulk-momentum phase signature.
///
/// phi(w) = sign(bulk_momentum[w]) in {-1, 0, +1}  — causal phase
/// g(P)   = |{phi(w_i) : w_i in intermediates}|    — generation (1, 2, 3)
/// Phi(P) = Sum phi(w_i)                            — net phase
/// Matter: Phi > 0, Antimatter: Phi < 0
pub struct GenerationSets {
    /// Generation 1 prism node indices — most abundant signature (electron-like).
    pub gen1: Vec<usize>,
    /// Generation 2 prism node indices — second most abundant (muon-like).
    pub gen2: Vec<usize>,
    /// Generation 3 prism node indices — third most abundant (tau-like).
    pub gen3: Vec<usize>,
    /// Anti-Generation 1 prism node indices — CPT conjugate (positron-like).
    pub anti1: Vec<usize>,
    /// Sterile prism node indices — prisms with Phi = 0 (fully phase-cancelled, dark matter).
    pub sterile: Vec<usize>,
    /// Average topological mass per generation: [gen1, gen2, gen3, anti1].
    pub mass: [f64; 4],
}

// --- Signature helpers ------------------------------------------------------

/// Encode the topological signature of a Prism as a packed u32.
///
/// Collects bulk_momentum from all components, sorts, takes four quartile
/// values [q0, q1, q2, q3], normalises to [-10, +10], offsets by +16 (fits
/// in 5 bits), and packs into bits [0..4], [5..9], [10..14], [15..19].
///
/// The anti-signature (CPT conjugate) reverses and negates: a_i = -c_{3-i} + 16.
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
#[allow(dead_code)]
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

// --- Private helpers --------------------------------------------------------

/// Count intersection of two SORTED slices.  O(|a| + |b|).
#[inline]
fn count_slice_intersection(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut c) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less    => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal   => { c += 1; i += 1; j += 1; }
        }
    }
    c
}

/// Collect intersection of two SORTED slices into a Vec<usize>.
#[inline]
fn extract_slice_intersection(a: &[u32], b: &[u32]) -> Vec<usize> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less    => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal   => {
                out.push(a[i] as usize);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Binary-search connectivity check on a directed CSR.
#[inline]
fn are_connected(csr: &CsrGraph<Directed>, u: usize, v: u32) -> bool {
    csr.has_edge(u, v)
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

// --- Main function ----------------------------------------------------------

/// Apply the Causal Prism contraction with particle classification.
///
/// Searches for K_{2,N} bipartite structures via 2-hop traversal on the
/// triangle-free Hasse CSR, classifies them by topological signature into
/// generations, and performs threat contraction.
///
/// The `vacuum_csr` is moved into `DefectOutput` (ownership transfer).
pub fn apply_defect(
    n: usize,
    vacuum_csr: CsrGraph<Directed>,
    bulk_momentum: Vec<i32>,
) -> (DefectOutput, TopologySummary, Vec<CausalPrism>) {

    // ====================================================================
    //  1. CSR helpers (Zero-Copy, sorted adjacency guaranteed by Phase 1)
    // ====================================================================

    // Build reverse (incoming) CSR: predecessors of each node.
    //
    // The directed Hasse CSR stores u->v (forward only).  For the belly
    // intersection we need parents(v) = {w : w->v}.  The forward-forward
    // candidate search finds v via u->w->v, then the belly is computed as
    // children(u) intersect parents(v) using this reverse CSR.
    //
    // Memory: same size as forward CSR.
    let rev_csr = vacuum_csr.reverse();

    // ====================================================================
    //  2. Core = top CORE_NUM/CORE_DEN nodes by degree
    // ====================================================================
    let target_core = (n * CORE_NUM / CORE_DEN).max(1);

    // Use undirected degree (out + in) for a balanced core at the diamond's centre.
    let mut by_deg: Vec<(usize, usize)> = (0..n)
        .map(|i| {
            let out_d = vacuum_csr.degree(i);
            let in_d  = rev_csr.degree(i);
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
             Core: {core_count} nodes (top {CORE_NUM}/{CORE_DEN}, cutoff degree >= {cutoff})"
        );
        println!("{}", msg);
        use std::fs::OpenOptions; use std::io::Write;
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("simulation.log") {
            writeln!(f, "{}", msg).ok();
        }
    }

    // ====================================================================
    //  3. Prism detection (Timelike K_{2,N} — Forward-Forward)
    //
    //  The Causal Prism is a TIMELIKE structure: Past Pole -> Belly -> Future Pole.
    //  K_{2,N} means two poles (u = past, v = future) with belly nodes w_i:
    //    u -> w_i  and  w_i -> v  for all i in {1...N}
    //  i.e., the belly = children(u) intersect parents(v).
    //
    //  Algorithm:
    //    For each u in core:
    //      For w in children(u):                  [1-hop: belly candidate]
    //        For v in children(w), v in core:     [2-hop: future pole candidate]
    //          belly = children(u) intersect parents(v)
    //          if |belly| >= MIN_PRISM_SHARED: Prism(u, v, belly)
    //    Select best partner per u (greedy maximum belly size).
    // ====================================================================
    println!("  Scanning for Causal Prisms (forward-forward, belly intersection)...");

    let mut prisms: Vec<CausalPrism> = Vec::new();
    let mut is_placed = vec![false; n];
    let total = core_nodes.len();
    let report_step = (total / 10).max(1);

    // Diagnostic: track intersection count distribution
    let mut max_isect_global = 0usize;
    let mut isect_histogram = vec![0usize; 20]; // counts for isect 0..19

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

        let nb_u = vacuum_csr.neighbors(u);
        // Minimum valence pruning: u needs >= MIN_PRISM_SHARED children
        // to form a K_{2,N} belly. Skip early to avoid the 2-hop traversal.
        if nb_u.len() < MIN_PRISM_SHARED { continue; }

        // 2-hop candidate collection via u->w->v (O(D^2) per node)
        //
        // Path: u -> w -> v  ("forward-forward")
        //   For each child w of u (u->w), find all children v of w (w->v).
        //   Any such v is a future pole candidate reachable in 2 forward hops.
        //   Completeness: if u and v are poles of a timelike K_{2,N}, then
        //   every belly node w_i satisfies u->w_i->v, so v is found through w_i.
        cands.clear();
        for &w in nb_u {
            let w = w as usize;
            for &v in vacuum_csr.neighbors(w) {
                let v = v as usize;
                if v != u && is_core[v] && !is_placed[v] {
                    cands.push(v as u32);
                }
            }
        }
        cands.sort_unstable();
        cands.dedup();

        // Find best Prism partner among local candidates
        let mut best_v           = usize::MAX;
        let mut best_count       = 0usize;
        let mut best_intermediates: Vec<usize> = Vec::new();

        for &v_u32 in &cands {
            let v = v_u32 as usize;
            if is_placed[v] { continue; }

            let pv     = rev_csr.neighbors(v);
            if pv.len() < MIN_PRISM_SHARED { continue; }
            let shared = count_slice_intersection(nb_u, pv);

            // Diagnostic tracking
            max_isect_global = max_isect_global.max(shared);
            if shared < 20 { isect_histogram[shared] += 1; }

            if shared >= MIN_PRISM_SHARED && shared > best_count {
                let intermediates = extract_slice_intersection(nb_u, pv);
                // Verify every intermediate is still available.
                if intermediates.iter().all(|&w| !is_placed[w]) {
                    best_count       = shared;
                    best_v           = v;
                    best_intermediates = intermediates;
                }
            }
        }

        if best_v == usize::MAX { continue; }

        // Commit Prism
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
        let msg = format!(
            "  Causal Prisms found: {} (K_{{2,N}}, N >= {})",
            prisms.len(), MIN_PRISM_SHARED
        );
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

    // ====================================================================
    //  4. Topological signature (diagnostic) + Vol I phase classification
    //
    //  phi(w) = sign(bulk_momentum[w]) in {-1, 0, +1}  — causal phase
    //  g(P) = |{phi(w_i) : w_i in intermediates}|       — generation (1,2,3)
    //  Phi(P) = Sum phi(w_i)                             — net phase
    //  Matter: Phi > 0, Antimatter: Phi < 0
    // ====================================================================

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
        net_phase: i32,      // Phi(P) = Sum phi(w_i)
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

    // ====================================================================
    //  5. Threat detection & contraction
    //
    //  Threat criterion: external node t connected to BOTH poles AND
    //  >= PRISM_THREAT intermediates.  Absorb into the higher-degree pole.
    // ====================================================================
    let mut merge_into: Vec<usize> = (0..n).collect();
    let mut is_merged  = vec![false; n];
    let mut merge_count = 0usize;
    let mut k5_minors: Vec<K5Minor> = Vec::new();

    // Undirected adjacency check: t<->v exists if t->v OR v->t in the directed Hasse.
    // The K_5 threat criterion is topological (undirected), not causal (directed).
    let connected_undirected = |u: usize, v: usize| -> bool {
        are_connected(&vacuum_csr, u, v as u32) || are_connected(&vacuum_csr, v, u as u32)
    };

    for (prism_idx, prism) in prisms.iter().enumerate() {
        let deg_o = vacuum_csr.degree(prism.origin);
        let deg_d = vacuum_csr.degree(prism.destination);
        let absorber = if deg_o >= deg_d { prism.origin } else { prism.destination };

        // Gather all external neighbours of the prism components (BOTH directions).
        // Forward (children) + reverse (parents) — CPT-symmetric threat scan.
        let mut threats: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &v in vacuum_csr.neighbors(prism.origin)       { threats.insert(v as usize); }
        for &v in vacuum_csr.neighbors(prism.destination)  { threats.insert(v as usize); }
        for &w in &prism.intermediates {
            for &v in vacuum_csr.neighbors(w) { threats.insert(v as usize); }
        }
        for &v in rev_csr.neighbors(prism.origin)       { threats.insert(v as usize); }
        for &v in rev_csr.neighbors(prism.destination)  { threats.insert(v as usize); }
        for &w in &prism.intermediates {
            for &v in rev_csr.neighbors(w) { threats.insert(v as usize); }
        }

        // Remove prism members from threat candidates
        threats.remove(&prism.origin);
        threats.remove(&prism.destination);
        for &w in &prism.intermediates { threats.remove(&w); }

        for t in threats {
            if is_placed[t] || is_merged[t] { continue; }
            let to_origin = connected_undirected(t, prism.origin);
            let to_dest   = connected_undirected(t, prism.destination);
            let to_inter: Vec<usize> = prism.intermediates.iter()
                .filter(|&&w| connected_undirected(t, w))
                .map(|&w| w)
                .collect();
            if (to_origin && to_dest) && to_inter.len() >= PRISM_THREAT {
                // Record K_5 minor before absorbing: {t, origin, dest, inter[0], inter[1]}
                let v0 = to_inter[0];
                let v1 = to_inter[1];
                k5_minors.push(K5Minor {
                    vertices: [t, prism.origin, prism.destination, v0, v1],
                    source_prism_idx: prism_idx,
                    z5_level: 0, // Will be filled if pts are available
                });

                merge_into[t] = absorber;
                is_merged[t]  = true;
                merge_count  += 1;
            }
        }
    }

    {
        let msg = format!("  Threat contractions: {merge_count} | K_5 minors: {}", k5_minors.len());
        println!("{}", msg);
        use std::fs::OpenOptions; use std::io::Write;
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("simulation.log") {
            writeln!(f, "{}", msg).ok();
        }
    }

    // ====================================================================
    //  6. Resolve transitive merge chains  (integer pointer chase)
    // ====================================================================
    for i in 0..n {
        let mut t = merge_into[i];
        while merge_into[t] != t { t = merge_into[t]; }
        merge_into[i] = t;
    }

    // Free rev_csr (1.18 GB at N=10M) — last used in threat detection above.
    // Dropping before defect edge collection reduces Phase 2 peak by ~1.18 GB.
    drop(rev_csr);

    // ====================================================================
    //  7. Build Defect CSR  (Zero-Copy)
    //
    //  Vacuum edges with merge_into applied, plus one completing edge
    //  per Prism: origin <-> destination.  This edge is always NEW because
    //  in a triangle-free graph the two poles are never directly connected.
    //  We do NOT add intermediate-intermediate edges (preserves bipartite).
    // ====================================================================
    let mut def_edges: Vec<(u32, u32)> = Vec::new();

    // The vacuum CSR is DIRECTED (forward-only: u->v where u < v causally).
    // Each edge appears exactly once.  Do NOT filter by index ordering
    // (u < v) — in the cell-sorted build_hasse_direct path, index order
    // does not match time order within the same quantised time layer.
    for u in 0..n {
        for &v_u32 in vacuum_csr.neighbors(u) {
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

    let defect_csr = CsrGraph::<Undirected>::new(adj_head_def, adj_data_def, n);

    // ====================================================================
    //  8. Core index vectors
    // ====================================================================
    let vacuum_core = core_nodes.clone();
    let defect_core: Vec<usize> = core_nodes.iter().filter(|&&i| !is_merged[i]).cloned().collect();

    // ====================================================================
    //  9. Generation classification (Vol I: phase class counting)
    //
    //  Gen g = prisms with g(P) = g distinct phase classes.
    //  Gen1 matter: g=1, Phi>0.  Anti1: g=1, Phi<0.
    // ====================================================================
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
                    // Phi = 0 within g=1 means all intermediates have bm = 0.
                    // Assign to gen1 (matter-neutral, still generation 1).
                    gen1_nodes.extend_from_slice(&c.nodes);
                }
            }
            2 => gen2_nodes.extend_from_slice(&c.nodes),
            3 => gen3_nodes.extend_from_slice(&c.nodes),
            _ => {} // g > 3 impossible with phi in {-1, 0, +1}
        }
    }

    // ====================================================================
    //  10. Mass Spectrum (Topological Inertia)
    //
    //  Mass = N (number of intermediates), averaged per generation.
    // ====================================================================
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

    // Diagnostics
    {
        let vac_e = vacuum_csr.n_edge_slots();            // directed: each edge stored once
        let def_e = defect_csr.n_edge_slots() / 2;        // undirected: each edge stored twice
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

    // Intermediate phase census (occupancy model diagnostic)
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
        println!(
            "  Intermediate phase census: phi=+1: {} ({:.4})  phi=0: {} ({:.4})  phi=-1: {} ({:.4})",
            phase_pos_count, phase_pos_count as f64 / phase_total,
            phase_zero_count, phase_zero_count as f64 / phase_total,
            phase_neg_count, phase_neg_count as f64 / phase_total
        );
    }

    // Per-generation prism counts
    let prisms_gen1 = classifications.iter().filter(|c| c.generation == 1).count();
    let prisms_gen2 = classifications.iter().filter(|c| c.generation == 2).count();
    let prisms_gen3 = classifications.iter().filter(|c| c.generation == 3).count();
    println!(
        "  Prism generation census: Gen1={prisms_gen1}  Gen2={prisms_gen2}  Gen3={prisms_gen3}  (total={})",
        prisms_gen1 + prisms_gen2 + prisms_gen3
    );

    // Phase-coherence mass decomposition (Theorem: zero free parameters)
    //
    //  M_grav(P) = N,  M_vis(P) = |Phi(P)|,  M_dark(P) = N - |Phi(P)|
    //  Omega_dark/Omega_vis = Sum(N - |Phi|) / Sum|Phi|
    //  Q_topo = Sum|Phi|^2 / Sum N^2   (topological charge ratio, intrinsic)
    //  alpha_EM  = Q_topo / (8 pi)     (observed coupling, geometric synthesis)
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
            "  Phase-coherence: Sum|Phi|={visible_mass_total}  Sum(N-|Phi|)={dark_mass_total}  SumN={grav_mass_total}\n  \
             Q_topo = Sum|Phi|^2/SumN^2 = {q_topo:.6}  |  alpha = Q/(8pi) = {alpha_em:.6}  (1/alpha = {:.1})\n  \
             Omega_linear = {omega_ratio:.4}  |  Omega_energy = 1/Q-1 = {omega_energy:.4}  [alpha(1+Omega)={:.6}  vs 1/(8pi)={:.6}]",
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

    // Sterile prism nodes (Phi = 0: fully phase-cancelled -> truly dark)
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

    // K_5 statistics for topology summary
    let k5_count = k5_minors.len();
    let mean_k5_z5_level = if k5_count > 0 {
        k5_minors.iter().map(|m| m.z5_level as f64).sum::<f64>() / k5_count as f64
    } else {
        0.0
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
        k5_count,
        mean_k5_z5_level,
    };

    (DefectOutput {
        vacuum_csr,
        defect_csr,
        merge_map: merge_into,
        vacuum_core,
        defect_core,
        generations: GenerationSets {
            gen1: gen1_nodes,
            gen2: gen2_nodes,
            gen3: gen3_nodes,
            anti1: anti1_nodes,
            sterile: sterile_nodes,
            mass: [mass_gen1, mass_gen2, mass_gen3, mass_anti1],
        },
        k5_minors,
    }, topology, prisms)
}

/// Unbiased K_{2,n} prism census: scan ALL nodes, no degree threshold, no placement.
///
/// Returns every valid K_{2,n} (n >= 3) subgraph in the Hasse diagram without
/// the core selection bias of `apply_defect`.  Prisms may share nodes (no greedy
/// placement), giving the complete topological census.
///
/// For each origin u with out-degree >= 3, we traverse u→w→v (2-hop forward)
/// and compute belly = children(u) ∩ parents(v).  Every (u,v) pair with
/// |belly| >= 3 emits a prism.  To avoid double-counting, we require u < v
/// (in node index order).
pub fn scan_all_prisms(
    vacuum_csr: &CsrGraph<Directed>,
    n: usize,
) -> Vec<CausalPrism> {
    let rev_csr = vacuum_csr.reverse();
    let min_belly = 3usize;

    let mut prisms: Vec<CausalPrism> = Vec::new();
    let mut cands: Vec<u32> = Vec::with_capacity(256);

    let report_step = (n / 20).max(1);

    for u in 0..n {
        if u % report_step == 0 && u > 0 {
            let pct = u as f64 / n as f64 * 100.0;
            eprintln!("  [Unbiased scan] {pct:.0}% ({u}/{n}) | Prisms: {}", prisms.len());
        }

        let nb_u = vacuum_csr.neighbors(u);
        if nb_u.len() < min_belly { continue; }

        // 2-hop forward: u → w → v
        cands.clear();
        for &w in nb_u {
            for &v in vacuum_csr.neighbors(w as usize) {
                let v_us = v as usize;
                // Require v > u to avoid double-counting (u,v) and (v,u).
                if v_us > u {
                    cands.push(v);
                }
            }
        }
        cands.sort_unstable();
        cands.dedup();

        for &v_u32 in &cands {
            let v = v_u32 as usize;
            let pv = rev_csr.neighbors(v);
            if pv.len() < min_belly { continue; }

            let shared = count_slice_intersection(nb_u, pv);
            if shared >= min_belly {
                let intermediates = extract_slice_intersection(nb_u, pv);
                prisms.push(CausalPrism {
                    origin: u,
                    destination: v,
                    intermediates,
                });
            }
        }
    }

    eprintln!("  [Unbiased scan] Complete: {} total prisms found", prisms.len());
    prisms
}

/// Unbiased K_{2,n} census keeping only the MAXIMAL prism per origin node.
///
/// For each origin u, emits only the prism with the largest belly (most
/// intermediates).  This selects the "most complete topological representation"
/// of each node's defect structure, without the top-10% core bias.
pub fn scan_maximal_prisms(
    vacuum_csr: &CsrGraph<Directed>,
    n: usize,
) -> Vec<CausalPrism> {
    let rev_csr = vacuum_csr.reverse();
    let min_belly = 3usize;

    // For each origin u, track best (largest belly) prism.
    let mut best: Vec<Option<CausalPrism>> = (0..n).map(|_| None).collect();

    let report_step = (n / 10).max(1);

    for u in 0..n {
        if u % report_step == 0 && u > 0 {
            eprintln!("  [Maximal scan] {}%", u * 100 / n);
        }

        let nb_u = vacuum_csr.neighbors(u);
        if nb_u.len() < min_belly { continue; }

        // 2-hop forward: u → w → v
        let mut cands: Vec<u32> = Vec::new();
        for &w in nb_u {
            for &v in vacuum_csr.neighbors(w as usize) {
                if (v as usize) > u {
                    cands.push(v);
                }
            }
        }
        cands.sort_unstable();
        cands.dedup();

        let mut best_belly = best[u].as_ref().map(|p| p.intermediates.len()).unwrap_or(0);

        for &v_u32 in &cands {
            let v = v_u32 as usize;
            let pv = rev_csr.neighbors(v);
            if pv.len() < min_belly { continue; }

            let shared = count_slice_intersection(nb_u, pv);
            if shared >= min_belly && shared > best_belly {
                let intermediates = extract_slice_intersection(nb_u, pv);
                best_belly = shared;
                best[u] = Some(CausalPrism {
                    origin: u,
                    destination: v,
                    intermediates,
                });
            }
        }
    }

    let prisms: Vec<CausalPrism> = best.into_iter().flatten().collect();
    eprintln!("  [Maximal scan] Complete: {} maximal prisms", prisms.len());
    prisms
}
