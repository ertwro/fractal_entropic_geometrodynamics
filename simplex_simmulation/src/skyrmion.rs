//! Phase 2 — K4 Topological Defect (Pure Integer Kuratowski Calculus)
//!
//! Zero floating-point.  The Strong Force is discrete combinatorics:
//!   - Core identification:  integer degree ranking (usize)
//!   - K5 threat detection:  integer mutual-neighbor counting
//!   - Vertex contraction:   boolean flags + index remapping
//!   - K4 completion:        index insertion into sorted edge list
//!
//! f64 only enters the simulation at Phase 3 (macroscopic d_S averages).

/// Fraction of nodes selected as topological core (numerator / denominator).
/// Top 10% by undirected Hasse degree ≈ the diamond's combinatorial center.
const CORE_NUM: usize = 1;
const CORE_DEN: usize = 10;

/// A core neighbour connected to ≥ K5_THREAT members of a K4 clique
/// threatens a forbidden K5 subgraph and must be contracted.
const K5_THREAT: usize = 3;

/// Apply the Kuratowski contraction.
///
/// Pure integer/boolean logic.  Returns
/// `(defect_rows, defect_cols, vacuum_core, defect_core)`.
///
/// `vacuum_core` = all core indices (for measuring the same region on the
/// unmodified vacuum graph).  `defect_core` = core minus merged-away nodes.
pub fn apply_defect(
    n: usize,
    edge_rows: &[u32],
    edge_cols: &[u32],
) -> (Vec<u32>, Vec<u32>, Vec<usize>, Vec<usize>) {

    // ════════════════════════════════════════════════════════════════
    //  1.  Undirected adjacency   (Vec<Vec<usize>>, sorted, deduped)
    // ════════════════════════════════════════════════════════════════
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (&r, &c) in edge_rows.iter().zip(edge_cols.iter()) {
        let ri = r as usize;
        let ci = c as usize;
        adj[ri].push(ci);
        adj[ci].push(ri);
    }
    for list in adj.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }

    // ════════════════════════════════════════════════════════════════
    //  2.  Core = top CORE_NUM/CORE_DEN nodes by degree  (integer)
    // ════════════════════════════════════════════════════════════════
    let target_core = n * CORE_NUM / CORE_DEN;
    let mut by_degree: Vec<(usize, usize)> = adj          // (degree, node)
        .iter()
        .enumerate()
        .map(|(i, a)| (a.len(), i))
        .collect();
    by_degree.sort_unstable_by(|a, b| b.0.cmp(&a.0));     // descending

    let degree_cutoff: usize = by_degree
        .get(target_core.saturating_sub(1))
        .map(|&(d, _)| d)
        .unwrap_or(0);

    let mut is_core: Vec<bool> = vec![false; n];
    let mut core_count: usize = 0;
    for &(deg, node) in &by_degree {
        if core_count >= target_core && deg < degree_cutoff {
            break;
        }
        if deg >= degree_cutoff {
            is_core[node] = true;
            core_count += 1;
        }
    }
    let core_nodes: Vec<usize> = (0..n).filter(|&i| is_core[i]).collect();

    // ════════════════════════════════════════════════════════════════
    //  3.  Core-density ranking   (integer: core-neighbour count)
    // ════════════════════════════════════════════════════════════════
    let mut core_by_density: Vec<(usize, usize)> = core_nodes   // (node, core_deg)
        .iter()
        .map(|&node| {
            let cd: usize = adj[node].iter().filter(|&&nb| is_core[nb]).count();
            (node, cd)
        })
        .collect();
    core_by_density.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    // ════════════════════════════════════════════════════════════════
    //  4.  Greedy K4 formation + K5 contraction   (bool + usize)
    // ════════════════════════════════════════════════════════════════
    let mut merge_into: Vec<usize> = (0..n).collect();    // identity map
    let mut is_merged: Vec<bool>  = vec![false; n];
    let mut is_placed: Vec<bool>  = vec![false; n];
    let mut k4_cliques: Vec<[usize; 4]> = Vec::new();
    let mut merge_count: usize = 0;

    for &(seed, _) in &core_by_density {
        if is_placed[seed] || is_merged[seed] { continue; }

        // Available core neighbours
        let available: Vec<usize> = adj[seed]
            .iter()
            .filter(|&&nb| is_core[nb] && !is_placed[nb] && !is_merged[nb])
            .cloned()
            .collect();
        if available.len() < 3 { continue; }

        // Score by mutual connectivity (integer count)
        let mut scored: Vec<(usize, usize)> = available
            .iter()
            .map(|&nb| {
                let mutual: usize = available
                    .iter()
                    .filter(|&&other| {
                        other != nb && adj[nb].binary_search(&other).is_ok()
                    })
                    .count();
                (nb, mutual)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        if scored.len() < 3 { continue; }
        let k4 = [seed, scored[0].0, scored[1].0, scored[2].0];
        k4_cliques.push(k4);
        for &node in &k4 { is_placed[node] = true; }

        // K5 threat: remaining neighbour connected to ≥ K5_THREAT members
        for &nb in &available {
            if is_placed[nb] || is_merged[nb] || k4.contains(&nb) { continue; }

            let connections: usize = k4
                .iter()
                .filter(|&&m| adj[nb].binary_search(&m).is_ok())
                .count();

            if connections >= K5_THREAT {
                // Absorb into highest-degree K4 member (integer comparison)
                let absorber = *k4
                    .iter()
                    .filter(|&&m| adj[nb].binary_search(&m).is_ok())
                    .max_by_key(|&&m| adj[m].len())
                    .unwrap();
                merge_into[nb] = absorber;
                is_merged[nb] = true;
                merge_count += 1;
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  5.  Resolve transitive merge chains   (integer pointer chase)
    // ════════════════════════════════════════════════════════════════
    for i in 0..n {
        let mut t = merge_into[i];
        while merge_into[t] != t { t = merge_into[t]; }
        merge_into[i] = t;
    }

    // ════════════════════════════════════════════════════════════════
    //  6.  Rebuild edge list   (integer sort + dedup, no HashSet)
    // ════════════════════════════════════════════════════════════════
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(edge_rows.len() + k4_cliques.len() * 6);

    // Original edges with merges applied
    for (&r, &c) in edge_rows.iter().zip(edge_cols.iter()) {
        let ri = merge_into[r as usize] as u32;
        let ci = merge_into[c as usize] as u32;
        if ri != ci {
            let (lo, hi) = if ri < ci { (ri, ci) } else { (ci, ri) };
            edges.push((lo, hi));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let surviving = edges.len();

    // K4 completion
    for k4 in &k4_cliques {
        let m = [
            merge_into[k4[0]] as u32,
            merge_into[k4[1]] as u32,
            merge_into[k4[2]] as u32,
            merge_into[k4[3]] as u32,
        ];
        for i in 0..4 {
            for j in (i + 1)..4 {
                if m[i] != m[j] {
                    let (lo, hi) = if m[i] < m[j] { (m[i], m[j]) } else { (m[j], m[i]) };
                    edges.push((lo, hi));
                }
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let k4_added = edges.len() - surviving;

    let new_rows: Vec<u32> = edges.iter().map(|&(r, _)| r).collect();
    let new_cols: Vec<u32> = edges.iter().map(|&(_, c)| c).collect();

    // ════════════════════════════════════════════════════════════════
    //  7.  Core index vectors
    // ════════════════════════════════════════════════════════════════
    let vacuum_core: Vec<usize> = core_nodes.clone();
    let defect_core: Vec<usize> = core_nodes
        .iter()
        .filter(|&&i| !is_merged[i])
        .cloned()
        .collect();

    // ── Diagnostics ─────────────────────────────────────────────────
    let delta: i64 = new_rows.len() as i64 - edge_rows.len() as i64;
    let sign = if delta >= 0 { "+" } else { "" };
    println!("[Phase 2] Kuratowski contraction (pure integer)");
    println!(
        "  Core: {} nodes (top {}/{} by degree, cutoff ≥ {})",
        core_count, CORE_NUM, CORE_DEN, degree_cutoff
    );
    println!("  K4 cliques: {}", k4_cliques.len());
    println!("  K5 contractions: {}", merge_count);
    println!("  K4 completion edges: +{}", k4_added);
    println!(
        "  Edges: {} → {} ({}{})",
        edge_rows.len(),
        new_rows.len(),
        sign,
        delta
    );
    println!("  Active core: {} / {}", defect_core.len(), vacuum_core.len());

    (new_rows, new_cols, vacuum_core, defect_core)
}
