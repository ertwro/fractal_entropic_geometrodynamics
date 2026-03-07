//! Zigzag loop decomposition of volume growth excess.
//!
//! Decomposes the undirected BFS volume excess (over directed BFS) into:
//! 1. Minimum backward-edge count (k) on shortest undirected paths
//! 2. Shortest undirected cycle length through each backward BFS-tree edge
//!
//! Key prediction: Hasse diagrams (transitive reductions) have NO triangles
//! (3-cycles), because if a→b→c then a→c is transitively reducible and removed.
//! The shortest undirected cycle is 4 (K_{2,2} diamond).
//!
//! If the adelic decomposition D_directed=4 + D_zigzag≈0.64 is correct:
//! - k=0 volume = directed BFS volume (causal/Archimedean, d_H=4)
//! - k≥1 volume = zigzag excess (non-Archimedean, the 0.64)
//! - Backward edges should participate in 4-cycles (K_{2,2}) and 6-cycles (K_{3,3})
//!
//! Usage: cargo run --release --example zigzag_decompose [N] [n_origins] [seed]
//! Default: N=100000, n_origins=200, seed=42

use feg_prism::graph::{CsrGraph, Directed, Undirected};
use feg_prism::phase1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// Directed (forward-only) BFS. Returns dist[v] (u32::MAX if unreachable).
fn bfs_directed(csr: &CsrGraph<Directed>, origin: usize, n: usize) -> Vec<u32> {
    let mut dist = vec![u32::MAX; n];
    let mut queue = VecDeque::new();
    dist[origin] = 0;
    queue.push_back(origin);
    while let Some(u) = queue.pop_front() {
        let d = dist[u];
        for &v in csr.neighbors(u) {
            let vi = v as usize;
            if dist[vi] == u32::MAX {
                dist[vi] = d + 1;
                queue.push_back(vi);
            }
        }
    }
    dist
}

/// Undirected BFS with backward-edge tracking.
///
/// For each node, computes:
/// - `dist[v]`: shortest undirected distance from origin
/// - `min_back[v]`: minimum number of backward (anti-causal) edges on any
///   shortest undirected path from origin to v
///
/// Also returns backward edges `(parent, child, parent_dist)` used in first discoveries.
fn bfs_undirected_tracked(
    sym: &CsrGraph<Undirected>,
    dir: &CsrGraph<Directed>,
    origin: usize,
    n: usize,
) -> (Vec<u32>, Vec<u32>, Vec<(usize, usize, u32)>) {
    let mut dist = vec![u32::MAX; n];
    let mut min_back = vec![u32::MAX; n];
    let mut back_edges: Vec<(usize, usize, u32)> = Vec::new();
    let mut queue = VecDeque::new();

    dist[origin] = 0;
    min_back[origin] = 0;
    queue.push_back(origin);

    while let Some(u) = queue.pop_front() {
        let d = dist[u];
        let b = min_back[u];

        for &v in sym.neighbors(u) {
            let vi = v as usize;
            let is_forward = dir.has_edge(u, v);
            let nb = b + if is_forward { 0 } else { 1 };

            if dist[vi] == u32::MAX {
                // First visit
                dist[vi] = d + 1;
                min_back[vi] = nb;
                queue.push_back(vi);
                if !is_forward {
                    back_edges.push((u, vi, d));
                }
            } else if dist[vi] == d + 1 && nb < min_back[vi] {
                // Same distance, fewer backward edges — update
                min_back[vi] = nb;
            }
        }
    }

    (dist, min_back, back_edges)
}

/// Count of common elements in two sorted slices.
fn sorted_intersection_count(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j) = (0, 0);
    let mut count = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

/// Check if two sorted slices share any element other than `excl`.
fn has_intersection_excl(a: &[u32], b: &[u32], excl: u32) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                if a[i] != excl {
                    return true;
                }
                i += 1;
                j += 1;
            }
        }
    }
    false
}

/// Find shortest undirected cycle through edge (u, v).
///
/// Returns cycle length (3, 4, 5) or 0 if none found within length 5.
/// Hasse diagrams have girth ≥ 4 (no triangles), so L=3 should never appear.
fn shortest_cycle_length(u: usize, v: usize, sym: &CsrGraph<Undirected>) -> u8 {
    let nu = sym.neighbors(u);
    let nv = sym.neighbors(v);

    // L=3 (triangle): common neighbor of u and v
    // Should be zero for Hasse diagrams (transitive reduction eliminates triangles).
    if sorted_intersection_count(nu, nv) > 0 {
        return 3;
    }

    // L=4 (diamond/K_{2,2}): path u-w1-w2-v where w1∈N(u), w2∈N(v), w1-w2 edge.
    // Equivalently: some neighbor of u shares a neighbor with v (excluding u,v).
    for &w in nu {
        if w as usize == v {
            continue;
        }
        let nw = sym.neighbors(w as usize);
        if has_intersection_excl(nw, nv, u as u32) {
            return 4;
        }
    }

    // L=5: path u-w1-w2-w3-v. Check: neighbor-of-neighbor of u intersects N(v).
    // Cost: O(deg^2 × deg) per edge — cap at a subsample.
    for &w1 in nu.iter().take(20) {
        if w1 as usize == v {
            continue;
        }
        for &w2 in sym.neighbors(w1 as usize).iter().take(20) {
            let w2i = w2 as usize;
            if w2i == u || w2i == v {
                continue;
            }
            let nw2 = sym.neighbors(w2i);
            if has_intersection_excl(nw2, nv, w1) {
                return 5;
            }
        }
    }

    0 // not found within L=5
}

/// Count 4-cycles through edge (u,v) in the undirected graph.
/// A 4-cycle u-w1-w2-v-u requires w1∈N(u)\{v}, w2∈N(v)\{u}, w1-w2 edge.
fn count_4cycles(u: usize, v: usize, sym: &CsrGraph<Undirected>) -> usize {
    let nv = sym.neighbors(v);
    let mut count = 0;
    for &w1 in sym.neighbors(u) {
        if w1 as usize == v {
            continue;
        }
        let nw1 = sym.neighbors(w1 as usize);
        // Count |N(w1) ∩ N(v)| excluding u
        let (mut i, mut j) = (0, 0);
        while i < nw1.len() && j < nv.len() {
            match nw1[i].cmp(&nv[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    if nw1[i] != u as u32 {
                        count += 1;
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
    }
    count
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let n_origins: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(42);

    println!("=== Zigzag Loop Decomposition ===");
    println!("  N = {},  origins = {},  seed = {}\n", n, n_origins, seed);

    // Phase 1: sprinkle + Hasse
    let mut rng = StdRng::seed_from_u64(seed);
    let (pts_raw, _half_t) = phase1::sprinkle(n, &mut rng);
    let (_sorted_coords, vacuum_csr, _momentum) = phase1::build_hasse_direct(&pts_raw);
    drop(pts_raw);

    println!("[1] Symmetrizing Hasse diagram...");
    let sym = vacuum_csr.symmetrize();

    let margin = n / 10;
    let origins: Vec<usize> = (0..n_origins)
        .map(|_| rng.gen_range(margin..n - margin))
        .collect();

    let max_r = 25usize;
    let max_k = 5usize;

    // Accumulators: per (r, k) node counts
    let mut sum_dir = vec![0.0f64; max_r + 1]; // cumulative directed volume
    let mut sum_und = vec![0.0f64; max_r + 1]; // cumulative undirected volume
    let mut decomp = vec![vec![0.0f64; max_k + 2]; max_r + 1]; // [r][k], k=max_k+1 is overflow

    // Cycle length histogram (over backward BFS-tree edges)
    let mut cycle_hist = [0usize; 7]; // [0]=unknown, [3],[4],[5],[6] = cycle lengths
    let mut four_cycle_counts: Vec<usize> = Vec::new(); // per-backward-edge 4-cycle count
    let mut total_back_edges = 0usize;

    // Shortcut gain histogram: dist_dir - dist_und for excess nodes
    let mut gain_hist = vec![0usize; 30];

    // Richness by BFS radius: (sum_richness, count) at each parent distance
    let mut richness_by_r: Vec<(f64, usize)> = vec![(0.0, 0); max_r + 1];
    // Richness histogram (log-binned): [0-1, 2-5, 6-15, 16-50, 51-150, 151-500, 501+]
    let richness_bins = [1, 5, 15, 50, 150, 500, usize::MAX];
    let mut richness_hist = vec![0usize; richness_bins.len()];

    println!("[2] Running {} BFS pairs (directed + undirected)...", n_origins);

    for (idx, &origin) in origins.iter().enumerate() {
        if idx > 0 && idx % 50 == 0 {
            println!("  Progress: {}/{}", idx, n_origins);
        }

        let dist_dir = bfs_directed(&vacuum_csr, origin, n);
        let (dist_und, min_back, back_edges) = bfs_undirected_tracked(&sym, &vacuum_csr, origin, n);

        // Accumulate shells
        for v in 0..n {
            // Directed
            if dist_dir[v] != u32::MAX {
                let r = dist_dir[v] as usize;
                if r <= max_r {
                    sum_dir[r] += 1.0;
                }
            }
            // Undirected (with k decomposition)
            if dist_und[v] != u32::MAX {
                let r = dist_und[v] as usize;
                if r <= max_r {
                    sum_und[r] += 1.0;
                    let k = (min_back[v] as usize).min(max_k + 1);
                    decomp[r][k] += 1.0;
                }

                // Shortcut gain
                if min_back[v] > 0 {
                    let gain = if dist_dir[v] != u32::MAX {
                        (dist_dir[v] as usize).saturating_sub(dist_und[v] as usize)
                    } else {
                        dist_und[v] as usize // "infinite" gain; cap at dist_und
                    };
                    if gain < gain_hist.len() {
                        gain_hist[gain] += 1;
                    }
                }
            }
        }

        // Stratified richness sampling: up to 50 backward edges per radius per BFS
        // This ensures we get data at ALL radii, not just the first few shells.
        let mut edges_by_r: Vec<Vec<(usize, usize)>> = vec![vec![]; max_r + 1];
        for &(u, v, parent_dist) in &back_edges {
            let r = parent_dist as usize;
            if r <= max_r && edges_by_r[r].len() < 50 {
                edges_by_r[r].push((u, v));
            }
        }

        // Compute richness for stratified sample
        for r in 0..=max_r {
            for &(u, v) in &edges_by_r[r] {
                total_back_edges += 1;
                let c4 = count_4cycles(u, v, &sym);
                four_cycle_counts.push(c4);

                richness_by_r[r].0 += c4 as f64;
                richness_by_r[r].1 += 1;

                for (bin, &upper) in richness_bins.iter().enumerate() {
                    if c4 <= upper {
                        richness_hist[bin] += 1;
                        break;
                    }
                }
            }
        }

        // Cycle check on small sample (already confirmed L=4 at 100%)
        for &(u, v, _) in back_edges.iter().take(50) {
            let cl = shortest_cycle_length(u, v, &sym);
            if (cl as usize) < cycle_hist.len() {
                cycle_hist[cl as usize] += 1;
            } else {
                cycle_hist[0] += 1;
            }
        }
    }

    let no = n_origins as f64;

    // === Output 1: Volume decomposition ===
    println!("\n=== Volume Decomposition: Directed vs Undirected ===");
    println!(
        "{:>3} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "r", "V_dir(r)", "V_und(r)", "excess", "d_H_dir", "d_H_und"
    );

    let mut cum_dir = vec![0.0f64; max_r + 1];
    let mut cum_und = vec![0.0f64; max_r + 1];
    cum_dir[0] = sum_dir[0] / no;
    cum_und[0] = sum_und[0] / no;
    for r in 1..=max_r {
        cum_dir[r] = cum_dir[r - 1] + sum_dir[r] / no;
        cum_und[r] = cum_und[r - 1] + sum_und[r] / no;
    }

    for r in 1..=max_r.min(20) {
        let excess = cum_und[r] - cum_dir[r];
        let dh_dir = if r >= 2 && cum_dir[r] > 0.0 && cum_dir[r - 1] > 0.0 {
            (cum_dir[r].ln() - cum_dir[r - 1].ln())
                / ((r as f64).ln() - ((r - 1) as f64).ln())
        } else {
            0.0
        };
        let dh_und = if r >= 2 && cum_und[r] > 0.0 && cum_und[r - 1] > 0.0 {
            (cum_und[r].ln() - cum_und[r - 1].ln())
                / ((r as f64).ln() - ((r - 1) as f64).ln())
        } else {
            0.0
        };
        println!(
            "{:3} {:10.1} {:10.1} {:10.1} {:8.3} {:8.3}",
            r, cum_dir[r], cum_und[r], excess, dh_dir, dh_und
        );
    }

    // === Output 2: Excess decomposed by backward-edge count ===
    println!("\n=== Excess Decomposition by Min Backward Edges (k) ===");
    println!(
        "{:>3} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "r", "excess", "k=0%", "k=1", "k=2", "k=3", "k=4", "k=5+"
    );

    let mut cum_k = vec![vec![0.0f64; max_k + 2]; max_r + 1];
    for r in 0..=max_r {
        for k in 0..=max_k + 1 {
            cum_k[r][k] = if r > 0 { cum_k[r - 1][k] } else { 0.0 } + decomp[r][k] / no;
        }
    }

    for r in 2..=max_r.min(15) {
        let excess = cum_und[r] - cum_dir[r];
        if excess < 1.0 {
            continue;
        }
        // k=0 should equal V_dir
        let k0_check = cum_k[r][0];
        let k0_pct = 100.0 * (k0_check - cum_dir[r]).abs() / cum_dir[r].max(1.0);
        println!(
            "{:3} {:10.1} {:>7.2}% {:8.1} {:8.1} {:8.1} {:8.1} {:8.1}",
            r,
            excess,
            k0_pct, // should be ~0% (k=0 ≈ V_dir)
            cum_k[r][1],
            cum_k[r][2],
            cum_k[r][3],
            cum_k[r][4],
            cum_k[r][5] + cum_k[r][max_k + 1],
        );
    }

    // === Output 3: Fraction of excess by k ===
    println!("\n=== Excess Fraction by k (percentage of excess volume) ===");
    println!(
        "{:>3} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "r", "excess", "%k=1", "%k=2", "%k=3", "%k≥4"
    );
    for r in 2..=max_r.min(15) {
        let excess = cum_und[r] - cum_dir[r];
        if excess < 10.0 {
            continue;
        }
        let k_ge4: f64 = (4..=max_k + 1).map(|k| cum_k[r][k]).sum();
        println!(
            "{:3} {:10.1} {:7.1}% {:7.1}% {:7.1}% {:7.1}%",
            r,
            excess,
            100.0 * cum_k[r][1] / excess,
            100.0 * cum_k[r][2] / excess,
            100.0 * cum_k[r][3] / excess,
            100.0 * k_ge4 / excess,
        );
    }

    // === Output 4: Cycle length histogram ===
    println!("\n=== Cycle Length Histogram (backward edges in BFS trees) ===");
    println!("Total backward edges sampled: {}", total_back_edges);
    if total_back_edges > 0 {
        for cl in 3..cycle_hist.len() {
            if cycle_hist[cl] > 0 {
                println!(
                    "  L={}: {:>8} ({:5.1}%)",
                    cl,
                    cycle_hist[cl],
                    100.0 * cycle_hist[cl] as f64 / total_back_edges as f64
                );
            }
        }
        println!(
            "  L>6: {:>8} ({:5.1}%)",
            cycle_hist[0],
            100.0 * cycle_hist[0] as f64 / total_back_edges as f64
        );
    }

    // 4-cycle richness: how many 4-cycles does each backward edge participate in?
    if !four_cycle_counts.is_empty() {
        let mut sorted = four_cycle_counts.clone();
        sorted.sort_unstable();
        let total: usize = sorted.iter().sum();
        let mean = total as f64 / sorted.len() as f64;
        let median = sorted[sorted.len() / 2];
        let max = *sorted.last().unwrap();
        let p90 = sorted[(sorted.len() as f64 * 0.9) as usize];

        println!("\n=== 4-Cycle Richness per Backward Edge ===");
        println!(
            "  mean={:.1}, median={}, p90={}, max={}",
            mean, median, p90, max
        );
        println!("  (Higher = edge embedded in richer K_{{m,n}} structure)");
    }

    // === Output 5: Richness by BFS radius ===
    println!("\n=== 4-Cycle Richness by BFS Radius (UV → IR flow) ===");
    println!(
        "{:>3} {:>8} {:>10} {:>12}",
        "r", "edges", "mean_rich", "prediction"
    );
    println!("  (If richness drives d_H excess, it should decrease with r)");
    for r in 0..=max_r.min(15) {
        let (sum_r, cnt) = richness_by_r[r];
        if cnt >= 10 {
            let mean_r = sum_r / cnt as f64;
            let dh_excess = if r >= 2 && cum_und[r] > 0.0 && cum_und[r - 1] > 0.0
                && cum_dir[r] > 0.0 && cum_dir[r - 1] > 0.0
            {
                let dh_und = (cum_und[r].ln() - cum_und[r - 1].ln())
                    / ((r as f64).ln() - ((r - 1) as f64).ln());
                let dh_dir = (cum_dir[r].ln() - cum_dir[r - 1].ln())
                    / ((r as f64).ln() - ((r - 1) as f64).ln());
                dh_und - dh_dir
            } else {
                0.0
            };
            println!(
                "{:3} {:8} {:10.1} {:>12.3}",
                r, cnt, mean_r, dh_excess
            );
        }
    }

    // === Output 5b: Richness histogram ===
    println!("\n=== 4-Cycle Richness Histogram ===");
    let labels = ["0-1", "2-5", "6-15", "16-50", "51-150", "151-500", "501+"];
    for (i, &count) in richness_hist.iter().enumerate() {
        if count > 0 {
            println!(
                "  {:>8}: {:>8} ({:5.1}%)",
                labels[i],
                count,
                100.0 * count as f64 / total_back_edges.max(1) as f64
            );
        }
    }

    // === Output 7: Shortcut gain histogram ===
    println!("\n=== Shortcut Gain (dist_dir - dist_und for k≥1 nodes) ===");
    let total_gain: usize = gain_hist.iter().sum();
    if total_gain > 0 {
        println!(
            "{:>5} {:>10} {:>8}",
            "gain", "count", "%"
        );
        for (g, &c) in gain_hist.iter().enumerate() {
            if c > 0 {
                println!(
                    "{:5} {:10} {:7.1}%",
                    g,
                    c,
                    100.0 * c as f64 / total_gain as f64
                );
            }
        }
    }

    // === Output 8: Adelic decomposition check ===
    println!("\n=== Adelic Decomposition Check ===");
    println!("  d_S^adelic = 2ln2/ln3 + 2ln3/ln4 + 2ln5/ln6 = {:.4}",
             2.0 * 2.0_f64.ln() / 3.0_f64.ln()
             + 2.0 * 3.0_f64.ln() / 4.0_f64.ln()
             + 2.0 * 5.0_f64.ln() / 6.0_f64.ln());
    println!("  Prediction: directed d_H = 4.0, undirected excess ≈ 0.64");
    for r in 2..=max_r.min(10) {
        let dh_dir = if cum_dir[r] > 0.0 && cum_dir[r - 1] > 0.0 {
            (cum_dir[r].ln() - cum_dir[r - 1].ln())
                / ((r as f64).ln() - ((r - 1) as f64).ln())
        } else {
            0.0
        };
        let dh_und = if cum_und[r] > 0.0 && cum_und[r - 1] > 0.0 {
            (cum_und[r].ln() - cum_und[r - 1].ln())
                / ((r as f64).ln() - ((r - 1) as f64).ln())
        } else {
            0.0
        };
        let excess_d = dh_und - dh_dir;
        println!(
            "  r={}: d_H_dir={:.3}, d_H_und={:.3}, excess={:.3}",
            r, dh_dir, dh_und, excess_d
        );
    }

    println!("\nDone.");
}
