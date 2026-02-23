//! Bridge to `causal_set_sim` — wraps sprinkle, Hasse, and Kuratowski defect.

use causal_set_sim::diamond;
use causal_set_sim::skyrmion::{self, CausalPrism, DefectResult, TopologySummary};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A causal graph: sprinkled points + Hasse diagram in CSR format.
pub struct CausalGraph {
    pub n: usize,
    pub coords: Vec<[f64; 4]>,
    pub adj_head: Vec<u32>,
    pub adj_data: Vec<u32>,
    pub bulk_momentum: Vec<i32>,
}

/// Result of Kuratowski contraction (Phase 2).
pub struct DefectData {
    pub result: DefectResult,
    pub topology: TopologySummary,
    pub prisms: Vec<CausalPrism>,
}

/// Description of a single detected Causal Prism K_{2,N}.
pub struct PrismInfo {
    pub origin: usize,
    pub destination: usize,
    pub belly: Vec<usize>,
    pub generation: i32, // 1,2,3 = matter; -1 = anti; 0 = sterile
}

impl CausalGraph {
    /// Sprinkle N events in a 4D causal diamond and build the Hasse diagram.
    ///
    /// Uses `build_hasse_sparse` for N ≤ 15 000 and `build_hasse_direct`
    /// for larger N, matching the Colisionador's tier selection.
    pub fn sprinkle_and_build(n: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let (pts_raw, _big_t) = diamond::sprinkle(n, &mut rng);

        let (coords, adj_head, adj_data, bulk_momentum) = if n <= 15_000 {
            diamond::build_hasse_sparse(&pts_raw)
        } else {
            diamond::build_hasse_direct(&pts_raw)
        };

        CausalGraph { n, coords, adj_head, adj_data, bulk_momentum }
    }

    /// Apply Kuratowski defect: detect Causal Prisms, classify generations,
    /// and contract K₅ threats.  Clones the CSR vectors so the graph remains
    /// usable after this call.
    pub fn apply_defect(&self) -> DefectData {
        let (result, topology, prisms) = skyrmion::apply_defect(
            self.n,
            self.adj_head.clone(),
            self.adj_data.clone(),
            self.bulk_momentum.clone(),
        );
        DefectData { result, topology, prisms }
    }

    /// Neighbours of node `u` in the Hasse diagram (forward direction).
    pub fn neighbours(&self, u: usize) -> &[u32] {
        let start = self.adj_head[u] as usize;
        let end = self.adj_head[u + 1] as usize;
        &self.adj_data[start..end]
    }

    /// Simulate cover time on a prism subgraph.
    ///
    /// A lazy random walker starts at `origin` and must visit every belly
    /// node before reaching `dest`.  Returns the number of ticks to
    /// cover all belly nodes (coupon-collector delay).
    ///
    /// The walk is on the symmetric (undirected) local subgraph induced
    /// by {origin, dest} ∪ belly, using the Hasse edges.
    pub fn prism_cover_time(&self, origin: usize, belly: &[usize], dest: usize, seed: u64) -> u32 {
        let mut rng = StdRng::seed_from_u64(seed);

        // Build local symmetric adjacency for the prism subgraph.
        // In K_{2,N}: origin connects to all belly, dest connects to all belly.
        // Belly nodes connect to origin and dest (and no other belly nodes).
        let mut nodes: Vec<usize> = vec![origin, dest];
        nodes.extend_from_slice(belly);

        // For each node, collect its local neighbours.
        use std::collections::{HashMap, HashSet};
        let node_set: HashSet<usize> = nodes.iter().copied().collect();
        let mut local_adj: HashMap<usize, Vec<usize>> = HashMap::new();

        for &u in &nodes {
            let mut nbrs = Vec::new();
            // Forward neighbours
            let start = self.adj_head[u] as usize;
            let end = self.adj_head[u + 1] as usize;
            for &v in &self.adj_data[start..end] {
                if node_set.contains(&(v as usize)) {
                    nbrs.push(v as usize);
                }
            }
            // Backward neighbours (reverse edges)
            for &w in &nodes {
                if w == u { continue; }
                let ws = self.adj_head[w] as usize;
                let we = self.adj_head[w + 1] as usize;
                for &v in &self.adj_data[ws..we] {
                    if v as usize == u && !nbrs.contains(&w) {
                        nbrs.push(w);
                    }
                }
            }
            local_adj.insert(u, nbrs);
        }

        // Walk until all belly nodes are visited.
        let belly_set: HashSet<usize> = belly.iter().copied().collect();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut current = origin;
        let mut ticks: u32 = 0;

        loop {
            if belly_set.contains(&current) {
                visited.insert(current);
            }
            if visited.len() == belly_set.len() {
                break;
            }

            let nbrs = local_adj.get(&current).unwrap();
            if nbrs.is_empty() { break; }
            let idx = rng.gen_range(0..nbrs.len());
            current = nbrs[idx];
            ticks += 1;

            // Safety: prevent infinite loop on degenerate graphs.
            if ticks > 100_000 { break; }
        }
        ticks
    }

    /// Simulate cover time on a prism subgraph, returning the full trajectory.
    ///
    /// Identical walk logic to `prism_cover_time`, but returns the sequence
    /// of visited node indices (one per tick) instead of just the count.
    /// The trajectory terminates when all belly nodes have been visited.
    pub fn prism_cover_trajectory(
        &self,
        origin: usize,
        belly: &[usize],
        dest: usize,
        seed: u64,
    ) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);

        // Build local symmetric adjacency for the prism subgraph.
        let mut nodes: Vec<usize> = vec![origin, dest];
        nodes.extend_from_slice(belly);

        use std::collections::{HashMap, HashSet};
        let node_set: HashSet<usize> = nodes.iter().copied().collect();
        let mut local_adj: HashMap<usize, Vec<usize>> = HashMap::new();

        for &u in &nodes {
            let mut nbrs = Vec::new();
            let start = self.adj_head[u] as usize;
            let end = self.adj_head[u + 1] as usize;
            for &v in &self.adj_data[start..end] {
                if node_set.contains(&(v as usize)) {
                    nbrs.push(v as usize);
                }
            }
            for &w in &nodes {
                if w == u { continue; }
                let ws = self.adj_head[w] as usize;
                let we = self.adj_head[w + 1] as usize;
                for &v in &self.adj_data[ws..we] {
                    if v as usize == u && !nbrs.contains(&w) {
                        nbrs.push(w);
                    }
                }
            }
            local_adj.insert(u, nbrs);
        }

        let belly_set: HashSet<usize> = belly.iter().copied().collect();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut current = origin;
        let mut trajectory: Vec<usize> = vec![origin];

        loop {
            if belly_set.contains(&current) {
                visited.insert(current);
            }
            if visited.len() == belly_set.len() {
                break;
            }

            let nbrs = local_adj.get(&current).unwrap();
            if nbrs.is_empty() { break; }
            let idx = rng.gen_range(0..nbrs.len());
            current = nbrs[idx];
            trajectory.push(current);

            if trajectory.len() > 100_000 { break; }
        }
        trajectory
    }

    /// K₃,₃ probe: for a prism, test which external vacuum neighbours
    /// would be accepted vs blocked by the K₃,₃ planarity constraint.
    ///
    /// An external node `w` is "rejected" if adding edges from `w` to
    /// both poles + one belly node would create K₃,₃ (3 sources connected
    /// to 3 sinks).  Returns `(accepted, rejected)` node index vectors.
    pub fn k33_probe(
        &self,
        origin: usize,
        dest: usize,
        belly: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        use std::collections::HashSet;
        let prism_set: HashSet<usize> = {
            let mut s = HashSet::new();
            s.insert(origin);
            s.insert(dest);
            for &b in belly { s.insert(b); }
            s
        };

        // Collect all external neighbours of the prism nodes.
        let mut external: HashSet<usize> = HashSet::new();
        for &u in prism_set.iter() {
            let start = self.adj_head[u] as usize;
            let end = self.adj_head[u + 1] as usize;
            for &v in &self.adj_data[start..end] {
                if !prism_set.contains(&(v as usize)) {
                    external.insert(v as usize);
                }
            }
            // Also check reverse edges.
            for w in 0..self.n {
                if prism_set.contains(&w) || external.contains(&w) { continue; }
                let ws = self.adj_head[w] as usize;
                let we = self.adj_head[w + 1] as usize;
                for &v in &self.adj_data[ws..we] {
                    if v as usize == u {
                        external.insert(w);
                    }
                }
            }
        }

        let mut accepted = Vec::new();
        let mut rejected = Vec::new();

        for &w in &external {
            // Count how many prism poles + belly nodes `w` already connects to.
            let mut pole_connections = 0u32;
            let mut belly_connections = 0u32;

            // Check w's forward neighbours.
            let ws = self.adj_head[w] as usize;
            let we = self.adj_head[w + 1] as usize;
            for &v in &self.adj_data[ws..we] {
                let vi = v as usize;
                if vi == origin || vi == dest {
                    pole_connections += 1;
                } else if belly.contains(&vi) {
                    belly_connections += 1;
                }
            }

            // Check reverse edges (prism nodes pointing to w).
            for &u in prism_set.iter() {
                let us = self.adj_head[u] as usize;
                let ue = self.adj_head[u + 1] as usize;
                for &v in &self.adj_data[us..ue] {
                    if v as usize == w {
                        if u == origin || u == dest {
                            pole_connections += 1;
                        } else if belly.contains(&u) {
                            belly_connections += 1;
                        }
                    }
                }
            }

            // K₃,₃ requires 3 nodes on each side. If w connects to both
            // poles AND at least one belly node, we'd have 3 "sources"
            // (origin, dest, belly_node) each connected to "sinks" (w +
            // others), creating a K₃,₃ minor.  Reject if this holds.
            if pole_connections >= 2 && belly_connections >= 1 {
                rejected.push(w);
            } else {
                accepted.push(w);
            }
        }

        accepted.sort();
        rejected.sort();
        (accepted, rejected)
    }

    /// Modulo interference: run walkers accumulating g^S mod p phase.
    ///
    /// Returns per-node `(node_index, arrivals, phase_sum, intensity)`.
    /// `intensity` = normalized constructive/destructive measure.
    pub fn modulo_interference(
        &self,
        n_walkers: usize,
        steps: usize,
        prime: u64,
        root: u64,
        seed: u64,
    ) -> Vec<(usize, u64, u64, f64)> {
        let mut rng = StdRng::seed_from_u64(seed);

        // Make symmetric adjacency for undirected walk.
        let (sym_head, sym_data) = diamond::make_symmetric(
            self.n,
            &self.adj_head,
            &self.adj_data,
        );

        // Per-node accumulators: (arrivals, phase_sum).
        let mut arrivals = vec![0u64; self.n];
        let mut phase_sum = vec![0u64; self.n];

        // Modular exponentiation helper.
        fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
            let mut result = 1u64;
            base %= modulus;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = result.wrapping_mul(base) % modulus;
                }
                exp >>= 1;
                base = base.wrapping_mul(base) % modulus;
            }
            result
        }

        for _ in 0..n_walkers {
            // Start at a random node.
            let start = rng.gen_range(0..self.n);
            let mut current = start;
            let mut step_count: u64 = 0;

            for _ in 0..steps {
                let s = sym_head[current] as usize;
                let e = sym_head[current + 1] as usize;
                if s == e { break; } // isolated node
                let idx = rng.gen_range(s..e);
                current = sym_data[idx] as usize;
                step_count += 1;

                // Accumulate g^step_count mod p at arrival node.
                let phase = mod_pow(root, step_count, prime);
                arrivals[current] += 1;
                phase_sum[current] = (phase_sum[current] + phase) % prime;
            }
        }

        // Compute intensity: |phase_sum/arrivals - p/2| / (p/2), normalized to [0, 1].
        let half_p = prime as f64 / 2.0;
        let mut results = Vec::new();
        for i in 0..self.n {
            if arrivals[i] > 0 {
                let avg_phase = phase_sum[i] as f64 / arrivals[i] as f64;
                let intensity = (avg_phase - half_p).abs() / half_p;
                results.push((i, arrivals[i], phase_sum[i], intensity));
            }
        }
        results
    }

    /// Causal slice: cut at given depth.
    ///
    /// Returns `(nodes_below, nodes_above, severed_edges)` where
    /// severed_edges are Hasse edges that cross the slice boundary.
    pub fn causal_slice(
        &self,
        depths: &[u32],
        depth: u32,
    ) -> (Vec<usize>, Vec<usize>, Vec<(u32, u32)>) {
        let mut below = Vec::new();
        let mut above = Vec::new();
        let mut severed = Vec::new();

        for i in 0..self.n {
            if depths[i] <= depth {
                below.push(i);
            } else {
                above.push(i);
            }
        }

        // Find edges that cross the boundary.
        for u in 0..self.n {
            let start = self.adj_head[u] as usize;
            let end = self.adj_head[u + 1] as usize;
            for &v in &self.adj_data[start..end] {
                let vi = v as usize;
                if (depths[u] <= depth && depths[vi] > depth)
                    || (depths[u] > depth && depths[vi] <= depth)
                {
                    severed.push((u as u32, v));
                }
            }
        }

        (below, above, severed)
    }

    /// Enumerate all directed edges (u → v) in the Hasse diagram.
    pub fn edges(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for u in 0..self.n {
            let start = self.adj_head[u] as usize;
            let end = self.adj_head[u + 1] as usize;
            for &v in &self.adj_data[start..end] {
                out.push((u as u32, v));
            }
        }
        out
    }

    /// Number of directed edges in the Hasse diagram.
    pub fn edge_count(&self) -> usize {
        self.adj_data.len()
    }
}

impl DefectData {
    /// Expose the full list of detected Causal Prisms for scene scripts.
    ///
    /// Each entry contains (origin, destination, belly_nodes, generation).
    /// Generation is classified from the bulk momentum phase signature.
    pub fn prism_list(&self) -> Vec<PrismInfo> {
        use std::collections::HashSet;
        // Build sets for each generation to classify prisms.
        let gen1_set: HashSet<usize> = self.result.gen1_nodes.iter().copied().collect();
        let gen2_set: HashSet<usize> = self.result.gen2_nodes.iter().copied().collect();
        let gen3_set: HashSet<usize> = self.result.gen3_nodes.iter().copied().collect();
        let anti1_set: HashSet<usize> = self.result.anti1_nodes.iter().copied().collect();

        self.prisms
            .iter()
            .map(|p| {
                // Classify by checking if the origin belongs to a generation set.
                let gen = if gen1_set.contains(&p.origin) { 1 }
                    else if gen2_set.contains(&p.origin) { 2 }
                    else if gen3_set.contains(&p.origin) { 3 }
                    else if anti1_set.contains(&p.origin) { -1 }
                    else { 0 };
                PrismInfo {
                    origin: p.origin,
                    destination: p.destination,
                    belly: p.intermediates.clone(),
                    generation: gen,
                }
            })
            .collect()
    }

    /// Extract prism metadata for visualisation.
    ///
    /// For the MVP we expose the generation node lists directly — the
    /// Python side can query which nodes belong to which generation.
    pub fn generation_nodes(&self) -> Vec<(Vec<usize>, i32)> {
        let mut gens = Vec::new();
        if !self.result.gen1_nodes.is_empty() {
            gens.push((self.result.gen1_nodes.clone(), 1));
        }
        if !self.result.gen2_nodes.is_empty() {
            gens.push((self.result.gen2_nodes.clone(), 2));
        }
        if !self.result.gen3_nodes.is_empty() {
            gens.push((self.result.gen3_nodes.clone(), 3));
        }
        if !self.result.anti1_nodes.is_empty() {
            gens.push((self.result.anti1_nodes.clone(), -1));
        }
        if !self.result.sterile_nodes.is_empty() {
            gens.push((self.result.sterile_nodes.clone(), 0));
        }
        gens
    }

    /// Defect CSR (after K₅ contraction).
    pub fn defect_edges(&self) -> Vec<(u32, u32)> {
        let n = self.result.def_head.len() - 1;
        let mut out = Vec::new();
        for u in 0..n {
            let start = self.result.def_head[u] as usize;
            let end = self.result.def_head[u + 1] as usize;
            for &v in &self.result.def_data[start..end] {
                out.push((u as u32, v));
            }
        }
        out
    }

    /// Merge map: `merge_into[i]` gives the canonical node after K₅
    /// contraction.  Identity for non-merged nodes.
    pub fn merge_map(&self) -> &[usize] {
        &self.result.merge_into
    }
}
