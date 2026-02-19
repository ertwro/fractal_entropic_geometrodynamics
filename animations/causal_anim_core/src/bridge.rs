//! Bridge to `causal_set_sim` — wraps sprinkle, Hasse, and Kuratowski defect.

use causal_set_sim::diamond;
use causal_set_sim::skyrmion::{self, DefectResult, TopologySummary};
use rand::rngs::StdRng;
use rand::SeedableRng;

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
        let (result, topology) = skyrmion::apply_defect(
            self.n,
            self.adj_head.clone(),
            self.adj_data.clone(),
            self.bulk_momentum.clone(),
        );
        DefectData { result, topology }
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
    /// Extract prism metadata for visualisation.
    ///
    /// Walks the generation node lists and groups nodes by the prisms they
    /// belong to (poles + belly).  Because `CausalPrism` is private in
    /// `skyrmion`, we reconstruct the prism structure from the defect CSR:
    /// a prism exists wherever two core nodes share ≥ 3 common neighbours
    /// in the vacuum CSR.
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
