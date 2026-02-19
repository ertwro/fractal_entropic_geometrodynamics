//! Stratified layout engine.
//!
//! Assigns node positions so that the vertical axis (Y) faithfully encodes
//! causal depth, while horizontal axes (X, Z) are relaxed via a spring
//! model within each causal layer.

use crate::bridge::CausalGraph;

/// Node positions and causal depth metadata.
pub struct LayoutEngine {
    pub positions: Vec<[f32; 3]>,
    depths: Vec<u32>,
    layers: Vec<Vec<usize>>,
}

impl LayoutEngine {
    /// Build initial layout from a `CausalGraph`.
    ///
    /// 1. Computes causal depth via reverse-adjacency DP.
    /// 2. Sets Y = depth (scaled).
    /// 3. Sets X from the spatial x-coordinate of the 4D sprinkle point,
    ///    providing a physically-motivated initial horizontal spread.
    /// 4. Sets Z from the spatial y-coordinate (used in 3D mode).
    pub fn new(graph: &CausalGraph) -> Self {
        let n = graph.n;

        // --- Compute causal depths ---
        // Build parent lists (reverse adjacency).
        let mut parents: Vec<Vec<usize>> = vec![vec![]; n];
        for u in 0..n {
            let start = graph.adj_head[u] as usize;
            let end = graph.adj_head[u + 1] as usize;
            for &v in &graph.adj_data[start..end] {
                parents[v as usize].push(u);
            }
        }

        // Forward DP: depth[v] = max(depth[parent] + 1).
        // Nodes are already time-sorted (sprinkle guarantees this after
        // build_hasse_sparse/direct which re-orders by time).
        let mut depths = vec![0u32; n];
        for v in 0..n {
            for &u in &parents[v] {
                depths[v] = depths[v].max(depths[u] + 1);
            }
        }

        // Group nodes by layer.
        let max_depth = depths.iter().copied().max().unwrap_or(0);
        let mut layers: Vec<Vec<usize>> = vec![vec![]; (max_depth + 1) as usize];
        for (i, &d) in depths.iter().enumerate() {
            layers[d as usize].push(i);
        }

        // --- Initial positions ---
        let y_scale = 1.0_f32;
        let mut positions = vec![[0.0f32; 3]; n];
        for i in 0..n {
            positions[i][0] = graph.coords[i][1] as f32; // x from 4D
            positions[i][1] = depths[i] as f32 * y_scale; // y = causal depth
            positions[i][2] = graph.coords[i][2] as f32; // z from 4D
        }

        LayoutEngine { positions, depths, layers }
    }

    /// Spring relaxation: repulsion within layers, attraction along Hasse
    /// edges, centering force.  Only X and Z coordinates are modified;
    /// Y (causal depth) is immutable.
    pub fn relax(&mut self, iterations: usize, adj_head: &[u32], adj_data: &[u32]) {
        let n = self.positions.len();
        let c_rep = 0.5_f32;
        let c_att = 0.1_f32;
        let c_grav = 0.01_f32;
        let dt = 0.05_f32;

        for _ in 0..iterations {
            let mut fx = vec![0.0f32; n];
            let mut fz = vec![0.0f32; n];

            // Repulsion within same layer (Coulomb).
            for layer in &self.layers {
                let len = layer.len();
                for i in 0..len {
                    let a = layer[i];
                    for j in (i + 1)..len {
                        let b = layer[j];
                        let dx = self.positions[a][0] - self.positions[b][0];
                        let dz = self.positions[a][2] - self.positions[b][2];
                        let dist_sq = dx * dx + dz * dz + 1e-4;
                        let inv_dist = 1.0 / dist_sq.sqrt();
                        let f = c_rep / dist_sq;
                        fx[a] += dx * inv_dist * f;
                        fz[a] += dz * inv_dist * f;
                        fx[b] -= dx * inv_dist * f;
                        fz[b] -= dz * inv_dist * f;
                    }
                }
            }

            // Attraction along Hasse edges (spring).
            for u in 0..n {
                let start = adj_head[u] as usize;
                let end = adj_head[u + 1] as usize;
                for &v in &adj_data[start..end] {
                    let v = v as usize;
                    let dx = self.positions[v][0] - self.positions[u][0];
                    let dz = self.positions[v][2] - self.positions[u][2];
                    fx[u] += dx * c_att;
                    fz[u] += dz * c_att;
                    fx[v] -= dx * c_att;
                    fz[v] -= dz * c_att;
                }
            }

            // Centering force.
            for i in 0..n {
                fx[i] -= self.positions[i][0] * c_grav;
                fz[i] -= self.positions[i][2] * c_grav;
            }

            // Integrate (Euler).  Y is frozen.
            for i in 0..n {
                self.positions[i][0] += fx[i] * dt;
                self.positions[i][2] += fz[i] * dt;
            }
        }
    }

    pub fn depth(&self, node: usize) -> u32 {
        self.depths[node]
    }

    pub fn max_depth(&self) -> u32 {
        self.depths.iter().copied().max().unwrap_or(0)
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, depth: u32) -> &[usize] {
        &self.layers[depth as usize]
    }
}
