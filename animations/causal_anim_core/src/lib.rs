//! CausalAnim Core — GPU-accelerated animation engine for the Kuratowski
//! Calculus.
//!
//! This crate provides:
//!   - **bridge**: connection to the `causal_set_sim` simulation engine
//!   - **layout**: stratified spring layout for Hasse diagrams
//!   - **timeline**: dual-clock (causal ticks / presentation seconds)
//!   - **lod**: level-of-detail culling
//!   - **renderer**: headless wgpu renderer (nodes + edges → PNG frames)
//!
//! The Python package `causal_anim` calls into this crate via PyO3.

pub mod bridge;
pub mod layout;
pub mod lod;
pub mod renderer;
pub mod timeline;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use bridge::{CausalGraph, DefectData};
use layout::LayoutEngine;
use renderer::{EdgeInstance, NodeInstance, Renderer};
use timeline::Timeline;

/// The main engine object exposed to Python.
///
/// Holds the causal graph, layout, defect data, and GPU renderer.
/// Python builds the scene declaratively and then calls rendering
/// methods to produce frames.
#[pyclass]
pub struct SceneEngine {
    graph: Option<CausalGraph>,
    defect: Option<DefectData>,
    layout: Option<LayoutEngine>,
    renderer: Renderer,
    edges_cache: Vec<(u32, u32)>,
    timeline: Timeline,
}

#[pymethods]
impl SceneEngine {
    /// Create a new scene engine with the given frame resolution.
    #[new]
    fn new(width: u32, height: u32) -> Self {
        SceneEngine {
            graph: None,
            defect: None,
            layout: None,
            renderer: Renderer::new(width, height),
            edges_cache: Vec::new(),
            timeline: Timeline::new(),
        }
    }

    /// Sprinkle N events and build the Hasse diagram.
    fn build_universe(&mut self, n: usize, seed: u64) {
        let graph = CausalGraph::sprinkle_and_build(n, seed);
        self.edges_cache = graph.edges();
        self.layout = Some(LayoutEngine::new(&graph));
        self.graph = Some(graph);
    }

    /// Run Kuratowski defect: detect Causal Prisms, classify generations,
    /// contract K₅ threats.
    fn apply_defect(&mut self) {
        let graph = self.graph.as_ref().expect("call build_universe first");
        self.defect = Some(graph.apply_defect());
    }

    /// Run spring relaxation on the layout.
    fn relax_layout(&mut self, iterations: u32) {
        let graph = self.graph.as_ref().expect("call build_universe first");
        let layout = self.layout.as_mut().expect("layout not initialised");
        layout.relax(iterations as usize, &graph.adj_head, &graph.adj_data);
    }

    /// Number of nodes in the current graph.
    fn node_count(&self) -> usize {
        self.graph.as_ref().map(|g| g.n).unwrap_or(0)
    }

    /// Number of edges in the current graph.
    fn edge_count(&self) -> usize {
        self.edges_cache.len()
    }

    /// Get flat list of node positions [x0,y0,z0, x1,y1,z1, ...].
    fn get_positions_flat(&self) -> Vec<f32> {
        self.layout
            .as_ref()
            .map(|l| l.positions.iter().flat_map(|p| p.iter().copied()).collect())
            .unwrap_or_default()
    }

    /// Get edges as flat list [u0,v0, u1,v1, ...].
    fn get_edges_flat(&self) -> Vec<u32> {
        self.edges_cache
            .iter()
            .flat_map(|&(u, v)| [u, v])
            .collect()
    }

    /// Get generation node indices: returns list of (node_indices, gen_id).
    fn get_generation_nodes(&self) -> Vec<(Vec<usize>, i32)> {
        self.defect
            .as_ref()
            .map(|d| d.generation_nodes())
            .unwrap_or_default()
    }

    /// Max causal depth in the layout.
    fn max_depth(&self) -> u32 {
        self.layout.as_ref().map(|l| l.max_depth()).unwrap_or(0)
    }

    /// Render a single frame to PNG bytes.
    ///
    /// Arguments are flat arrays to minimise PyO3 overhead:
    ///   node_data: [x,y,z,radius, r,g,b,a, ...]  (8 floats per node)
    ///   edge_data: [sx,sy,sz,width, ex,ey,ez,pad, r,g,b,a, ...] (12 floats per edge)
    ///   camera: [cx, cy, zoom]
    ///   bg: [r, g, b]  (0.0–1.0)
    fn render_png<'py>(
        &self,
        py: Python<'py>,
        node_data: Vec<f32>,
        edge_data: Vec<f32>,
        camera: [f32; 3],
        bg: [f64; 3],
    ) -> Bound<'py, PyBytes> {
        let nodes: Vec<NodeInstance> = node_data
            .chunks_exact(8)
            .map(|c| NodeInstance {
                position: [c[0], c[1], c[2]],
                radius: c[3],
                color: [c[4], c[5], c[6], c[7]],
            })
            .collect();

        let edges: Vec<EdgeInstance> = edge_data
            .chunks_exact(12)
            .map(|c| EdgeInstance {
                start: [c[0], c[1], c[2]],
                width: c[3],
                end: [c[4], c[5], c[6]],
                _pad: 0.0,
                color: [c[8], c[9], c[10], c[11]],
            })
            .collect();

        let png = self.renderer.render_png(
            &nodes,
            &edges,
            [camera[0], camera[1]],
            camera[2],
            bg,
        );
        PyBytes::new(py, &png)
    }

    /// Convenience: render the full graph with uniform styling.
    ///
    /// Returns PNG bytes.  Good for quick previews.
    fn render_full<'py>(
        &self,
        py: Python<'py>,
        node_color: [f32; 4],
        node_radius: f32,
        edge_color: [f32; 4],
        edge_width: f32,
        camera_x: f32,
        camera_y: f32,
        camera_zoom: f32,
        bg: [f64; 3],
    ) -> Bound<'py, PyBytes> {
        let layout = self.layout.as_ref().expect("no layout");
        let nodes: Vec<NodeInstance> = layout
            .positions
            .iter()
            .map(|&pos| NodeInstance { position: pos, radius: node_radius, color: node_color })
            .collect();

        let edges: Vec<EdgeInstance> = self
            .edges_cache
            .iter()
            .map(|&(u, v)| {
                let s = layout.positions[u as usize];
                let e = layout.positions[v as usize];
                EdgeInstance { start: s, width: edge_width, end: e, _pad: 0.0, color: edge_color }
            })
            .collect();

        let png = self.renderer.render_png(
            &nodes,
            &edges,
            [camera_x, camera_y],
            camera_zoom,
            bg,
        );
        PyBytes::new(py, &png)
    }

    // ─── New bridge methods for Phase 4 animations ────────────────────────

    /// Simulate cover time on a single prism subgraph.
    ///
    /// Returns the number of lazy-walk ticks to visit all belly nodes.
    fn prism_cover_time(&self, origin: usize, belly: Vec<usize>, dest: usize, seed: u64) -> u32 {
        let graph = self.graph.as_ref().expect("call build_universe first");
        graph.prism_cover_time(origin, &belly, dest, seed)
    }

    /// Simulate cover time on a prism subgraph, returning the full trajectory.
    ///
    /// Returns the sequence of visited node indices (one per tick).
    fn prism_cover_trajectory(&self, origin: usize, belly: Vec<usize>, dest: usize, seed: u64) -> Vec<usize> {
        let graph = self.graph.as_ref().expect("call build_universe first");
        graph.prism_cover_trajectory(origin, &belly, dest, seed)
    }

    /// K₃,₃ probe: for a prism, identify which external neighbour edges
    /// are blocked by K₃,₃ vs accepted.
    ///
    /// Returns (accepted_node_indices, rejected_node_indices).
    fn k33_probe(&self, origin: usize, dest: usize, belly: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
        let graph = self.graph.as_ref().expect("call build_universe first");
        graph.k33_probe(origin, dest, &belly)
    }

    /// Modulo interference: run walkers accumulating g^S mod p phase.
    ///
    /// Returns per-node (node_index, arrivals, phase_sum, intensity).
    fn modulo_interference(
        &self,
        n_walkers: usize,
        steps: usize,
        prime: u64,
        root: u64,
        seed: u64,
    ) -> Vec<(usize, u64, u64, f64)> {
        let graph = self.graph.as_ref().expect("call build_universe first");
        graph.modulo_interference(n_walkers, steps, prime, root, seed)
    }

    /// Causal slice: cut at given depth.
    ///
    /// Returns (nodes_below, nodes_above, severed_edges).
    fn causal_slice(&self, depth: u32) -> (Vec<usize>, Vec<usize>, Vec<(u32, u32)>) {
        let graph = self.graph.as_ref().expect("call build_universe first");
        let layout = self.layout.as_ref().expect("layout not initialised");
        // Build depth array from layout.
        let n = graph.n;
        let depths: Vec<u32> = (0..n).map(|i| layout.depth(i)).collect();
        graph.causal_slice(&depths, depth)
    }

    /// Get all detected prisms: returns list of (origin, dest, belly, generation).
    fn get_prisms(&self) -> Vec<(usize, usize, Vec<usize>, i32)> {
        self.defect
            .as_ref()
            .map(|d| {
                d.prism_list()
                    .into_iter()
                    .map(|p| (p.origin, p.destination, p.belly, p.generation))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get node depth (causal layer) for a specific node.
    fn node_depth(&self, node: usize) -> u32 {
        self.layout.as_ref().map(|l| l.depth(node)).unwrap_or(0)
    }

    // ─── Timeline pass-through ───────────────────────────────────────────

    fn timeline_rush(&mut self, ticks: u64, duration_secs: f64) {
        self.timeline.rush(ticks, duration_secs);
    }
    fn timeline_slow_motion(&mut self, ticks: u64, duration_secs: f64) {
        self.timeline.slow_motion(ticks, duration_secs);
    }
    fn timeline_pause(&mut self, duration_secs: f64) {
        self.timeline.pause(duration_secs);
    }
    fn timeline_set_pace(&mut self, ticks_per_second: f64) {
        self.timeline.set_pace(ticks_per_second);
    }
    fn timeline_total_duration(&self) -> f64 {
        self.timeline.total_duration()
    }
}

/// Python module entry point.
#[pymodule]
fn causal_anim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SceneEngine>()?;
    Ok(())
}
