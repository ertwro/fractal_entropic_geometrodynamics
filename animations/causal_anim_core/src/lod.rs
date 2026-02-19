//! Level-of-detail manager.
//!
//! Selects which nodes and edges to render based on the camera's visible
//! region and the total count of elements in the frustum.

/// Visual detail level, selected automatically from node count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailLevel {
    /// > 10⁶ visible nodes: render as density heatmap.
    Cosmic,
    /// 10⁴ – 10⁶: instanced points only, no edges.
    Galactic,
    /// 10³ – 10⁴: points + major edges.
    Stellar,
    /// < 10³: full nodes, all edges, labels.
    Atomic,
    /// Single prism in focus: full K_{2,N} detail.
    PrismFocus,
}

impl DetailLevel {
    pub fn from_visible_count(count: usize) -> Self {
        match count {
            0..=999 => DetailLevel::Atomic,
            1_000..=9_999 => DetailLevel::Stellar,
            10_000..=999_999 => DetailLevel::Galactic,
            _ => DetailLevel::Cosmic,
        }
    }

    /// Whether individual edges should be drawn at this level.
    pub fn draw_edges(self) -> bool {
        matches!(self, DetailLevel::Atomic | DetailLevel::Stellar | DetailLevel::PrismFocus)
    }

    /// Whether node labels / annotations are visible.
    pub fn draw_labels(self) -> bool {
        matches!(self, DetailLevel::Atomic | DetailLevel::PrismFocus)
    }

    /// Base node radius multiplier.
    pub fn node_radius_scale(self) -> f32 {
        match self {
            DetailLevel::PrismFocus => 3.0,
            DetailLevel::Atomic => 1.5,
            DetailLevel::Stellar => 0.8,
            DetailLevel::Galactic => 0.3,
            DetailLevel::Cosmic => 0.1,
        }
    }
}

/// Camera-axis-aligned bounding box for frustum culling.
pub struct ViewBounds {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
}

impl ViewBounds {
    /// Build from camera centre + zoom (half-extents) + aspect ratio.
    pub fn from_camera(cx: f32, cy: f32, zoom: f32, aspect: f32) -> Self {
        let half_w = zoom * aspect;
        let half_h = zoom;
        ViewBounds {
            x_min: cx - half_w,
            x_max: cx + half_w,
            y_min: cy - half_h,
            y_max: cy + half_h,
        }
    }

    /// Test whether a 2D point is inside (uses X and Y of position).
    pub fn contains(&self, pos: &[f32; 3]) -> bool {
        pos[0] >= self.x_min
            && pos[0] <= self.x_max
            && pos[1] >= self.y_min
            && pos[1] <= self.y_max
    }
}

/// Filter nodes and edges to those within the view bounds.
///
/// Returns `(visible_node_indices, visible_edge_indices)` where edge
/// indices refer to the flat edge list returned by `CausalGraph::edges()`.
pub fn cull(
    positions: &[[f32; 3]],
    edges: &[(u32, u32)],
    bounds: &ViewBounds,
) -> (Vec<usize>, Vec<usize>) {
    let node_visible: Vec<bool> = positions.iter().map(|p| bounds.contains(p)).collect();

    let visible_nodes: Vec<usize> = node_visible
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None })
        .collect();

    let visible_edges: Vec<usize> = edges
        .iter()
        .enumerate()
        .filter_map(|(i, &(u, v))| {
            if node_visible[u as usize] || node_visible[v as usize] {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    (visible_nodes, visible_edges)
}
