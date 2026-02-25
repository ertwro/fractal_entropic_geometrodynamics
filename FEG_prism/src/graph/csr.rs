// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Juan Pablo Silva Alvarado
// Fractal Entropic Geometrodynamics — DOI: 10.5281/zenodo.18769707

//! Type-safe Compressed Sparse Row (CSR) graph with compile-time direction tag.
//!
//! `CsrGraph<Directed>` stores forward-only edges (past → future).
//! `CsrGraph<Undirected>` stores both directions (symmetric).
//!
//! The direction marker prevents passing a directed Hasse diagram to spectral
//! walkers (which require the symmetric graph) at compile time.

use std::marker::PhantomData;

/// Zero-sized marker: edges are forward-only (u → v where t_u < t_v).
pub struct Directed;

/// Zero-sized marker: edges stored in both directions.
pub struct Undirected;

/// Compressed Sparse Row graph with direction tag.
///
/// Storage: `head[u]..head[u+1]` indexes into `data` to give neighbors of `u`.
/// `head` has length `n_nodes + 1`, `data` has length equal to total edge slots.
///
/// Neighbor lists within each row are sorted ascending for binary search.
pub struct CsrGraph<D = Directed> {
    head: Vec<u32>,
    data: Vec<u32>,
    n_nodes: usize,
    _dir: PhantomData<D>,
}

impl<D> CsrGraph<D> {
    /// Construct from pre-built CSR arrays.
    ///
    /// # Panics
    /// Panics if `head.len() != n_nodes + 1`.
    pub fn new(head: Vec<u32>, data: Vec<u32>, n_nodes: usize) -> Self {
        assert_eq!(head.len(), n_nodes + 1, "head.len() must be n_nodes + 1");
        Self { head, data, n_nodes, _dir: PhantomData }
    }

    /// Sorted neighbor slice for node `u`.
    #[inline]
    pub fn neighbors(&self, u: usize) -> &[u32] {
        let start = self.head[u] as usize;
        let end = self.head[u + 1] as usize;
        &self.data[start..end]
    }

    /// Degree of node `u`.
    #[inline]
    pub fn degree(&self, u: usize) -> usize {
        (self.head[u + 1] - self.head[u]) as usize
    }

    /// Binary-search edge test (neighbors are sorted).
    #[inline]
    pub fn has_edge(&self, u: usize, v: u32) -> bool {
        self.neighbors(u).binary_search(&v).is_ok()
    }

    /// Number of nodes in the graph.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Total number of edge slots stored. For undirected graphs this counts
    /// each undirected edge twice.
    #[inline]
    pub fn n_edge_slots(&self) -> usize {
        self.data.len()
    }

    /// Borrow the raw CSR arrays (for interop with legacy functions).
    #[inline]
    pub fn raw(&self) -> (&[u32], &[u32]) {
        (&self.head, &self.data)
    }

    /// Consume and return the raw CSR arrays.
    pub fn into_raw(self) -> (Vec<u32>, Vec<u32>) {
        (self.head, self.data)
    }

    /// Borrow head array.
    #[inline]
    pub fn head(&self) -> &[u32] {
        &self.head
    }

    /// Borrow data array.
    #[inline]
    pub fn data(&self) -> &[u32] {
        &self.data
    }
}

impl CsrGraph<Directed> {
    /// Build the symmetric (undirected) version by adding reverse edges.
    ///
    /// Each directed edge u→v produces both u↔v entries. Neighbor lists
    /// are sorted ascending.
    pub fn symmetrize(&self) -> CsrGraph<Undirected> {
        let n = self.n_nodes;
        // Count total degree for symmetric graph
        let mut deg = vec![0u32; n];
        for u in 0..n {
            for &v in self.neighbors(u) {
                deg[u] += 1;
                deg[v as usize] += 1;
            }
        }

        // Build head from degrees
        let mut head = vec![0u32; n + 1];
        for i in 0..n {
            head[i + 1] = head[i] + deg[i];
        }
        let total = head[n] as usize;
        let mut data = vec![0u32; total];

        // Fill data (using pos as write cursor)
        let mut pos = head[..n].to_vec();
        for u in 0..n {
            for &v in self.neighbors(u) {
                data[pos[u] as usize] = v;
                pos[u] += 1;
                data[pos[v as usize] as usize] = u as u32;
                pos[v as usize] += 1;
            }
        }

        // Sort each neighbor list
        for u in 0..n {
            let start = head[u] as usize;
            let end = head[u + 1] as usize;
            data[start..end].sort_unstable();
        }

        CsrGraph::new(head, data, n)
    }

    /// Build a reversed directed graph (swap edge direction).
    pub fn reverse(&self) -> CsrGraph<Directed> {
        let n = self.n_nodes;
        let mut deg = vec![0u32; n];
        for &v in &self.data {
            deg[v as usize] += 1;
        }

        let mut head = vec![0u32; n + 1];
        for i in 0..n {
            head[i + 1] = head[i] + deg[i];
        }
        let mut data = vec![0u32; self.data.len()];
        let mut pos = head[..n].to_vec();
        for u in 0..n {
            for &v in self.neighbors(u) {
                data[pos[v as usize] as usize] = u as u32;
                pos[v as usize] += 1;
            }
        }

        for u in 0..n {
            let start = head[u] as usize;
            let end = head[u + 1] as usize;
            data[start..end].sort_unstable();
        }

        CsrGraph::new(head, data, n)
    }
}

impl CsrGraph<Undirected> {
    /// Extract unique edges (u < v) for eigendecomposition.
    pub fn unique_edges(&self) -> Vec<(u32, u32)> {
        let mut edges = Vec::new();
        for u in 0..self.n_nodes {
            for &v in self.neighbors(u) {
                if (u as u32) < v {
                    edges.push((u as u32, v));
                }
            }
        }
        edges
    }
}

/// Build a directed CSR from unsorted (row, col) edge pairs.
///
/// The resulting graph has `n_nodes` nodes and sorted neighbor lists.
pub fn build_directed_csr(n_nodes: usize, rows: &[u32], cols: &[u32]) -> CsrGraph<Directed> {
    let mut head = vec![0u32; n_nodes + 1];
    for &r in rows {
        head[r as usize + 1] += 1;
    }
    for i in 0..n_nodes {
        head[i + 1] += head[i];
    }
    let mut data = vec![0u32; rows.len()];
    let mut pos = head.clone();
    for (&r, &c) in rows.iter().zip(cols) {
        data[pos[r as usize] as usize] = c;
        pos[r as usize] += 1;
    }
    // Sort each neighbor list
    for u in 0..n_nodes {
        let start = head[u] as usize;
        let end = head[u + 1] as usize;
        data[start..end].sort_unstable();
    }
    CsrGraph::new(head, data, n_nodes)
}

/// Build an undirected CSR from unsorted (row, col) edge pairs where u < v.
///
/// Automatically adds reverse edges.
pub fn build_undirected_csr(n_nodes: usize, rows: &[u32], cols: &[u32]) -> CsrGraph<Undirected> {
    let mut deg = vec![0u32; n_nodes];
    for (&r, &c) in rows.iter().zip(cols) {
        deg[r as usize] += 1;
        deg[c as usize] += 1;
    }
    let mut head = vec![0u32; n_nodes + 1];
    for i in 0..n_nodes {
        head[i + 1] = head[i] + deg[i];
    }
    let total = head[n_nodes] as usize;
    let mut data = vec![0u32; total];
    let mut pos = head[..n_nodes].to_vec();
    for (&r, &c) in rows.iter().zip(cols) {
        data[pos[r as usize] as usize] = c;
        pos[r as usize] += 1;
        data[pos[c as usize] as usize] = r;
        pos[c as usize] += 1;
    }
    for u in 0..n_nodes {
        let start = head[u] as usize;
        let end = head[u + 1] as usize;
        data[start..end].sort_unstable();
    }
    CsrGraph::new(head, data, n_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directed_basics() {
        // Triangle: 0→1, 0→2, 1→2
        let g = build_directed_csr(3, &[0, 0, 1], &[1, 2, 2]);
        assert_eq!(g.n_nodes(), 3);
        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 1);
        assert_eq!(g.degree(2), 0);
        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(0, 2));
        assert!(!g.has_edge(2, 0));
    }

    #[test]
    fn symmetrize_directed() {
        let g = build_directed_csr(3, &[0, 0, 1], &[1, 2, 2]);
        let s = g.symmetrize();
        assert_eq!(s.n_nodes(), 3);
        assert_eq!(s.degree(0), 2);
        assert_eq!(s.degree(1), 2);
        assert_eq!(s.degree(2), 2);
        assert!(s.has_edge(2, 0));
        assert!(s.has_edge(2, 1));
    }

    #[test]
    fn unique_edges_undirected() {
        let s = build_undirected_csr(3, &[0, 0, 1], &[1, 2, 2]);
        let edges = s.unique_edges();
        assert_eq!(edges.len(), 3);
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(0, 2)));
        assert!(edges.contains(&(1, 2)));
    }

    #[test]
    fn reverse_directed() {
        let g = build_directed_csr(3, &[0, 0, 1], &[1, 2, 2]);
        let r = g.reverse();
        assert_eq!(r.degree(0), 0);
        assert_eq!(r.degree(1), 1);
        assert_eq!(r.degree(2), 2);
        assert!(r.has_edge(1, 0));
        assert!(r.has_edge(2, 0));
        assert!(r.has_edge(2, 1));
    }
}
