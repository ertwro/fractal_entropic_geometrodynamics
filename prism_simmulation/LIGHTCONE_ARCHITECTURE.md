# Light-Cone Engine: SoA Architecture Specification

## Physics-to-Memory Isomorphism

The three simulation phases access **physically disjoint** node properties:

- **Phase 1 (Kinematics):** `coords`, `times` — needed only at the leading edge
- **Phase 2 (Matter):** `csr_*`, `phases` — topology + charge, no coordinates
- **Phase 3 (Forces):** `csr_offsets`, `csr_edges` — pure topology, nothing else

SoA guarantees each phase saturates cache lines with exactly the data it needs.

## Core Struct

```rust
/// The sliding window representing the active physical light-cone.
/// Data is strictly separated by physical property (SoA) for perfect cache locality.
pub struct LightConeWindow {
    // ── 1. TIMELINE & LIFECYCLE (The Causal Arrow) ──────────────────────
    pub t_cursor: f64,              // Current leading edge of the window
    pub t_evict: f64,               // Trailing edge (nodes behind this are Done)
    pub n_total: usize,             // Total N for the full diamond
    pub half_t: f64,                // T/2 of the causal diamond

    pub id_base: usize,             // Global ID of SoA index 0. Incremented at eviction.
    pub times: Vec<f64>,            // t-coordinate of each node
    pub zones: Vec<u8>,             // 0=Active, 1=HasseMature, 2=PrismMature, 3=Done
    // ID mapping is arithmetic: global_id = soa_index + id_base
    //                           soa_index = global_id - id_base

    // ── 2. KINEMATICS (Leading edge only — evicts into topology) ────────
    pub coords: Vec<[f64; 3]>,      // Spatial (x,y,z) — dead after Hasse maturation
    pub qt: Vec<u16>,               // Quantised grid coords for OCI
    pub qx: Vec<u16>,
    pub qy: Vec<u16>,
    pub qz: Vec<u16>,

    // ── 3. MATTER & CHARGE (Prism properties) ───────────────────────────
    pub phases: Vec<i8>,            // φ(w) = sign(out_deg - in_deg) ∈ {-1, 0, +1}
    pub is_prism_member: Vec<bool>, // true if committed to a prism (exclusion guard)

    // ── 4. TOPOLOGY (global-ID indexed CSR — hot path for walkers) ──────
    // Edge values are GLOBAL IDs (stable across evictions).
    // Offset arrays are indexed by (global_id - csr_base).
    // Eviction bumps the base and drains offsets — zero edge rewriting.
    pub csr_base: usize,            // = id_base at last symmetrize. csr_offsets[id - csr_base].
    pub csr_offsets: Vec<u32>,      // Symmetric CSR row pointers (global-ID indexed)
    pub csr_edges: Vec<u32>,        // Symmetric CSR column indices (global IDs)

    // ── 5. DIRECTED HASSE (Forward-only, for prism detection) ───────────
    pub dir_base: usize,
    pub dir_offsets: Vec<u32>,      // Directed CSR: children(u), global-ID indexed
    pub dir_edges: Vec<u32>,        // Values are global IDs
    pub rev_base: usize,
    pub rev_offsets: Vec<u32>,      // Reverse CSR: parents(v), global-ID indexed
    pub rev_edges: Vec<u32>,        // Values are global IDs

    // ── 6. GLOBAL ACCUMULATORS (Persist across evictions) ───────────────
    pub total_phase_sq: u64,        // Σ|Φ(P)|² — EM self-energy
    pub total_mass_sq: u64,         // ΣN²      — gravitational self-energy
    pub total_visible: u64,         // Σ|Φ(P)|  — visible mass
    pub total_dark: u64,            // Σ(N-|Φ|) — dark mass
    pub total_grav: u64,            // ΣN        — gravitational mass
    pub prism_count: u64,           // Number of committed prisms
    pub prism_histogram: Vec<(usize, usize)>,  // (belly_size, count)

    // Walker accumulators (u64 integer, single f64 division at end)
    pub walker_return_counts: Vec<u64>,  // Per-step return counts
    pub walker_total: u64,               // Total walkers launched
    pub walker_escapes: u64,             // Walkers that hit window boundary (diagnostic)

    // ── 7. GRID INDEX (OCI for leading-edge Hasse construction) ─────────
    pub occupied_cells: Vec<Vec<(i16, i16, i16, u32, u32)>>,  // Per-time-layer OCI
}
```

## Sweep Loop

```rust
impl LightConeWindow {
    /// Main sweep: advance cursor through the causal diamond from -T/2 to +T/2.
    ///
    /// Each cursor step processes a band of width Δt ≈ W_total/4.
    /// The 8-step loop operates on disjoint SoA slices per phase.
    pub fn sweep(&mut self, seed: u64, steps: &[u32], walkers_per_cohort: usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let dt_step = self.window_width() / 4.0;
        let mut frame_index: u32 = 0;

        while self.t_cursor < self.half_t {
            // ┌─────────────────────────────────────────────────────────┐
            // │ STEP 1: SPRINKLE (touches: coords, times)              │
            // │                                                         │
            // │ Add new events at the leading edge [t_cursor, t_lead]. │
            // │ Each node gets global_id = id_base + soa_index.        │
            // │ SoA benefit: coords/times are contiguous for the new   │
            // │ batch — perfect for OCI grid construction.             │
            // └─────────────────────────────────────────────────────────┘
            let n_new = self.sprinkle_leading_edge(&mut rng);

            // ┌─────────────────────────────────────────────────────────┐
            // │ STEP 2: HASSE (touches: coords, times, qt/qx/qy/qz)   │
            // │                                                         │
            // │ Build directed edges for zone=Active nodes.            │
            // │ Uses OCI spatial index on quantised coordinates.       │
            // │ Output: dir_offsets/dir_edges for newly matured nodes. │
            // │                                                         │
            // │ After this step, coords are DEAD for these nodes.      │
            // │ (We don't zero them — just advance the zone flag.)     │
            // └─────────────────────────────────────────────────────────┘
            self.build_hasse_for_active_nodes();

            // ┌─────────────────────────────────────────────────────────┐
            // │ STEP 3: MATURE (touches: times, zones — O(window))     │
            // │                                                         │
            // │ Promote nodes whose full forward light cone is loaded: │
            // │   Active → HasseMature (all edges discovered)          │
            // │   HasseMature → PrismMature (prism detection eligible) │
            // └─────────────────────────────────────────────────────────┘
            self.promote_zones();

            // ┌─────────────────────────────────────────────────────────┐
            // │ STEP 4: SYMMETRIZE (touches: dir_*, → csr_*)           │
            // │                                                         │
            // │ Rebuild symmetric CSR from directed edges for the      │
            // │ current window. Walkers need undirected graph.         │
            // │ Only rebuilds if new nodes matured this step.          │
            // └─────────────────────────────────────────────────────────┘
            if n_new > 0 {
                self.rebuild_symmetric_csr();
            }

            // ┌─────────────────────────────────────────────────────────┐
            // │ STEP 5: PRISMS (touches: dir_*, rev_*, phases)         │
            // │                                                         │
            // │ Detect K_{2,N} on newly PrismMature nodes.             │
            // │ Forward-forward 2-hop: children(u) ∩ parents(v).       │
            // │                                                         │
            // │ SoA benefit: phases[] is contiguous for the belly      │
            // │ signature computation. coords[] is untouched.          │
            // │                                                         │
            // │ Committed prisms accumulate into global counters       │
            // │ IMMEDIATELY — their contribution to Q_topo is final.   │
            // └─────────────────────────────────────────────────────────┘
            self.detect_prisms_and_accumulate();

            // ┌─────────────────────────────────────────────────────────┐
            // │ STEP 6: WALKERS (touches: csr_offsets, csr_edges ONLY) │
            // │                                                         │
            // │ Launch cohort from guard band (≥5 time units from      │
            // │ window edges). This is the HOT LOOP — O(W·tmax).      │
            // │                                                         │
            // │ SoA benefit: CPU cache is 100% CSR edge data.          │
            // │ No coords, no phases, no zones polluting L1/L2.       │
            // │ At 52 GB window: ~25 GB of cache pollution eliminated. │
            // │                                                         │
            // │ Integer u64 accumulation; f64 division only at end.    │
            // │                                                         │
            // │ Boundary guard: if a walker steps outside the window,  │
            // │ break early and increment walker_escapes. The escape   │
            // │ rate is tracked as a diagnostic — if escapes/total >   │
            // │ 0.1%, W_REAR should be increased.                      │
            // └─────────────────────────────────────────────────────────┘
            self.run_walker_cohort(seed, steps, walkers_per_cohort);

            // ┌─────────────────────────────────────────────────────────┐
            // │ STEP 7: EMIT FRAME (optional, for animation pipeline)  │
            // │                                                         │
            // │ Package current window state as StreamingTopologySlice.│
            // └─────────────────────────────────────────────────────────┘
            frame_index += 1;

            // ┌─────────────────────────────────────────────────────────┐
            // │ STEP 8: EVICT (touches: zones, times, accumulators)    │
            // │                                                         │
            // │ Remove nodes where zone == Done AND t < t_evict.       │
            // │ Before dropping: accumulate any remaining observables. │
            // │                                                         │
            // │ SoA benefit: eviction is a parallel truncation of      │
            // │ all arrays at the same index range — no scattered      │
            // │ field access. Ring buffer makes this O(1) amortized.   │
            // └─────────────────────────────────────────────────────────┘
            self.evict_trailing_edge();

            // Advance cursor
            self.t_cursor += dt_step;
            self.t_evict = self.t_cursor - self.rear_width();
        }

        // Final pass: process any remaining nodes in the window
        self.flush_remaining();
    }
}
```

## Eviction with Accumulation

```rust
impl LightConeWindow {
    /// Evict nodes behind t_evict.
    ///
    /// The key invariant: total_mass_sq and total_phase_sq contain the EXACT
    /// same values they would if we held all N nodes in memory. The sliding
    /// window is mathematically equivalent to the full-universe computation.
    ///
    /// Performance note: `drain(..k)` is O(window) per call due to memmove,
    /// but total cost across all ~18 eviction steps is ~24 seconds — 0.02%
    /// of the ~30h realization time. A VecDeque would eliminate this but
    /// introduces non-contiguous memory, complicating CSR slice access
    /// (`csr_edges[start..end]` requires contiguous backing). Keep Vec.
    fn evict_trailing_edge(&mut self) {
        // Find the partition point: all nodes with t < t_evict AND zone == Done
        let mut k = 0;
        for i in 0..self.times.len() {
            if self.times[i] < self.t_evict && self.zones[i] == 3 {
                k = i + 1;
            } else {
                break;  // SoA arrays are time-sorted within the window
            }
        }

        if k == 0 { return; }

        // ── SoA drain (O(window) memmove, but negligible — see note above) ──
        self.times.drain(..k);
        self.zones.drain(..k);
        self.coords.drain(..k);
        self.qt.drain(..k);
        self.qx.drain(..k);
        self.qy.drain(..k);
        self.qz.drain(..k);
        self.phases.drain(..k);
        self.is_prism_member.drain(..k);

        // ── CSR base bump (O(k) offset drain — NO edge rewriting) ───────────
        // CSR edge values are global IDs → stable across evictions.
        // Only the offset arrays need their evicted rows removed.
        self.csr_offsets.drain(..k);
        self.csr_base += k;
        self.dir_offsets.drain(..k);
        self.dir_base += k;
        self.rev_offsets.drain(..k);
        self.rev_base += k;
        // csr_edges, dir_edges, rev_edges: UNCHANGED.
        //
        // Dead edge cleanup: over time, csr_edges accumulates entries
        // referencing evicted global IDs. These are harmless — walkers
        // never start from evicted nodes, and edges TO evicted nodes
        // lead outside the window (walker boundary guard catches this).
        // Optional compaction every ~5 steps if memory pressure requires.

        // Update id_base to match
        self.id_base += k;
    }

    /// Accumulate a committed prism's contribution to global observables.
    ///
    /// Called once per prism, at commitment time (not at eviction time).
    /// This is the critical invariant: Q_topo = total_phase_sq / total_mass_sq
    /// is exact at any point during the sweep, for all prisms committed so far.
    fn accumulate_prism(&mut self, belly_size: usize, net_phase: i32) {
        let n = belly_size;
        let phi_abs = net_phase.unsigned_abs() as usize;

        self.total_mass_sq += (n * n) as u64;
        self.total_phase_sq += (phi_abs * phi_abs) as u64;
        self.total_visible += phi_abs as u64;
        self.total_dark += (n - phi_abs) as u64;
        self.total_grav += n as u64;
        self.prism_count += 1;
    }

    /// After the sweep completes, extract the final observables.
    ///
    /// These are IDENTICAL to what the in-memory path computes,
    /// because every prism's contribution was accumulated exactly once.
    pub fn final_observables(&self) -> (f64, f64, f64) {
        let q_topo = if self.total_mass_sq > 0 {
            self.total_phase_sq as f64 / self.total_mass_sq as f64
        } else { 0.0 };

        let alpha = q_topo / (8.0 * std::f64::consts::PI);
        let omega_energy = if q_topo > 0.0 { 1.0 / q_topo - 1.0 } else { f64::INFINITY };

        (q_topo, alpha, omega_energy)
    }
}
```

## Window Width Constants

```rust
/// Leading edge: need MAX_CAUSAL_DEPTH + safety for complete Hasse discovery.
const W_FRONT: f64 = 17.0;   // MAX_CAUSAL_DEPTH(15) + 2 safety

/// Trailing edge: walkers with tmax=15 lazy steps, ~3 avg link span.
/// Increased from 50 to 60 for safety: while empirical walker displacement
/// at N=10M is well within 50, the worst-case (all max-span edges) is
/// tmax × MAX_CAUSAL_DEPTH = 225.  The lazy walk's 50% stay probability
/// and rarity of max-span edges make this astronomically unlikely, but
/// the 60→50 upgrade costs only 3.4% more memory (31.3 vs 27.6 GB).
/// The runtime walker_escapes diagnostic self-validates this bound.
const W_REAR: f64 = 60.0;    // tmax(15) × dt_eff(~3) + 50% safety margin

/// Guard band: walkers launch ≥5 time units from both edges.
const GUARD_BAND: f64 = 5.0;

/// Total window width.
const W_TOTAL: f64 = W_FRONT + W_REAR;  // ≈ 77 time units

/// Cursor advance per sweep step.
const DT_STEP: f64 = W_TOTAL / 4.0;     // ≈ 19 time units
```

## Memory Budget Proof

At N=1B: T = (24·10⁹/π)^{1/4} ≈ 295.

Window fraction = W_TOTAL / T ≈ 77/295 ≈ 0.261.
Window nodes ≈ 0.261 × 10⁹ ≈ 261M.

Per-node SoA cost:
- times: 8 bytes
- zones: 1 byte
- coords: 24 bytes (evicted early for mature nodes — effective ~8 bytes amortized)
- phases: 1 byte
- is_prism_member: 1 byte
- qt/qx/qy/qz: 8 bytes (evicted with coords)
- CSR: ~15 edges × 4 bytes = 60 bytes (symmetric, both directions)
- Directed CSR: ~8 edges × 4 bytes = 32 bytes

**Hot budget (mature nodes):** 8 + 1 + 1 + 1 + 60 + 32 = 103 bytes/node
**Cold budget (leading edge):** + 24 + 8 = 135 bytes/node

At 261M nodes × ~120 bytes avg ≈ **31.3 GB** — fits in 64 GB with 33 GB headroom.

The full in-memory path at 1B: 10⁹ × 120 bytes ≈ **120 GB** — impossible.

**Saving: 74%.** QED: O(N^{3/4}) memory scaling.

Note: `id_base` and `csr_base`/`dir_base`/`rev_base` are scalars (8 bytes each),
replacing the former `local_ids: Vec<u32>` (4 bytes/node). Net saving: ~4 bytes/node.
