"""
Scene — the top-level container that composes animations, timeline,
and camera into a rendered video.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
from pathlib import Path
from typing import List, Optional

from .annotate import Annotate
from .camera import Camera, CameraAction, CameraState, lerp_camera
from .primitives import (
    Animation,
    BuildCausalClosure,
    CausalSlice,
    ContractK5,
    DetectPrism,
    DetectThreat,
    DiffuseWalkers,
    Highlight,
    InterferenceField,
    ModuloPhaseWalk,
    ProbeVacuumEdge,
    ReduceHasse,
    Sprinkle,
    TimerOverlay,
    TraversePrism,
)
from .timeline import Timeline

# ─── Colour palette (from RFC) ────────────────────────────────────────────

BG_DEFAULT = (0x1D / 255, 0x35 / 255, 0x57 / 255)

COLOR_VACUUM  = (0xCE / 255, 0xD4 / 255, 0xDA / 255, 0.9)
COLOR_HASSE   = (0xF1 / 255, 0xFA / 255, 0xEE / 255, 1.0)
COLOR_EDGE    = (0x6C / 255, 0x75 / 255, 0x7D / 255, 0.4)

GEN_COLORS = {
    1:  (0x2A / 255, 0x9D / 255, 0x8F / 255, 1.0),  # teal
    2:  (0xE9 / 255, 0xC4 / 255, 0x6A / 255, 1.0),  # amber
    3:  (0xE7 / 255, 0x6F / 255, 0x51 / 255, 1.0),  # terracotta
    -1: (0x48 / 255, 0xBF / 255, 0xE3 / 255, 1.0),  # cyan (anti)
    0:  (0x8D / 255, 0x99 / 255, 0xAE / 255, 1.0),  # grey (sterile)
}

# Additional colours for Phase 4 primitives.
COLOR_WALKER_PULSE = (1.0, 0.95, 0.4, 1.0)       # bright yellow
COLOR_WALKER_VISITED = (0.2, 0.85, 0.4, 1.0)      # green
COLOR_PROBE_ACCEPTED = (0.25, 0.55, 1.0, 1.0)     # blue beam
COLOR_PROBE_REJECTED = (1.0, 1.0, 1.0, 0.9)       # white flash
COLOR_SLICE_LINE = (1.0, 1.0, 1.0, 0.8)           # bright white
COLOR_SEVERED = (1.0, 1.0, 1.0, 1.0)              # bright white for severed edges


def _lerp_color(a, b, t):
    """Linear interpolation between two RGBA tuples."""
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(len(a)))


def _heat_color(intensity):
    """Map intensity [0, 1] to deep blue → white → orange colour ramp."""
    if intensity < 0.5:
        # Deep blue to white.
        t = intensity * 2.0
        return (t, t, 1.0, 1.0)
    else:
        # White to orange.
        t = (intensity - 0.5) * 2.0
        return (1.0, 1.0 - 0.4 * t, 1.0 - 0.8 * t, 1.0)


def _timer_position(pos_str, width, height):
    """Convert position string to pixel coordinates."""
    margin_x = int(width * 0.05)
    margin_y = int(height * 0.08)
    if pos_str == "top-right":
        return (width - margin_x, margin_y)
    elif pos_str == "top-left":
        return (margin_x, margin_y)
    elif pos_str == "bottom-right":
        return (width - margin_x, height - margin_y)
    else:
        return (margin_x, height - margin_y)


class Scene:
    """Declarative scene description.

    Usage::

        scene = Scene("my_scene")
        scene.play(Sprinkle(N=500, seed=42))
        scene.wait(1.0)
        scene.export("output.mp4")
    """

    def __init__(
        self,
        name: str,
        resolution: tuple[int, int] = (1920, 1080),
        fps: int = 60,
        background: tuple[float, float, float] = BG_DEFAULT,
    ) -> None:
        self.name = name
        self.width, self.height = resolution
        self.fps = fps
        self.background = background
        self.timeline = Timeline()
        self.camera = Camera()

        # Ordered list of (presentation_time, item).
        self._events: List[tuple[float, object]] = []

        # Rust engine (lazy-initialised on first Sprinkle).
        self._engine = None

    # ── Public API ────────────────────────────────────────────────────

    def play(self, *animations: Animation | CameraAction | Annotate, duration: float | None = None) -> None:
        """Schedule one or more animations to play in parallel at the
        current presentation time."""
        t = self.timeline.total_duration
        for anim in animations:
            self._events.append((t, anim))

    def wait(self, seconds: float = 1.0) -> None:
        """Advance presentation time without changing the graph."""
        self.timeline.pause(seconds)

    def wait_ticks(self, ticks: int) -> None:
        """Advance causal time at the current pace."""
        self.timeline.wait_ticks(ticks)

    # ── Export ─────────────────────────────────────────────────────────

    def export(self, path: str = "output.mp4") -> None:
        """Render all frames and encode to video.

        Requires ``ffmpeg`` on PATH for the final encode step.
        """
        try:
            import causal_anim_core  # type: ignore[import-not-found]
        except ImportError:
            print(
                "[CausalAnim] causal_anim_core not found. "
                "Build with: cd animations && maturin develop --release"
            )
            return

        total = self.timeline.total_duration
        n_frames = max(1, int(total * self.fps))

        # ── Phase 1: build the universe (first Sprinkle wins) ─────
        sprinkle_ev = None
        for _, ev in self._events:
            if isinstance(ev, Sprinkle):
                sprinkle_ev = ev
                break

        if sprinkle_ev is None:
            print("[CausalAnim] No Sprinkle found in scene — nothing to render.")
            return

        engine = causal_anim_core.SceneEngine(self.width, self.height)
        engine.build_universe(sprinkle_ev.N, sprinkle_ev.seed)
        engine.relax_layout(50)
        self._engine = engine

        # Detect if we have defect operations and run defect if needed.
        has_defect = any(
            isinstance(ev, (DetectPrism, DetectThreat, ContractK5,
                            TraversePrism, TimerOverlay, ProbeVacuumEdge))
            for _, ev in self._events
        )
        if has_defect:
            engine.apply_defect()

        # ── Phase 2: pre-computation for new primitives ──────────
        frames_dir = Path(path).with_suffix("") / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        positions_flat = engine.get_positions_flat()
        n_nodes = engine.node_count()
        n_edges = engine.edge_count()
        edges_flat = engine.get_edges_flat()

        # Restructure positions for easier access.
        base_positions = [
            [positions_flat[i * 3], positions_flat[i * 3 + 1], positions_flat[i * 3 + 2]]
            for i in range(n_nodes)
        ]

        # Compute auto-camera bounds.
        if n_nodes > 0:
            xs = [p[0] for p in base_positions]
            ys = [p[1] for p in base_positions]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            auto_zoom = span * 0.6
        else:
            cx, cy, auto_zoom = 0.0, 0.0, 10.0

        cam = CameraState(x=cx, y=cy, zoom=auto_zoom)

        # ── Pre-compute data for Phase 4 primitives ──────────────
        # Each pre-computation stores data keyed by the primitive's id().

        # TraversePrism: pre-compute full trajectory via prism_cover_trajectory.
        traverse_data: dict = {}   # id(TraversePrism) → {trajectory, cover_times, ...}
        for _, ev in self._events:
            if isinstance(ev, TraversePrism):
                trajectory = engine.prism_cover_trajectory(
                    ev.prism.origin,
                    ev.prism.belly,
                    ev.prism.destination,
                    42,
                )
                belly_set = set(ev.prism.belly)
                # Build visited_at_tick: belly_node → first-visit tick
                visited_at_tick = {}
                for tick, node in enumerate(trajectory):
                    if node in belly_set and node not in visited_at_tick:
                        visited_at_tick[node] = tick
                # Build trail edges from trajectory
                trail_edges = [
                    (trajectory[i - 1], trajectory[i], i)
                    for i in range(1, len(trajectory))
                ]
                cover_times = [len(trajectory)]
                ev._cover_times = cover_times
                traverse_data[id(ev)] = {
                    "trajectory": trajectory,
                    "trail_edges": trail_edges,
                    "cover_times": cover_times,
                    "max_cover": len(trajectory),
                    "visited_at_tick": visited_at_tick,
                    "prism_nodes": set([ev.prism.origin, ev.prism.destination] + ev.prism.belly),
                    "belly_set": belly_set,
                    "finished_at_frame": None,
                }

        # ProbeVacuumEdge: pre-compute accepted/rejected.
        probe_data: dict = {}
        for _, ev in self._events:
            if isinstance(ev, ProbeVacuumEdge):
                accepted, rejected = engine.k33_probe(
                    ev.prism.origin, ev.prism.destination, ev.prism.belly,
                )
                # Limit to n_probes total.
                all_probes = [(ni, True) for ni in accepted] + [(ni, False) for ni in rejected]
                all_probes = all_probes[:ev.n_probes]
                probe_data[id(ev)] = {
                    "probes": all_probes,  # list of (node_idx, is_accepted)
                    "prism_nodes": set([ev.prism.origin, ev.prism.destination] + ev.prism.belly),
                }

        # ModuloPhaseWalk + InterferenceField: pre-compute interference.
        interference_data: dict = {}
        for _, ev in self._events:
            if isinstance(ev, ModuloPhaseWalk):
                results = engine.modulo_interference(
                    ev.n_walkers, ev.steps, ev.prime, ev.root, 42,
                )
                # Build per-node intensity map.
                intensity_map = {}
                for node_idx, arrivals, phase_sum, intensity in results:
                    intensity_map[node_idx] = intensity
                interference_data[id(ev)] = {
                    "intensity_map": intensity_map,
                    "max_intensity": max((v for v in intensity_map.values()), default=1.0),
                }

        # CausalSlice: pre-compute slice.
        slice_data: dict = {}
        for _, ev in self._events:
            if isinstance(ev, CausalSlice):
                max_d = engine.max_depth()
                slice_depth = int(ev.depth_fraction * max_d)
                below, above, severed = engine.causal_slice(slice_depth)
                # Compute slice area as the horizontal extent of nodes at the slice depth.
                slice_y = None
                area = 1.0
                if below:
                    # Find nodes nearest to the slice boundary.
                    boundary_nodes = [i for i in range(n_nodes) if engine.node_depth(i) == slice_depth]
                    if boundary_nodes:
                        bxs = [positions[i][0] for i in boundary_nodes]
                        slice_y = positions[boundary_nodes[0]][1]
                        area = max(max(bxs) - min(bxs), 0.01)
                slice_data[id(ev)] = {
                    "below": set(below),
                    "above": set(above),
                    "severed": severed,
                    "severed_set": set((u, v) for u, v in severed),
                    "slice_depth": slice_depth,
                    "slice_y": slice_y,
                    "n_severed": len(severed),
                    "area": area,
                }

        # ── Phase 3: per-frame rendering ─────────────────────────

        # Build event schedule: for each event, compute its start frame
        # and duration in frames.
        event_schedule = []
        for t_start, ev in self._events:
            # Default durations for primitives.
            dur = 3.0  # default
            if isinstance(ev, Annotate):
                dur = ev.duration
            elif isinstance(ev, CameraAction):
                dur = ev.duration if hasattr(ev, 'duration') else 1.0
            elif isinstance(ev, TraversePrism):
                td = traverse_data.get(id(ev))
                dur = (td["max_cover"] / 10.0 + 1.0) if td else 5.0
            elif isinstance(ev, ProbeVacuumEdge):
                dur = ev.n_probes * 1.5
            elif isinstance(ev, (ModuloPhaseWalk, InterferenceField)):
                dur = 8.0
            elif isinstance(ev, CausalSlice):
                dur = 10.0
            f_start = int(t_start * self.fps)
            f_end = int((t_start + dur) * self.fps)
            event_schedule.append((f_start, f_end, ev))

        # Pre-compute the set of all prism nodes (for causal fog exclusion).
        all_prism_nodes: set = set()
        for _, ev in self._events:
            if isinstance(ev, (DetectPrism, TraversePrism)):
                prim = ev if isinstance(ev, DetectPrism) else ev.prism
                all_prism_nodes.add(prim.origin)
                all_prism_nodes.add(prim.destination)
                all_prism_nodes.update(prim.belly)

        # Pre-compute fog alpha per node (depth-based dimming).
        max_depth = engine.max_depth()
        fog_alphas: list[float] = [1.0] * n_nodes
        if max_depth > 0:
            for i in range(n_nodes):
                if i not in all_prism_nodes:
                    depth_frac = engine.node_depth(i) / max(max_depth, 1)
                    fog_alphas[i] = 0.3 + 0.7 * depth_frac

        print(f"[CausalAnim] Rendering {n_frames} frames ({self.width}x{self.height} @ {self.fps} fps)")

        for fi in range(n_frames):
            t = fi / self.fps
            progress = t / total if total > 0 else 1.0

            # Determine which nodes are visible (progressive reveal for Sprinkle).
            visible_frac = min(1.0, progress * 2)  # reveal over first half
            n_visible = max(1, int(n_nodes * visible_frac))

            # Per-frame copy of positions (jitter may perturb these).
            positions = [list(p) for p in base_positions]

            # Per-node colour/radius overrides for this frame.
            node_colors = [COLOR_VACUUM] * n_nodes
            node_radii = [0.08] * n_nodes

            # Per-edge colour/width overrides.  We'll build edge_data after.
            edge_overrides: dict = {}  # (u,v) → (r, g, b, a, width)

            # Extra edges to add this frame (e.g. probe beams).
            extra_edges: list = []

            # Collect active events for this frame.
            active_events = [
                (f_start, f_end, ev) for f_start, f_end, ev in event_schedule
                if f_start <= fi < f_end
            ]

            for f_start, f_end, ev in active_events:
                local_progress = (fi - f_start) / max(1, f_end - f_start)

                if isinstance(ev, DetectPrism):
                    gen_col = GEN_COLORS.get(ev.generation, COLOR_VACUUM)
                    for n_idx in [ev.origin, ev.destination]:
                        node_colors[n_idx] = gen_col
                        node_radii[n_idx] = 0.15
                    for n_idx in ev.belly:
                        node_colors[n_idx] = gen_col
                        node_radii[n_idx] = 0.12

                elif isinstance(ev, TraversePrism):
                    td = traverse_data.get(id(ev))
                    if td:
                        trajectory = td["trajectory"]
                        max_ct = td["max_cover"]
                        visited_at_tick = td["visited_at_tick"]
                        belly_set = td["belly_set"]
                        belly_list = ev.prism.belly
                        gen_col = GEN_COLORS.get(ev.prism.generation, COLOR_VACUUM)

                        # Current tick from trajectory
                        current_tick = min(int(local_progress * max_ct), max_ct - 1)

                        # Walker position from trajectory
                        walker_node = trajectory[current_tick]

                        # Build visited set up to current_tick
                        visited_now = set()
                        for bnode, tick in visited_at_tick.items():
                            if tick <= current_tick:
                                visited_now.add(bnode)
                        all_visited = len(visited_now) == len(belly_set)

                        # Track finished frame
                        if all_visited and td["finished_at_frame"] is None:
                            td["finished_at_frame"] = fi

                        # --- Poles: 30% brighter, larger radius ---
                        pole_col = _lerp_color(gen_col, (1.0, 1.0, 1.0, 1.0), 0.3)
                        for n_idx in [ev.prism.origin, ev.prism.destination]:
                            node_colors[n_idx] = pole_col
                            node_radii[n_idx] = 0.18

                        # --- Belly nodes ---
                        for n_idx in belly_list:
                            if n_idx in visited_now:
                                node_colors[n_idx] = COLOR_WALKER_VISITED
                                node_radii[n_idx] = 0.12
                            else:
                                node_colors[n_idx] = (0.4, 0.4, 0.4, 0.6)
                                node_radii[n_idx] = 0.10

                        # --- Revisit flash: walker on already-visited belly node ---
                        if walker_node in belly_set and walker_node in visited_now:
                            # Check if this is a revisit (not the first visit)
                            first_visit = visited_at_tick.get(walker_node, current_tick)
                            if current_tick > first_visit:
                                node_colors[walker_node] = (0.7, 0.2, 0.2, 1.0)
                                node_radii[walker_node] = 0.14

                        # --- Walker pulse on current node ---
                        if not all_visited:
                            node_colors[walker_node] = COLOR_WALKER_PULSE
                            node_radii[walker_node] = 0.18

                        # --- Last-node flare: 1 belly node remaining ---
                        unvisited = belly_set - visited_now
                        if len(unvisited) == 1 and not all_visited:
                            last_node = next(iter(unvisited))
                            pulse_r = 0.14 + 0.06 * math.sin(fi * 0.3)
                            node_colors[last_node] = (1.0, 0.9, 0.3, 1.0)
                            node_radii[last_node] = pulse_r

                        # --- Completion white flare ---
                        finished_frame = td["finished_at_frame"]
                        if finished_frame is not None:
                            flare_age = fi - finished_frame
                            if flare_age < 15:
                                flare_alpha = 1.0 - flare_age / 15.0
                                for n_idx in td["prism_nodes"]:
                                    if n_idx < n_nodes:
                                        node_colors[n_idx] = _lerp_color(
                                            node_colors[n_idx],
                                            (1.0, 1.0, 1.0, 1.0),
                                            flare_alpha,
                                        )

                        # --- Layout jitter: belly nodes vibrate ---
                        if not all_visited:
                            for n_idx in belly_list:
                                amp = 0.03 * (1.0 - local_progress)
                                if n_idx == walker_node:
                                    amp *= 2.0
                                jitter = amp * math.sin(fi * 0.4 + n_idx * 1.7)
                                positions[n_idx] = [
                                    positions[n_idx][0] + jitter,
                                    positions[n_idx][1],
                                    positions[n_idx][2],
                                ]

                        # --- Ghost trails: exponentially decaying past edges ---
                        for from_node, to_node, tick in td["trail_edges"]:
                            if tick > current_tick:
                                break
                            age = current_tick - tick
                            alpha = gen_col[3] * math.exp(-age / 30.0)
                            if alpha < 0.02:
                                continue
                            extra_edges.append({
                                "from": from_node, "to": to_node,
                                "color": (gen_col[0], gen_col[1], gen_col[2], alpha),
                                "width": 0.015,
                            })

                elif isinstance(ev, TimerOverlay):
                    # Timer overlays are handled via annotations in the frame.
                    pass  # Annotation injected below.

                elif isinstance(ev, ProbeVacuumEdge):
                    pd = probe_data.get(id(ev))
                    if pd:
                        probes = pd["probes"]
                        prism_nodes = pd["prism_nodes"]
                        # Highlight prism nodes.
                        gen_col = GEN_COLORS.get(ev.prism.generation, COLOR_VACUUM)
                        for n_idx in prism_nodes:
                            if n_idx < n_nodes:
                                node_colors[n_idx] = gen_col
                                node_radii[n_idx] = 0.13

                        # Sequential probes: each gets ~1/(n_probes) of the duration.
                        n_probes = len(probes)
                        if n_probes > 0:
                            probe_idx = min(int(local_progress * n_probes), n_probes - 1)
                            for pi in range(probe_idx + 1):
                                node_idx, is_accepted = probes[pi]
                                if node_idx >= n_nodes:
                                    continue
                                probe_local = (local_progress * n_probes) - pi
                                probe_local = max(0.0, min(1.0, probe_local))

                                if is_accepted:
                                    # Blue beam that sticks.
                                    alpha = min(1.0, probe_local * 2)
                                    col = (COLOR_PROBE_ACCEPTED[0], COLOR_PROBE_ACCEPTED[1],
                                           COLOR_PROBE_ACCEPTED[2], alpha)
                                    node_colors[node_idx] = col
                                    node_radii[node_idx] = 0.1
                                    # Draw edge from probe to nearest prism pole.
                                    extra_edges.append({
                                        "from": node_idx, "to": ev.prism.origin,
                                        "color": col, "width": 0.03,
                                    })
                                else:
                                    # White flash that fades.
                                    if probe_local < 0.5:
                                        flash = probe_local * 2
                                        col = (1.0, 1.0, 1.0, flash)
                                    else:
                                        fade = 1.0 - (probe_local - 0.5) * 2
                                        col = (1.0, 0.3, 0.3, max(0.0, fade))
                                    node_colors[node_idx] = col
                                    node_radii[node_idx] = 0.1 + 0.08 * (1.0 - probe_local)
                                    if probe_local < 0.8:
                                        extra_edges.append({
                                            "from": node_idx, "to": ev.prism.origin,
                                            "color": (1.0, 1.0, 1.0, max(0.0, 0.8 - probe_local)),
                                            "width": 0.02,
                                        })

                elif isinstance(ev, ModuloPhaseWalk):
                    # Walkers are visualised as phase labels; the heavy
                    # computation is done in InterferenceField.  Here we
                    # just lightly highlight active nodes.
                    ifd = interference_data.get(id(ev))
                    if ifd:
                        reveal = local_progress
                        for node_idx, intensity in ifd["intensity_map"].items():
                            if node_idx < n_nodes and intensity > 0.1:
                                alpha = min(1.0, reveal * 2) * 0.5
                                node_colors[node_idx] = (0.6, 0.8, 1.0, alpha)

                elif isinstance(ev, InterferenceField):
                    ifd = interference_data.get(id(ev.walk))
                    if ifd:
                        reveal = min(1.0, local_progress * 1.5)
                        for node_idx, intensity in ifd["intensity_map"].items():
                            if node_idx < n_nodes:
                                col = _heat_color(intensity)
                                # Fade in the heatmap.
                                alpha = col[3] * reveal
                                node_colors[node_idx] = (col[0], col[1], col[2], alpha)
                                node_radii[node_idx] = 0.06 + 0.1 * intensity

                elif isinstance(ev, CausalSlice):
                    sd = slice_data.get(id(ev))
                    if sd:
                        # Animate the slice sweeping from bottom to current depth.
                        sweep_progress = min(1.0, local_progress * 1.5)
                        # Dim nodes above the current sweep position.
                        for n_idx in sd["above"]:
                            if n_idx < n_nodes:
                                node_colors[n_idx] = (
                                    COLOR_VACUUM[0], COLOR_VACUUM[1],
                                    COLOR_VACUUM[2], 0.3,
                                )
                        # Highlight severed edges.
                        if sweep_progress > 0.3:
                            sev_reveal = min(1.0, (sweep_progress - 0.3) / 0.4)
                            for u, v in sd["severed"]:
                                edge_overrides[(u, v)] = (1.0, 1.0, 1.0, sev_reveal, 0.04)

            # --- Causal fog: apply pre-computed depth-based dimming ---
            for i in range(n_nodes):
                if fog_alphas[i] < 1.0:
                    r, g, b, a = node_colors[i]
                    node_colors[i] = (r, g, b, a * fog_alphas[i])

            # --- Glow nodes (fake bloom): collect nodes that need halos ---
            glow_nodes: list = []  # [(position, glow_radius, glow_color)]
            for f_start, f_end, ev in active_events:
                if isinstance(ev, TraversePrism):
                    td = traverse_data.get(id(ev))
                    if td:
                        # Poles glow
                        for n_idx in [ev.prism.origin, ev.prism.destination]:
                            if n_idx < n_visible:
                                gc = node_colors[n_idx]
                                glow_nodes.append((
                                    positions[n_idx],
                                    node_radii[n_idx] * 3.0,
                                    (gc[0], gc[1], gc[2], 0.12),
                                ))
                        # Walker pulse glows
                        traj = td["trajectory"]
                        max_ct = td["max_cover"]
                        lp = (fi - f_start) / max(1, f_end - f_start)
                        ct = min(int(lp * max_ct), max_ct - 1)
                        wn = traj[ct]
                        if wn < n_visible:
                            wc = node_colors[wn]
                            glow_nodes.append((
                                positions[wn],
                                node_radii[wn] * 3.0,
                                (wc[0], wc[1], wc[2], 0.12),
                            ))

            # Build node data: [x, y, z, radius, r, g, b, a] per node
            # Emit glow halos first (drawn behind), then regular nodes.
            node_data: list[float] = []
            for pos, grad, gcol in glow_nodes:
                node_data.extend([pos[0], pos[1], pos[2], grad])
                node_data.extend(gcol)
            for i in range(n_visible):
                p = positions[i]
                r = node_radii[i]
                c = node_colors[i]
                node_data.extend([p[0], p[1], p[2], r])
                node_data.extend(c)

            # Build edge data: [sx,sy,sz,width, ex,ey,ez,pad, r,g,b,a]
            edge_data: list[float] = []
            for ei in range(0, len(edges_flat), 2):
                u = edges_flat[ei]
                v = edges_flat[ei + 1]
                if u < n_visible and v < n_visible:
                    pu = positions[u]
                    pv = positions[v]
                    override = edge_overrides.get((u, v))
                    if override:
                        r, g, b, a, w = override
                        edge_data.extend([pu[0], pu[1], pu[2], w])
                        edge_data.extend([pv[0], pv[1], pv[2], 0.0])
                        edge_data.extend([r, g, b, a])
                    else:
                        edge_data.extend([pu[0], pu[1], pu[2], 0.02])
                        edge_data.extend([pv[0], pv[1], pv[2], 0.0])
                        edge_data.extend(COLOR_EDGE)

            # Add extra edges (probe beams, etc.).
            for ee in extra_edges:
                if ee["from"] < n_nodes and ee["to"] < n_nodes:
                    pf = positions[ee["from"]]
                    pt = positions[ee["to"]]
                    c = ee["color"]
                    edge_data.extend([pf[0], pf[1], pf[2], ee["width"]])
                    edge_data.extend([pt[0], pt[1], pt[2], 0.0])
                    edge_data.extend(c)

            # Apply camera actions.
            for f_start, f_end, ev in active_events:
                if isinstance(ev, CameraAction):
                    local_t = (fi - f_start) / max(1, f_end - f_start)
                    cam = lerp_camera(cam, ev, local_t)

            camera_arr = [cam.x, cam.y, cam.zoom]
            png_bytes = engine.render_png(
                node_data, edge_data, camera_arr, list(self.background)
            )

            frame_path = frames_dir / f"frame_{fi:06d}.png"
            frame_path.write_bytes(bytes(png_bytes))

            if fi % self.fps == 0:
                print(f"  frame {fi}/{n_frames}  t={t:.2f}s")

        # ── Phase 3: encode video ─────────────────────────────────
        output = Path(path)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(output),
        ]
        print(f"[CausalAnim] Encoding → {output}")
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"[CausalAnim] Done: {output}")
        except FileNotFoundError:
            print(
                "[CausalAnim] ffmpeg not found. "
                f"Frames saved in {frames_dir}/ — encode manually."
            )
        except subprocess.CalledProcessError as e:
            print(f"[CausalAnim] ffmpeg failed: {e.stderr.decode()}")

    # ── Snapshot ──────────────────────────────────────────────────────

    def snapshot(self, path: str = "snapshot.png") -> None:
        """Render a single frame of the current state."""
        try:
            import causal_anim_core  # type: ignore[import-not-found]
        except ImportError:
            print("[CausalAnim] causal_anim_core not found.")
            return

        sprinkle_ev = None
        for _, ev in self._events:
            if isinstance(ev, Sprinkle):
                sprinkle_ev = ev
                break

        if sprinkle_ev is None:
            print("[CausalAnim] No Sprinkle in scene.")
            return

        engine = causal_anim_core.SceneEngine(self.width, self.height)
        engine.build_universe(sprinkle_ev.N, sprinkle_ev.seed)
        engine.relax_layout(50)

        # Auto-camera.
        positions_flat = engine.get_positions_flat()
        n = engine.node_count()
        if n > 0:
            xs = positions_flat[0::3]
            ys = positions_flat[1::3]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            zoom = span * 0.6
        else:
            cx, cy, zoom = 0.0, 0.0, 10.0

        png = engine.render_full(
            node_color=list(COLOR_VACUUM),
            node_radius=0.08,
            edge_color=list(COLOR_EDGE),
            edge_width=0.02,
            camera_x=cx,
            camera_y=cy,
            camera_zoom=zoom,
            bg=list(self.background),
        )
        Path(path).write_bytes(bytes(png))
        print(f"[CausalAnim] Snapshot saved: {path}")
