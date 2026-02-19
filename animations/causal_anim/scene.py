"""
Scene — the top-level container that composes animations, timeline,
and camera into a rendered video.
"""

from __future__ import annotations

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
    ContractK5,
    DetectPrism,
    DetectThreat,
    DiffuseWalkers,
    Highlight,
    ReduceHasse,
    Sprinkle,
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
        has_defect = any(isinstance(ev, (DetectPrism, DetectThreat, ContractK5)) for _, ev in self._events)
        if has_defect:
            engine.apply_defect()

        # ── Phase 2: per-frame rendering ──────────────────────────
        frames_dir = Path(path).with_suffix("") / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        positions_flat = engine.get_positions_flat()
        n_nodes = engine.node_count()
        n_edges = engine.edge_count()
        edges_flat = engine.get_edges_flat()

        # Restructure positions for easier access.
        positions = [
            [positions_flat[i * 3], positions_flat[i * 3 + 1], positions_flat[i * 3 + 2]]
            for i in range(n_nodes)
        ]

        # Compute auto-camera bounds.
        if n_nodes > 0:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            auto_zoom = span * 0.6
        else:
            cx, cy, auto_zoom = 0.0, 0.0, 10.0

        cam = CameraState(x=cx, y=cy, zoom=auto_zoom)

        print(f"[CausalAnim] Rendering {n_frames} frames ({self.width}x{self.height} @ {self.fps} fps)")

        for fi in range(n_frames):
            t = fi / self.fps
            progress = t / total if total > 0 else 1.0

            # Determine which nodes are visible (progressive reveal for Sprinkle).
            visible_frac = min(1.0, progress * 2)  # reveal over first half
            n_visible = max(1, int(n_nodes * visible_frac))

            # Build node data: [x, y, z, radius, r, g, b, a] per node
            node_data: list[float] = []
            for i in range(n_visible):
                p = positions[i]
                node_data.extend([p[0], p[1], p[2], 0.08])
                node_data.extend(COLOR_VACUUM)

            # Build edge data: [sx,sy,sz,width, ex,ey,ez,pad, r,g,b,a]
            edge_data: list[float] = []
            for ei in range(0, len(edges_flat), 2):
                u = edges_flat[ei]
                v = edges_flat[ei + 1]
                if u < n_visible and v < n_visible:
                    pu = positions[u]
                    pv = positions[v]
                    edge_data.extend([pu[0], pu[1], pu[2], 0.02])
                    edge_data.extend([pv[0], pv[1], pv[2], 0.0])
                    edge_data.extend(COLOR_EDGE)

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
