"""
Dual-clock timeline: causal ticks (τ) and presentation seconds (t).

Python wrapper that mirrors the Rust `Timeline` and also drives the
frame-level interpolation for the Python scene loop.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class TimeSegment:
    kind: str          # "rush" | "slow_motion" | "pause" | "normal"
    ticks: int = 0
    duration: float = 0.0
    ticks_per_second: float = 10.0
    causal_start: int = 0
    presentation_start: float = 0.0


class Timeline:
    """Manages the mapping τ ↔ t."""

    def __init__(self) -> None:
        self._segments: List[TimeSegment] = []
        self._current_causal: int = 0
        self._current_presentation: float = 0.0
        self._default_pace: float = 10.0

    # ── Segment builders ──────────────────────────────────────────────

    def rush(self, ticks: int, duration: float) -> None:
        """Compress *ticks* causal steps into *duration* seconds."""
        self._segments.append(TimeSegment(
            kind="rush", ticks=ticks, duration=duration,
            causal_start=self._current_causal,
            presentation_start=self._current_presentation,
        ))
        self._current_causal += ticks
        self._current_presentation += duration

    def slow_motion(self, ticks: int, duration: float) -> None:
        """Stretch *ticks* causal steps over *duration* seconds."""
        self._segments.append(TimeSegment(
            kind="slow_motion", ticks=ticks, duration=duration,
            causal_start=self._current_causal,
            presentation_start=self._current_presentation,
        ))
        self._current_causal += ticks
        self._current_presentation += duration

    def pause(self, duration: float) -> None:
        """Freeze the graph; advance only presentation time."""
        self._segments.append(TimeSegment(
            kind="pause", duration=duration,
            causal_start=self._current_causal,
            presentation_start=self._current_presentation,
        ))
        self._current_presentation += duration

    def set_pace(self, ticks_per_second: float) -> None:
        """Set the default pace for subsequent ``wait_ticks`` calls."""
        self._default_pace = ticks_per_second

    def wait_ticks(self, ticks: int) -> None:
        """Advance *ticks* causal steps at the current default pace."""
        duration = ticks / self._default_pace
        self._segments.append(TimeSegment(
            kind="normal", ticks=ticks, duration=duration,
            ticks_per_second=self._default_pace,
            causal_start=self._current_causal,
            presentation_start=self._current_presentation,
        ))
        self._current_causal += ticks
        self._current_presentation += duration

    # ── Queries ───────────────────────────────────────────────────────

    @property
    def total_duration(self) -> float:
        return self._current_presentation

    @property
    def total_causal_ticks(self) -> int:
        return self._current_causal

    def presentation_to_causal(self, t: float) -> float:
        """Map presentation time *t* to an interpolated causal tick."""
        for seg in reversed(self._segments):
            if t >= seg.presentation_start:
                dt = t - seg.presentation_start
                if seg.kind in ("rush", "slow_motion"):
                    progress = max(0.0, min(1.0, dt / seg.duration)) if seg.duration > 0 else 1.0
                    return seg.causal_start + progress * seg.ticks
                elif seg.kind == "pause":
                    return float(seg.causal_start)
                else:  # normal
                    return seg.causal_start + dt * seg.ticks_per_second
        return 0.0

    def segment_progress(self, t: float) -> float:
        """Fractional progress (0–1) within the active segment."""
        for seg in reversed(self._segments):
            if t >= seg.presentation_start:
                dt = t - seg.presentation_start
                return max(0.0, min(1.0, dt / seg.duration)) if seg.duration > 0 else 1.0
        return 0.0
