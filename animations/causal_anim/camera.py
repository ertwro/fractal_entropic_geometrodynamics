"""
Camera controller for 2D orthographic and future 3D perspective views.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .primitives import DetectPrism


@dataclass
class CameraState:
    """Current camera parameters."""
    x: float = 0.0
    y: float = 0.0
    zoom: float = 10.0   # half-extent in world units


class Camera:
    """Declarative camera commands.

    Each method returns a camera *action* that the scene plays over a
    given duration (lerped linearly unless easing is added later).
    """

    def __init__(self) -> None:
        self.state = CameraState()
        self._actions: list = []

    def focus_on(
        self,
        target: "DetectPrism | int",
        zoom: float = 4.0,
        duration: float = 1.0,
    ) -> "CameraAction":
        """Move the camera to centre on *target* and zoom in."""
        return CameraAction(kind="focus", target=target, zoom=zoom, duration=duration)

    def pull_back(self, scale: float = 2.0, duration: float = 1.0) -> "CameraAction":
        """Zoom out by *scale* factor."""
        return CameraAction(kind="pull_back", scale=scale, duration=duration)

    def orbit(self, angle: float = 360.0, duration: float = 3.0) -> "CameraAction":
        """Rotate around the scene centre (3D mode)."""
        return CameraAction(kind="orbit", angle=angle, duration=duration)

    def move_to(self, x: float, y: float, zoom: float, duration: float = 1.0) -> "CameraAction":
        """Move the camera to an explicit position."""
        return CameraAction(kind="move", target_x=x, target_y=y, zoom=zoom, duration=duration)


@dataclass
class CameraAction:
    """A camera transition to be played by the scene."""
    kind: str
    target: object = None
    zoom: float = 10.0
    scale: float = 1.0
    angle: float = 0.0
    duration: float = 1.0
    target_x: float = 0.0
    target_y: float = 0.0

    def resolve_target(self, positions: list[list[float]]) -> tuple[float, float]:
        """Compute the camera centre from the target (prism or node id)."""
        from .primitives import DetectPrism

        if isinstance(self.target, DetectPrism):
            # Centre on midpoint between poles.
            o = self.target.origin
            d = self.target.destination
            cx = (positions[o][0] + positions[d][0]) / 2
            cy = (positions[o][1] + positions[d][1]) / 2
            return (cx, cy)
        elif isinstance(self.target, int):
            return (positions[self.target][0], positions[self.target][1])
        return (self.target_x, self.target_y)


def lerp_camera(start: CameraState, action: CameraAction, t: float) -> CameraState:
    """Linearly interpolate camera state at fraction *t* ∈ [0, 1]."""
    if action.kind == "pull_back":
        return CameraState(
            x=start.x,
            y=start.y,
            zoom=start.zoom * (1.0 + (action.scale - 1.0) * t),
        )
    elif action.kind in ("focus", "move"):
        tx = action.target_x
        ty = action.target_y
        return CameraState(
            x=start.x + (tx - start.x) * t,
            y=start.y + (ty - start.y) * t,
            zoom=start.zoom + (action.zoom - start.zoom) * t,
        )
    return start
