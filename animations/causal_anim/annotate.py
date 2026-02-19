"""
Text and LaTeX annotation overlay.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Annotate:
    """Place a text or LaTeX annotation on the scene.

    Parameters
    ----------
    text : str
        The annotation text.  Strings containing ``$`` are treated as
        LaTeX and rendered via an external ``latexmk`` call (future).
    position : tuple[float, float]
        Normalised screen position (0,0 = bottom-left; 1,1 = top-right).
    duration : float
        How long the annotation stays visible (presentation seconds).
    style : str
        ``"plain"`` for simple text, ``"latex"`` for LaTeX rendering.
    font_size : int
        Font size in points (for plain text).
    color : str
        Hex colour string.
    """
    text: str
    position: tuple[float, float] = (0.5, 0.5)
    duration: float = 3.0
    style: str = "latex"
    font_size: int = 24
    color: str = "#F1FAEE"

    @property
    def is_latex(self) -> bool:
        return "$" in self.text or self.style == "latex"

    def screen_xy(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalised position to pixel coordinates."""
        return (int(self.position[0] * width), int((1.0 - self.position[1]) * height))
