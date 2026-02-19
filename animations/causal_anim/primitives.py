"""
Animation atoms — logical operations of the Kuratowski Calculus.

Each class describes a discrete topological operation, NOT a geometric
transformation.  The Scene translates these into frame-level rendering
commands.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════════════
# Phase 1 — Vacuum Generation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Sprinkle:
    """Poisson-sprinkle N events into a 4D causal diamond."""
    N: int
    seed: int = 0

    # Filled after execution
    _node_count: int = 0


@dataclass
class BuildCausalClosure:
    """Render ALL causal relations (before transitive reduction).

    Visual: explosion of O(N²) grey edges.
    """
    sprinkle: Sprinkle
    _edge_count: int = 0


@dataclass
class ReduceHasse:
    """Animate the transitive reduction.

    Visual: redundant edges fade out; surviving edges brighten.
    """
    closure: BuildCausalClosure


# ═══════════════════════════════════════════════════════════════════════
# Phase 2 — Emergence of Matter
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DetectPrism:
    """Highlight a detected Causal Prism K_{2,N}.

    Visual: halo + bundled Bézier + generation colour + convex bubble.
    """
    origin: int
    destination: int
    belly: List[int]
    generation: int = 1  # 1,2,3 = Gen; -1 = Anti; 0 = Sterile


@dataclass
class DetectThreat:
    """Mark an external node that threatens to create K₅.

    Visual: node blinks red, connecting edges turn red, "K₅!" indicator.
    """
    threat_node: int
    prism: DetectPrism


@dataclass
class ContractK5:
    """Animate the K₅ threat absorption into a pole.

    Visual: threat node contracts into absorber pole (ease-in-out),
    edges redirect, flash on absorption.
    """
    threat: DetectThreat
    absorber: str = "max_degree"  # "max_degree" | "origin" | "destination"


# ═══════════════════════════════════════════════════════════════════════
# Phase 3 — Spectral Flow and Electromagnetism
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DiffuseWalkers:
    """Visualise random walkers diffusing on the Hasse skeleton.

    Each walker is a luminous particle with a decaying trail.
    """
    n_walkers: int = 100
    origins: str | List[int] = "uniform"  # "uniform" | "core" | "gen1" | list
    steps: int = 30


@dataclass
class DirectedFlux:
    """Visualise directed causal flux between prism generations.

    Attraction: teal convergent streamlines.
    Repulsion:  terracotta divergent streamlines.
    """
    sources: List[DetectPrism] = field(default_factory=list)
    targets: List[DetectPrism] = field(default_factory=list)
    flux_type: str = "attraction"  # "attraction" | "repulsion"


@dataclass
class ShowSpectralDimension:
    """Overlay a live d_S(t) mini-chart while walkers diffuse."""
    walkers: Optional[DiffuseWalkers] = None
    position: str = "bottom-right"


# ═══════════════════════════════════════════════════════════════════════
# Composition helpers
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Highlight:
    """Temporarily highlight a set of nodes and/or edges."""
    nodes: Optional[List[int]] = None
    edges: Optional[List[tuple]] = None
    color: str = "#FFFFFF"
    duration: float = 1.0


# Type alias for anything playable.
Animation = (
    Sprinkle
    | BuildCausalClosure
    | ReduceHasse
    | DetectPrism
    | DetectThreat
    | ContractK5
    | DiffuseWalkers
    | DirectedFlux
    | ShowSpectralDimension
    | Highlight
)
