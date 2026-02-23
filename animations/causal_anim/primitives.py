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


# ═══════════════════════════════════════════════════════════════════════
# Phase 4 — Advanced Visualization Primitives
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TraversePrism:
    """Walker pulse traversing a specific prism, counting cover time.

    A lazy random walker starts at the origin pole and must visit every
    belly node before reaching the destination.  The number of ticks
    required is the topological mass proxy (coupon-collector delay).
    """
    prism: DetectPrism
    n_pulses: int = 5
    # Filled after execution
    _cover_times: List[int] = field(default_factory=list)


@dataclass
class TimerOverlay:
    """Dynamic on-screen tick counter bound to a TraversePrism."""
    traversal: TraversePrism
    position: str = "top-right"  # "top-left" | "top-right" | etc.
    label: str = "ticks"


@dataclass
class ProbeVacuumEdge:
    """Animate attempted edges from vacuum nodes to a prism.

    Accepted edges glow blue; K₃,₃-blocked edges flash white and shatter.
    Visualises the structural impossibility of charge screening.
    """
    prism: DetectPrism
    n_probes: int = 8


@dataclass
class ModuloPhaseWalk:
    """Walkers carrying visible g^S mod p phase counters.

    Each walker accumulates a multiplicative phase g^S mod p as it
    traverses the Hasse skeleton.  The result is a discrete analogue
    of quantum interference — without complex numbers.
    """
    n_walkers: int = 30
    origins: str | List[int] = "uniform"
    steps: int = 40
    prime: int = 65537
    root: int = 3


@dataclass
class InterferenceField:
    """Color all nodes by NTT interference intensity.

    Hot (orange/white) = constructive, cold (deep blue) = destructive.
    """
    walk: ModuloPhaseWalk


@dataclass
class CausalSlice:
    """2D plane cutting through the Hasse diagram at a given causal depth.

    Counts and displays severed links vs slice area, revealing the
    holographic relationship between bulk gravity and boundary entropy.
    """
    depth_fraction: float = 0.5  # 0.0=bottom, 1.0=top
    display_count: bool = True


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
    | TraversePrism
    | TimerOverlay
    | ProbeVacuumEdge
    | ModuloPhaseWalk
    | InterferenceField
    | CausalSlice
)
