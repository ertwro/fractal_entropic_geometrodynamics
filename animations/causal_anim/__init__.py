"""
CausalAnim — Programmatic animation engine for quantum gravity on graphs.

Usage::

    from causal_anim import Scene, Sprinkle, ReduceHasse, DetectPrism

    scene = Scene("electron_genesis", resolution=(1920, 1080), fps=60)
    scene.play(Sprinkle(N=500, seed=42))
    scene.wait(1.0)
    scene.export("electron_genesis.mp4")
"""

from .annotate import Annotate
from .camera import Camera, CameraAction, CameraState
from .primitives import (
    BuildCausalClosure,
    CausalSlice,
    ContractK5,
    DetectPrism,
    DetectThreat,
    DiffuseWalkers,
    DirectedFlux,
    Highlight,
    InterferenceField,
    ModuloPhaseWalk,
    ProbeVacuumEdge,
    ReduceHasse,
    ShowSpectralDimension,
    Sprinkle,
    TimerOverlay,
    TraversePrism,
)
from .scene import Scene
from .timeline import Timeline

__all__ = [
    # Scene
    "Scene",
    "Timeline",
    # Primitives
    "Sprinkle",
    "BuildCausalClosure",
    "ReduceHasse",
    "DetectPrism",
    "DetectThreat",
    "ContractK5",
    "DiffuseWalkers",
    "DirectedFlux",
    "ShowSpectralDimension",
    "Highlight",
    "TraversePrism",
    "TimerOverlay",
    "ProbeVacuumEdge",
    "ModuloPhaseWalk",
    "InterferenceField",
    "CausalSlice",
    # Camera
    "Camera",
    "CameraAction",
    "CameraState",
    # Annotations
    "Annotate",
]
