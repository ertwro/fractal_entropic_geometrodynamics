#!/usr/bin/env python3
"""
modulo_clockwork.py
───────────────────
Animation 3: Quantum Interference Without i

Narrative:
  Act I   (6s):  Sprinkle vacuum, build Hasse, camera on a central node
  Act II  (12s): ModuloPhaseWalk — walkers carry g^S mod p phase
                  Show phase values ticking as odometers
  Act III (8s):  InterferenceField — node heatmap reveals constructive (bright)
                  and destructive (dark) fringes
  Act IV  (4s):  Closing annotation

Usage:
  cd animations
  maturin develop --release
  python scenes/prism_simulation/modulo_clockwork.py
"""

from causal_anim import (
    Annotate,
    Camera,
    InterferenceField,
    ModuloPhaseWalk,
    ReduceHasse,
    BuildCausalClosure,
    Scene,
    Sprinkle,
)

scene = Scene("modulo_clockwork", resolution=(3840, 2160), fps=60)

# ═══════════════════════════════════════════════════════════════════════
# ACT I — The Vacuum Lattice
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "Each walker carries a modular phase: $g^S \\bmod p$.",
    position=(0.5, 0.9), duration=3.0,
))

vacuum = Sprinkle(N=800, seed=77)
scene.timeline.rush(ticks=800, duration=3.0)
scene.play(vacuum)
scene.wait(0.5)

closure = BuildCausalClosure(vacuum)
scene.timeline.rush(ticks=2000, duration=1.5)
scene.play(closure)

hasse = ReduceHasse(closure)
scene.timeline.slow_motion(ticks=1, duration=1.5)
scene.play(hasse)
scene.wait(0.5)

# ═══════════════════════════════════════════════════════════════════════
# ACT II — The Phase Walkers
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "30 walkers diffuse through the Hasse diagram,\n"
    "each accumulating a discrete phase.",
    position=(0.5, 0.9), duration=4.0, style="plain",
))

walk = ModuloPhaseWalk(
    n_walkers=30,
    origins="uniform",
    steps=40,
    prime=65537,
    root=3,
)

scene.timeline.set_pace(ticks_per_second=8)
scene.play(walk)
scene.wait(10.0)

scene.play(Annotate(
    r"Phase $= 3^{S} \bmod 65537$",
    position=(0.5, 0.1), duration=3.0,
))
scene.wait(2.0)

# ═══════════════════════════════════════════════════════════════════════
# ACT III — The Interference Pattern
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "Where phases align: constructive.\n"
    "Where they cancel: destructive.",
    position=(0.5, 0.9), duration=4.0, style="plain",
))

interference = InterferenceField(walk=walk)
scene.play(interference)
scene.wait(6.0)

scene.play(Annotate(
    "Hot nodes: many arrivals, coherent phase.\n"
    "Cold nodes: cancellation by modular arithmetic.",
    position=(0.5, 0.1), duration=3.0, style="plain",
))
scene.wait(2.0)

# ═══════════════════════════════════════════════════════════════════════
# ACT IV — Closing
# ═══════════════════════════════════════════════════════════════════════

scene.play(Camera().pull_back(scale=2.0, duration=2.0))
scene.play(Annotate(
    "It isn't a wave. It's a clock.",
    position=(0.5, 0.5), duration=4.0, style="plain",
))
scene.wait(3.0)

# ── Render ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scene.export("modulo_clockwork.mp4")
