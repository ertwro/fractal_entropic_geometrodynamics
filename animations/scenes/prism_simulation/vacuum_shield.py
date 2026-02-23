#!/usr/bin/env python3
"""
vacuum_shield.py
────────────────
Animation 2: K_{3,3} Charge Screening

Narrative:
  Act I   (6s):  Sprinkle vacuum, build Hasse, detect Gen1 prism, zoom in
  Act II  (15s): ProbeVacuumEdge — 8 sequential probes from external nodes
                  Accepted: blue beam sticks
                  Rejected: white K_{3,3} flash, beam shatters
  Act III (5s):  Closing annotation

Usage:
  cd animations
  maturin develop --release
  python scenes/prism_simulation/vacuum_shield.py
"""

from causal_anim import (
    Annotate,
    Camera,
    DetectPrism,
    ProbeVacuumEdge,
    ReduceHasse,
    BuildCausalClosure,
    Scene,
    Sprinkle,
)

scene = Scene("vacuum_shield", resolution=(3840, 2160), fps=60)

# ═══════════════════════════════════════════════════════════════════════
# ACT I — The Prism Emerges
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "A charged particle sits inside the vacuum.",
    position=(0.5, 0.9), duration=3.0, style="plain",
))

vacuum = Sprinkle(N=500, seed=42)
scene.timeline.rush(ticks=500, duration=2.5)
scene.play(vacuum)
scene.wait(0.5)

closure = BuildCausalClosure(vacuum)
scene.timeline.rush(ticks=1000, duration=1.0)
scene.play(closure)

hasse = ReduceHasse(closure)
scene.timeline.slow_motion(ticks=1, duration=1.5)
scene.play(hasse)
scene.wait(0.5)

# Detect a Gen1 prism and zoom into it.
prism = DetectPrism(
    origin=42, destination=187,
    belly=[91, 103, 156],
    generation=1,
)
scene.timeline.slow_motion(ticks=1, duration=2.0)
scene.play(prism)

scene.play(Camera().focus_on(prism, zoom=4.0), duration=1.5)
scene.wait(1.0)

# ═══════════════════════════════════════════════════════════════════════
# ACT II — Probing the Shield
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "Can an external edge penetrate the prism?",
    position=(0.5, 0.9), duration=3.0, style="plain",
))
scene.wait(1.0)

probe = ProbeVacuumEdge(prism=prism, n_probes=8)
scene.timeline.set_pace(ticks_per_second=5)
scene.play(probe)
scene.wait(12.0)

scene.play(Annotate(
    r"$K_{3,3}$ planarity: the graph itself forbids the connection.",
    position=(0.5, 0.1), duration=3.5,
))
scene.wait(2.0)

# ═══════════════════════════════════════════════════════════════════════
# ACT III — Closing
# ═══════════════════════════════════════════════════════════════════════

scene.play(Camera().pull_back(scale=3.0, duration=2.0))
scene.play(Annotate(
    "This isn't a force pushing the edge away;\n"
    "it is a structural impossibility.",
    position=(0.5, 0.5), duration=5.0, style="plain",
))
scene.wait(3.0)

# ── Render ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scene.export("vacuum_shield.mp4")
