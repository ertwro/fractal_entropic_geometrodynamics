#!/usr/bin/env python3
"""
coupon_collector.py
───────────────────
Animation 1: Mass as Topological Delay (Coupon-Collector Cover Time)

Narrative:
  Act I   (8s):  Sprinkle vacuum, build Hasse, detect two prisms
                  (K_{2,3} Gen1 left, K_{2,6} Gen3 right)
  Act II  (15s): Simultaneous TraversePrism on both with TimerOverlay
                  Gen1 finishes fast (~12 ticks), Gen3 struggles (~41 ticks)
  Act III (5s):  Closing annotation

Usage:
  cd animations
  maturin develop --release
  python scenes/prism_simulation/coupon_collector.py
"""

from causal_anim import (
    Annotate,
    Camera,
    DetectPrism,
    ReduceHasse,
    BuildCausalClosure,
    Scene,
    Sprinkle,
    TimerOverlay,
    TraversePrism,
)

scene = Scene("coupon_collector", resolution=(3840, 2160), fps=60)

# ═══════════════════════════════════════════════════════════════════════
# ACT I — The Vacuum and Two Prisms
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "Two prisms, two masses, one vacuum.",
    position=(0.5, 0.9), duration=3.0, style="plain",
))

vacuum = Sprinkle(N=500, seed=42)
scene.timeline.rush(ticks=500, duration=3.0)
scene.play(vacuum)
scene.wait(1.0)

closure = BuildCausalClosure(vacuum)
scene.timeline.rush(ticks=1000, duration=1.5)
scene.play(closure)

hasse = ReduceHasse(closure)
scene.timeline.slow_motion(ticks=1, duration=2.0)
scene.play(hasse)
scene.wait(1.0)

# Detect a Gen1 (electron-like) prism — K_{2,3}.
prism_gen1 = DetectPrism(
    origin=42, destination=187,
    belly=[91, 103, 156],
    generation=1,
)

# Detect a Gen3 (tau-like) prism — K_{2,6} (larger belly → higher mass).
prism_gen3 = DetectPrism(
    origin=15, destination=310,
    belly=[48, 72, 119, 201, 245, 289],
    generation=3,
)

scene.timeline.slow_motion(ticks=1, duration=2.0)
scene.play(prism_gen1, prism_gen3)

scene.play(Annotate(
    r"$K_{2,3}$ (Gen 1) vs $K_{2,6}$ (Gen 3)",
    position=(0.5, 0.1), duration=3.0,
))
scene.wait(1.5)

# ═══════════════════════════════════════════════════════════════════════
# ACT II — The Race: Cover Time Reveals Mass
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "A walker must visit every belly node. How long does it take?",
    position=(0.5, 0.9), duration=4.0, style="plain",
))

# Focus camera to show both prisms.
scene.play(Camera().move_to(x=0.0, y=5.0, zoom=8.0, duration=1.0))
scene.wait(0.5)

# Launch simultaneous traversals — both start at pace 10.
traverse_gen1 = TraversePrism(prism=prism_gen1, n_pulses=5)
traverse_gen3 = TraversePrism(prism=prism_gen3, n_pulses=5)

timer_gen1 = TimerOverlay(traversal=traverse_gen1, position="top-left", label="Gen1 ticks")
timer_gen3 = TimerOverlay(traversal=traverse_gen3, position="top-right", label="Gen3 ticks")

scene.timeline.set_pace(ticks_per_second=10)
scene.play(traverse_gen1, traverse_gen3, timer_gen1, timer_gen3)

# Gen1 finishes fast (~12 ticks ≈ 1.2s) — let it run.
scene.wait(2.0)

# "DONE" annotation over Gen1.
scene.play(Annotate(
    "DONE",
    position=(0.25, 0.5), duration=3.0, style="plain",
))

# Camera zooms into Gen3 prism — the struggle.
scene.play(Camera().move_to(x=0.3, y=5.0, zoom=3.0, duration=1.5))
scene.wait(1.5)

# Slow-motion for Gen3's endgame — stretch the final ticks.
scene.timeline.slow_motion(ticks=15, duration=4.0)
scene.wait(4.0)

# Gen3 completes — white flare handled by scene.py.
scene.wait(1.0)

# Pull back to show both prisms settled.
scene.play(Camera().move_to(x=0.0, y=5.0, zoom=8.0, duration=1.0))
scene.wait(1.0)

scene.play(Annotate(
    r"$\langle\tau_{\mathrm{cover}}\rangle = N \cdot H_N$"
    " — the coupon-collector bound.",
    position=(0.5, 0.1), duration=4.0,
))
scene.wait(3.0)

# ═══════════════════════════════════════════════════════════════════════
# ACT III — Closing
# ═══════════════════════════════════════════════════════════════════════

scene.play(Camera().pull_back(scale=3.0, duration=2.0))
scene.play(Annotate(
    "Mass is the topological delay of a\n"
    "quantum state trying to remember itself.",
    position=(0.5, 0.5), duration=5.0, style="plain",
))
scene.wait(3.0)

# ── Render ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scene.export("coupon_collector.mp4")
