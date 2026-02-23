#!/usr/bin/env python3
"""
holographic_slice.py
────────────────────
Animation 4: Gravity as Causal Severing

Narrative:
  Act I   (6s):  Sprinkle large-ish vacuum (N=2000), full Hasse
  Act II  (10s): CausalSlice sweeps from bottom to middle
                  Severed links glow white, counter ticks up
  Act III (8s):  Freeze at midpoint, display severed_links / slice_area
  Act IV  (4s):  Closing annotation

Usage:
  cd animations
  maturin develop --release
  python scenes/prism_simulation/holographic_slice.py
"""

from causal_anim import (
    Annotate,
    Camera,
    CausalSlice,
    ReduceHasse,
    BuildCausalClosure,
    Scene,
    Sprinkle,
)

scene = Scene("holographic_slice", resolution=(3840, 2160), fps=60)

# ═══════════════════════════════════════════════════════════════════════
# ACT I — A Large Vacuum
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "2000 causal events, linked by the speed of light.",
    position=(0.5, 0.9), duration=3.0, style="plain",
))

vacuum = Sprinkle(N=2000, seed=99)
scene.timeline.rush(ticks=2000, duration=3.0)
scene.play(vacuum)
scene.wait(0.5)

closure = BuildCausalClosure(vacuum)
scene.timeline.rush(ticks=5000, duration=1.5)
scene.play(closure)

hasse = ReduceHasse(closure)
scene.timeline.slow_motion(ticks=1, duration=1.5)
scene.play(hasse)
scene.wait(0.5)

# ═══════════════════════════════════════════════════════════════════════
# ACT II — The Slice Descends
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "A causal slice cuts through the diagram.\n"
    "Every severed link is an information channel lost.",
    position=(0.5, 0.9), duration=4.0, style="plain",
))
scene.wait(1.0)

causal_slice = CausalSlice(depth_fraction=0.5, display_count=True)
scene.timeline.set_pace(ticks_per_second=5)
scene.play(causal_slice)
scene.wait(8.0)

# ═══════════════════════════════════════════════════════════════════════
# ACT III — The Holographic Count
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    "Severed links / slice area",
    position=(0.5, 0.15), duration=4.0, style="plain",
))

scene.play(Annotate(
    r"$\frac{L_{\mathrm{severed}}}{A_{\mathrm{slice}}} \approx 4.16$"
    r"$\quad \propto \frac{1}{4G}$",
    position=(0.5, 0.5), duration=5.0,
))
scene.wait(6.0)

# ═══════════════════════════════════════════════════════════════════════
# ACT IV — Closing
# ═══════════════════════════════════════════════════════════════════════

scene.play(Camera().pull_back(scale=2.0, duration=2.0))
scene.play(Annotate(
    "Gravity is the thermodynamic cost\n"
    "of cutting a causal graph.",
    position=(0.5, 0.5), duration=4.0, style="plain",
))
scene.wait(3.0)

# ── Render ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scene.export("holographic_slice.mp4")
