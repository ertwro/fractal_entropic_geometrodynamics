#!/usr/bin/env python3
"""
electron_genesis.py
───────────────────
Escena: Un electrón (Prisma K_{2,3}) cristaliza desde el vacío cuántico.

Narrativa:
  1. El vacío se siembra (Poisson sprinkle)
  2. Se establece el orden causal (closure → reducción)
  3. Una fluctuación topológica forma un K_{2,3}
  4. Un nodo externo amenaza con crear K₅
  5. La contracción de Kuratowski absorbe la amenaza
  6. Un walker difunde por el grafo, queda atrapado en el prisma
  7. La demora de residencia revela la masa

Uso:
  cd animations
  maturin develop --release
  python scenes/prism_simmulation/electron_genesis.py
"""

from causal_anim import (
    Annotate,
    Camera,
    ContractK5,
    DetectPrism,
    DetectThreat,
    DiffuseWalkers,
    ReduceHasse,
    BuildCausalClosure,
    Scene,
    ShowSpectralDimension,
    Sprinkle,
)

scene = Scene("electron_genesis", resolution=(3840, 2160), fps=60)

# ═══════════════════════════════════════════════════════════════════════
# ACTO I — El Vacío
# ═══════════════════════════════════════════════════════════════════════

scene.play(Annotate(
    r"El vacío es un diagrama de Hasse finito, libre de triángulos.",
    position=(0.5, 0.9), duration=3.0, style="plain",
))

vacuum = Sprinkle(N=500, seed=42)
scene.timeline.rush(ticks=500, duration=3.0)
scene.play(vacuum)
scene.wait(1.0)

closure = BuildCausalClosure(vacuum)
scene.timeline.rush(ticks=1000, duration=1.5)
scene.play(closure)

scene.play(Annotate(
    r"$u \prec v \iff (t_v - t_u)^2 > |\Delta\vec{x}|^2$",
    position=(0.5, 0.1), duration=2.0,
))
scene.wait(1.0)

hasse = ReduceHasse(closure)
scene.timeline.slow_motion(ticks=1, duration=2.5)
scene.play(hasse)

scene.play(Annotate(
    "Reducción transitiva: solo quedan las relaciones de cobertura.",
    position=(0.5, 0.1), duration=2.0, style="plain",
))
scene.wait(2.0)

# ═══════════════════════════════════════════════════════════════════════
# ACTO II — La Cristalización
# ═══════════════════════════════════════════════════════════════════════

prism = DetectPrism(
    origin=42, destination=187,
    belly=[91, 103, 156],
    generation=1,
)

scene.play(Camera().focus_on(prism, zoom=4.0), duration=1.5)
scene.wait(0.5)

scene.timeline.slow_motion(ticks=1, duration=3.0)
scene.play(prism)

scene.play(Annotate(
    r"$\mathcal{P}(u, v, W) \cong K_{2,3} \quad M = \kappa \cdot 3$",
    position=(0.5, 0.85), duration=3.0,
))
scene.wait(2.0)

# ═══════════════════════════════════════════════════════════════════════
# ACTO III — La Amenaza de Kuratowski
# ═══════════════════════════════════════════════════════════════════════

threat = DetectThreat(threat_node=220, prism=prism)
scene.timeline.slow_motion(ticks=1, duration=2.0)
scene.play(threat)

scene.play(Annotate(
    r"Amenaza $K_5$: nodo 220 conecta a ambos polos + 2 intermediarios.",
    position=(0.5, 0.1), duration=2.5,
))
scene.wait(1.5)

contraction = ContractK5(threat, absorber="max_degree")
scene.timeline.slow_motion(ticks=1, duration=2.5)
scene.play(contraction)

scene.play(Annotate(
    "Contracción de vértice: planaridad restaurada.",
    position=(0.5, 0.1), duration=2.0, style="plain",
))
scene.wait(2.0)

# ═══════════════════════════════════════════════════════════════════════
# ACTO IV — La Masa como Demora Causal
# ═══════════════════════════════════════════════════════════════════════

scene.play(Camera().pull_back(scale=2.0, duration=1.0))

walkers = DiffuseWalkers(
    n_walkers=50,
    origins=[prism.origin],
    steps=30,
)
ds_plot = ShowSpectralDimension(walkers, position="bottom-right")

scene.timeline.set_pace(ticks_per_second=10)
scene.play(walkers, ds_plot)
scene.wait(3.0)

scene.play(Annotate(
    r"$\langle\tau_{\mathrm{res}}\rangle \propto N$"
    " — La masa es demora topológica.",
    position=(0.5, 0.85), duration=4.0,
))
scene.wait(3.0)

# ═══════════════════════════════════════════════════════════════════════
# EPÍLOGO
# ═══════════════════════════════════════════════════════════════════════

scene.play(Camera().pull_back(scale=10.0, duration=3.0))
scene.play(Annotate(
    "Un electrón no es una partícula puntual.\n"
    "Es una bifurcación causal que el vacío no puede simplificar.",
    position=(0.5, 0.5), duration=5.0, style="plain",
))
scene.wait(3.0)

# ── Render ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scene.export("electron_genesis.mp4")
