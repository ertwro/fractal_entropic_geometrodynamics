# CausalAnim

Programmatic animation engine for quantum gravity on graphs, topological
logic, and causal sets.  Built for the Kuratowski Calculus of
**Modulo Synthesis**.

Unlike continuous animation engines (manim, Motion Canvas), CausalAnim
animates **discrete logical operations** — sprinkles, transitive
reductions, Kuratowski contractions, and spectral flows — on a GPU-
accelerated rendering backend.

```
┌─────────────────────────────────────────────────────────┐
│                 Scene script (.py)                       │
│       from causal_anim import Scene, Sprinkle, ...      │
├─────────────────────────────────────────────────────────┤
│           causal_anim  (Python package)                  │
│     Scene DSL · Timeline · Camera · Annotations         │
├─────────────────────┬───────────────────────────────────┤
│     PyO3 bridge     │     causal_anim_core (Rust)       │
│                     │  Layout · LOD · wgpu Renderer     │
├─────────────────────┴───────────────────────────────────┤
│            causal_set_sim  (Rust crate, existing)       │
│   sprinkle · build_hasse · apply_defect · run_walkers   │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

| Tool     | Version | Purpose                        |
|----------|---------|--------------------------------|
| Rust     | ≥ 1.75  | Compile the GPU core           |
| Python   | ≥ 3.10  | Run scene scripts              |
| maturin  | ≥ 1.0   | Build Rust→Python extension    |
| ffmpeg   | any     | Encode PNG frames to video     |
| GPU      | Vulkan, Metal, or DX12 | wgpu rendering    |

Install maturin if you don't have it:

```bash
pip install maturin
```

### Build

```bash
cd animations
maturin develop --release
```

This compiles the Rust crate `causal_anim_core` (with wgpu, PyO3, and
the simulation engine linked in) and installs it as a Python extension
module alongside the pure-Python `causal_anim` package.

### Run the example scene

```bash
python scenes/prism_simulation/electron_genesis.py
```

Produces `electron_genesis.mp4` — a ~55-second animation of an electron
(Causal Prism K_{2,3}) crystallising from the quantum vacuum.

### Take a single snapshot

```python
from causal_anim import Scene, Sprinkle

scene = Scene("test", resolution=(1920, 1080))
scene.play(Sprinkle(N=500, seed=42))
scene.snapshot("hasse_500.png")
```

---

## Architecture

### Two-Layer Design

**Python layer** (`causal_anim/`) — what the user writes.  Declarative,
high-level, manim-style.  Describes *what* to animate.

**Rust layer** (`causal_anim_core/`) — what does the work.  Computes
graph topology, spring layouts, and GPU-rendered frames.  Linked
directly to the simulation engine (`causal_set_sim`) with zero
serialisation overhead.

### Rust Modules

| Module        | File          | Responsibility                                   |
|---------------|---------------|--------------------------------------------------|
| `bridge`      | `bridge.rs`   | Wraps `causal_set_sim`: sprinkle, Hasse, defect  |
| `layout`      | `layout.rs`   | Stratified spring layout (Y = depth, X/Z = spring) |
| `timeline`    | `timeline.rs` | Dual clock: τ (causal ticks) ↔ t (seconds)       |
| `lod`         | `lod.rs`      | Level-of-detail culling by visible node count     |
| `renderer`    | `renderer.rs` | Headless wgpu: instanced SDF circles + line quads |
| `lib`         | `lib.rs`      | PyO3 entry point: `SceneEngine` class             |

### Python Modules

| Module         | File             | Responsibility                              |
|----------------|------------------|---------------------------------------------|
| `primitives`   | `primitives.py`  | Animation atoms (Sprinkle, DetectPrism, ...) |
| `scene`        | `scene.py`       | Scene composition, export, snapshot          |
| `timeline`     | `timeline.py`    | Dual clock (Python mirror of Rust timeline)  |
| `camera`       | `camera.py`      | Camera controller with focus/orbit/pull_back |
| `annotate`     | `annotate.py`    | Text and LaTeX annotations                   |

---

## Core Concepts

### 1. Animation Atoms (Primitives)

Every animation corresponds to a **logical operation** of the Kuratowski
Calculus, not a geometric transformation.

| Primitive               | Physics operation                          | Visual effect                            |
|-------------------------|--------------------------------------------|------------------------------------------|
| `Sprinkle(N, seed)`     | Poisson sprinkle in 4D causal diamond      | Nodes appear progressively               |
| `BuildCausalClosure(s)` | Compute all causal relations               | Dense grey edge burst                    |
| `ReduceHasse(c)`        | Transitive reduction                       | Redundant edges fade out                 |
| `DetectPrism(o,d,W,g)`  | Identify K_{2,N} bipartite structure       | Halo, bundled Bézier, generation colour  |
| `DetectThreat(t, p)`    | Mark node threatening K₅                   | Red blink, "K₅!" indicator              |
| `ContractK5(threat)`    | Absorb threat into pole                    | Node contracts into absorber, flash      |
| `DiffuseWalkers(W,o,s)` | Random walk on Hasse skeleton              | Luminous particles with trail            |
| `DirectedFlux(src,tgt)` | Directed transmission between generations  | Convergent/divergent streamlines         |
| `ShowSpectralDimension` | d_S(t) overlay                             | Live mini-chart                          |
| `Highlight(nodes,edges)`| Temporary emphasis                         | Colour pulse                             |

### 2. The Dual Timeline

CausalAnim maintains two independent clocks:

```
τ (causal ticks)     0   1   2   3   4   5   6   7   ...
                     ●───●───●───●───●───●───●───●
                     │       │           │
                     ▼       ▼           ▼
t (viewer seconds)  0.0    0.5   1.0   1.2  1.5  2.0  ...
                     ●──────●─────●──●───●────●──
                    sprinkle   reduce   K₅!  walkers
```

**τ** is the graph's internal clock — strictly integer, monotone.
Each tick is a discrete event (a sprinkled node, a walker step, an
edge reduced).

**t** is the viewer's clock — continuous float, measured in seconds.
The mapping τ → t is controlled by pace functions:

```python
scene.timeline.rush(ticks=10000, duration=3.0)     # fast-forward
scene.timeline.slow_motion(ticks=1, duration=2.5)   # dramatic reveal
scene.timeline.pause(duration=4.0)                   # freeze for narration
scene.timeline.set_pace(ticks_per_second=10)         # constant rate
```

**`scene.wait(seconds)`** advances only presentation time (for the
human).  **`scene.wait_ticks(n)`** advances causal time (for the
physics).

### 3. The Layout

The layout engine assigns 2D/3D positions with a single invariant:

> **Y never lies.**  If u ≺ v in the causal order, then y(u) < y(v).
> Always.

- **Y axis** = causal depth (longest chain from past boundary).
  Computed once, immutable.
- **X, Z axes** = spatial degrees of freedom.  Initialised from the
  4D sprinkle coordinates, then refined by spring relaxation within
  each causal layer.

The relaxation uses three forces:
- Coulomb repulsion between same-layer nodes (prevents overlap)
- Spring attraction along Hasse edges (preserves locality)
- Centering force (prevents drift)

### 4. Level of Detail

For graphs with millions of nodes, the renderer automatically selects
a detail level based on the number of visible nodes in the camera
frustum:

| Level           | Visible nodes | Rendering                        |
|-----------------|---------------|----------------------------------|
| **Cosmic**      | > 10⁶         | Density heatmap                  |
| **Galactic**    | 10⁴ – 10⁶    | Instanced points, no edges       |
| **Stellar**     | 10³ – 10⁴    | Points + major edges             |
| **Atomic**      | < 10³         | Full nodes, all edges, labels    |
| **PrismFocus**  | 1 prism       | Full K_{2,N} detail              |

### 5. Colour Palette

| Element            | Hex       | RGB              | Usage                |
|--------------------|-----------|------------------|----------------------|
| Background         | `#1D3557` | (29, 53, 87)     | Scene background     |
| Vacuum nodes       | `#CED4DA` | (206, 212, 218)  | Non-prism nodes      |
| Hasse edges        | `#6C757D` | (108, 117, 125)  | Causal links (α=0.4) |
| Gen 1 (electron)   | `#2A9D8F` | (42, 157, 143)   | Teal                 |
| Gen 2 (muon)       | `#E9C46A` | (233, 196, 106)  | Amber                |
| Gen 3 (tau)        | `#E76F51` | (231, 111, 81)   | Terracotta           |
| Anti-Gen 1         | `#48BFE3` | (72, 191, 227)   | Cyan                 |
| Sterile (DM)       | `#8D99AE` | (141, 153, 174)  | Grey-blue            |
| K₅ threat          | `#E63946` | (230, 57, 70)    | Red alarm            |
| Contraction flash  | `#FFFFFF` | (255, 255, 255)  | White                |

---

## API Reference

### Scene

```python
from causal_anim import Scene

scene = Scene(
    name="my_scene",
    resolution=(1920, 1080),  # width × height in pixels
    fps=60,                    # frames per second
    background=(0.11, 0.21, 0.34),  # RGB floats (default: #1D3557)
)
```

#### `scene.play(*animations, duration=None)`

Schedule one or more animations to play in parallel at the current
presentation time.

```python
scene.play(Sprinkle(N=500, seed=42))
scene.play(prism, Camera().focus_on(prism, zoom=4.0))  # parallel
```

#### `scene.wait(seconds)`

Advance presentation time without changing the graph.  Use for pauses
between acts or while annotations are visible.

#### `scene.wait_ticks(ticks)`

Advance causal time at the currently configured pace.

#### `scene.export(path)`

Render all frames to PNG, then encode to video with ffmpeg.

```python
scene.export("output.mp4")
```

If ffmpeg is not installed, frames are saved in `output/frames/` for
manual encoding.

#### `scene.snapshot(path)`

Render the current state as a single PNG image.

```python
scene.snapshot("preview.png")
```

### Timeline

```python
scene.timeline.rush(ticks=10000, duration=3.0)
scene.timeline.slow_motion(ticks=1, duration=2.5)
scene.timeline.pause(duration=4.0)
scene.timeline.set_pace(ticks_per_second=10)
scene.timeline.wait_ticks(50)
```

| Method                 | τ advances? | t advances? | Use case                |
|------------------------|-------------|-------------|-------------------------|
| `rush(ticks, dur)`     | yes         | yes         | Bulk sprinkle           |
| `slow_motion(ticks,d)` | yes         | yes         | Dramatic reveal         |
| `pause(dur)`           | no          | yes         | Narration / equation    |
| `set_pace(tps)`        | —           | —           | Set default rate        |
| `wait_ticks(n)`        | yes         | yes         | Advance at default pace |

### Camera

```python
from causal_anim import Camera

cam = Camera()
scene.play(cam.focus_on(prism, zoom=4.0), duration=1.5)
scene.play(cam.pull_back(scale=2.0, duration=1.0))
scene.play(cam.move_to(x=0, y=10, zoom=20, duration=2.0))
```

### Annotations

```python
from causal_anim import Annotate

scene.play(Annotate(
    r"$\mathcal{P}(u, v, W) \cong K_{2,3}$",
    position=(0.5, 0.85),   # normalised screen coords
    duration=3.0,
    style="latex",           # or "plain"
    color="#F1FAEE",
))
```

---

## Example: Electron Genesis

The reference scene (`scenes/prism_simulation/electron_genesis.py`)
tells the story of an electron crystallising from the quantum vacuum
in four acts:

| Act   | Title                    | Physics                                | Duration |
|-------|--------------------------|----------------------------------------|----------|
| **I** | The Vacuum               | Sprinkle → causal closure → reduction  | ~10 s    |
| **II**| The Crystallisation      | Prism K_{2,3} detection                | ~8 s     |
| **III**| The Kuratowski Threat   | K₅ threat → vertex contraction         | ~10 s    |
| **IV**| Mass as Causal Delay    | Walker diffusion, τ_res ∝ N            | ~10 s    |
|       | Epilogue                 | Pull back, closing annotation          | ~8 s     |

```python
from causal_anim import Scene, Sprinkle, ReduceHasse, DetectPrism, ...

scene = Scene("electron_genesis", resolution=(3840, 2160), fps=60)

# Act I
vacuum = Sprinkle(N=500, seed=42)
scene.timeline.rush(ticks=500, duration=3.0)
scene.play(vacuum)
# ...

scene.export("electron_genesis.mp4")
```

---

## Project Structure

```
animations/
├── README.md                               ← This file
├── RFC_001_causal_anim.md                  ← Architecture design document
├── pyproject.toml                          ← Python build config (maturin)
├── .gitignore
│
├── causal_anim_core/                       ← Rust GPU engine
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                          ← PyO3 module: SceneEngine
│       ├── bridge.rs                       ← causal_set_sim wrapper
│       ├── layout.rs                       ← Stratified spring layout
│       ├── timeline.rs                     ← Dual clock (τ, t)
│       ├── lod.rs                          ← Level-of-detail manager
│       └── renderer.rs                     ← Headless wgpu renderer
│
├── causal_anim/                            ← Python scripting DSL
│   ├── __init__.py                         ← Public re-exports
│   ├── primitives.py                       ← Animation atoms
│   ├── scene.py                            ← Scene composition + export
│   ├── timeline.py                         ← Timeline (Python mirror)
│   ├── camera.py                           ← Camera controller
│   └── annotate.py                         ← Text / LaTeX overlays
│
├── scenes/                                 ← Scene scripts
│   └── prism_simulation/
│       └── electron_genesis.py             ← Reference scene
│
└── renders/                                ← Video output (gitignored)
```

---

## Rendering Pipeline

```
Scene script (.py)
    │
    ▼
SceneEngine.build_universe(N, seed)
    │  Calls: causal_set_sim::sprinkle() + build_hasse_direct()
    │
    ├──► LayoutEngine::new()          Compute causal depths (DP)
    │        │                        Initialise X,Z from 4D coords
    │        └► relax(50)             Spring relaxation within layers
    │
    ├──► apply_defect()               K₂,N detection + K₅ contraction
    │
    └──► Per-frame render loop
            │
            ├──► Build NodeInstance[]  (position, radius, colour)
            ├──► Build EdgeInstance[]  (start, end, colour, width)
            ├──► Upload to GPU        (wgpu instance buffers)
            │
            ├──► Render pass 1: edges (instanced triangle-strip quads)
            ├──► Render pass 2: nodes (instanced SDF circles + alpha)
            │
            ├──► Readback texture → CPU
            ├──► Encode PNG
            └──► ffmpeg → MP4
```

---

## Dependencies

### Rust (`causal_anim_core`)

| Crate             | Version | Role                               |
|-------------------|---------|-------------------------------------|
| `causal_set_sim`  | local   | Simulation engine (sprinkle, Hasse) |
| `wgpu`            | 24      | GPU rendering (Vulkan/Metal/DX12)   |
| `pyo3`            | 0.28    | Rust ↔ Python bridge                |
| `image`           | 0.25    | PNG encoding                        |
| `bytemuck`        | 1       | GPU buffer marshalling              |
| `glam`            | 0.29    | Linear algebra                      |
| `rayon`           | 1.10    | CPU parallelism                     |
| `rand`            | 0.8     | RNG (matches simulation engine)     |
| `pollster`        | 0.4     | Blocking async (wgpu futures)       |

### Python (`causal_anim`)

Zero external dependencies.  The Rust extension (`causal_anim_core`)
is the only non-stdlib import.

### System

| Tool    | Required for        |
|---------|---------------------|
| ffmpeg  | Video encoding      |
| GPU     | wgpu rendering      |
| latexmk | LaTeX annotations (future) |

---

## Design Decisions

See [RFC-001](RFC_001_causal_anim.md) for the full architectural
rationale.  Key decisions:

1. **Hybrid Rust + Python** — Rust for GPU compute (10M nodes @ 60 FPS),
   Python for scripting ergonomics (30-line scene descriptions).

2. **Y axis = causal depth** — The vertical axis never lies.  Horizontal
   positions are aesthetically free; chronological order is sacred.

3. **Animation atoms = logical operations** — We don't animate pixel
   movement.  We animate sprinkles, reductions, contractions, and
   walker diffusion.

4. **Dual timeline** — τ (causal ticks) and t (viewer seconds) advance
   independently, connected by pace functions.

5. **Direct crate linkage** — No CSV serialisation between the simulation
   engine and the renderer.  `causal_set_sim` is a Cargo path dependency;
   CSR arrays pass in-memory.
