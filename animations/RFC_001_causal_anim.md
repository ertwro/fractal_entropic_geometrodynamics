# RFC-001: CausalAnim — Motor de Animación para Gravedad Cuántica de Grafos

**Estado:** Propuesta
**Autor:** Arquitectura de Modulo Synthesis
**Fecha:** 2026-02-17
**Componentes afectados:** `animations/`, `prism_simmulation/`

---

## 0. Resumen Ejecutivo

CausalAnim es un motor de animación programático diseñado para renderizar
la física discreta del Cálculo de Kuratowski. A diferencia de motores
continuos (manim, Motion Canvas), CausalAnim no interpola vectores en
$\mathbb{R}^3$ sobre un tiempo $t$ continuo. Anima **operaciones lógicas**
sobre grafos causales discretos: sprinkles, reducciones transitivas,
contracciones de Kuratowski y flujos espectrales.

**Arquitectura:** Núcleo de cómputo en Rust (layout, LOD, instanced
rendering vía `wgpu`) + capa de scripting en Python (escenas declarativas
vía PyO3). El Colisionador existente se enlaza directamente como
dependencia de crate, sin serialización intermedia.

---

## 1. Filosofía de Renderizado: El Layout

### 1.1 El Problema

Un DAG puro de N > 10⁴ nodos renderizado ingenuamente produce una bola
de pelo incomprensible. Los algoritmos force-directed clásicos
(Fruchterman-Reingold, ForceAtlas2) ignoran la estructura causal y tratan
al grafo como no-dirigido, destruyendo la semántica física.

### 1.2 La Solución: Layout Estratificado con Relajación Lateral

El layout opera en dos ejes ortogonales con semánticas distintas:

```
Eje Y (vertical) ≡ Profundidad causal τ
   ↑  Determinista: y(v) = longitud de la cadena máxima desde ∂⁻ hasta v
   ↑  Se calcula una sola vez; es inmutable durante la animación
   ↑  Preserva el orden parcial: u ≺ v ⟹ y(u) < y(v)

Eje X,Z (horizontal) ≡ Grados de libertad espaciales emergentes
   ←→ No-determinista: posición dentro de la anticadena (capa causal)
   ←→ Se relaja con un spring-layout modificado DENTRO de cada capa
   ←→ Repulsión entre nodos de la misma capa (∝ 1/r²)
   ←→ Atracción suave hacia vecinos Hasse de capas adyacentes (spring)
```

**Propiedad fundamental:** el eje vertical nunca miente. Si $u \prec v$,
entonces $u$ aparece visualmente debajo de $v$. Siempre. La "geografía"
horizontal es estéticamente libre, pero la "cronología" vertical es
sagrada.

### 1.3 Relajación Dentro de Capas (Intra-Layer Spring)

Para cada capa causal $L_k = \{v : \text{depth}(v) = k\}$:

```
F_repulsión(v_i, v_j) = C_rep / ||x_i - x_j||²    (Coulomb, dentro de L_k)
F_atracción(v_i, v_j) = C_att · ||x_i - x_j||       (Spring, si (v_i, v_j) ∈ E_H con |depth| = 1)
F_gravedad(v_i)        = -C_grav · x_i               (Centering, evita dispersión)
```

Las constantes `C_rep`, `C_att`, `C_grav` se ajustan automáticamente
según $|L_k|$ para que cada capa tenga un radio visual proporcional a
$\sqrt{|L_k|}$. La relajación se ejecuta en GPU (compute shader wgpu)
con convergencia típica en ~50 iteraciones.

### 1.4 Resaltado de Prismas Causales

Los Prismas $K_{2,N}$ son las estructuras más importantes del universo.
Su layout recibe tratamiento especial:

```
Prisma K_{2,N}:
        v (polo futuro)     ← Nodo grande, color generación
       /|\  ···  |\
      / | \      | \
    w₁  w₂ w₃  wₙ₋₁ wₙ   ← Nodos medianos, halo translúcido
      \ | /      | /
       \|/  ···  |/
        u (polo pasado)     ← Nodo grande, color generación

Renderizado:
  - Polos u, v: radio 3×, borde luminoso del color de generación
  - Belly W: radio 1.5×, arco horizontal equiespaciado
  - Aristas internas: bundled (curvas de Bézier convergentes), grosor 2×
  - Envolvente: burbuja translúcida convexa (convex hull con padding)
```

Colores de generación (del codebase existente fig_sim_mass):

| Generación | Color              | Interpretación        |
|------------|--------------------|-----------------------|
| Gen1       | Teal `#2A9D8F`     | Electrón/positrón     |
| Gen2       | Ámbar `#E9C46A`    | Muón                  |
| Gen3       | Terracota `#E76F51`| Tau                   |
| Anti1      | Teal invertido     | Antimateria Gen1      |
| Estéril    | Gris `#8D99AE`     | Materia oscura (C6)   |
| Vacío      | Gris tenue `#CED4DA` | Nodos no-prisma     |

### 1.5 Niveles de Detalle (LOD)

Para manejar N = 10⁷ nodos sin colapsar el framerate:

| Escala             | N visible     | Renderizado                              |
|--------------------|---------------|------------------------------------------|
| **Cósmica**        | > 10⁶         | Campo de densidad (heatmap 2D/3D)        |
| **Galáctica**      | 10⁴ – 10⁶    | Puntos instanciados, aristas ocultas     |
| **Estelar**        | 10³ – 10⁴    | Puntos + aristas principales             |
| **Atómica**        | < 10³         | Nodos completos, etiquetas, belly arcs   |
| **Prism-focus**    | 1 prisma      | Full K_{2,N} con animación de walkers    |

La transición entre niveles usa alpha-blending suave (0.3s de
crossfade). La cámara define el nivel activo según el conteo de nodos
en el frustum.

---

## 2. Átomos de Animación: Las Primitivas del API

### 2.1 Principio de Diseño

Cada primitiva del API corresponde a una **operación lógica** del
Cálculo de Kuratowski, no a una transformación geométrica. El motor
traduce internamente cada operación lógica a una secuencia de
transformaciones visuales.

```
Operación lógica          →  Efecto visual
────────────────────────     ──────────────────────────────────
Sprinkle(N)               →  Aparición estocástica de puntos
Imply(u, v)               →  Flecha causal dibujada
ReduceHasse()             →  Fade-out de aristas redundantes
DetectPrism(u, v, W)      →  Halo + bundling + color generación
ContractK5(threat)        →  Absorción del nodo amenaza en polo
DiffuseWalkers(W)         →  Partículas moviéndose por el grafo
DirectedFlux(src, tgt)    →  Flechas con color de atracción/repulsión
```

### 2.2 Catálogo de Primitivas

#### Fase 1 — Generación de Vacío

```python
class Sprinkle:
    """Siembra de Poisson en diamante causal 4D."""
    def __init__(self, N: int, seed: int = 0):
        self.N = N
        self.seed = seed
    # Visual: los nodos aparecen uno a uno (o en ráfagas)
    # en posiciones (x, y) mapeadas desde las coordenadas 4D.
    # La velocidad de aparición escala con presentation_rate.

class Imply:
    """Dibuja una relación causal u ≺ v."""
    def __init__(self, u: int, v: int, style: str = "arrow"):
        self.u = u
        self.v = v
        self.style = style  # "arrow" | "line" | "glow"
    # Visual: línea dirigida de u a v, fade-in de 0.1s.
    # Si style="glow", la línea pulsa brevemente al crearse.

class BuildCausalClosure:
    """Dibuja TODAS las relaciones causales (antes de reducción)."""
    def __init__(self, sprinkle: Sprinkle):
        self.sprinkle = sprinkle
    # Visual: explosión de aristas (O(N²) posibles).
    # Se renderiza como ráfaga densa gris claro.
    # Propósito: mostrar el "ruido" previo a la reducción.

class ReduceHasse:
    """Animación de la reducción transitiva."""
    def __init__(self, closure: BuildCausalClosure):
        self.closure = closure
    # Visual: las aristas redundantes (transitivas) se desvanecen
    # simultáneamente, dejando solo el esqueleto de Hasse.
    # Duración: 1-2s. Efecto: de bola de pelo a estructura limpia.
    # Color: aristas supervivientes pasan de gris a blanco.
```

#### Fase 2 — Emergencia de Materia

```python
class DetectPrism:
    """Resalta un Prisma Causal K_{2,N} detectado."""
    def __init__(self, origin: int, destination: int,
                 belly: list[int], generation: int):
        self.origin = origin
        self.destination = destination
        self.belly = belly
        self.generation = generation  # 1, 2, 3, -1 (anti), 0 (estéril)
    # Visual:
    #   1. Halo pulsante en nodos del prisma (0.3s)
    #   2. Aristas internas se colorean y engrosan (bundled Bézier)
    #   3. Burbuja convexa translúcida envuelve la estructura
    #   4. Etiqueta flotante: "K_{2,N}" con N = |belly|
    #   5. Color según tabla de generaciones (§1.4)

class DetectThreat:
    """Marca un nodo externo que amenaza con crear K₅."""
    def __init__(self, threat_node: int, prism: DetectPrism):
        self.threat_node = threat_node
        self.prism = prism
    # Visual:
    #   1. El nodo amenaza parpadea en rojo (#E63946)
    #   2. Las aristas conectoras al prisma se colorean rojo
    #   3. Un indicador "K₅!" aparece brevemente
    #   4. Líneas discontinuas muestran el minor K₅ potencial

class ContractK5:
    """Animación de absorción: el nodo amenaza colapsa en el polo."""
    def __init__(self, threat: DetectThreat, absorber: str = "max_degree"):
        self.threat = threat
        self.absorber = absorber  # "max_degree" | "origin" | "destination"
    # Visual:
    #   1. El nodo amenaza se contrae hacia el polo absorbente (0.5s ease-in-out)
    #   2. Las aristas del nodo amenaza se redirigen al polo (morph)
    #   3. Flash blanco en el polo al absorber (0.1s)
    #   4. El polo crece ligeramente (radio += factor)
    #   5. La etiqueta de masa se actualiza: N → N+1
    #   6. El indicator "K₅!" se convierte en "✓ Planar"
```

#### Fase 3 — Flujo Espectral y Electromagnético

```python
class DiffuseWalkers:
    """Visualización de random walkers difundiendo por el grafo."""
    def __init__(self, graph: ReduceHasse, n_walkers: int = 100,
                 origins: list[int] | str = "uniform",
                 steps: int = 30):
        self.graph = graph
        self.n_walkers = n_walkers
        self.origins = origins  # "uniform" | "core" | "gen1" | lista
        self.steps = steps
    # Visual:
    #   Cada walker es una partícula luminosa (radio 2px, trail de 5 pasos)
    #   que salta entre nodos siguiendo la lazy random walk (50% quieto,
    #   50% vecino aleatorio). Color del walker hereda del nodo origen.
    #   La "cola" (trail) deja un rastro decayente que muestra el camino.
    #   Los retornos al origen producen un flash.

class DirectedFlux:
    """Visualización del flujo causal dirigido entre generaciones."""
    def __init__(self, sources: list[DetectPrism],
                 targets: list[DetectPrism],
                 flux_type: str = "attraction"):
        self.sources = sources
        self.targets = targets
        self.flux_type = flux_type  # "attraction" | "repulsion"
    # Visual:
    #   Walkers dirigidos (solo futuro) fluyen de sources a targets.
    #   Atracción: flechas teal convergentes, grosor ∝ T_{A→B}
    #   Repulsión: flechas terracota divergentes, grosor ∝ T_{A→A}
    #   Se renderiza como campo de flujo (streamlines) sobre el grafo.

class ShowSpectralDimension:
    """Overlay del gráfico d_S(t) en tiempo real."""
    def __init__(self, walkers: DiffuseWalkers,
                 position: str = "bottom-right"):
        self.walkers = walkers
        self.position = position
    # Visual:
    #   Gráfico 2D incrustado (miniatura) mostrando d_S(t) vs t.
    #   La curva se dibuja progresivamente a medida que los walkers
    #   avanzan. Línea horizontal punteada en d_S = 2 (UV) y d_S = 4 (IR).
```

#### Primitivas de Composición

```python
class Annotate:
    """Añade texto o ecuación LaTeX a la escena."""
    def __init__(self, text: str, position: tuple[float, float],
                 duration: float = 3.0, style: str = "latex"):
        ...
    # Soporta LaTeX renderizado a textura (vía latexmk + rsvg).

class Camera:
    """Control de cámara: zoom, pan, foco en estructura."""
    def focus_on(self, target: DetectPrism | int, zoom: float = 1.0): ...
    def orbit(self, angle: float, duration: float): ...
    def pull_back(self, scale: float, duration: float): ...

class Highlight:
    """Resalta temporalmente un conjunto de nodos/aristas."""
    def __init__(self, nodes: list[int] = None,
                 edges: list[tuple[int, int]] = None,
                 color: str = "#FFFFFF", duration: float = 1.0): ...
```

### 2.3 Composición de Escenas

Las primitivas se componen en una `Scene`, que es la unidad atómica
de renderizado (equivale a un `Scene` de manim):

```python
class Scene:
    """Contenedor secuencial de operaciones lógicas."""
    def __init__(self, name: str, resolution: tuple = (1920, 1080),
                 fps: int = 60, background: str = "#1D3557"):
        self.name = name
        self.timeline = Timeline()
        ...

    def play(self, *animations, duration: float = None):
        """Ejecuta animaciones en paralelo."""
        ...

    def wait(self, seconds: float = 1.0):
        """Pausa de presentación (tiempo del espectador)."""
        ...

    def wait_ticks(self, ticks: int):
        """Avanza N ticks causales del grafo subyacente."""
        ...
```

---

## 3. El Manejo del Tiempo: El Timeline Dual

### 3.1 Dos Relojes Independientes

CausalAnim mantiene dos relojes que avanzan de forma independiente:

```
┌─────────────────────────────────────────────────────────────┐
│ τ (causal ticks)     0   1   2   3   4   5   6   7   ...   │
│                      ●───●───●───●───●───●───●───●          │
│                      │       │           │                   │
│                      ▼       ▼           ▼                   │
│ t (viewer seconds)  0.0    0.5   1.0   1.2  1.5  2.0  ...  │
│                      ●──────●─────●──●───●────●──           │
│                     sprinkle   reduce   K₅!  walkers        │
└─────────────────────────────────────────────────────────────┘
```

**τ (ticks causales):** El reloj interno del grafo. Cada tick corresponde
a un evento discreto: un nodo sprinkleado, un paso de walker, una arista
reducida. Es estrictamente entero y monótono.

**t (segundos de presentación):** El reloj del espectador humano. Es
continuo (float) y su relación con τ es controlada por el `pace` de
la escena.

### 3.2 La Función de Pace (Ritmo)

El mapeo τ → t se controla con una función de pace que define cuántos
segundos de presentación consume cada tick causal:

```python
class Timeline:
    def __init__(self):
        self.segments = []   # lista de (τ_start, τ_end, pace_fn)

    def set_pace(self, ticks_per_second: float):
        """Pace constante: N ticks causales por segundo de video."""
        ...

    def rush(self, ticks: int, duration: float):
        """Comprime N ticks en duration segundos (fast-forward)."""
        ...

    def slow_motion(self, ticks: int, duration: float):
        """Estira N ticks sobre duration segundos (cámara lenta)."""
        ...

    def pause(self, duration: float):
        """Congela τ, avanza solo t (para narración/anotación)."""
        ...
```

**Ejemplos de uso:**

| Momento narrativo                  | τ ticks | t segundos | Pace         |
|------------------------------------|---------|------------|--------------|
| Sprinkle de 10⁴ nodos             | 10000   | 3.0        | rush         |
| Cierre causal (ráfaga de aristas)  | ~10⁶    | 1.5        | rush extremo |
| Reducción transitiva (fade-out)    | 1       | 2.0        | slow_motion  |
| Detección del primer prisma        | 1       | 3.0        | slow_motion  |
| Pausa para ecuación $M = \kappa N$ | 0       | 4.0        | pause        |
| Amenaza K₅ y contracción           | 5       | 5.0        | slow_motion  |
| Difusión de 100 walkers, 30 pasos  | 3000    | 6.0        | pace normal  |

### 3.3 wait() — Semántica Clara

```python
# Pausa de PRESENTACIÓN: el grafo no avanza, el espectador lee
scene.wait(2.0)  # 2 segundos reales. τ no cambia.

# Avance CAUSAL: el grafo avanza N ticks, la duración visual es automática
scene.wait_ticks(50)  # 50 ticks causales. t avanza según el pace activo.
```

La distinción es fundamental: `wait()` es para el humano, `wait_ticks()`
es para la física.

---

## 4. Stack Tecnológico

### 4.1 Decisión de Arquitectura

**Elección: Híbrido Rust + Python (PyO3).**

Justificación:

| Criterio              | 100% Python (manim) | 100% Rust (Bevy/wgpu) | Híbrido (Rust+PyO3) |
|-----------------------|---------------------|----------------------|---------------------|
| Rendimiento 10M nodos | ✗ Imposible         | ✓ 60 FPS nativo      | ✓ 60 FPS (Rust GPU) |
| Ergonomía de scripting| ✓ Pythónico         | ✗ Verboso             | ✓ Python + Rust core|
| Integración Colision. | ✗ Serializar CSV    | ✓ Enlace directo     | ✓ PyO3 ↔ crate dep  |
| Ecosistema LaTeX      | ✓ manim lo tiene    | ✗ Hay que construirlo | ✓ Subprocess latex  |
| Curva de aprendizaje  | ✓ Baja              | ✗ Alta                | ◐ Media             |
| Reproducibilidad      | ✓ Script = video    | ✓ Determinista        | ✓ Script = video    |

**El 100% Rust se descarta** porque la ergonomía de scripting de escenas
en Rust (lifetimes, borrowing, traits) haría que escribir una escena se
sienta como escribir un driver de kernel. El objetivo es que un físico
pueda describir una escena en 30 líneas de Python.

**El 100% Python se descarta** porque manim colapsa alrededor de N = 5000
nodos (Cairo no escala), y necesitamos N ≥ 10⁶ para visualizaciones
fieles a la simulación.

### 4.2 Componentes del Stack

```
┌──────────────────────────────────────────────────────────┐
│                    Script de Escena (.py)                 │
│         from causal_anim import Scene, Sprinkle, ...     │
├──────────────────────────────────────────────────────────┤
│              causal_anim (Python package)                 │
│     Scene DSL · Timeline · Camera · LaTeX renderer       │
├──────────────────────┬───────────────────────────────────┤
│   PyO3 bridge        │     Opcional: Jupyter widget      │
│   causal_anim_core   │     (live preview en notebook)    │
├──────────────────────┴───────────────────────────────────┤
│              causal_anim_core (Rust crate)                │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │  Layout   │  │  LOD     │  │  Renderer (wgpu)     │   │
│  │  Engine   │  │  Manager │  │                      │   │
│  │           │  │          │  │  - Instanced points  │   │
│  │ Stratified│  │ Frustum  │  │  - Bézier edges      │   │
│  │ + Spring  │  │ culling  │  │  - Glow/bloom post   │   │
│  │ (GPU      │  │ + LOD    │  │  - Text atlas        │   │
│  │  compute) │  │ switch   │  │  - Frame export      │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  causal_set_sim (crate dependency, existente)    │    │
│  │  sprinkle() · build_hasse_direct() ·             │    │
│  │  apply_defect() · run_walkers()                  │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 4.3 Dependencias Concretas

**Rust (causal_anim_core/Cargo.toml):**
```toml
[dependencies]
causal_set_sim = { path = "../prism_simmulation" }  # Colisionador
wgpu = "24"                    # GPU rendering (Vulkan/Metal/DX12)
winit = "0.30"                 # Windowing (preview mode)
glam = "0.29"                  # Vectores/matrices GPU-friendly
bytemuck = "1.21"              # Pod casting para buffers GPU
image = "0.25"                 # Exportar frames PNG
rayon = "1.10"                 # Paralelismo CPU (layout)
pyo3 = { version = "0.23", features = ["extension-module"] }

[lib]
crate-type = ["cdylib"]        # Compilar como .so para Python
name = "causal_anim_core"
```

**Python (pyproject.toml):**
```toml
[project]
name = "causal-anim"
requires-python = ">=3.10"
dependencies = []              # Zero deps puras; el core es Rust

[build-system]
requires = ["maturin>=1.0"]
build-backend = "maturin"

[tool.maturin]
features = ["pyo3/extension-module"]
```

### 4.4 Pipeline de Renderizado

```
Escena (.py)
    │
    ▼
causal_anim_core::build_scene()
    │
    ├──▶ Fase Layout (GPU compute shader, ~50 iter)
    │       Input:  CSR (adj_head, adj_data) + depths
    │       Output: Vec<[f32; 3]> posiciones por nodo
    │
    ├──▶ Fase LOD (CPU, O(N))
    │       Input:  posiciones + camera frustum
    │       Output: sets de nodos/aristas visibles por nivel
    │
    ├──▶ Fase Render (GPU render pipeline, por frame)
    │       Pass 1: Instanced circles (nodos)
    │       Pass 2: Instanced lines/Bézier (aristas)
    │       Pass 3: Glow bloom (post-process)
    │       Pass 4: Text overlay (anotaciones)
    │       Output: RGBA frame buffer
    │
    └──▶ Fase Encode (CPU, paralelo)
            Input:  secuencia de frames PNG
            Output: video MP4/WebM (vía ffmpeg subprocess)
```

Framerate objetivo: **60 FPS** en preview, **4K@60** en export.

### 4.5 Modos de Ejecución

| Modo          | Uso                               | Salida              |
|---------------|-----------------------------------|----------------------|
| `preview`     | Ventana interactiva con controles | Ventana winit + wgpu |
| `export`      | Renderizado offline a video       | frames/ → MP4        |
| `jupyter`     | Widget inline en notebook         | Canvas WebGPU        |
| `snapshot`    | Frame individual de alta res      | PNG/SVG              |

---

## 5. Ejemplo Completo: Nacimiento de un Electrón desde el Vacío

```python
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
"""
from causal_anim import (
    Scene, Sprinkle, BuildCausalClosure, ReduceHasse,
    DetectPrism, DetectThreat, ContractK5,
    DiffuseWalkers, ShowSpectralDimension,
    Annotate, Camera
)

scene = Scene("electron_genesis", resolution=(3840, 2160), fps=60)

# ═══════════════════════════════════════════════════════
# ACTO I — El Vacío
# ═══════════════════════════════════════════════════════

scene.play(Annotate(
    r"$\text{El vacío es un diagrama de Hasse finito, libre de triángulos.}$",
    position=(0.5, 0.9), duration=3.0
))

# Siembra 500 eventos en el diamante causal
vacuum = Sprinkle(N=500, seed=42)
scene.timeline.rush(ticks=500, duration=3.0)      # 500 nodos en 3 segundos
scene.play(vacuum)

scene.wait(1.0)

# Cierre causal: mostrar TODAS las relaciones (bola de pelo)
closure = BuildCausalClosure(vacuum)
scene.timeline.rush(ticks=closure.edge_count, duration=1.5)
scene.play(closure)

scene.play(Annotate(
    r"$u \prec v \iff (t_v - t_u)^2 > |\Delta\vec{x}|^2$",
    position=(0.5, 0.1), duration=2.0
))
scene.wait(1.0)

# Reducción transitiva: de O(N²) aristas a O(N·D) con D ≤ 15
hasse = ReduceHasse(closure)
scene.timeline.slow_motion(ticks=1, duration=2.5)
scene.play(hasse)

scene.play(Annotate(
    r"$\text{Reducción transitiva: solo quedan las relaciones de cobertura.}$",
    position=(0.5, 0.1), duration=2.0
))
scene.wait(2.0)

# ═══════════════════════════════════════════════════════
# ACTO II — La Cristalización
# ═══════════════════════════════════════════════════════

# El Colisionador detecta un prisma K_{2,3}
# (en la práctica, hasse.detect_prisms() llama a skyrmion::apply_defect)
prism = DetectPrism(
    origin=42, destination=187,
    belly=[91, 103, 156],
    generation=1                           # Gen1 → electrón
)

scene.play(Camera().focus_on(prism, zoom=4.0), duration=1.5)
scene.wait(0.5)

scene.timeline.slow_motion(ticks=1, duration=3.0)
scene.play(prism)

scene.play(Annotate(
    r"$\mathcal{P}(u, v, W) \cong K_{2,3} \quad M = \kappa \cdot 3$",
    position=(0.5, 0.85), duration=3.0
))
scene.wait(2.0)

# ═══════════════════════════════════════════════════════
# ACTO III — La Amenaza de Kuratowski
# ═══════════════════════════════════════════════════════

# Un nodo externo conecta a ambos polos y 2 intermediarios → K₅
threat = DetectThreat(threat_node=220, prism=prism)
scene.timeline.slow_motion(ticks=1, duration=2.0)
scene.play(threat)

scene.play(Annotate(
    r"$\text{Amenaza } K_5\text{: nodo 220 conecta a ambos polos + 2 intermediarios}$",
    position=(0.5, 0.1), duration=2.5
))
scene.wait(1.5)

# Contracción: el nodo amenaza se absorbe en el polo de mayor grado
contraction = ContractK5(threat, absorber="max_degree")
scene.timeline.slow_motion(ticks=1, duration=2.5)
scene.play(contraction)

scene.play(Annotate(
    r"$\text{Contracción de vértice: planaridad restaurada.}$",
    position=(0.5, 0.1), duration=2.0
))
scene.wait(2.0)

# ═══════════════════════════════════════════════════════
# ACTO IV — La Masa como Demora Causal
# ═══════════════════════════════════════════════════════

scene.play(Camera().pull_back(scale=2.0, duration=1.0))

# Lanzar walkers desde el polo pasado del prisma
walkers = DiffuseWalkers(
    graph=hasse,
    n_walkers=50,
    origins=[prism.origin],
    steps=30
)
ds_plot = ShowSpectralDimension(walkers, position="bottom-right")

scene.timeline.set_pace(ticks_per_second=10)    # 10 pasos/segundo
scene.play(walkers, ds_plot)

scene.play(Annotate(
    r"$\langle\tau_{\mathrm{res}}\rangle \propto N$"
    r"$\quad\text{— La masa es demora topológica.}$",
    position=(0.5, 0.85), duration=4.0
))
scene.wait(3.0)

# ═══════════════════════════════════════════════════════
# EPÍLOGO
# ═══════════════════════════════════════════════════════

scene.play(Camera().pull_back(scale=10.0, duration=3.0))
scene.play(Annotate(
    r"$\text{Un electrón no es una partícula puntual.}$"
    "\n"
    r"$\text{Es una bifurcación causal que el vacío no puede simplificar.}$",
    position=(0.5, 0.5), duration=5.0
))
scene.wait(3.0)

# Renderizar
scene.export("electron_genesis.mp4")
```

**Duración total estimada:** ~55 segundos.
**Nodos renderizados:** 500 (escala atómica, full detail).
**Frame count:** 55 × 60 = 3300 frames @ 4K.

---

## 6. Estructura de Directorios Propuesta

```
animations/
├── RFC_001_causal_anim.md              ← Este documento
├── causal_anim_core/                   ← Crate Rust (motor)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                      ← PyO3 module root
│       ├── layout.rs                   ← Stratified spring layout (GPU)
│       ├── lod.rs                      ← Level-of-detail manager
│       ├── renderer.rs                 ← wgpu render pipeline
│       ├── timeline.rs                 ← Dual clock (τ, t)
│       └── bridge.rs                   ← Adaptador CSR ↔ causal_set_sim
├── causal_anim/                        ← Python package (scripting DSL)
│   ├── __init__.py                     ← Re-exports públicos
│   ├── scene.py                        ← Scene, play(), wait()
│   ├── primitives.py                   ← Sprinkle, Imply, ReduceHasse, ...
│   ├── camera.py                       ← Camera controls
│   ├── annotate.py                     ← LaTeX text overlays
│   └── timeline.py                     ← Timeline Python wrapper
├── pyproject.toml                      ← Build config (maturin)
├── scenes/                             ← Scripts de escenas
│   ├── prism_simmulation/              ← Escenas del colisionador
│   ├── modulo_synthesis_vol_I/         ← Escenas de Vol. I
│   ├── modulo_synthesis_vol_II/        ← Escenas de Vol. II
│   └── modulo_synthesis_kuratowski_calculus/
└── renders/                            ← Output de videos (gitignored)
```

---

## 7. Fases de Implementación

### Fase 0 — Scaffolding (1 semana)

- [ ] Crear crate `causal_anim_core` con `Cargo.toml` enlazando a
      `causal_set_sim` como dependencia
- [ ] Configurar PyO3 + maturin para build del módulo Python
- [ ] Stub de `Scene.export()` que produce un PNG estático de un
      sprinkle de 100 nodos (layout + wgpu + frame export)
- [ ] CI: `cargo test` + `maturin develop` + `python -c "import causal_anim"`

### Fase 1 — Layout Engine (2 semanas)

- [ ] Implementar cálculo de profundidad causal (longest path from ∂⁻)
- [ ] Implementar relajación spring intra-capa (CPU primero, GPU después)
- [ ] Test: layout de Hasse con N=1000 produce y-monotonía estricta
- [ ] Benchmark: layout de N=100k converge en < 1s

### Fase 2 — Renderer Básico (2 semanas)

- [ ] Pipeline wgpu: instanced circles + instanced lines
- [ ] Post-process: bloom/glow para nodos de prisma
- [ ] Export a PNG frames + ffmpeg → MP4
- [ ] Preview mode con winit (pan, zoom, orbit con mouse)

### Fase 3 — Primitivas de Animación (3 semanas)

- [ ] Sprinkle (aparición progresiva)
- [ ] BuildCausalClosure + ReduceHasse (fade in/out de aristas)
- [ ] DetectPrism (halo, bundling, burbuja convexa)
- [ ] DetectThreat + ContractK5 (parpadeo, absorción, morph)
- [ ] DiffuseWalkers (partículas con trail)
- [ ] Timeline dual (τ, t) con rush/slow_motion/pause

### Fase 4 — Pulido y Escenas (2 semanas)

- [ ] Annotate con LaTeX rendering
- [ ] ShowSpectralDimension (gráfico incrustado)
- [ ] Camera system completo (focus, orbit, pull_back)
- [ ] Escena de referencia: `electron_genesis.py` renderizada completa
- [ ] LOD system para N > 10⁴

### Fase 5 — Optimización GPU (continua)

- [ ] Migrar layout spring a compute shader
- [ ] Frustum culling en GPU
- [ ] Benchmark: N = 10⁷ a ≥ 30 FPS en preview mode
- [ ] Benchmark: N = 10⁶ a 60 FPS en export mode

---

## 8. Decisiones de Diseño Abiertas

| # | Pregunta                                                  | Opciones                       | Decisión pendiente |
|---|-----------------------------------------------------------|--------------------------------|--------------------|
| 1 | ¿Soporte 3D rotable o solo 2D plano para export?          | 2D / 3D / ambos               | Ambos (toggle)     |
| 2 | ¿El script `.py` debería soportar hot-reload en preview?   | Sí / No                       | Sí (watchdog)      |
| 3 | ¿Formato intermedio de escena serializada?                  | JSON / MessagePack / ninguno  | Pendiente          |
| 4 | ¿Soporte para audio sync (narración + animación)?          | Sí / futuro                   | Futuro (post-MVP)  |
| 5 | ¿Exportar a WebGPU interactivo (web embed)?                | Sí / futuro                   | Futuro (Fase 6)    |

---

## Apéndice A: Glosario de Mapeos Físico → Visual

| Concepto físico              | Representación visual                                |
|------------------------------|------------------------------------------------------|
| Evento (nodo)                | Círculo con radio ∝ grado Hasse                      |
| Relación causal (arista)     | Línea dirigida (flecha tenue, gris)                  |
| Reducción transitiva         | Fade-out de aristas redundantes                      |
| Profundidad causal           | Coordenada Y (más profundo = más arriba)             |
| Anticadena (capa)            | Franja horizontal de nodos                           |
| Prisma K_{2,N}               | Burbuja coloreada con bundled Bézier                 |
| Polo pasado u                | Círculo grande, borde luminoso inferior              |
| Polo futuro v                | Círculo grande, borde luminoso superior              |
| Belly W                      | Arco de N círculos medianos, distribuidos horizontal  |
| Masa topológica N            | Etiqueta numérica + grosor de burbuja                |
| Amenaza K₅                   | Parpadeo rojo + líneas discontinuas                  |
| Contracción                  | Animación de absorción (ease-in-out)                 |
| Walker (random walk)         | Partícula luminosa con trail decayente               |
| Retorno al origen            | Flash en nodo de partida                             |
| Dimensión espectral d_S      | Gráfico 2D incrustado (miniatura)                    |
| Flujo de atracción           | Streamlines teal convergentes                        |
| Flujo de repulsión           | Streamlines terracota divergentes                    |
| Generación (1,2,3)           | Color del prisma (teal, ámbar, terracota)            |
| Materia oscura (estéril)     | Gris translúcido, sin flujo dirigido                 |

---

## Apéndice B: Paleta de Colores

```
Background:     #1D3557  (azul oscuro profundo)
Vacuum nodes:   #CED4DA  (gris claro)
Vacuum edges:   #6C757D  (gris medio, alpha 0.4)
Hasse reduced:  #F1FAEE  (blanco cálido)

Gen1 (e⁻):     #2A9D8F  (teal)
Gen2 (μ):      #E9C46A  (ámbar)
Gen3 (τ):      #E76F51  (terracota)
Anti1 (e⁺):    #48BFE3  (cian claro)
Sterile (DM):  #8D99AE  (gris azulado)

Threat K₅:     #E63946  (rojo alarma)
Contraction:   #FFFFFF  (flash blanco)
Walker trail:  #F4A261  (naranja suave, alpha decay)
Return flash:  #FFBE0B  (dorado)

Annotation:    #F1FAEE  (blanco cálido)
Grid lines:    #457B9D  (azul medio, alpha 0.15)
```
