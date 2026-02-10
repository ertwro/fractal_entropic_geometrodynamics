"""
Generate a random 2D Causal Set Hasse diagram as a standalone TikZ file.

The script sprinkles points into a 2D causal diamond using light-cone
coordinates, builds the causal order, takes the transitive reduction
(Hasse diagram), and exports the result as compilable TikZ.

Usage:
    python generate_hasse.py
"""
from __future__ import annotations

import random
from pathlib import Path

import networkx as nx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_POINTS = 25
OUTPUT = Path(__file__).resolve().parent / "fig1_2_hasse.tex"

# Colour palette (must match the main book)
COL_STRUCTURE = "1A535C"
COL_GOLD      = "C6892A"


def generate_causal_set_tikz(
    filename: Path = OUTPUT,
    num_points: int = NUM_POINTS,
    seed: int = SEED,
) -> None:
    """Sprinkle a causal set and write its Hasse diagram to *filename*."""
    print(f"Generating Causal Set Hasse Diagram → {filename}")
    random.seed(seed)

    # Sprinkle in light-cone coordinates (u, v) ∈ [0, 1]²
    points = [(random.random(), random.random()) for _ in range(num_points)]
    # Sort by time t = u + v for a natural vertical layout
    points.sort(key=lambda p: p[0] + p[1])

    # Build the full causal-order DAG
    G = nx.DiGraph()
    for i in range(num_points):
        G.add_node(i, pos=points[i])

    for i in range(num_points):
        ui, vi = points[i]
        for j in range(num_points):
            if i == j:
                continue
            uj, vj = points[j]
            if ui < uj and vi < vj:
                G.add_edge(i, j)

    # Transitive reduction → Hasse diagram
    H = nx.transitive_reduction(G)

    # --- Write TikZ ---
    with open(filename, "w") as f:
        f.write(
            r"""\documentclass[tikz,border=2mm]{standalone}
\definecolor{structure}{HTML}{"""
            + COL_STRUCTURE
            + r"""}
\definecolor{main}{HTML}{"""
            + COL_GOLD
            + r"""}
\begin{document}
\begin{tikzpicture}[
    scale=5,
    event/.style={circle, fill=structure, inner sep=1.5pt},
    link/.style={draw=gray!50, thin}
]
"""
        )

        # Nodes (x = v − u, t = v + u)
        for i, (u, v) in enumerate(points):
            x = v - u
            t = u + v
            f.write(f"\\node[event] ({i}) at ({x:.3f}, {t:.3f}) {{}};\n")

        # Edges
        for u_node, v_node in H.edges():
            f.write(f"\\draw[link] ({u_node}) -- ({v_node});\n")

        f.write(r"\end{tikzpicture}" "\n" r"\end{document}" "\n")

    print(f"  ✓ Written {filename.name}")


if __name__ == "__main__":
    generate_causal_set_tikz()
