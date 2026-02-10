"""
Figure generation for Modulo Synthesis (Volumes I & II).

Generates publication-quality matplotlib plots:
  - Fig 1.1: Lattice vs Poisson Sprinkling under Lorentz boost
  - Fig 1.5: Spectral Dimension Flow (2D → 4D)
  - Fig 1.7: Cosmic Bounce (scale factor + density)
  - Fig 2.6: Lambda Decay (geometric vs constant Λ)

Usage:
    python generate_plots.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------
FIGURE_DIR = Path(__file__).resolve().parent
SEED = 42

# Colour palette (matches TikZ figures)
TEAL      = "#1A535C"
GOLD      = "#C6892A"
SLATE     = "#2E4057"
BURGUNDY  = "#8B2635"
GREEN     = "#3A7D44"

# Publication-quality defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("font", family="serif", size=11)
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["figure.dpi"] = 300


# ---------------------------------------------------------------------------
# Fig 1.1 — Lattice vs Poisson Sprinkling
# ---------------------------------------------------------------------------
def plot_lattice_vs_sprinkling() -> None:
    """Demonstrate Lorentz invariance of Poisson sprinkling vs lattice."""
    print("Generating Figure 1.1: Lattice vs Sprinkling…")
    rng = np.random.default_rng(SEED)

    beta = 0.8
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    W = 2.5                     # plot half-width

    # --- Regular lattice in (x, t) ---
    grid = np.linspace(-10, 10, 15)
    X, T = np.meshgrid(grid, grid)
    x_lat, t_lat = X.ravel(), T.ravel()

    # Boost the lattice
    xp_lat = gamma * (x_lat - beta * t_lat)
    tp_lat = gamma * (t_lat - beta * x_lat)

    # --- Poisson sprinkling in light-cone coordinates ---
    n_pts = 500
    u = rng.uniform(-5, 5, n_pts)
    v = rng.uniform(-5, 5, n_pts)
    x_poi = (v - u) / 2.0
    t_poi = (v + u) / 2.0

    # Boost in light-cone coords preserves area element
    rapidity = np.arctanh(beta)
    up = u * np.exp(-rapidity)
    vp = v * np.exp(rapidity)
    xp_poi = (vp - up) / 2.0
    tp_poi = (vp + up) / 2.0

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (x_lat,  t_lat,  "Lattice (Rest Frame)",       "black",  "o"),
        (xp_lat, tp_lat, r"Lattice (Boosted $\beta=0.8$)", "red", "o"),
        (x_poi,  t_poi,  "Causal Set (Rest Frame)",    "black",  "x"),
        (xp_poi, tp_poi, r"Causal Set (Boosted $\beta=0.8$)", "blue", "x"),
    ]
    xlabels = [r"$x$", r"$x'$", r"$x$", r"$x'$"]
    ylabels = [r"$t$", r"$t'$", r"$t$", r"$t'$"]

    for ax, (px, pt, title, col, mk), xl, yl in zip(
        axes.ravel(), panels, xlabels, ylabels
    ):
        mask = (np.abs(px) < W) & (np.abs(pt) < W)
        ax.scatter(px[mask], pt[mask], c=col, s=10, marker=mk)
        ax.set_title(rf"\textbf{{{title}}}")
        ax.set_aspect("equal")
        ax.set_xlim(-W, W)
        ax.set_ylim(-W, W)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig1_1_lattice_vs_sprinkling.pdf")
    plt.close(fig)
    print("  ✓ Saved fig1_1_lattice_vs_sprinkling.pdf")


# ---------------------------------------------------------------------------
# Fig 1.5 — Spectral Dimension Flow
# ---------------------------------------------------------------------------
def plot_spectral_dimension() -> None:
    """Plot dS(k) = 4 − 2k² / (k² + M_Pl²) with M_Pl = 1."""
    print("Generating Figure 1.5: Spectral Dimension…")

    k = np.logspace(-2, 2, 500)
    d_S = 4.0 - 2.0 * k**2 / (k**2 + 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(k, d_S, lw=2.5, color=TEAL)
    ax.axhline(4, ls="--", color="gray", alpha=0.5, label="Macroscopic (4D)")
    ax.axhline(2, ls="--", color="gray", alpha=0.5, label="Microscopic (2D)")

    ax.set_xlabel(r"Energy Scale $k / M_{\mathrm{Pl}}$", fontsize=12)
    ax.set_ylabel(r"Spectral Dimension $d_S(k)$", fontsize=12)
    ax.set_title(r"\textbf{Dimensional Reduction: From 4D to 2D}", fontsize=14)
    ax.set_ylim(1.5, 4.5)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    ax.text(0.02, 3.8, r"\textbf{General Relativity}", fontsize=12, color=SLATE)
    ax.text(10, 2.2, r"\textbf{Quantum Gravity}", fontsize=12, color=BURGUNDY)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig1_5_spectral_dimension.pdf")
    plt.close(fig)
    print("  ✓ Saved fig1_5_spectral_dimension.pdf")


# ---------------------------------------------------------------------------
# Fig 1.7 — Cosmic Bounce
# ---------------------------------------------------------------------------
def plot_cosmic_bounce() -> None:
    """Toy bounce model: a(t) = √(a_min² + t²)."""
    print("Generating Figure 1.7: Cosmic Bounce…")

    a_min_sq = 0.2
    t = np.linspace(-3, 3, 500)
    a = np.sqrt(a_min_sq + t**2)
    rho = 1.0 / a**4
    rho /= rho.max()  # normalise

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Scale factor (left axis)
    ax1.plot(t, a, color=TEAL, lw=2.5, label=r"$a(t)$")
    ax1.set_xlabel(r"Cosmic Time $t$ (Planck units)", fontsize=12)
    ax1.set_ylabel(r"Scale Factor $a(t)$", color=TEAL, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=TEAL)

    # Density (right axis)
    ax2 = ax1.twinx()
    ax2.plot(t, rho, color=BURGUNDY, lw=2, ls="--", label=r"$\rho(t)$")
    ax2.set_ylabel(
        r"Density $\rho / \rho_{\mathrm{max}}$", color=BURGUNDY, fontsize=12
    )
    ax2.tick_params(axis="y", labelcolor=BURGUNDY)

    ax1.axvline(0, color="gray", ls=":", alpha=0.5)
    ax1.text(0.1, 0.4, r"\textbf{Bounce Point}", transform=ax1.transAxes)
    fig.suptitle(
        r"\textbf{The Cosmic Bounce: Avoiding Singularity}", fontsize=14, y=0.98
    )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig1_7_cosmic_bounce.pdf")
    plt.close(fig)
    print("  ✓ Saved fig1_7_cosmic_bounce.pdf")


# ---------------------------------------------------------------------------
# Fig 2.6 — Lambda Decay
# ---------------------------------------------------------------------------
def plot_lambda_decay() -> None:
    r"""Geometric Λ(t) ∼ t⁻² vs constant Λ."""
    print("Generating Figure 2.6: Lambda Decay…")

    t = np.logspace(0, 60, 500)
    Lambda_t = 1.0 / t**2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(t, Lambda_t, color=GREEN, lw=2.5,
              label=r"Geometric Decay $\Lambda(t) \sim t^{-2}$")
    ax.axhline(1e-120, ls="--", color="red", lw=1.5,
               label=r"Standard Model $\Lambda = \text{const}$ (Future Heat Death)")

    # Mark "Today"
    t_now = 1e60
    ax.axvline(t_now, color="gray", ls=":", alpha=0.6)
    ax.scatter([t_now], [1e-120], color="black", zorder=10, s=40)
    ax.text(t_now * 2, 1e-110, r"\textbf{Today}", fontsize=12)

    ax.set_xlabel(r"Cosmic Time $t$ (Planck units)", fontsize=12)
    ax.set_ylabel(r"Cosmological Constant $\Lambda$", fontsize=12)
    ax.set_title(r"\textbf{The Fate of the Vacuum}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig2_6_lambda_decay.pdf")
    plt.close(fig)
    print("  ✓ Saved fig2_6_lambda_decay.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_lattice_vs_sprinkling()
    plot_spectral_dimension()
    plot_cosmic_bounce()
    plot_lambda_decay()
    print("\nAll figures generated successfully.")
