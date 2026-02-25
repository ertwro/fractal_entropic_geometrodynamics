"""Shared publication style and colour palette for all FEG figures."""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colourblind-safe Okabe-Ito palette
C = {
    "vac":     "#4C72B0",
    "def":     "#C44E52",
    "gen1":    "#55A868",
    "gen2":    "#8172B2",
    "gen3":    "#CCB974",
    "anti1":   "#64B5CD",
    "flux_a":  "#C44E52",
    "flux_r":  "#4C72B0",
    "dark":    "#2f2f2f",
    "vis":     "#E8A838",
    "sterile": "#999999",
    "fit":     "#2f2f2f",
    "planck":  "#E8A838",
}


def apply_style(*, compact=False):
    """Set matplotlib rcParams for publication figures.

    Parameters
    ----------
    compact : bool
        If True, use smaller fonts suitable for 2×2 composite panels.
    """
    if compact:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "lines.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linewidth": 0.3,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        })
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linewidth": 0.4,
        })


def savefig(fig, name, out_dir):
    """Save figure as PDF + PNG, then close it."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    print(f"  [+] {name}.pdf / .png")
    plt.close(fig)
