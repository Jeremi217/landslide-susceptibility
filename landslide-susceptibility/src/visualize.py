"""
Generuje statyczne mapy wynikowe:
  1. slope_map.png   – mapa kątów nachylenia (ciągła skala barwna)
  2. tin_map.png     – wizualizacja TIN z kolorowaniem wg nachylenia
  3. risk_map.png    – mapa klas ryzyka osuwiskowego
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as mtri
from matplotlib.colors import BoundaryNorm, ListedColormap
from pathlib import Path

from risk import RISK_CLASSES, RISK_COLORS, RISK_LABELS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _save(fig, filename: str, dpi: int = 200):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[visualize] Zapisano: {path}")
    plt.close(fig)


def plot_slope_raster(Z: np.ndarray, meta: dict, slopes_grid: np.ndarray = None):
    cellsize = meta["cellsize"]

    if slopes_grid is None:
        dy, dx = np.gradient(Z, cellsize, cellsize)
        slopes_grid = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slopes_grid, cmap="YlOrRd", vmin=0, vmax=45, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Slope angle [°]", fontsize=11)
    ax.set_title("Slope Map (from DTM)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    _save(fig, "slope_map.png")


def plot_tin(points: np.ndarray, tri, slopes: np.ndarray):
    fig, ax = plt.subplots(figsize=(11, 9))

    mpl_tri = mtri.Triangulation(points[:, 0], points[:, 1], tri.simplices)

    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=0, vmax=45)
    slopes_clipped = np.clip(slopes, 0, 45)
    tpc = ax.tripcolor(mpl_tri, facecolors=slopes_clipped,
                       cmap=cmap, norm=norm, edgecolors="none")

    cbar = fig.colorbar(tpc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Slope angle [°]", fontsize=11)
    ax.set_aspect("equal")
    ax.set_title("TIN — Slope per Triangle (Delaunay)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    _save(fig, "tin_map.png")


def plot_risk_map(risk_grid: np.ndarray, Z: np.ndarray, meta: dict):
    cmap = ListedColormap(RISK_COLORS)
    bounds = [i - 0.5 for i in range(len(RISK_CLASSES) + 1)]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(11, 9))

    # Hillshade jako tło
    Z_filled = np.where(np.isnan(Z), np.nanmean(Z), Z)
    dy, dx = np.gradient(Z_filled, meta["cellsize"], meta["cellsize"])
    hillshade = -dx * 0.5 + dy * 0.3
    hillshade = (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min() + 1e-10)
    ax.imshow(hillshade, cmap="gray", alpha=0.35, aspect="equal")

    risk_float = risk_grid.astype(float)
    risk_float[risk_grid == -1] = np.nan
    ax.imshow(risk_float, cmap=cmap, norm=norm, alpha=0.80, aspect="equal")

    patches = [mpatches.Patch(color=RISK_COLORS[i], label=RISK_LABELS[i])
               for i in range(len(RISK_CLASSES))]
    patches.append(mpatches.Patch(color="white", label="No data"))
    ax.legend(handles=patches, loc="lower right",
              title="Landslide risk", fontsize=9,
              title_fontsize=10, framealpha=0.9)

    ax.set_title("Landslide Susceptibility Map\n(based on DTM slope analysis via Delaunay TIN)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    _save(fig, "risk_map.png")


def plot_slope_histogram(slopes: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 4))
    valid = slopes[~np.isnan(slopes)]
    ax.hist(valid, bins=60, color="#fd8d3c", edgecolor="white", linewidth=0.3)

    for t in [5, 15, 25, 35]:
        ax.axvline(t, color="#333", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(t + 0.3, ax.get_ylim()[1] * 0.95, f"{t}°",
                fontsize=8, va="top", color="#333")

    ax.set_xlabel("Slope angle [°]", fontsize=11)
    ax.set_ylabel("Number of triangles", fontsize=11)
    ax.set_title("Distribution of Slope Angles (TIN triangles)", fontsize=13)
    ax.set_xlim(0, 60)
    _save(fig, "slope_histogram.png")
