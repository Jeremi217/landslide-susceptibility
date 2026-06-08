import argparse
import sys
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loader import load_asc, get_xy_grid
from triangulation import build_tin, compute_slope_per_triangle, get_triangle_centroids, export_ply
from risk import classify_slopes, rasterize_risk
from visualize import plot_slope_raster, plot_tin, plot_risk_map, plot_slope_histogram


def main():
    parser = argparse.ArgumentParser(
        description="Landslide Susceptibility Analysis from DTM (ASC) using Delaunay TIN"
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Ścieżka do pliku NMT w formacie ASC (domyślnie: data/nmt.asc)"
    )
    parser.add_argument(
        "--max-points", "-n",
        type=int,
        default=80_000,
        help="Maks. liczba punktów do triangulacji (domyślnie: 80000)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Landslide Susceptibility Analysis")
    print("  Delaunay TIN + DTM slope classification")
    print("=" * 60)

    t0 = time.time()

    # 1. Wczytaj NMT
    print("\n[1/5] Wczytywanie NMT...")
    if args.input is None:
        input_path = Path(__file__).resolve().parent / "data" / "nmt.asc"
    else:
        input_path = Path(args.input)
        if not input_path.is_absolute() and not input_path.exists():
            input_path = Path(__file__).resolve().parent / input_path
    Z, meta = load_asc(str(input_path))
    X, Y = get_xy_grid(Z, meta)

    # 2. Mapa nachylenia rastrowego
    print("\n[2/5] Generowanie mapy nachylenia (rastrowej)...")
    plot_slope_raster(Z, meta)

    # 3. Delaunay + kąty nachylenia
    print("\n[3/5] Triangulacja Delaunaya...")
    points, tri = build_tin(X, Y, Z, max_points=args.max_points)
    slopes = compute_slope_per_triangle(points, tri)
    centroids = get_triangle_centroids(points, tri)

    export_ply(points, tri, slopes, str(Path(__file__).resolve().parent / "output" / "model.ply"))

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.RdYlGn_r
    slopes_norm = np.clip(slopes / 45.0, 0, 1)
    colors = cmap(slopes_norm)

    verts = [[points[tri.simplices[i, 0]],
              points[tri.simplices[i, 1]],
              points[tri.simplices[i, 2]]] for i in range(len(tri.simplices))]

    poly = Poly3DCollection(verts, facecolors=colors, edgecolors='none', alpha=0.9)
    ax.add_collection3d(poly)

    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_zlim(points[:, 2].min(), points[:, 2].max())
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Terrain Model')

    plt.tight_layout()
    plt.show()

    # 4. Klasyfikacja ryzyka
    print("\n[4/5] Klasyfikacja ryzyka osuwiskowego...")
    risk = classify_slopes(slopes)
    risk_grid = rasterize_risk(centroids, risk, Z, meta)

    # 5. Wizualizacje
    print("\n[5/5] Generowanie map wynikowych...")
    plot_tin(points, tri, slopes)
    plot_risk_map(risk_grid, Z, meta)
    plot_slope_histogram(slopes)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Gotowe! Czas przetwarzania: {elapsed:.1f} s")
    print(f"  Wyniki zapisane w katalogu: output/")
    print(f"    - slope_map.png       (nachylenie rastrowe)")
    print(f"    - tin_map.png         (TIN Delaunaya z nachyleniem)")
    print(f"    - risk_map.png        (mapa ryzyka osuwiskowego)")
    print(f"    - slope_histogram.png (rozkład kątów)")
    print("=" * 60)


if __name__ == "__main__":
    main()
