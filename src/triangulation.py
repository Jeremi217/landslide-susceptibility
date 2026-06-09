"""
Buduje triangulację Delaunaya na punktach NMT,
a następnie oblicza kąt nachylenia każdego trójkąta, według:
  1. Spłaszcz siatkę rastra do chmury punktów (X, Y, Z).
  2. Triangulacja Delaunaya w 2D (scipy.spatial.Delaunay) na (X, Y).
  3. Dla każdego trójkąta wyznacz wektor normalny do płaszczyzny.
  4. Kąt między normalną a osią Z = kąt nachylenia (slope).
"""

import numpy as np
from scipy.spatial import Delaunay
import struct
import matplotlib.pyplot as plt


def build_tin(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
              max_points: int = 80_000) -> tuple[np.ndarray, np.ndarray]:

    # Spłaszcz i usuń NaN
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z.ravel()

    valid = ~np.isnan(z_flat)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    z_flat = z_flat[valid]

    n_points = len(x_flat)
    print(f"[triangulation] Liczba punktów (po usunięciu NaN): {n_points:,}")

    # Dla dużych rastrów próbkuj punkty
    if n_points > max_points:
        print(f"[triangulation] Próbkowanie do {max_points:,} punktów (raster zbyt duży)...")
        idx = np.random.choice(n_points, max_points, replace=False)
        idx.sort()
        x_flat = x_flat[idx]
        y_flat = y_flat[idx]
        z_flat = z_flat[idx]

    points = np.column_stack([x_flat, y_flat, z_flat])

    print("[triangulation] Buduję triangulację Delaunaya (może chwilę potrwać)...")
    tri = Delaunay(points[:, :2])  # triangulacja w 2D (X, Y)
    print(f"[triangulation] Liczba trójkątów: {len(tri.simplices):,}")

    return points, tri


def compute_slope_per_triangle(points: np.ndarray,
                                tri: Delaunay) -> np.ndarray:

    simplices = tri.simplices  # (M, 3) – indeksy wierzchołków

    A = points[simplices[:, 0]]  # (M, 3)
    B = points[simplices[:, 1]]
    C = points[simplices[:, 2]]

    AB = B - A
    AC = C - A

    normals = np.cross(AB, AC)  # (M, 3)

    # Długość wektora normalnego
    norm_lengths = np.linalg.norm(normals, axis=1)

    # Unikaj dzielenia przez zero
    valid = norm_lengths > 1e-10
    slopes = np.full(len(simplices), np.nan)

    cos_angle = np.abs(normals[valid, 2]) / norm_lengths[valid]
    cos_angle = np.clip(cos_angle, 0, 1)
    slopes[valid] = np.degrees(np.arccos(cos_angle))

    print(f"[triangulation] Kąt nachylenia: min={np.nanmin(slopes):.1f}°, "
          f"max={np.nanmax(slopes):.1f}°, "
          f"średnia={np.nanmean(slopes):.1f}°")

    return slopes


def get_triangle_centroids(points: np.ndarray, tri: Delaunay) -> np.ndarray:
    simplices = tri.simplices
    A = points[simplices[:, 0], :2]
    B = points[simplices[:, 1], :2]
    C = points[simplices[:, 2], :2]
    return (A + B + C) / 3.0

def export_ply(points: np.ndarray, tri, slopes: np.ndarray, filepath: str):

    # Mapuj nachylenie (0-45°)
    cmap = plt.cm.RdYlGn_r
    slopes_norm = np.clip(slopes / 45.0, 0, 1)
    colors = (cmap(slopes_norm)[:, :3] * 255).astype(np.uint8)

    vertices = points  # (N, 3)
    faces = tri.simplices  # (M, 3)

    with open(filepath, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))

        for v in vertices:
            f.write(struct.pack('<fff', float(v[0]), float(v[1]), float(v[2])))

        for i, face in enumerate(faces):
            f.write(struct.pack('<B', 3))
            f.write(struct.pack('<iii', int(face[0]), int(face[1]), int(face[2])))
            r, g, b = colors[i]
            f.write(struct.pack('<BBB', int(r), int(g), int(b)))

    print(f"[ply] Zapisano: {filepath}")