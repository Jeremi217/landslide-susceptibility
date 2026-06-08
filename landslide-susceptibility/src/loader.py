"""
Wczytuje plik NMT w formacie ASC (ASCII Grid) z geoportalu.
Zwraca siatkę wysokości (numpy array) oraz metadane georeferncyjne.
"""

import numpy as np
from pathlib import Path


def load_asc(filepath: str) -> tuple[np.ndarray, dict]:

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")

    meta = {}
    header_lines = 6

    with open(filepath, "r") as f:
        for i in range(header_lines):
            line = f.readline().strip().split()
            key = line[0].lower()
            # obsługa wariantów: xllcorner / xllcenter
            meta[key] = float(line[1]) if "." in line[1] else int(line[1])

    Z = np.loadtxt(filepath, skiprows=header_lines, dtype=np.float32)

    # Zamień wartości NoData na NaN
    nodata_val = meta.get("nodata_value", meta.get("nodata", -9999))
    Z[Z == nodata_val] = np.nan

    print(f"[loader] Wczytano: {filepath.name}")
    print(f"         Rozmiar: {Z.shape[0]} wierszy x {Z.shape[1]} kolumn")
    print(f"         Rozdzielczość: {meta.get('cellsize', '?')} m")
    print(f"         Zakres wysokości: {np.nanmin(Z):.1f} – {np.nanmax(Z):.1f} m n.p.m.")

    return Z, meta


def get_xy_grid(Z: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:

    cellsize = meta["cellsize"]
    xll = meta.get("xllcorner", meta.get("xllcenter", 0))
    yll = meta.get("yllcorner", meta.get("yllcenter", 0))

    nrows, ncols = Z.shape

    # X rośnie w prawo, Y rośnie w górę
    x = xll + (np.arange(ncols) + 0.5) * cellsize
    y = yll + (np.arange(nrows - 1, -1, -1) + 0.5) * cellsize

    X, Y = np.meshgrid(x, y)
    return X, Y
