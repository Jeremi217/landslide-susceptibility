"""
Klasyfikuje ryzyko osuwiskowe na podstawie kąta nachylenia stoku.

Progi oparte na literaturze geomorfologicznej:
  - Varnes (1978), Hungr et al. (2014)
  - Typowe wartości dla gruntów niespoistych i spoistych w Karpatach/Sudetach

Klasy ryzyka:
  0 – Brak ryzyka       (slope < 5°)
  1 – Niskie            (5° – 15°)
  2 – Umiarkowane       (15° – 25°)
  3 – Wysokie           (25° – 35°)
  4 – Bardzo wysokie    (> 35°)
"""

import numpy as np

# Definicja klas: (próg_dolny, próg_górny, etykieta, kolor_hex)
RISK_CLASSES = [
    (0,   5,  "No risk",       "#d4f1d4"),
    (5,   15, "Low",           "#ffffb2"),
    (15,  25, "Moderate",      "#fecc5c"),
    (25,  35, "High",          "#fd8d3c"),
    (35, 999, "Very High",     "#e31a1c"),
]

RISK_LABELS  = [r[2] for r in RISK_CLASSES]
RISK_COLORS  = [r[3] for r in RISK_CLASSES]
RISK_THRESHOLDS = [(r[0], r[1]) for r in RISK_CLASSES]


def classify_slopes(slopes: np.ndarray) -> np.ndarray:
    risk = np.full(slopes.shape, -1, dtype=np.int8)

    for cls_idx, (lo, hi, *_) in enumerate(RISK_CLASSES):
        mask = (slopes >= lo) & (slopes < hi)
        risk[mask] = cls_idx

    valid = ~np.isnan(slopes)
    counts = {RISK_LABELS[i]: int(np.sum(risk[valid] == i))
              for i in range(len(RISK_CLASSES))}

    print("[risk] Rozkład klas ryzyka (liczba trójkątów):")
    total = sum(counts.values())
    for label, count in counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"       {label:<15}: {count:>8,}  ({pct:5.1f}%)")

    return risk


def rasterize_risk(centroids: np.ndarray, risk: np.ndarray,
                   Z: np.ndarray, meta: dict) -> np.ndarray:
    """
    Projektuje klasyfikację ryzyka z trójkątów TIN z powrotem na siatkę rastra.
    Każda komórka rastra dostaje klasę ryzyka najbliższego centroidu trójkąta.
    """
    cellsize = meta["cellsize"]
    xll = meta.get("xllcorner", meta.get("xllcenter", 0))
    yll = meta.get("yllcorner", meta.get("yllcenter", 0))
    nrows, ncols = Z.shape

    risk_grid = np.full((nrows, ncols), -1, dtype=np.int8)

    # Zamień współrzędne centroidów na indeksy rastra
    col_idx = ((centroids[:, 0] - xll) / cellsize).astype(int)
    # Geoportal: wiersz 0 = północ (max Y), więc odwracamy
    row_idx = (nrows - 1 - ((centroids[:, 1] - yll) / cellsize).astype(int))

    # Filtruj centroidy poza zasięgiem rastra
    valid = (col_idx >= 0) & (col_idx < ncols) & \
            (row_idx >= 0) & (row_idx < nrows) & \
            (risk >= 0)

    risk_grid[row_idx[valid], col_idx[valid]] = risk[valid]

    # Wypełnij komórki bez przypisanego centroidu, interpolacją NN
    # – prosta iteracja po indeksach z scipy
    from scipy.ndimage import label, distance_transform_edt

    known = risk_grid >= 0
    if known.any():
        _, nearest = distance_transform_edt(~known, return_indices=True)
        risk_grid = risk_grid[nearest[0], nearest[1]]
        # Przywróć NaN tam gdzie oryginalny NMT był NaN
        risk_grid[np.isnan(Z)] = -1

    print(f"[risk] Siatka ryzyka: {nrows}x{ncols}, "
          f"pokrycie: {100*known.mean():.1f}% przed interpolacją")

    return risk_grid
