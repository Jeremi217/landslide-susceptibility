# Landslide Susceptibility Analysis from DTM

A Python pipeline for landslide susceptibility mapping using Digital Terrain Model (DTM) data. The analysis is based on slope angle classification derived from a **Delaunay TIN (Triangulated Irregular Network)** built from DTM point data.

## Methodology

1. **DTM loading** — reads ASC (ASCII Grid) files from Polish Geoportal (GUGiK)
2. **Delaunay triangulation** — builds a TIN from the DTM point cloud using `scipy.spatial.Delaunay`
3. **Slope calculation** — computes the inclination angle of each triangle from its normal vector (cross product of edge vectors)
4. **Risk classification** — classifies slope angles into 5 landslide susceptibility classes based on geomorphological literature (Varnes 1978, Hungr et al. 2014)
5. **Rasterization** — projects triangle-level risk back onto the original raster grid
6. **Visualization** — generates static maps using `matplotlib`

### Risk Classification Thresholds

| Class        | Slope angle | Color  |
|--------------|-------------|--------|
| No risk      | < 5°        | green  |
| Low          | 5° – 15°    | yellow |
| Moderate     | 15° – 25°   | orange |
| High         | 25° – 35°   | red    |
| Very High    | > 35°       | dark red |

## Data Source

DTM data: [GUGiK Geoportal](https://www.geoportal.gov.pl/) — NMT in ASC format (1m or 5m resolution)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Place your .asc DTM file in the data/ directory, then run:
python main.py --input data/nmt.asc

# For large rasters, limit the number of TIN points (default: 80000):
python main.py --input data/nmt.asc --max-points 50000
```

## Output

All results are saved in the `output/` directory:

| File | Description |
|------|-------------|
| `slope_map.png` | Continuous slope angle map (raster gradient) |
| `tin_map.png` | Delaunay TIN coloured by slope angle per triangle |
| `risk_map.png` | Landslide susceptibility map with hillshade background |
| `slope_histogram.png` | Distribution of slope angles across all TIN triangles |

## Project Structure

```
landslide-susceptibility/
├── data/
│   └── nmt.asc          ← place your DTM file here
├── output/              ← generated maps
├── src/
│   ├── loader.py        ← ASC loading and XY grid generation
│   ├── triangulation.py ← Delaunay TIN + slope calculation
│   ├── risk.py          ← risk classification + rasterization
│   └── visualize.py     ← all plots and maps
├── main.py              ← main pipeline
├── requirements.txt
└── README.md
```

## Technical Notes

- For rasters larger than `--max-points`, the pipeline randomly samples points before triangulation to maintain performance. The sampling is spatially uniform.
- Slope per triangle is computed as the angle between the triangle's normal vector and the vertical axis: `slope = arccos(|n_z| / |n|)`
- Risk grid is produced by assigning each raster cell the risk class of its nearest triangle centroid (nearest-neighbour interpolation).

## References

- Varnes, D.J. (1978). *Slope movement types and processes*. Transportation Research Board Special Report 176.
- Hungr, O., Leroueil, S., Picarelli, L. (2014). *The Varnes classification of landslide types, an update*. Landslides, 11(2), 167–194.
