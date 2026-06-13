"""Build a regular grid of candidate sites over Bavaria (CheckList Step 1).

Each cell (default 5 km x 5 km) is one candidate solar site. We keep both the
cell polygon (for map rendering) and its centroid (for feature sampling).

Run:
    python -m src.grid
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from . import config
from .gee_auth import get_bavaria_gdf

# EPSG:3035 = LAEA Europe, metric CRS -> regular km cells.
_METRIC = "EPSG:3035"


def build_grid(cell_km: float | None = None) -> gpd.GeoDataFrame:
    cell_m = (cell_km or config.GRID_CELL_KM) * 1000.0

    bavaria = get_bavaria_gdf().to_crs(_METRIC)
    bav_poly = bavaria.geometry.union_all()
    minx, miny, maxx, maxy = bav_poly.bounds

    cells = []
    xs = np.arange(minx, maxx + cell_m, cell_m)
    ys = np.arange(miny, maxy + cell_m, cell_m)
    for x in xs:
        for y in ys:
            cell = box(x, y, x + cell_m, y + cell_m)
            if cell.intersects(bav_poly):
                cells.append(cell.intersection(bav_poly))

    grid = gpd.GeoDataFrame(geometry=cells, crs=_METRIC)
    grid["cell_id"] = range(len(grid))
    grid = grid.to_crs("EPSG:4326")

    cent = grid.to_crs(_METRIC).geometry.centroid.to_crs("EPSG:4326")
    grid["lon"] = cent.x.values
    grid["lat"] = cent.y.values

    grid.to_file(config.F_GRID, driver="GeoJSON")
    print(f"[grid] {len(grid)} cells of {cell_km or config.GRID_CELL_KM} km "
          f"-> {config.F_GRID.name}")
    return grid


if __name__ == "__main__":
    build_grid()
