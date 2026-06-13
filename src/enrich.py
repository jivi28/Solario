"""Enrich the grid features with two extra signals (no model retrain needed):

  * dist_powerline_m -> powers the descriptive `factor_grid` factor.
  * protected         -> WDPA protected-area flag -> `protected area` exclusion.

Power lines come from OSM (Overpass), or an official
`data/raw/powerlines_bavaria.geojson` if you drop one in. WDPA is sampled from
GEE at the grid centroids only (small, ~8 requests).

Run:
    python -m src.enrich            # enrich data/interim/grid_features.csv
"""
from __future__ import annotations

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from . import config
from .gee_auth import get_bavaria, init
from .labels import _query_overpass_raw  # shared robust Overpass POST
from .sample_features import CHUNK

_POWERLINE_QUERY = """
[out:json][timeout:180];
area["name"="Bayern"]["admin_level"="4"]->.bav;
(
  way["power"="line"](area.bav);
  way["power"="minor_line"](area.bav);
);
out geom;
"""


# --------------------------------------------------------------------------- #
# Power lines -> distance
# --------------------------------------------------------------------------- #
def ensure_powerlines() -> gpd.GeoDataFrame:
    """Load powerlines geojson, fetching from OSM (and caching) if missing."""
    if config.F_POWERLINES.exists():
        gdf = gpd.read_file(config.F_POWERLINES)
        print(f"[enrich] using {config.F_POWERLINES.name} ({len(gdf)} lines)")
        return gdf

    elements = _query_overpass_raw(_POWERLINE_QUERY)
    lines = []
    for el in elements or []:
        if el.get("type") == "way" and el.get("geometry"):
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(coords) >= 2:
                lines.append({"osm_id": el["id"], "geometry": LineString(coords)})
    gdf = gpd.GeoDataFrame(lines, crs="EPSG:4326")
    gdf.to_file(config.F_POWERLINES, driver="GeoJSON")
    print(f"[enrich] fetched {len(gdf)} power lines from OSM -> {config.F_POWERLINES.name}")
    return gdf


def add_powerline_distance(df: pd.DataFrame, lines: gpd.GeoDataFrame) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    ).to_crs("EPSG:3035")
    lines_m = lines.to_crs("EPSG:3035")[["geometry"]]
    joined = gpd.sjoin_nearest(pts, lines_m, distance_col="dist_powerline_m")
    joined = joined[~joined.index.duplicated(keep="first")]
    df = df.drop(columns=["dist_powerline_m"], errors="ignore")
    df["dist_powerline_m"] = joined["dist_powerline_m"].round(0).values
    return df


# --------------------------------------------------------------------------- #
# WDPA protected areas -> flag
# --------------------------------------------------------------------------- #
def add_protected_flag(df: pd.DataFrame) -> pd.DataFrame:
    init()
    wdpa = ee.FeatureCollection(config.WDPA).filterBounds(get_bavaria())
    # Paint onto a canvas with a real projection (a bare constant image has none,
    # so reduceRegions would return no value). 100 m is plenty for a 5 km grid.
    mask = (ee.Image.constant(0)
            .paint(wdpa, 1)
            .rename("protected")
            .reproject(crs="EPSG:3857", scale=100)
            .unmask(0))

    pts = df.assign(_idx=range(len(df)))[["_idx", "lon", "lat"]].to_dict("records")
    out = []
    for i in range(0, len(pts), CHUNK):
        chunk = pts[i:i + CHUNK]
        feats = [ee.Feature(ee.Geometry.Point([p["lon"], p["lat"]]), {"_idx": p["_idx"]})
                 for p in chunk]
        # Sample at the mask's native 100 m scale (not 10 m) so the value resolves.
        sampled = mask.reduceRegions(
            collection=ee.FeatureCollection(feats),
            reducer=ee.Reducer.first().setOutputs(["protected"]),  # name it explicitly
            scale=100,
        )
        out.extend(f["properties"] for f in sampled.getInfo()["features"])
        print(f"[enrich] WDPA sampled {min(i + CHUNK, len(pts))}/{len(pts)}")

    flags = pd.DataFrame(out).set_index("_idx").sort_index()
    df = df.drop(columns=["protected"], errors="ignore")
    prot = flags["protected"] if "protected" in flags else 0
    df["protected"] = pd.Series(prot, index=range(len(df))).fillna(0).astype(int).values
    return df


def enrich_grid() -> pd.DataFrame:
    df = pd.read_csv(config.F_GRID_FEATURES)
    df = add_powerline_distance(df, ensure_powerlines())
    df = add_protected_flag(df)
    df.to_csv(config.F_GRID_FEATURES, index=False)
    n_prot = int((df["protected"] == 1).sum())
    print(f"[enrich] grid: dist_powerline_m filled, {n_prot} protected cells "
          f"-> {config.F_GRID_FEATURES.name}")
    return df


if __name__ == "__main__":
    enrich_grid()
