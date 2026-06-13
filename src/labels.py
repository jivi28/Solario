"""Build training labels for the PU suitability classifier.

Positives  (y=1): existing solar farms in Bavaria, fetched from OpenStreetMap
                  via the Overpass API (no API key needed).
Background (y=0): random points across Bavaria, kept away from any solar farm.

Run:
    python -m src.labels
"""
from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, shape

from . import config
from .gee_auth import get_bavaria_gdf

# Overpass QL: solar plants & generators inside the Bavaria admin area.
_OVERPASS_QUERY = """
[out:json][timeout:180];
area["name"="Bayern"]["admin_level"="4"]->.bav;
(
  way["power"="plant"]["plant:source"="solar"](area.bav);
  relation["power"="plant"]["plant:source"="solar"](area.bav);
  way["power"="generator"]["generator:source"="solar"](area.bav);
  node["power"="generator"]["generator:source"="solar"](area.bav);
);
out center;
"""


# HTTP statuses that just mean "this mirror is busy/picky" -> try the next one.
_RETRYABLE = {406, 429, 500, 502, 503, 504}


def _query_overpass_raw(query: str) -> list[dict] | None:
    """POST `query` to each Overpass mirror in turn; return elements or None.

    Reusable by any caller (solar farms, power lines, ...). Returns None only
    when every endpoint failed.
    """
    headers = {
        "User-Agent": config.OVERPASS_UA,
        "Accept": "application/json",
    }
    for url in config.OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(
                url, data={"data": query}, headers=headers, timeout=200
            )
        except requests.RequestException as exc:
            print(f"[overpass] {url} -> request error ({exc.__class__.__name__}); "
                  f"trying next mirror")
            continue

        if resp.status_code in _RETRYABLE:
            print(f"[overpass] {url} -> HTTP {resp.status_code}; trying next mirror\n"
                  f"          body: {resp.text[:800].strip()!r}")
            continue
        if not resp.ok:
            print(f"[overpass] {url} -> HTTP {resp.status_code} (non-retryable)\n"
                  f"          body: {resp.text[:1000].strip()!r}; trying next mirror")
            continue

        try:
            return resp.json().get("elements", [])
        except ValueError:
            print(f"[overpass] {url} -> non-JSON response; trying next mirror\n"
                  f"          body: {resp.text[:800].strip()!r}")
            continue
    return None


def _query_overpass() -> list[dict] | None:
    """Solar-farm query against the Overpass mirrors."""
    return _query_overpass_raw(_OVERPASS_QUERY)


def _elements_to_gdf(elements: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for el in elements:
        if "center" in el:                       # ways / relations
            lon, lat = el["center"]["lon"], el["center"]["lat"]
        elif "lon" in el:                        # nodes
            lon, lat = el["lon"], el["lat"]
        else:
            continue
        rows.append({"osm_id": el["id"], "osm_type": el["type"],
                     "geometry": Point(lon, lat)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def fetch_solar_farms() -> gpd.GeoDataFrame:
    """Bavaria solar installations as a point GeoDataFrame.

    Tries each Overpass mirror; on total failure, falls back to a cached
    download (`config.F_SOLAR`) if one exists instead of crashing the pipeline.
    """
    elements = _query_overpass()

    if elements is None:
        if config.F_SOLAR.exists():
            gdf = gpd.read_file(config.F_SOLAR)
            print(f"[labels] all Overpass mirrors failed; using cache "
                  f"{config.F_SOLAR.name} ({len(gdf)} installations)")
            return gdf
        raise RuntimeError(
            "All Overpass endpoints failed and no cache exists at "
            f"{config.F_SOLAR}. Re-run later or drop a solar_farms.geojson there."
        )

    gdf = _elements_to_gdf(elements)
    if gdf.empty:
        raise RuntimeError("Overpass returned 0 solar installations for Bavaria "
                           "— check the query before continuing.")
    gdf.to_file(config.F_SOLAR, driver="GeoJSON")   # refresh cache
    print(f"[labels] fetched {len(gdf)} solar installations -> {config.F_SOLAR.name}")
    return gdf


def _random_points_in(polygon, n: int, rng: np.random.Generator) -> list[Point]:
    """Rejection-sample `n` points uniformly inside a (multi)polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    pts: list[Point] = []
    while len(pts) < n:
        xs = rng.uniform(minx, maxx, n)
        ys = rng.uniform(miny, maxy, n)
        for x, y in zip(xs, ys):
            p = Point(x, y)
            if polygon.contains(p):
                pts.append(p)
                if len(pts) == n:
                    break
    return pts


def build_labels() -> gpd.GeoDataFrame:
    """Combine positives + sampled negatives into one labelled GeoDataFrame."""
    rng = np.random.default_rng(config.RANDOM_SEED)

    bavaria = get_bavaria_gdf().to_crs("EPSG:4326")
    bav_poly = bavaria.geometry.union_all()

    pos = fetch_solar_farms()
    if config.MAX_POSITIVES and len(pos) > config.MAX_POSITIVES:
        pos = pos.sample(config.MAX_POSITIVES, random_state=config.RANDOM_SEED)
        pos = pos.reset_index(drop=True)
        print(f"[labels] capped positives to {len(pos)} (of full download) "
              f"for tractable sampling")
    n_neg = len(pos) * config.NEG_PER_POS

    # Buffer positives (in metres) to keep negatives clear of real farms.
    pos_m = pos.to_crs("EPSG:3035")
    keepout = pos_m.geometry.buffer(config.POS_BUFFER_M).union_all()

    neg_pts: list[Point] = []
    while len(neg_pts) < n_neg:
        cand = _random_points_in(bav_poly, n_neg - len(neg_pts), rng)
        cand_gdf = gpd.GeoDataFrame(geometry=cand, crs="EPSG:4326").to_crs("EPSG:3035")
        mask = ~cand_gdf.geometry.within(keepout)
        neg_pts.extend(gpd.GeoSeries(cand, crs="EPSG:4326")[mask.values].tolist())

    neg = gpd.GeoDataFrame(
        {"label": 0, "geometry": neg_pts[:n_neg]}, crs="EPSG:4326"
    )
    pos = pos[["geometry"]].copy()
    pos["label"] = 1

    labels = pd.concat([pos, neg], ignore_index=True)
    labels = gpd.GeoDataFrame(labels, crs="EPSG:4326")
    labels["lon"] = labels.geometry.x
    labels["lat"] = labels.geometry.y
    labels.to_file(config.F_LABELS, driver="GeoJSON")
    print(f"[labels] {int((labels.label==1).sum())} positives + "
          f"{int((labels.label==0).sum())} negatives -> {config.F_LABELS.name}")
    return labels


if __name__ == "__main__":
    t0 = time.time()
    build_labels()
    print(f"[labels] done in {time.time()-t0:.1f}s")
