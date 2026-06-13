"""Google Earth Engine initialization + shared GEE asset helpers.

Run directly for a connectivity check:
    python -m src.gee_auth
"""
from __future__ import annotations

import ee

from . import config


def init() -> None:
    """Initialize Earth Engine, authenticating interactively on first run."""
    try:
        ee.Initialize(project=config.GEE_PROJECT or None)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=config.GEE_PROJECT or None)


def get_bavaria() -> "ee.Geometry":
    """Bavaria boundary as an ee.Geometry (FAO GAUL, bbox fallback)."""
    try:
        fc = (
            ee.FeatureCollection(config.GAUL_L1)
            .filter(ee.Filter.eq("ADM1_NAME", config.BAVARIA_ADM1_NAME))
        )
        geom = fc.geometry()
        # Force evaluation so we fall back if the filter returns nothing.
        if geom.area().getInfo() <= 0:
            raise ValueError("empty Bavaria geometry")
        return geom
    except Exception:
        return ee.Geometry.Rectangle(list(config.BAVARIA_BBOX))


def get_bavaria_gdf():
    """Bavaria boundary as a GeoDataFrame (EPSG:4326), cached to data/raw.

    Pulls the polygon from GEE once and writes it to `config.F_BAVARIA` so the
    purely-local steps (labels, grid) don't need a GEE round-trip every time.
    """
    import geopandas as gpd

    if config.F_BAVARIA.exists():
        return gpd.read_file(config.F_BAVARIA)

    import geemap

    init()
    geom = get_bavaria()
    fc = ee.FeatureCollection([ee.Feature(geom)])
    gdf = geemap.ee_to_gdf(fc).set_crs("EPSG:4326")
    gdf.to_file(config.F_BAVARIA, driver="GeoJSON")
    return gdf


def get_embedding_image(year: int | None = None) -> "ee.Image":
    """AlphaEarth annual embedding mosaic for `year`, clipped to nothing.

    Returns a single 64-band image (the annual layer for that year).
    """
    year = year or config.EMBEDDING_YEAR
    coll = ee.ImageCollection(config.EMBEDDING_COLLECTION).filterDate(
        f"{year}-01-01", f"{year}-12-31"
    )
    return coll.mosaic().select(config.EMBEDDING_BANDS)


def _connectivity_check() -> None:
    init()
    bav = get_bavaria()
    area_km2 = bav.area().divide(1e6).getInfo()
    img = get_embedding_image()
    n_bands = img.bandNames().size().getInfo()
    # Sample one pixel near Munich to confirm the embedding has data.
    pt = ee.Geometry.Point([11.58, 48.14])
    sample = img.reduceRegion(ee.Reducer.first(), pt, scale=10).getInfo()
    have_data = sample.get("A00") is not None
    print(f"[ok] GEE initialized (project={config.GEE_PROJECT or 'default'})")
    print(f"[ok] Bavaria area ~ {area_km2:,.0f} km^2")
    print(f"[ok] AlphaEarth embedding bands: {n_bands} (year {config.EMBEDDING_YEAR})")
    print(f"[ok] Embedding sampled at Munich: {'data present' if have_data else 'NO DATA'}")


if __name__ == "__main__":
    _connectivity_check()
