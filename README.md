# Solar-Farm Land-Suitability Classifier (Bavaria)

ML classification branch of the Energy/AI hackathon project
(*Direction 8 — Satellite Intelligence for Energy*).

It learns **where good solar-farm land is in Bavaria** from Google DeepMind's
**AlphaEarth Foundations** satellite embeddings (64-dim vector per 10 m patch, in
Google Earth Engine) — no vision model, no GPU. Output is a map-ready GeoJSON of
5 km grid cells scored **good / okay / bad**, which the frontend branch renders
green → yellow → red.

## The idea (PU / proxy labels)

There's no ground-truth "this cell is suitable" dataset, so:

- **Positives** = existing solar farms (OpenStreetMap). Developers already built
  where conditions are good → these are examples of *suitable* land.
- **Background** = random Bavaria points away from any farm.
- A RandomForest on the AlphaEarth embedding learns *what suitable solar land
  looks like*; its predicted probability **is** the suitability score.
- Undeveloped cells that *look like* existing farms → high score → candidate sites.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set GEE_PROJECT=your-gcp-project-id
earthengine authenticate      # one-time; you already have GEE access
python -m src.gee_auth        # connectivity check
```

## Run

```bash
python -m src.pipeline          # full run (Tier-2 features)
python -m src.pipeline --tier1  # embeddings-only, fastest
```

Or step by step: `labels → grid → sample_features labels → train →
sample_features grid → predict` (see `src/`).

## Data sources

| Data | Source | Role |
|---|---|---|
| AlphaEarth embeddings | GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | features (64-dim) |
| Solar farms | OpenStreetMap via Overpass API | positive labels |
| Bavaria boundary | GEE `FAO/GAUL/2015/level1` | region / grid |
| Land cover 10 m | GEE `ESA/WorldCover/v200` | exclusion mask + feature |
| Slope | GEE `USGS/SRTMGL1_003` | terrain feature |
| GHI | GEE `ECMWF/ERA5_LAND/MONTHLY_AGGR` | irradiance feature |
| Power lines | `data/raw/powerlines_bavaria.geojson` (optional) | grid-proximity feature |

## Output

`outputs/bavaria_suitability.geojson` — one feature per 5 km cell with
`score` ∈ [0,1], `suitability_class` (good/okay/bad/excluded) and a feature
breakdown (slope, ghi, dist_powerline_m, landcover) for the map tooltip.

## Layout

```
src/config.py          all tunables / paths / dataset ids
src/gee_auth.py        GEE init + Bavaria + embedding helpers
src/labels.py          Overpass solar farms + random negatives
src/grid.py            Bavaria 5 km grid
src/sample_features.py sample AlphaEarth (+Tier-2) at points
src/train.py           RandomForest + spatial hold-out + save model
src/predict.py         score grid -> GeoJSON
src/pipeline.py        run all steps
notebooks/01_train_suitability.ipynb   metrics + map preview
```
