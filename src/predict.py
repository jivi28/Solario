"""Score the Bavaria grid and write the map-ready, *explainable* GeoJSON.

For each cell we keep the raw features (slope, ghi, dist_powerline_m, landcover),
the RandomForest probability (`score`) and class, and add interpretable
explanation fields so a user can click a cell and immediately see *why* it is
excluded / bad / okay / good:

    factor_sun, factor_terrain, factor_landuse, factor_grid, factor_model  (0..1)
    exclusion_reason, decision_reason, top_positive_factors,
    top_negative_factors, tooltip

Reruns prediction only (no retrain, no GEE sampling):
    python -m src.predict
"""
from __future__ import annotations

import joblib
import geopandas as gpd
import numpy as np
import pandas as pd

from . import config

# A factor counts as a clear positive / negative driver beyond these cut-offs.
POS, NEG = 0.60, 0.40

# Human phrasing for each interpretable factor, by sentiment.
_PHRASES = {
    ("factor_sun", "+"): "good solar resource", ("factor_sun", "-"): "low solar resource",
    ("factor_terrain", "+"): "flat terrain", ("factor_terrain", "-"): "steep terrain",
    ("factor_landuse", "+"): "suitable land use", ("factor_landuse", "-"): "poor land use",
    ("factor_grid", "+"): "close to grid", ("factor_grid", "-"): "far from grid",
    ("factor_model", "+"): "high model score", ("factor_model", "-"): "low model score",
}
_FACTORS = ["factor_model", "factor_sun", "factor_terrain", "factor_landuse", "factor_grid"]
_HEADLINE = {"good": "Good candidate", "okay": "Okay candidate", "bad": "Weak candidate"}


# --------------------------------------------------------------------------- #
# Classification + exclusion
# --------------------------------------------------------------------------- #
def _classify(score: float, excluded: bool) -> str:
    if excluded:
        return "excluded"
    if score >= config.CLASS_THRESHOLDS["good"]:
        return "good"
    if score >= config.CLASS_THRESHOLDS["okay"]:
        return "okay"
    return "bad"


def _exclusion_reason(landcover, has_score: bool, protected: bool = False) -> str | None:
    """Return a human exclusion reason, or None if the cell is buildable."""
    if not has_score:
        return "missing data"
    if protected:
        return "protected area"
    if pd.isna(landcover):
        return None
    code = int(landcover)
    if code in config.WORLDCOVER_EXCLUDE:
        return config.WORLDCOVER_NAMES.get(code, "excluded landcover")
    return None


# --------------------------------------------------------------------------- #
# Interpretable factor scores (all 0..1, higher = better for solar)
# --------------------------------------------------------------------------- #
def _factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # Sun: percentile rank of annual GHI across the grid.
    f["factor_sun"] = (df["ghi"].rank(pct=True)
                       if df.get("ghi") is not None and df["ghi"].notna().any()
                       else np.nan)

    # Terrain: flatness from slope (0 deg -> 1.0, >=15 deg -> 0).
    f["factor_terrain"] = ((1 - df["slope"] / 15.0).clip(0, 1)
                           if df.get("slope") is not None and df["slope"].notna().any()
                           else np.nan)

    # Land use: excluded classes -> 0, else WorldCover suitability weight.
    def _lu(c):
        if pd.isna(c):
            return np.nan
        c = int(c)
        if c in config.WORLDCOVER_EXCLUDE:
            return 0.0
        return config.WORLDCOVER_SUITABILITY.get(c, 0.5)
    f["factor_landuse"] = (df["landcover"].map(_lu)
                           if "landcover" in df else np.nan)

    # Grid: closer to a power line is better; NaN when no powerline data exists.
    if "dist_powerline_m" in df and df["dist_powerline_m"].notna().any():
        f["factor_grid"] = 1 - df["dist_powerline_m"].rank(pct=True)
    else:
        f["factor_grid"] = np.nan

    # Model: the RandomForest suitability probability itself.
    f["factor_model"] = df["score"]
    return f.round(3)


# --------------------------------------------------------------------------- #
# Per-cell narrative
# --------------------------------------------------------------------------- #
def _drivers(row) -> tuple[list[str], list[str]]:
    """Split available factors into positive / negative driver phrases."""
    pos, neg = [], []
    for fac in _FACTORS:
        v = row[fac]
        if pd.isna(v):
            continue
        if v >= POS:
            pos.append((v, _PHRASES[(fac, "+")]))
        elif v <= NEG:
            neg.append((v, _PHRASES[(fac, "-")]))
    pos = [p for _, p in sorted(pos, key=lambda t: -t[0])]
    neg = [p for _, p in sorted(neg, key=lambda t: t[0])]
    return pos, neg


def _explain(row) -> dict:
    cls = row["suitability_class"]
    reason = row["exclusion_reason"]
    grid_unavailable = pd.isna(row["factor_grid"])

    if cls == "excluded":
        if reason == "missing data":
            tip = "Excluded: missing satellite/landcover data for this cell."
            dec = "Excluded: missing data"
        else:
            tip = f"Excluded: {reason}, not suitable for ground-mounted solar."
            dec = f"Excluded: {reason}"
        return {"decision_reason": dec, "top_positive_factors": "",
                "top_negative_factors": "", "tooltip": tip}

    pos, neg = _drivers(row)
    score = row["score"]
    qual = "high" if score >= POS else ("moderate" if score >= config.CLASS_THRESHOLDS["okay"] else "low")
    head = _HEADLINE[cls]

    main = [f"{qual} model score ({score:.2f})"] + [p for p in pos if p != "high model score"]
    tip = f"{head}: " + ", ".join(main) + "."
    if neg:
        tip += " Limited by " + ", ".join(neg) + "."
    if grid_unavailable:
        tip += " Grid proximity unavailable."

    dec = f"{head} (score {score:.2f}); +: {', '.join(pos) or 'none'}; -: {', '.join(neg) or 'none'}"
    return {"decision_reason": dec,
            "top_positive_factors": ", ".join(pos),
            "top_negative_factors": ", ".join(neg),
            "tooltip": tip}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def predict() -> gpd.GeoDataFrame:
    bundle = joblib.load(config.F_MODEL)
    model, feats = bundle["model"], bundle["features"]

    df = pd.read_csv(config.F_GRID_FEATURES)
    grid = gpd.read_file(config.F_GRID)[["cell_id", "geometry"]]

    valid = df.dropna(subset=feats).copy()
    valid["score"] = model.predict_proba(valid[feats].values)[:, 1]
    df = df.merge(valid[["cell_id", "score"]], on="cell_id", how="left")

    # Exclusion + class.
    landcover = df["landcover"] if "landcover" in df else pd.Series(np.nan, index=df.index)
    protected = df["protected"] if "protected" in df else pd.Series(0, index=df.index)
    df["exclusion_reason"] = [
        _exclusion_reason(lc, not pd.isna(s), bool(p))
        for lc, s, p in zip(landcover, df["score"], protected)
    ]
    excluded = df["exclusion_reason"].notna()
    df["suitability_class"] = [
        _classify(0.0 if pd.isna(s) else s, bool(e))
        for s, e in zip(df["score"], excluded)
    ]
    df["exclusion_reason"] = df["exclusion_reason"].fillna("")  # "" = not excluded

    # Interpretable factor scores + narrative.
    df = pd.concat([df, _factor_scores(df)], axis=1)
    narrative = df.apply(_explain, axis=1, result_type="expand")
    df = pd.concat([df, narrative], axis=1)

    df["score"] = df["score"].round(4)
    df["factor_model"] = df["score"]

    keep = [
        "cell_id", "lon", "lat", "score", "suitability_class",
        "slope", "ghi", "dist_powerline_m", "landcover", "protected",
        "factor_sun", "factor_terrain", "factor_landuse", "factor_grid", "factor_model",
        "exclusion_reason", "decision_reason",
        "top_positive_factors", "top_negative_factors", "tooltip",
    ]
    keep = [c for c in keep if c in df.columns]
    out = grid.merge(df[keep], on="cell_id", how="left")
    out.to_file(config.F_OUTPUT, driver="GeoJSON")

    counts = out["suitability_class"].value_counts().to_dict()
    print(f"[predict] wrote {len(out)} cells x {len(keep)+1} cols -> {config.F_OUTPUT.name}")
    print(f"[predict] class counts: {counts}")
    return out


if __name__ == "__main__":
    predict()
