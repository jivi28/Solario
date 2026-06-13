"""Run the whole classification pipeline end to end.

    python -m src.pipeline           # full run (Tier-2 features on)
    python -m src.pipeline --tier1   # embeddings-only (fastest)

Steps: labels -> grid -> sample(labels) -> train -> sample(grid) -> predict.
"""
from __future__ import annotations

import argparse

from . import labels, grid, sample_features, train, enrich, predict


def main(tier2: bool = True) -> None:
    print("== 1/7 labels ==");        labels.build_labels()
    print("== 2/7 grid ==");          grid.build_grid()
    print("== 3/7 sample labels =="); sample_features.run("labels", tier2=tier2)
    print("== 4/7 train ==");         train.train()
    print("== 5/7 sample grid ==");   sample_features.run("grid", tier2=tier2)
    print("== 6/7 enrich grid ==");   enrich.enrich_grid()  # powerlines + WDPA
    print("== 7/7 predict ==");       predict.predict()
    print("\nDone -> outputs/bavaria_suitability.geojson")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", action="store_true",
                    help="embeddings only (skip slope/ghi/landcover/powerline)")
    args = ap.parse_args()
    main(tier2=not args.tier1)
