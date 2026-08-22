#!/usr/bin/env python3
"""
train_classifier.py -- supervised classification with spatially blocked folds.

Trains the tree ensemble of Section 2.7 on the feature table from
build_features.py and reports per-class performance under the blocking scheme
of Section 2.8.

Two points of method are load-bearing and are implemented explicitly rather
than left to defaults:

  Blocking. Folds are formed over spatial blocks of plots, not over rows. Plots
  of one association cluster within metres of each other, so a random split
  would place near-duplicate observations on both sides and inflate accuracy
  without improving generalisation. Blocks are built by binning plot
  coordinates onto a coarse grid; every plot in a block goes to the same fold.

  Nested tuning. The hyper-parameter search runs inside the training folds
  only. A single search over all data, followed by cross-validation of the
  chosen configuration, leaks the held-out fold into model selection.

Usage
-----
    python3 train_classifier.py --features features.csv \
        --plots reference_plots.csv --folds 5 --block-km 2.0
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, GroupKFold

# Primary model and its benchmark, each with its own search space. Section 2.7
# specifies a gradient-boosted ensemble trained under the same protocol, so it
# is run here rather than merely described.
MODELS = {
    "Random forest": (
        RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1),
        {"n_estimators": [300, 600],
         "max_features": ["sqrt", 0.3],
         "min_samples_leaf": [1, 3]}),
    "Gradient boosting": (
        HistGradientBoostingClassifier(class_weight="balanced"),
        {"max_iter": [200, 400],
         "learning_rate": [0.05, 0.1],
         "max_leaf_nodes": [15, 31]}),
}


def spatial_blocks(plots_csv, block_km):
    """Assign each plot to a coarse spatial block; folds respect the blocks."""
    p = pd.read_csv(plots_csv)
    p = p[p["group"] == "halophytic"][["plot", "lon", "lat"]].drop_duplicates()
    # Degrees per block at this latitude; longitude is compressed by cos(lat).
    dlat = block_km / 111.32
    dlon = block_km / (111.32 * np.cos(np.radians(p["lat"].mean())))
    p["block"] = (np.floor(p["lat"] / dlat).astype(int).astype(str) + "_"
                  + np.floor(p["lon"] / dlon).astype(int).astype(str))
    return p[["plot", "block"]]


def run(features_csv, plots_csv, folds, block_km, seed):
    df = pd.read_csv(features_csv).merge(
        spatial_blocks(plots_csv, block_km), on="plot", how="left")
    df = df.dropna(subset=["block"])

    drop = ["plot", "epoch", "subtype", "block", "n_obs"]
    X = df.drop(columns=drop).to_numpy(float)
    y = df["subtype"].to_numpy()
    groups = df["block"].to_numpy()

    n_blocks = len(np.unique(groups))
    folds = min(folds, n_blocks)
    print(f"[info] {len(df)} rows, {X.shape[1]} predictors, "
          f"{n_blocks} spatial blocks, {folds} folds")
    print(f"[info] class counts: {dict(pd.Series(y).value_counts())}")

    outer = GroupKFold(n_splits=folds)
    results = {}
    for name, (est, grid) in MODELS.items():
        if hasattr(est, "random_state"):
            est.set_params(random_state=seed)
        y_true, y_pred, chosen = [], [], []
        for k, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
            # Inner search sees the training folds only.
            inner = GroupKFold(n_splits=min(4, len(np.unique(groups[tr]))))
            gs = GridSearchCV(est, grid, cv=inner, scoring="f1_macro",
                              n_jobs=1)
            gs.fit(X[tr], y[tr], groups=groups[tr])
            y_true.extend(y[te])
            y_pred.extend(gs.best_estimator_.predict(X[te]))
            chosen.append(gs.best_params_)
        results[name] = (y_true, y_pred, chosen)

        print(f"\n===== {name} =====")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))
        print(f"kappa {cohen_kappa_score(y_true, y_pred):.3f}")
        for par in grid:
            vals = [c[par] for c in chosen]
            mode = max(set(map(str, vals)), key=lambda v: list(map(str, vals)).count(v))
            print(f"  {par:18s} search {grid[par]}  modal choice {mode} "
                  f"({list(map(str, vals)).count(mode)}/{len(vals)} folds)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--plots", default="reference_plots.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-km", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.features, a.plots, a.folds, a.block_km, a.seed)
