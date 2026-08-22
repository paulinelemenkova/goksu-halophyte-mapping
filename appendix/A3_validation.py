#!/usr/bin/env python3
"""
A3_validation.py -- spatially blocked cross-validation and accuracy assessment.

Listing A3 of the manuscript. Trains the tree ensembles under the blocking
scheme of Section 2.8 and reports the metrics of Section 2.10.

Why blocking. Plots of one association cluster within metres of each other. A
random split places near-duplicate observations on both sides and reports an
accuracy that reflects memorisation rather than generalisation. Plots are
therefore binned onto a coarse grid and every plot in a cell is assigned whole
to one fold.

Why nested tuning. A single hyper-parameter search over all data, followed by
cross-validation of the winner, leaks the held-out fold into model selection.
The search here runs inside the training folds only.

Why quantity and allocation disagreement. The agreement coefficient is
uninformative beyond overall accuracy and is sensitive to class prevalence.
Decomposing the total disagreement separates "the wrong number of pixels per
class" from "the right number in the wrong places", which are different
failures with different remedies.

Usage
-----
    python3 A3_validation.py --features features.csv --plots reference_plots.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, GroupKFold

MODELS = {
    "Random forest": (
        RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1),
        {"n_estimators": [300, 600], "max_features": ["sqrt", 0.3],
         "min_samples_leaf": [1, 3]}),
    "Gradient boosting": (
        HistGradientBoostingClassifier(class_weight="balanced"),
        {"max_iter": [200, 400], "learning_rate": [0.05, 0.1],
         "max_leaf_nodes": [15, 31]}),
}


def spatial_blocks(plots_csv, block_km):
    """Bin plot coordinates onto a grid; the longitude step is compressed by
    the cosine of latitude so cells are square on the ground."""
    p = pd.read_csv(plots_csv)
    p = p[p["group"] == "halophytic"][["plot", "lon", "lat"]].drop_duplicates()
    dlat = block_km / 111.32
    dlon = block_km / (111.32 * np.cos(np.radians(p["lat"].mean())))
    p["block"] = (np.floor(p["lat"] / dlat).astype(int).astype(str) + "_"
                  + np.floor(p["lon"] / dlon).astype(int).astype(str))
    return p[["plot", "block"]]


def disagreement(cm):
    """Pontius and Millones quantity and allocation disagreement."""
    p = cm / cm.sum()
    q = 0.5 * np.abs(p.sum(1) - p.sum(0)).sum()
    a = sum(2 * min(p[i].sum() - p[i, i], p[:, i].sum() - p[i, i])
            for i in range(cm.shape[0]))
    return float(q), float(a)


def evaluate(X, y, groups, est, grid, folds, seed, labels):
    true, pred, chosen = [], [], []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        inner = GroupKFold(n_splits=min(4, len(np.unique(groups[tr]))))
        if hasattr(est, "random_state"):
            est.set_params(random_state=seed)
        gs = GridSearchCV(est, grid, cv=inner, scoring="f1_macro")
        gs.fit(X[tr], y[tr], groups=groups[tr])
        true.extend(y[te])
        pred.extend(gs.best_estimator_.predict(X[te]))
        chosen.append(gs.best_params_)
    cm = confusion_matrix(true, pred, labels=labels)
    pr, rc, f1, sup = precision_recall_fscore_support(
        true, pred, labels=labels, zero_division=0)
    q, a = disagreement(cm)
    return {"cm": cm, "producer": rc, "user": pr, "f1": f1, "support": sup,
            "oa": float(np.trace(cm) / cm.sum()),
            "kappa": float(cohen_kappa_score(true, pred)),
            "quantity": q, "allocation": a, "chosen": chosen}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--plots", default="reference_plots.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-km", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.features).merge(
        spatial_blocks(a.plots, a.block_km), on="plot",
        how="left").dropna(subset=["block"])
    drop = ["plot", "epoch", "subtype", "n_obs", "block"]
    X = df.drop(columns=[c for c in drop if c in df.columns]).to_numpy(float)
    y = df["subtype"].to_numpy()
    g = df["block"].to_numpy()
    labels = sorted(set(y))
    folds = min(a.folds, len(np.unique(g)))
    print(f"[info] {len(df)} rows, {X.shape[1]} predictors, "
          f"{len(np.unique(g))} blocks, {folds} folds")

    for name, (est, grid) in MODELS.items():
        r = evaluate(X, y, g, est, grid, folds, a.seed, labels)
        print(f"\n===== {name} =====")
        print(f"  overall accuracy {r['oa']:.3f}   kappa {r['kappa']:.3f}")
        print(f"  quantity disagreement {r['quantity']:.3f}   "
              f"allocation disagreement {r['allocation']:.3f}")
        print(f"  {'class':28s}{'producer':>10s}{'user':>8s}{'F1':>8s}{'n':>6s}")
        for i, c in enumerate(labels):
            print(f"  {c:28s}{r['producer'][i]:10.3f}{r['user'][i]:8.3f}"
                  f"{r['f1'][i]:8.3f}{r['support'][i]:6d}")


if __name__ == "__main__":
    main()
