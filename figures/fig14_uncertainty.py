#!/usr/bin/env python3
"""
fig14_uncertainty.py -- Figure 14 of the Goksu Delta halophyte manuscript.

Spatially explicit classification confidence, with the confusion structure that
explains it.

(a) Normalised class-probability entropy of Eq. (9) mapped for the baseline
    epoch. Zero denotes a fully determined assignment and one a uniform
    posterior over the four classes. The threshold used for the change
    filtering of Section 2.11 is marked on the colour bar, so the map shows
    directly which ground supports a change statement and which does not.

(b) Confusion matrix from the spatially blocked cross-validation of
    Section 2.8, row-normalised so each row sums to 100 per cent. This is the
    diagnostic counterpart of panel (a): the entropy map says where the
    classifier is unsure, the matrix says which classes it confuses. The pair
    contributing the largest off-diagonal share is annotated.

The matrix is computed under blocking rather than in sample. An in-sample
confusion matrix would report the forest's memory of its training plots and
would be close to diagonal regardless of how separable the classes are.

INPUT
-----
    fig09_map_baseline_entropy.tif   entropy raster, scaled by 1000
    features.csv                     plot features
    reference_plots.csv              coordinates, for the spatial blocks

USAGE
-----
    python3 fig14_uncertainty.py --entropy-max 0.85
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.ticker import AutoMinorLocator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
    "axes.labelpad": 2, "axes.titlepad": 3, "axes.linewidth": 0.8,
    "savefig.dpi": 600,
})

NAMES = ["Swamp", "Temporary floodplain", "Temporary ponds",
         "Terrestrial saline flats"]
SHORT = {"Swamp": "Swamp", "Temporary floodplain": "Floodplain",
         "Temporary ponds": "Ponds", "Terrestrial saline flats": "Saline flats"}

CREDIT = (
    "(a) Normalised class-probability entropy of Eq. (9) for the baseline epoch, "
    "bounded on [0, 1] over four classes; the dashed mark on the colour bar is the "
    "threshold of {emax} used for the change filtering of Sect. 2.11, above which "
    "{above:.0f} per cent of classified pixels fall. Median entropy over the classified "
    "domain is {med:.2f}. (b) Confusion matrix from the spatially blocked "
    "cross-validation of Sect. 2.8 over {n} plot-epoch rows, row-normalised to "
    "percentages, so the diagonal is producer's accuracy. Overall accuracy is "
    "{acc:.0f} per cent. Blocking matters: an in-sample matrix would report the "
    "forest's memory of its own training plots. Software used for plotting figure: "
    "Python {py} with Matplotlib {mpl}, scikit-learn {sk} and rasterio {rio}. "
    "Source: authors"
)


def spatial_blocks(plots_csv, block_km=2.0):
    p = pd.read_csv(plots_csv)
    p = p[p["group"] == "halophytic"][["plot", "lon", "lat"]].drop_duplicates()
    dlat = block_km / 111.32
    dlon = block_km / (111.32 * np.cos(np.radians(p["lat"].mean())))
    p["block"] = (np.floor(p["lat"] / dlat).astype(int).astype(str) + "_"
                  + np.floor(p["lon"] / dlon).astype(int).astype(str))
    return p[["plot", "block"]]


def blocked_confusion(features_csv, plots_csv, folds, seed):
    """Out-of-fold predictions under the blocking of Section 2.8."""
    df = pd.read_csv(features_csv).merge(
        spatial_blocks(plots_csv), on="plot", how="left").dropna(
            subset=["block"])
    drop = ["plot", "epoch", "subtype", "n_obs", "block"]
    X = df.drop(columns=[c for c in drop if c in df.columns]).to_numpy(float)
    y = df["subtype"].to_numpy()
    g = df["block"].to_numpy()
    folds = min(folds, len(np.unique(g)))

    true, pred = [], []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        clf = RandomForestClassifier(
            n_estimators=600, max_features=0.3, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=seed,
            n_jobs=-1).fit(X[tr], y[tr])
        true.extend(y[te])
        pred.extend(clf.predict(X[te]))
    cm = confusion_matrix(true, pred, labels=NAMES)
    acc = 100.0 * np.trace(cm) / cm.sum()
    return cm, acc, len(df)


def panel_entropy(ax, path, entropy_max):
    with rasterio.open(path) as src:
        e = src.read(1).astype("float32")
        bounds = src.bounds
    e = np.where(e < 0, np.nan, e / 1000.0)
    im = ax.imshow(e, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest",
                   extent=[bounds.left, bounds.right,
                           bounds.bottom, bounds.top])
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)

    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Normalised entropy", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    cb.ax.yaxis.set_minor_locator(AutoMinorLocator())
    cb.ax.axhline(entropy_max, color="#00c8ff", lw=1.4, ls="--")

    v = e[np.isfinite(e)]
    stats = {"median": float(np.median(v)),
             "above": 100.0 * float((v > entropy_max).mean()),
             "n": int(v.size)}
    geo = (bounds.right - bounds.left) / (bounds.top - bounds.bottom)
    return cb, stats, geo


def panel_confusion(ax, cm, acc):
    row = cm.sum(axis=1, keepdims=True)
    pct = 100.0 * cm / np.maximum(row, 1)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, origin="upper")
    n = len(NAMES)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if pct[i, j] > 55 else "#1a1a1a")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([SHORT[c] for c in NAMES], rotation=30, ha="right")
    ax.set_yticklabels([SHORT[c] for c in NAMES])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class")
    ax.tick_params(which="both", length=0)

    off = pct.copy()
    np.fill_diagonal(off, -1)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="#d62828", lw=1.8, zorder=4))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Row percentage", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    return cb, (NAMES[i], NAMES[j], pct[i, j])


def build(entropy_tif, features_csv, plots_csv, entropy_max, folds, seed,
          prefix):
    for f in (entropy_tif, features_csv, plots_csv):
        if not os.path.exists(f):
            sys.exit(f"ERROR: {f} not found.")

    cm, acc, n_rows = blocked_confusion(features_csv, plots_csv, folds, seed)
    print(f"[info] blocked cross-validation over {n_rows} rows, "
          f"overall accuracy {acc:.1f} %")

    fig = plt.figure(figsize=(6.89, 4.3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    cb_a, st, geo = panel_entropy(ax_a, entropy_tif, entropy_max)
    cb_b, worst = panel_confusion(ax_b, cm, acc)
    print(f"[info] median entropy {st['median']:.3f}; "
          f"{st['above']:.1f} % of {st['n']} classified pixels above "
          f"{entropy_max}")
    print(f"[info] largest off-diagonal: {SHORT[worst[0]]} misread as "
          f"{SHORT[worst[1]]} in {worst[2]:.0f} % of its reference plots")

    # Equal panel heights, letters on one level, as in Figures 7 and 12.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()
    fig_w_in, fig_h_in = fig.get_size_inches()
    k = fig_h_in / fig_w_in
    pa, pb = ax_a.get_position(), ax_b.get_position()
    H = min(pa.height, pb.height)
    top = max(pa.y1, pb.y1)
    ax_a.set_aspect("auto")
    ax_b.set_aspect("auto")
    ax_a.set_position([pa.x0, top - H, H * k * geo, H])
    wb = H * k
    ax_b.set_position([pb.x0, top - H, wb, H])
    cb_a.ax.set_position([pa.x0 + H * k * geo + 0.012, top - H, 0.018, H])
    cb_b.ax.set_position([pb.x0 + wb + 0.012, top - H, 0.018, H])

    fig.canvas.draw()
    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
        fig.text(ax.get_position().x0, top + 0.016, tag, fontsize=11,
                 fontweight="bold", va="bottom", ha="left")

    import sklearn
    import textwrap
    fig.canvas.draw()
    fig_h = fig_h_in * fig.dpi
    fig_w = fig_w_in * fig.dpi
    text = CREDIT.format(
        emax=entropy_max, above=st["above"], med=st["median"], n=n_rows,
        acc=acc, py=".".join(map(str, sys.version_info[:3])),
        mpl=matplotlib.__version__, sk=sklearn.__version__,
        rio=rasterio.__version__)
    t = None
    for ncol in range(170, 60, -2):
        if t is not None:
            t.remove()
        t = fig.text(0.0, 0.0, textwrap.fill(text, ncol), fontsize=8,
                     color="#555555", va="top", ha="left", linespacing=1.25)
        fig.canvas.draw()
        if t.get_window_extent(r).width <= fig_w:
            break
    fig.canvas.draw()
    low = min(ax_a.get_tightbbox(r).y0, ax_b.get_tightbbox(r).y0) / fig_h
    t.set_position((0.0, low - 0.020))
    fig.canvas.draw()
    gap = low * fig_h - t.get_window_extent(r).y1
    print(f"[check] gap between panels and credit: {gap:.0f} px "
          "(must be positive)")

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf and {prefix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entropy", default="fig09_map_baseline_entropy.tif")
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--plots", default="reference_plots.csv")
    ap.add_argument("--entropy-max", type=float, default=0.85)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="fig14_uncertainty")
    a = ap.parse_args()
    build(a.entropy, a.features, a.plots, a.entropy_max, a.folds, a.seed,
          a.prefix)
