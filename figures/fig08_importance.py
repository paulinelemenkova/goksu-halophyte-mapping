#!/usr/bin/env python3
"""
fig08_importance.py -- Figure 8 of the Goksu Delta halophyte manuscript.

Predictor importance for the halophyte habitat classification, comparing the
impurity-based and permutation-based rankings and aggregating them by feature
family.

(a) The highest-ranked individual predictors under both measures, drawn as
    paired bars. Impurity importance is reported for continuity with the wider
    literature but is biased toward predictors that offer many split points;
    permutation importance is measured on held-out blocks and is the measure
    used for interpretation, as stated in Section 2.7.

(b) The same importances summed within feature families, so that the question
    "which index matters" is separated from "which descriptor of that index
    matters". Families are the four spectral indices on one axis and the
    descriptor types -- mean level, harmonic coefficients, amplitude, phase,
    fit residual, median, interquartile range -- on the other.

Both measures are computed under the spatial blocking of Section 2.8:
permutation importance evaluated in-sample would reward predictors the forest
has memorised, exactly as an in-sample separability measure would.

INPUT
-----
    features.csv        from build_features.py
    reference_plots.csv for the spatial blocks

USAGE
-----
    python3 fig08_importance.py --top 14
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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

# Spectral index families, with the property each is meant to carry.
INDEX_LABEL = {
    "ndvi": "NDVI\ngreenness",
    "ndmi": "NDMI\nmoisture",
    "crsi": "CRSI\ncanopy salinity",
    "si": "SI\nsoil salinity",
}
INDEX_COLOUR = {"ndvi": "#4C9F70", "ndmi": "#3B7EA1",
                "crsi": "#C1663A", "si": "#9B59B6"}

# Descriptor types within each index series.
DESC_LABEL = {
    "a0": "mean level",
    "c": "harmonic coefficients",
    "amp1": "amplitude",
    "pha1": "phase",
    "rmse": "fit residual",
    "med": "median",
    "iqr": "interquartile range",
}

BAR_IMP = "#8FA9BF"      # impurity
BAR_PERM = "#243B53"     # permutation

CREDIT = (
    "Importances from a random forest of {ntree} trees trained under the {nf} spatial "
    "folds of Sect. 2.8 on {n} plot-epoch rows and {p} predictors. Impurity importance "
    "is the mean decrease in Gini impurity averaged over folds; permutation importance "
    "is the mean drop in balanced accuracy over {nrep} permutations of each predictor in "
    "the held-out block, averaged over folds, and is the measure used for interpretation "
    "because impurity importance rewards predictors offering many split points. Both are "
    "normalised to sum to one within a measure so the two rankings are comparable. Error "
    "bars in (a) span one standard deviation across folds. No terrain predictors are "
    "present in this feature table: it is built from the plot index series alone. "
    "Software used for plotting figure: Python {py} with Matplotlib {mpl}, scikit-learn "
    "{sk} and pandas {pd}. Source: authors"
)


def parse_name(col):
    """Split a predictor name into (index, descriptor type)."""
    idx, _, rest = col.partition("_")
    if rest.startswith("c") and rest[1:].isdigit():
        return idx, "c"
    return idx, rest


def pretty(col):
    idx, desc = parse_name(col)
    short = {"a0": "mean", "amp1": "amp", "pha1": "phase", "rmse": "resid",
             "med": "median", "iqr": "IQR"}
    tail = short.get(desc, col.split("_", 1)[1])
    return f"{idx.upper()} {tail}"


def spatial_blocks(plots_csv, block_km=2.0):
    p = pd.read_csv(plots_csv)
    p = p[p["group"] == "halophytic"][["plot", "lon", "lat"]].drop_duplicates()
    dlat = block_km / 111.32
    dlon = block_km / (111.32 * np.cos(np.radians(p["lat"].mean())))
    p["block"] = (np.floor(p["lat"] / dlat).astype(int).astype(str) + "_"
                  + np.floor(p["lon"] / dlon).astype(int).astype(str))
    return p[["plot", "block"]]


def importances(X, y, groups, cols, folds, ntree, nrep, seed):
    """Impurity and out-of-fold permutation importance, per fold."""
    imp, perm = [], []
    cv = GroupKFold(n_splits=folds)
    for k, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        clf = RandomForestClassifier(
            n_estimators=ntree, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        imp.append(clf.feature_importances_)
        r = permutation_importance(
            clf, X[te], y[te], n_repeats=nrep, random_state=seed,
            scoring="balanced_accuracy", n_jobs=1)
        perm.append(r.importances_mean)
        print(f"[info]   fold {k}: {len(te)} held out")
    imp, perm = np.array(imp), np.array(perm)
    # Normalise within a measure so the two rankings are comparable; clip the
    # permutation values at zero, since a negative drop means the predictor was
    # doing nothing and the sign is noise.
    perm = np.clip(perm, 0, None)
    return (imp.mean(0) / imp.mean(0).sum(), imp.std(0) / imp.mean(0).sum(),
            perm.mean(0) / max(perm.mean(0).sum(), 1e-12),
            perm.std(0) / max(perm.mean(0).sum(), 1e-12))


def panel_top(ax, cols, imp, imp_sd, perm, perm_sd, top):
    order = np.argsort(perm)[::-1][:top][::-1]
    ypos = np.arange(len(order))
    hgt = 0.38
    ax.barh(ypos + hgt / 2, perm[order], hgt, xerr=perm_sd[order],
            color=BAR_PERM, edgecolor="none", label="permutation, out of fold",
            error_kw=dict(lw=0.6, ecolor="0.35"), zorder=3)
    ax.barh(ypos - hgt / 2, imp[order], hgt, xerr=imp_sd[order],
            color=BAR_IMP, edgecolor="none", label="impurity, in sample",
            error_kw=dict(lw=0.6, ecolor="0.55"), zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([pretty(cols[i]) for i in order], fontsize=8)
    ax.set_xlabel("Relative importance")
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", axis="x", lw=0.5, color="0.88", zorder=0)
    ax.grid(which="minor", axis="x", lw=0.3, color="0.94", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92,
              edgecolor="0.6", borderpad=0.35)
    ax.text(0.01, 0.985, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def panel_family(ax, cols, imp, perm):
    idx_keys = list(INDEX_LABEL)
    desc_keys = list(DESC_LABEL)
    agg_i = {k: [0.0, 0.0] for k in idx_keys}
    agg_d = {k: [0.0, 0.0] for k in desc_keys}
    for c, a, b in zip(cols, imp, perm):
        i, d = parse_name(c)
        if i in agg_i:
            agg_i[i][0] += a
            agg_i[i][1] += b
        if d in agg_d:
            agg_d[d][0] += a
            agg_d[d][1] += b

    labels = ([INDEX_LABEL[k] for k in idx_keys]
              + [DESC_LABEL[k] for k in desc_keys])
    vals_i = [agg_i[k][0] for k in idx_keys] + [agg_d[k][0] for k in desc_keys]
    vals_p = [agg_i[k][1] for k in idx_keys] + [agg_d[k][1] for k in desc_keys]

    x = np.arange(len(labels))
    wid = 0.38
    ax.bar(x - wid / 2, vals_i, wid, color=BAR_IMP, edgecolor="none",
           label="impurity", zorder=3)
    ax.bar(x + wid / 2, vals_p, wid, color=BAR_PERM, edgecolor="none",
           label="permutation", zorder=3)
    # Separate the two groupings.
    ax.axvline(len(idx_keys) - 0.5, color="0.45", lw=0.8, ls="--", zorder=2)
    ax.text(len(idx_keys) / 2 - 0.5, ax.get_ylim()[1], "spectral index",
            ha="center", va="bottom", fontsize=8.5, color="#444444")
    ax.text((len(idx_keys) + len(labels)) / 2 - 0.5, ax.get_ylim()[1],
            "descriptor type", ha="center", va="bottom", fontsize=8.5,
            color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.6)
    ax.set_ylabel("Summed relative importance")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", axis="y", lw=0.5, color="0.88", zorder=0)
    ax.grid(which="minor", axis="y", lw=0.3, color="0.94", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92,
              edgecolor="0.6", borderpad=0.35)
    ax.text(0.01, 0.985, "(b)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")
    return dict(zip(labels, zip(vals_i, vals_p)))


def add_credit(fig, ax_bottom, text, fontsize=8, pad_in=0.14):
    import textwrap
    r = fig.canvas.get_renderer()
    fig_w = fig.get_size_inches()[0] * fig.dpi
    t = None
    for ncol in range(170, 60, -2):
        if t is not None:
            t.remove()
        t = fig.text(0.0, 0.0, textwrap.fill(text, ncol), fontsize=fontsize,
                     color="#555555", va="top", ha="left", linespacing=1.25)
        fig.canvas.draw()
        if t.get_window_extent(r).width <= fig_w:
            break
    fig.canvas.draw()
    low = ax_bottom.get_tightbbox(r).y0
    t.set_position((0.0, low / (fig.get_size_inches()[1] * fig.dpi)
                    - pad_in / fig.get_size_inches()[1]))
    return t


def covers_data(bb, ax, renderer):
    return sum(1 for pa in ax.patches
               if pa.get_window_extent(renderer).overlaps(bb))


def build(features_csv, plots_csv, folds, top, ntree, nrep, seed, prefix):
    try:
        df = pd.read_csv(features_csv)
    except FileNotFoundError:
        sys.exit(f"ERROR: {features_csv} not found. Run build_features.py first.")
    df = df.merge(spatial_blocks(plots_csv), on="plot", how="left")
    df = df.dropna(subset=["block"])

    drop = ["plot", "epoch", "subtype", "n_obs", "block"]
    cols = [c for c in df.columns if c not in drop]
    X = df[cols].to_numpy(float)
    y = df["subtype"].to_numpy()
    groups = df["block"].to_numpy()
    folds = min(folds, len(np.unique(groups)))
    print(f"[info] {len(df)} rows, {len(cols)} predictors, {folds} folds")

    imp, imp_sd, perm, perm_sd = importances(
        X, y, groups, cols, folds, ntree, nrep, seed)

    rank_i = [cols[i] for i in np.argsort(imp)[::-1][:5]]
    rank_p = [cols[i] for i in np.argsort(perm)[::-1][:5]]
    print(f"[info] top 5 by impurity   : {', '.join(pretty(c) for c in rank_i)}")
    print(f"[info] top 5 by permutation: {', '.join(pretty(c) for c in rank_p)}")
    shared = len(set(rank_i) & set(rank_p))
    print(f"[check] rankings share {shared} of their top 5")

    fig = plt.figure(figsize=(6.89, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0])
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    panel_top(ax_a, cols, imp, imp_sd, perm, perm_sd, top)
    fam = panel_family(ax_b, cols, imp, perm)
    print("[info] summed permutation importance by family:")
    for k, (a, b) in sorted(fam.items(), key=lambda t: -t[1][1]):
        print(f"[info]   {k.replace(chr(10), ' '):26s} impurity {a:.3f}  "
              f"permutation {b:.3f}")

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for tag, ax in (("a", ax_a), ("b", ax_b)):
        lb = ax.get_legend().get_window_extent(r)
        print(f"[check] panel ({tag}) legend covers "
              f"{covers_data(lb, ax, r)} bars (must be 0)")
        clash = [t.get_text()[:18] for t in ax.texts
                 if t.get_window_extent(renderer=r).overlaps(lb)]
        print(f"[check] panel ({tag}) legend overlaps {len(clash)} labels"
              + (f": {clash}" if clash else ""))

    import sklearn
    add_credit(fig, ax_b, CREDIT.format(
        ntree=ntree, nf=folds, n=len(df), p=len(cols), nrep=nrep,
        py=".".join(map(str, sys.version_info[:3])),
        mpl=matplotlib.__version__, sk=sklearn.__version__, pd=pd.__version__))

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf and {prefix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--plots", default="reference_plots.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="fig08_importance")
    a = ap.parse_args()
    build(a.features, a.plots, a.folds, a.top, a.trees, a.repeats, a.seed,
          a.prefix)
