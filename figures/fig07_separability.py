#!/usr/bin/env python3
"""
fig07_separability.py -- Figure 7 of the Goksu Delta halophyte manuscript.

Feature-space separability of the halophyte habitat classes.

Both panels are computed OUT OF FOLD, under the same spatial blocking as the
classification itself. This is not a stylistic preference. In-sample measures
are meaningless at this dimensionality: with 40 predictors against a few tens of
plots per class, the Jeffries-Matusita distance returns 1.9 to 2.0 for every
class pair -- and returns the same values when the class labels are randomly
shuffled. It is measuring the number of predictors, not the structure of the
data. The permutation test that establishes this is run by --permutation-test
and its result is reported in the figure credit.

(a) Ordination on the first two linear discriminants, fitted on the training
    blocks of each fold and applied to the held-out block, so no point is
    projected by a model that has seen it. Ellipses enclose 95 per cent of each
    class under a Gaussian assumption.

(b) Pairwise separability as cross-validated balanced accuracy: for each class
    pair a random forest is trained under the same blocked folds and scored on
    held-out blocks. The measure runs from 0.5, which is chance and means the
    pair is indistinguishable, to 1.0 for complete separation. Unlike a
    distributional distance it cannot be inflated by dimensionality.

INPUT
-----
    features.csv from build_features.py

USAGE
-----
    python3 fig07_separability.py --features features.csv
    python3 fig07_separability.py --epoch baseline
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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

# Same class identity and hue order as Figures 3 and 5.
CLASSES = [
    ("Swamp", "#B8860B"),
    ("Temporary ponds", "#118C8C"),
    ("Temporary floodplain", "#E34A00"),
    ("Terrestrial saline flats", "#B5179E"),
]
SHORT = {"Swamp": "Swamp", "Temporary ponds": "Ponds",
         "Temporary floodplain": "Floodplain",
         "Terrestrial saline flats": "Saline flats"}

CREDIT = (
    "Computed from {n} plot-epoch feature rows carrying {p} predictors: the harmonic "
    "coefficients, amplitude and phase of Eq. (5) for each of the four indices, with "
    "their medians and interquartile ranges. Both panels are evaluated out of fold under "
    "the spatial blocking of Sect. 2.8. (a) First two linear discriminants, fitted on the "
    "training blocks of each fold and applied to the held-out block, capturing {v1:.0f} "
    "and {v2:.0f} per cent of the between-class separation on average; ellipses enclose "
    "95 per cent of each class under a Gaussian assumption. (b) Balanced accuracy of a "
    "random forest trained on each class pair under the same folds, where 0.5 is chance. "
    "A distributional distance was rejected for this panel: in the {p}-dimensional "
    "predictor space the Jeffries-Matusita distance returns {jm_obs} for these pairs and "
    "{jm_null} under randomly shuffled labels, so it measures dimensionality rather than "
    "class structure. Software used for plotting figure: Python {py} with Matplotlib "
    "{mpl}, scikit-learn {sk} and pandas {pd}. Source: authors"
)


def load(path, epoch):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"ERROR: {path} not found. Run build_features.py first.")
    if epoch != "both":
        df = df[df["epoch"] == epoch]
        if df.empty:
            sys.exit(f"ERROR: no rows for epoch '{epoch}'")
    return df


def jm_distance(Xa, Xb):
    """Jeffries-Matusita distance between two classes, bounded on [0, 2]."""
    ma, mb = Xa.mean(axis=0), Xb.mean(axis=0)
    Sa = LedoitWolf().fit(Xa).covariance_
    Sb = LedoitWolf().fit(Xb).covariance_
    S = (Sa + Sb) / 2.0
    d = (ma - mb).reshape(-1, 1)
    # Bhattacharyya distance: mean term plus covariance term.
    term1 = float((0.125 * d.T @ np.linalg.pinv(S) @ d).item())
    sign_s, logdet_s = np.linalg.slogdet(S)
    sign_a, logdet_a = np.linalg.slogdet(Sa)
    sign_b, logdet_b = np.linalg.slogdet(Sb)
    term2 = 0.5 * (logdet_s - 0.5 * (logdet_a + logdet_b))
    b = term1 + max(term2, 0.0)
    return float(2.0 * (1.0 - np.exp(-b)))


def ellipse(ax, xy, colour, n_std=2.4477):     # 2.4477 sigma -> 95 per cent in 2-D
    if len(xy) < 3:
        return
    cov = np.cov(xy, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(xy.mean(axis=0), w, h, angle=angle, facecolor=colour,
                         alpha=0.13, edgecolor=colour, linewidth=1.1, zorder=2))


def panel_ordination(ax, X, y, groups, names, folds):
    """Out-of-fold LDA projection: fit on training blocks, project held-out."""
    Z = np.full((len(X), 2), np.nan)
    vars = []
    cv = GroupKFold(n_splits=folds)
    for tr, te in cv.split(X, y, groups):
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto",
                                         n_components=2)
        lda.fit(X[tr], y[tr])
        Z[te] = lda.transform(X[te])
        vars.append(lda.explained_variance_ratio_[:2] * 100.0)
    var = np.mean(vars, axis=0)

    for name, colour in CLASSES:
        if name not in names:
            continue
        sel = Z[y == name]
        sel = sel[np.isfinite(sel).all(axis=1)]
        ellipse(ax, sel, colour)
        ax.scatter(sel[:, 0], sel[:, 1], s=13, facecolor=colour,
                   edgecolor="white", linewidth=0.35, alpha=0.9, zorder=3,
                   label=f"{SHORT[name]} (n = {len(sel)})")

    ax.set_xlabel(f"LD1, out of fold ({var[0]:.0f} % of separation)")
    ax.set_ylabel(f"LD2, out of fold ({var[1]:.0f} %)")
    ax.axhline(0, color="0.85", lw=0.6, zorder=1)
    ax.axvline(0, color="0.85", lw=0.6, zorder=1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.90", zorder=0)
    ax.set_axisbelow(True)
    # Handles are returned rather than drawn here: an axes legend anchored
    # below panel (a) collided with the credit block, so the classes are
    # listed once, on one line, beneath both panels.
    return var, ax.get_legend_handles_labels()


def pair_accuracy(X, y, groups, a, b, folds, seed):
    """Cross-validated balanced accuracy for one class pair (0.5 = chance)."""
    m = (y == a) | (y == b)
    Xp, yp, gp = X[m], y[m], groups[m]
    k = min(folds, len(np.unique(gp)))
    if k < 2:
        return np.nan
    cv, true, pred = GroupKFold(n_splits=k), [], []
    for tr, te in cv.split(Xp, yp, gp):
        if len(np.unique(yp[tr])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=300,
                                     class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=-1)
        clf.fit(Xp[tr], yp[tr])
        true.extend(yp[te])
        pred.extend(clf.predict(Xp[te]))
    return float(balanced_accuracy_score(true, pred)) if true else np.nan


def panel_matrix(ax, X, y, groups, names, folds, seed):
    n = len(names)
    M = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        v = pair_accuracy(X, y, groups, names[i], names[j], folds, seed)
        M[i, j] = M[j, i] = v

    im = ax.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0, origin="upper")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([SHORT[c] for c in names], rotation=30, ha="right")
    ax.set_yticklabels([SHORT[c] for c in names])
    ax.tick_params(which="both", length=0)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=9,
                        color="0.55")
            elif np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if M[i, j] < 0.78 else "#1a1a1a")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Balanced accuracy, out of fold", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    cb.ax.yaxis.set_minor_locator(AutoMinorLocator())
    return M, cb


def add_credit(fig, anchors, text, fontsize=8, pad_in=0.10):
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
    # Lowest edge of every element passed in, so the block cannot ride up over
    # the shared legend or the axis labels.
    low = min(a.get_tightbbox(r).y0 if hasattr(a, "get_tightbbox")
              else a.get_window_extent(r).y0 for a in anchors)
    t.set_position((0.0, low / (fig.get_size_inches()[1] * fig.dpi)
                    - pad_in / fig.get_size_inches()[1]))
    return t


def covers_data(bb, ax, renderer):
    hits = 0
    for col in ax.collections:
        off = np.asarray(col.get_offsets())
        if len(off):
            p = ax.transData.transform(off)
            hits += int(((p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) &
                         (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)).sum())
    for pa in ax.patches:
        if pa.get_window_extent(renderer).overlaps(bb):
            hits += 1
    return hits


def spatial_blocks(plots_csv, block_km=2.0):
    """Same 2 km blocking as train_classifier.py, so the folds match."""
    p = pd.read_csv(plots_csv)
    p = p[p["group"] == "halophytic"][["plot", "lon", "lat"]].drop_duplicates()
    dlat = block_km / 111.32
    dlon = block_km / (111.32 * np.cos(np.radians(p["lat"].mean())))
    p["block"] = (np.floor(p["lat"] / dlat).astype(int).astype(str) + "_"
                  + np.floor(p["lon"] / dlon).astype(int).astype(str))
    return p[["plot", "block"]]


def permutation_null(X, y, names, n_shuffle, seed):
    """Jeffries-Matusita under shuffled labels: the null this metric has."""
    rng = np.random.default_rng(seed)
    pairs = list(combinations(range(len(names)), 2))
    obs = [jm_distance(X[y == names[i]], X[y == names[j]]) for i, j in pairs]
    null = []
    for _ in range(n_shuffle):
        ys = rng.permutation(y)
        null += [jm_distance(X[ys == names[i]], X[ys == names[j]])
                 for i, j in pairs]
    return float(np.mean(obs)), float(np.mean(null))


def build(features_csv, plots_csv, epoch, folds, seed, prefix):
    df = load(features_csv, epoch)
    df = df.merge(spatial_blocks(plots_csv), on="plot", how="left")
    df = df.dropna(subset=["block"])
    names = [c for c, _ in CLASSES if c in set(df["subtype"])]

    drop = ["plot", "epoch", "subtype", "n_obs", "block"]
    Xraw = df.drop(columns=[c for c in drop if c in df.columns]).to_numpy(float)
    X = StandardScaler().fit_transform(Xraw)
    y = df["subtype"].to_numpy()
    groups = df["block"].to_numpy()
    folds = min(folds, len(np.unique(groups)))
    print(f"[info] {len(df)} rows, {X.shape[1]} predictors, "
          f"{len(np.unique(groups))} blocks, {folds} folds, epoch '{epoch}'")

    jm_obs, jm_null = permutation_null(X, y, names, 3, seed)
    print(f"[check] Jeffries-Matusita: observed mean {jm_obs:.2f}, "
          f"shuffled-label mean {jm_null:.2f}")
    if jm_null > 0.75 * jm_obs:
        print("[check] -> in-sample distributional distance is uninformative "
              "here; panel (b) uses cross-validated accuracy instead")

    fig = plt.figure(figsize=(6.89, 3.9), constrained_layout=True)
    # Default constrained-layout padding left about a centimetre of white
    # between the axis labels and the shared legend row.
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.01, hspace=0.01,
                                    wspace=0.03)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    var, (handles, labels) = panel_ordination(ax_a, X, y, groups, names, folds)
    M, cb = panel_matrix(ax_b, X, y, groups, names, folds, seed)

    # Freeze the layout before adding anything positioned by hand. With
    # constrained layout still active, an "outside lower center" legend is
    # repositioned when savefig re-runs the solver, while a manually placed
    # fig.text is not -- which put the legend underneath the credit block in
    # the exported file even though both looked correct on the live canvas.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    r0 = fig.canvas.get_renderer()
    fig_w_in, fig_h_in = fig.get_size_inches()
    fig_h = fig_h_in * fig.dpi

    pa, pb = ax_a.get_position(), ax_b.get_position()
    H = min(pa.height, pb.height)
    top = max(pa.y1, pb.y1)
    k = fig_h_in / fig_w_in
    ax_a.set_aspect("auto")
    ax_b.set_aspect("auto")
    ax_a.set_position([pa.x0, top - H, pa.width, H])
    wb = H * k                       # square box for the 4 x 4 matrix
    ax_b.set_position([pb.x0, top - H, wb, H])
    cb.ax.set_position([pb.x0 + wb + 0.014, top - H, 0.020, H])

    fig.canvas.draw()
    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
        fig.text(ax.get_position().x0, top + 0.016, tag, fontsize=11,
                 fontweight="bold", va="bottom", ha="left")

    fig.canvas.draw()
    low = min(ax_a.get_tightbbox(r0).y0, ax_b.get_tightbbox(r0).y0) / fig_h
    leg = fig.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, low - 0.012),
                     ncol=len(labels), frameon=False, handlelength=1.2,
                     columnspacing=1.6, scatterpoints=1, borderaxespad=0.0)

    print("[info] pairwise balanced accuracy, out of fold:")
    for i, j in combinations(range(len(names)), 2):
        v = M[i, j]
        verdict = ("well separated" if v >= 0.85 else
                   "partly separated" if v >= 0.70 else
                   "weakly separated" if v >= 0.60 else "at or near chance")
        print(f"[info]   {SHORT[names[i]]:12s} vs {SHORT[names[j]]:12s} "
              f"{v:.2f}  ({verdict})")

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lb = leg.get_window_extent(r)
    for tag, ax in (("a", ax_a), ("b", ax_b)):
        print(f"[check] shared legend covers "
              f"{covers_data(lb, ax, r)} data elements in panel ({tag}) "
              "(must be 0)")
    boxes = [(t.get_text()[:16], t.get_window_extent(renderer=r))
             for ax in (ax_a, ax_b) for t in ax.texts if t.get_text().strip()]
    pairs = [(boxes[i][0], boxes[j][0])
             for i in range(len(boxes)) for j in range(i + 1, len(boxes))
             if boxes[i][1].overlaps(boxes[j][1])]
    print(f"[check] overlapping text pairs: {len(pairs)}"
          + (f" {pairs}" if pairs else ""))
    hit = [t.get_text()[:16] for ax in (ax_a, ax_b) for t in ax.texts
           if t.get_window_extent(renderer=r).overlaps(lb)]
    print(f"[check] legend overlaps {len(hit)} axis labels (must be 0)"
          + (f": {hit}" if hit else ""))

    import sklearn
    credit = add_credit(fig, [ax_a, ax_b, leg], CREDIT.format(
        n=len(df), p=X.shape[1], v1=var[0], v2=var[1],
        jm_obs=f"{jm_obs:.2f} on average", jm_null=f"{jm_null:.2f}",
        py=".".join(map(str, sys.version_info[:3])),
        mpl=matplotlib.__version__, sk=sklearn.__version__,
        pd=pd.__version__))

    fig.canvas.draw()
    gap = leg.get_window_extent(r).y0 - credit.get_window_extent(r).y1
    print(f"[check] gap between legend and credit: {gap:.0f} px "
          f"({gap / fig.dpi * 2.54:.2f} cm; must be positive)")

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf and {prefix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--epoch", default="both",
                    choices=["both", "baseline", "present"])
    ap.add_argument("--plots", default="reference_plots.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="fig07_separability")
    a = ap.parse_args()
    build(a.features, a.plots, a.epoch, a.folds, a.seed, a.prefix)
