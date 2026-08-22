#!/usr/bin/env python3
"""
fig05_phenology_curves.py -- Figure 5 of the Goksu Delta halophyte manuscript.

Mean annual phenological trajectories of the halophyte habitat classes.

(a)-(c) Seasonal course of greenness (NDVI), moisture (NDMI) and canopy salinity
        response (CRSI), each as the class median with an interquartile envelope,
        computed from all cloud-free observations at the reference plots of that
        class and smoothed over a moving window in day-of-year.
(d)     Class separability as a function of day of year, so the seasonal windows
        in which the classes are actually distinguishable can be read directly
        rather than inferred by eye from the three panels above. Separability is
        the mean pairwise standardised distance over the three index axes,

            d_ij(t) = mean_k |mu_ik(t) - mu_jk(t)| / sqrt(s_ik(t)^2 + s_jk(t)^2)

        averaged over all class pairs i < j, where mu and s are the class mean
        and standard deviation of index k inside the window centred on t. Larger
        is better separated; the value is unitless.

INPUT
-----
    plot_timeseries.csv with columns
        plot, subtype, sensor, date, doy, ndvi, ndmi, crsi, si

    Produce it with make_plot_timeseries.js in Earth Engine, which already has
    the 149 halophytic plot coordinates embedded. Nothing here invents
    reflectance: with no CSV the script stops and says so.

USAGE
-----
    python3 fig05_phenology_curves.py
    python3 fig05_phenology_curves.py --csv plot_timeseries.csv --window 30
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
from matplotlib.ticker import AutoMinorLocator

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

# Same four classes and the same hue identity as Figure 3, but darkened: the
# bright yellow and cyan that read well over the green relief basemap are
# illegible as thin lines on white.
CLASSES = [
    ("Swamp", "#B8860B"),                  # dark goldenrod  (yellow in Fig. 3)
    ("Temporary ponds", "#118C8C"),        # dark teal       (cyan in Fig. 3)
    ("Temporary floodplain", "#E34A00"),   # orangered
    ("Terrestrial saline flats", "#B5179E"),  # dark magenta
]

INDICES = [
    ("ndvi", "NDVI", "greenness"),
    ("ndmi", "NDMI", "moisture"),
    ("crsi", "CRSI", "salinity"),
]

MONTH_TICKS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

CREDIT = (
    "Index values sampled at the {nplot} halophytic releve plots of the published survey "
    "from all Landsat Collection 2 Level-2 scenes passing the cloud, cloud-shadow and "
    "cirrus mask (QA_PIXEL bits 1-4), {nobs} plot-date observations in total. NDVI after "
    "Tucker (1979), NDMI after Gao (1996), CRSI after Scudiero et al. (2015). Curves are "
    "class medians within a centred moving window of {win} days in day-of-year, with the "
    "envelope spanning the interquartile range; windows holding fewer than {minn} "
    "observations are left blank rather than interpolated. Separability in panel (d) is "
    "the mean pairwise standardised distance across the three index axes. Software used "
    "for plotting figure: Python {py} with Matplotlib {mpl} and pandas {pd}. "
    "Source: authors"
)


def load(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(
            f"ERROR: {path} not found.\n"
            "Figure 5 is built from reflectance sampled at the reference plots;\n"
            "this script will not invent phenology. Run make_plot_timeseries.js\n"
            "in Earth Engine (the plot coordinates are already embedded in it),\n"
            "download the CSV, and re-run."
        )
    need = {"subtype", "doy", "ndvi", "ndmi", "crsi"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing column(s): {sorted(missing)}")
    df = df.dropna(subset=["doy"])
    df["doy"] = df["doy"].astype(int)
    return df


def rolling_stats(df, col, window, min_n, grid):
    """Median, quartiles, mean and sd of `col` in a circular DOY window."""
    out = {}
    d = df["doy"].to_numpy()
    v = df[col].to_numpy()
    half = window / 2.0
    for t in grid:
        # Circular distance in day-of-year, so December and January share a
        # window instead of the series being cut at the year boundary.
        dist = np.abs(d - t)
        dist = np.minimum(dist, 365.25 - dist)
        sel = v[dist <= half]
        sel = sel[np.isfinite(sel)]
        if len(sel) < min_n:
            out[t] = (np.nan,) * 5
        else:
            out[t] = (np.median(sel), np.percentile(sel, 25),
                      np.percentile(sel, 75), sel.mean(), sel.std(ddof=1))
    return np.array([out[t] for t in grid])


def panel_index(ax, df, col, label, subtitle, window, min_n, grid, tag):
    for name, colour in CLASSES:
        sub = df[df["subtype"] == name]
        if sub.empty:
            continue
        st = rolling_stats(sub, col, window, min_n, grid)
        ax.fill_between(grid, st[:, 1], st[:, 2], color=colour, alpha=0.16, lw=0)
        ax.plot(grid, st[:, 0], color=colour, lw=1.5, label=name)

    ax.set_xlim(1, 365)
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel(f"{label}\n({subtitle})")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.88", zorder=0)
    ax.grid(which="minor", lw=0.3, color="0.94", zorder=0)
    ax.set_axisbelow(True)



def panel_separability(ax, df, window, min_n, grid):
    names = [n for n, _ in CLASSES if n in set(df["subtype"])]
    stats = {n: {c: rolling_stats(df[df["subtype"] == n], c, window, min_n, grid)
                 for c, _, _ in INDICES} for n in names}

    curves = {}
    for a, b in combinations(names, 2):
        acc = []
        for c, _, _ in INDICES:
            sa, sb = stats[a][c], stats[b][c]
            denom = np.sqrt(sa[:, 4] ** 2 + sb[:, 4] ** 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                acc.append(np.abs(sa[:, 3] - sb[:, 3]) / denom)
        curves[(a, b)] = np.nanmean(np.vstack(acc), axis=0)

    mean_sep = np.nanmean(np.vstack(list(curves.values())), axis=0)
    ax.plot(grid, mean_sep, color="#333333", lw=2.0, zorder=4,
            label="mean over all class pairs")
    for (a, b), y in curves.items():
        ax.plot(grid, y, lw=0.9, alpha=0.55, zorder=3)

    if np.isfinite(mean_sep).any():
        best = int(grid[np.nanargmax(mean_sep)])
        ax.axvline(best, color="#333333", lw=0.8, ls=":", zorder=2)
        # Anchored low on the y-axis rather than at the peak: the pairwise
        # curves fan out across the upper half of this panel and the label
        # sat on them.
        ax.annotate(f"most separable\naround day {best}",
                    xy=(best, 0.20), xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=8.5, va="bottom", ha="left", color="#333333")
        print(f"[info] peak mean separability at day {best} "
              f"({np.nanmax(mean_sep):.2f})")

    ax.set_xlim(1, 365)
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Separability\n(std. distance)")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.88", zorder=0)
    ax.grid(which="minor", lw=0.3, color="0.94", zorder=0)
    ax.set_axisbelow(True)
    # Below the axes: every interior corner of this panel carries either the
    # pairwise curves or the peak annotation.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False,
              handlelength=1.8, borderpad=0.3)



def covers_data(bb, ax, renderer):
    """Count data elements under a display-space bbox (must be zero)."""
    hits = 0
    for ln in ax.get_lines():
        d = np.column_stack(ln.get_data())
        d = d[np.isfinite(d).all(axis=1)]
        if d.shape[0] > 2:
            p = ax.transData.transform(d)
            hits += int(((p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) &
                         (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)).sum())
    for col in ax.collections:
        if col.get_window_extent(renderer).overlaps(bb):
            hits += 1
    return hits


def add_credit(fig, ax_bottom, text, fontsize=8, pad_in=0.14):
    """Wrap the credit to the measured figure width, hung below the axes."""
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


def build(csv_path, window, min_n, prefix):
    df = load(csv_path)
    grid = np.arange(1, 366, 5)

    print(f"[info] {len(df)} plot-date observations, "
          f"{df['plot'].nunique() if 'plot' in df else '?'} plots")
    for name, _ in CLASSES:
        n = int((df["subtype"] == name).sum())
        print(f"[info]   {name:26s} {n:6d} observations")

    fig = plt.figure(figsize=(6.89, 7.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.03, hspace=0.055)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.15])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    for ax, (col, label, sub), tag in zip(axes[:3], INDICES, "abc"):
        panel_index(ax, df, col, label, sub, window, min_n, grid, tag)
    panel_separability(axes[3], df, window, min_n, grid)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.30),
                   ncol=4, frameon=False, handlelength=1.6,
                   columnspacing=1.4)

    fig.canvas.draw()
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()

    # Panel letters above the upper-left corner of each panel, outside the
    # axes. Figure coordinates rather than axes-relative ones, so the four sit
    # at one offset instead of drifting with the differing panel heights.
    for ax, tag in zip(axes, "abcd"):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.052, pos.y1 + 0.004, f"({tag})", fontsize=11,
                 fontweight="bold", va="bottom", ha="left")

    fig.canvas.draw()
    for tag, ax in zip("abcd", axes):
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(r)
        print(f"[check] panel ({tag}) legend covers "
              f"{covers_data(lb, ax, r)} data elements (must be 0)")
        clash = [t.get_text() for t in ax.texts
                 if t.get_window_extent(renderer=r).overlaps(lb)]
        print(f"[check] panel ({tag}) legend overlaps {len(clash)} labels "
              "(must be 0)" + (f": {clash}" if clash else ""))
    boxes = [(t.get_text()[:18], t.get_window_extent(renderer=r))
             for ax in axes for t in ax.texts if t.get_text().strip()]
    pairs = [(boxes[i][0], boxes[j][0])
             for i in range(len(boxes)) for j in range(i + 1, len(boxes))
             if boxes[i][1].overlaps(boxes[j][1])]
    print(f"[check] overlapping text pairs: {len(pairs)}"
          + (f" {pairs}" if pairs else ""))

    add_credit(fig, axes[3], CREDIT.format(
        nplot=df["plot"].nunique() if "plot" in df else "n",
        nobs=len(df), win=window, minn=min_n,
        py=".".join(map(str, sys.version_info[:3])),
        mpl=matplotlib.__version__, pd=pd.__version__))

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf and {prefix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="plot_timeseries.csv")
    ap.add_argument("--window", type=int, default=30,
                    help="moving window width in days of year")
    ap.add_argument("--min-obs", type=int, default=8,
                    help="minimum observations per window; below this, blank")
    ap.add_argument("--prefix", default="fig05_phenology_curves")
    a = ap.parse_args()
    build(a.csv, a.window, a.min_obs, a.prefix)
