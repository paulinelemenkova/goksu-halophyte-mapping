#!/usr/bin/env python3
"""
fig04_image_inventory.py -- Figure 4 of the Goksu Delta halophyte manuscript.

Inventory of the usable satellite scenes by sensor and acquisition date.

(a) Scene timeline: every archive scene intersecting the delta, one marker per
    scene, arranged by sensor and coloured by the fraction of the study area
    that survives cloud, shadow and cirrus masking. The two analysis epochs are
    shaded, so the reader can see at a glance how much usable imagery each one
    rests on.
(b) Seasonal sampling density: usable scenes per calendar month in each epoch.
    This is the panel that matters for the phenological reconstruction of
    Section 2.5 -- a harmonic fit is only as good as the seasonal coverage
    behind it, and gaps in a particular month are a limitation to report, not
    to hide.

INPUT
-----
    scene_inventory.csv with columns
        sensor      Landsat 5 TM | Landsat 7 ETM+ | Landsat 8 OLI |
                    Landsat 9 OLI-2 | Sentinel-2A | Sentinel-2B | Sentinel-2C
        date        ISO yyyy-MM-dd
        cloud_pct   scene-level cloud cover, per cent
        usable_pct  per cent of the study-area footprint passing the mask

    Produce it with make_scene_inventory.js in Earth Engine. Nothing in this
    script invents scene metadata: if the CSV is absent it stops and says so.

USAGE
-----
    python3 fig04_image_inventory.py                      # scene_inventory.csv
    python3 fig04_image_inventory.py --csv my_scenes.csv
    python3 fig04_image_inventory.py --usable-threshold 70
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
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

# Sensor rows, newest at the top of the panel.
SENSOR_ORDER = [
    "Sentinel-2C", "Sentinel-2B", "Sentinel-2A",
    "Landsat 9 OLI-2", "Landsat 8 OLI", "Landsat 7 ETM+", "Landsat 5 TM",
]

# Analysis epochs. The baseline brackets the 2004-2005 field survey; the
# present epoch is the Sentinel-2 record. Edit here, not in the plotting code.
EPOCHS = [
    ("Baseline epoch", "2003-01-01", "2007-12-31", "#c8b89a"),
    ("Present epoch", "2021-01-01", "2026-12-31", "#8fbcd4"),
]

MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

CREDIT = (
    "Scene inventory exported from the Landsat Collection 2 Level-2 and Sentinel-2 "
    "Level-2A archives for the study-area footprint (33.86-34.11 E, 36.24-36.42 N). "
    "Usable fraction is the share of the footprint passing the cloud, cloud-shadow "
    "and cirrus mask: QA_PIXEL bits 1-4 for Landsat, Cloud Score+ cs_cdf >= {cs} for "
    "Sentinel-2, the latter preferred over the scene classification band because that "
    "band over-flags bright saline crusts as cloud. A scene counts as usable where the "
    "usable fraction reaches {thr:.0f} per cent. Software used for plotting figure: "
    "Python {py} with Matplotlib {mpl} and pandas {pd}. Source: authors"
)


def load(path, threshold):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(
            f"ERROR: {path} not found.\n"
            "Figure 4 is built from real archive metadata; this script will not\n"
            "invent scene dates or cloud cover. Run make_scene_inventory.js in\n"
            "Earth Engine, download the two CSVs, concatenate them, and re-run."
        )
    missing = {"sensor", "date", "usable_pct"} - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing column(s): {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["usable"] = df["usable_pct"] >= threshold
    df["month"] = df["date"].dt.month
    df["epoch"] = None
    for name, lo, hi, _ in EPOCHS:
        sel = (df["date"] >= lo) & (df["date"] <= hi)
        df.loc[sel, "epoch"] = name
    return df


def panel_a(ax, df):
    """Scene timeline, one row per sensor, colour = usable fraction."""
    present = [s for s in SENSOR_ORDER if s in set(df["sensor"])]
    ypos = {s: i for i, s in enumerate(present)}

    for name, lo, hi, colour in EPOCHS:
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=colour,
                   alpha=0.35, lw=0, zorder=0)

    sc = None
    for s in present:
        sub = df[df["sensor"] == s]
        sc = ax.scatter(sub["date"], np.full(len(sub), ypos[s]),
                        c=sub["usable_pct"], cmap="viridis", vmin=0, vmax=100,
                        s=7, marker="|", linewidths=0.7, zorder=3)

    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)
    ax.set_ylim(-0.7, len(present) - 0.3)
    ax.set_xlabel("Acquisition date")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", axis="x", lw=0.5, color="0.85", zorder=1)
    ax.grid(which="minor", axis="x", lw=0.3, color="0.92", zorder=1)
    ax.set_axisbelow(True)

    cb = ax.figure.colorbar(sc, ax=ax, pad=0.012, fraction=0.035)
    cb.set_label("Usable fraction of study area (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    cb.ax.yaxis.set_minor_locator(AutoMinorLocator())

    handles = [Patch(facecolor=c, alpha=0.35, label=n) for n, _, _, c in EPOCHS]
    # Below the sensor rows: the bottom-left corner of the panel holds the
    # earliest Landsat 5 markers, which a legend anchored there covered.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False, ncol=len(EPOCHS), handlelength=1.4,
              borderpad=0.35, columnspacing=1.6)
    ax.text(0.004, 0.97, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")

    # An interval with no scenes at all means it was never exported, not that
    # no imagery exists. Label it, so the blank stretch cannot be misread.
    gaps = _empty_spans(df)
    for lo, hi in gaps:
        mid = lo + (hi - lo) / 2
        ax.text(mid, ax.get_ylim()[1] - 0.35, "not exported", ha="center",
                va="top", fontsize=8, style="italic", color="#777777")


def _empty_spans(df, min_years=2):
    """Multi-year stretches inside the axis range that hold no scenes."""
    yrs = sorted(df["date"].dt.year.unique())
    spans, run = [], []
    for y in range(min(yrs), max(yrs) + 1):
        if y not in yrs:
            run.append(y)
        elif run:
            if len(run) >= min_years:
                spans.append((pd.Timestamp(f"{run[0]}-01-01"),
                              pd.Timestamp(f"{run[-1]}-12-31")))
            run = []
    if len(run) >= min_years:
        spans.append((pd.Timestamp(f"{run[0]}-01-01"),
                      pd.Timestamp(f"{run[-1]}-12-31")))
    return spans


def panel_b(ax, df):
    """Usable scenes per calendar month, one bar group per epoch."""
    names = [n for n, _, _, _ in EPOCHS]
    colours = {n: c for n, _, _, c in EPOCHS}
    width = 0.8 / len(names)
    x = np.arange(12)

    for k, name in enumerate(names):
        sub = df[(df["epoch"] == name) & df["usable"]]
        counts = [int((sub["month"] == m).sum()) for m in range(1, 13)]
        ax.bar(x + (k - (len(names) - 1) / 2) * width, counts, width * 0.92,
               color=colours[name], edgecolor="0.25", linewidth=0.5,
               label=f"{name} (n = {sum(counts)})", zorder=3)

    # Headroom for the legend: without it the legend box has nowhere to sit
    # that is not on top of a bar.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.42)

    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_xlabel("Month of acquisition")
    ax.set_ylabel("Usable scenes")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", axis="y", lw=0.5, color="0.85", zorder=0)
    ax.grid(which="minor", axis="y", lw=0.3, color="0.92", zorder=0)
    ax.set_axisbelow(True)
    # Offset right of the "(b)" panel letter, which a legend anchored at the
    # very left edge covered.
    ax.legend(loc="upper left", bbox_to_anchor=(0.058, 0.99), frameon=True,
              framealpha=0.92, edgecolor="0.6", borderpad=0.35)
    ax.text(0.004, 0.97, "(b)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def covers_data(bb, ax, renderer=None):
    """Count data elements under a display-space bbox (must be zero).

    Scatter points are tested vertex-by-vertex; bars are tested by their
    rendered extent. An earlier version pushed bar vertices through
    get_patch_transform() alone, which maps unit space to DATA space, not to
    display space -- so bar overlaps were silently reported as zero.
    """
    hits = 0
    for col in ax.collections:
        off = np.asarray(col.get_offsets())
        if len(off):
            p = ax.transData.transform(off)
            hits += int(((p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) &
                         (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)).sum())
    r = renderer or ax.figure.canvas.get_renderer()
    for pa in ax.patches:
        if pa.get_window_extent(r).overlaps(bb):
            hits += 1
    return hits


def add_credit(fig, ax_bottom, text, fontsize=8, pad_in=0.14):
    """Wrap the credit to the measured figure width and hang it just below."""
    import textwrap
    r = fig.canvas.get_renderer()
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    t = None
    for ncol in range(160, 60, -2):
        if t is not None:
            t.remove()
        t = fig.text(0.0, 0.0, textwrap.fill(text, ncol), fontsize=fontsize,
                     color="#555555", va="top", ha="left", linespacing=1.25)
        fig.canvas.draw()
        if t.get_window_extent(r).width <= fig_w_px:
            break
    fig.canvas.draw()
    # Use the axes' TIGHT bbox, which includes the tick labels and the x-axis
    # label. Measuring only ax.texts and ax.patches ignored both, so the credit
    # was hung above them and overlapped "Month of acquisition".
    low = ax_bottom.get_tightbbox(r).y0
    y = low / (fig.get_size_inches()[1] * fig.dpi)
    t.set_position((0.0, y - pad_in / fig.get_size_inches()[1]))
    return t


def build(csv_path, threshold, cs_thresh, prefix):
    df = load(csv_path, threshold)

    n = len(df)
    n_use = int(df["usable"].sum())
    print(f"[info] {n} scenes in inventory; {n_use} usable at >= {threshold}%")
    for name, _, _, _ in EPOCHS:
        sub = df[df["epoch"] == name]
        print(f"[info]   {name}: {len(sub)} scenes, "
              f"{int(sub['usable'].sum())} usable")
    empty = [MONTHS[m - 1] for m in range(1, 13)
             if not ((df["month"] == m) & df["usable"]).any()]
    if empty:
        print(f"[warn] months with no usable scene at all: {empty}")

    fig = plt.figure(figsize=(6.89, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    panel_a(ax_a, df)
    panel_b(ax_b, df)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for tag, ax in (("a", ax_a), ("b", ax_b)):
        leg = ax.get_legend()
        lb = leg.get_window_extent(r)
        print(f"[check] panel ({tag}) legend covers "
              f"{covers_data(lb, ax, r)} data elements (must be 0)")
        # Legend-vs-label: ax.texts holds the panel letter and any annotation,
        # and this check is what caught the legend sitting on "(b)".
        clash = [t.get_text().replace("\n", " ") for t in ax.texts
                 if t.get_window_extent(renderer=r).overlaps(lb)]
        print(f"[check] panel ({tag}) legend overlaps {len(clash)} labels "
              "(must be 0)" + (f": {clash}" if clash else ""))
    boxes = [(t.get_text()[:18], t.get_window_extent(renderer=r))
             for ax in (ax_a, ax_b) for t in ax.texts if t.get_text().strip()]
    pairs = [(boxes[i][0], boxes[j][0])
             for i in range(len(boxes)) for j in range(i + 1, len(boxes))
             if boxes[i][1].overlaps(boxes[j][1])]
    print(f"[check] overlapping text pairs: {len(pairs)}"
          + (f" {pairs}" if pairs else ""))

    add_credit(fig, ax_b, CREDIT.format(
        cs=cs_thresh, thr=threshold,
        py=".".join(map(str, sys.version_info[:3])),
        mpl=matplotlib.__version__, pd=pd.__version__))

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf and {prefix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="scene_inventory.csv")
    ap.add_argument("--usable-threshold", type=float, default=70.0,
                    help="per cent of the footprint that must pass the mask")
    ap.add_argument("--cs-threshold", type=float, default=0.60,
                    help="Cloud Score+ cs_cdf threshold used in the export")
    ap.add_argument("--prefix", default="fig04_image_inventory")
    a = ap.parse_args()
    build(a.csv, a.usable_threshold, a.cs_threshold, a.prefix)
