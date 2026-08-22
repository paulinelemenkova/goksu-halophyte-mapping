#!/usr/bin/env python3
"""
fig13_timeseries.py -- Figure 13 of the Goksu Delta halophyte manuscript.

Annual mapped area of each halophyte habitat class across the record, against
the surface-water and climatic-water-balance series that plausibly drive it.

(a) Class area by year, with the two analysis epochs shaded. A band around
    each line gives the year-to-year variability that the classifier alone
    produces; it is NOT a confidence interval on the ecological area, and is
    drawn so that a trend can be judged against the noise it sits in.
(b) Surface-water extent and the climatic water balance for the same years,
    which are the candidate drivers of any trend in (a).
(c) Usable scenes per year, because a change in the number of observations
    changes how well the harmonic model is constrained and can therefore move
    the class areas on its own.

Panel (c) is not decoration. An annual classification is only as stable as its
input, and a series that tracks scene count rather than surface condition would
be an artefact. Plotting the three together lets that be checked rather than
assumed.

INPUT
-----
    annual_series.csv from make_annual_series.js, one row per year with
    columns year, n_scenes, hist, water_px, precip_mm, pet_mm. The hist column
    is the Earth Engine frequency histogram, a JSON dictionary of class code to
    pixel count.

USAGE
-----
    python3 fig13_timeseries.py --csv annual_series.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import sys

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

CLASSES = [
    ("0", "Swamp", "#B8860B"),
    ("1", "Temporary floodplain", "#E34A00"),
    ("2", "Temporary ponds", "#118C8C"),
    ("3", "Terrestrial saline flats", "#B5179E"),
]
EPOCHS = [("Baseline epoch", 2003, 2007, "#c8b89a"),
          ("Present epoch", 2021, 2026, "#8fbcd4")]

PIXEL_HA = 30.0 * 30.0 / 1e4      # nominal 30 m cell

CREDIT = (
    "Annual classifications computed in Earth Engine with a random forest trained on "
    "the same {p} predictors and the same {n} reference rows as the local model, and "
    "applied to a harmonic fit of each year in turn; only per-class pixel counts are "
    "returned, so no annual raster is transferred. Areas are pixel counts scaled by the "
    "nominal cell area. The shaded band in (a) spans one standard deviation of the "
    "first differences of each series, which is the year-to-year movement the "
    "classifier produces on its own; it is not a confidence interval on habitat extent. "
    "Surface water is JRC Global Surface Water yearly history and the water balance is "
    "ERA5-Land precipitation less potential evaporation. Because the cloud and local "
    "forests are different implementations, the two epochs common to both are the check "
    "on this series. Software used for plotting figure: Python {py} with Matplotlib "
    "{mpl} and pandas {pd}. Source: authors"
)


def parse_hist(cell):
    """Earth Engine returns the histogram as a JSON dictionary in one cell."""
    if isinstance(cell, dict):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        return json.loads(cell)
    except json.JSONDecodeError:
        return ast.literal_eval(cell)


def load(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(
            f"ERROR: {path} not found.\n"
            "Figure 13 needs one classification per year across the record.\n"
            "Run make_annual_series.js in Earth Engine, after pasting in the\n"
            "training FeatureCollection written by make_training_fc.py, and\n"
            "download annual_series.csv. This script will not interpolate a\n"
            "yearly series from the two epochs."
        )
    for key, name, _ in CLASSES:
        df[name] = [parse_hist(h).get(key, 0) * PIXEL_HA for h in df["hist"]]
    df["water_ha"] = df["water_px"] * PIXEL_HA
    df["balance_mm"] = df["precip_mm"] - df["pet_mm"]
    return df.sort_values("year")


def shade_epochs(ax):
    for name, lo, hi, colour in EPOCHS:
        ax.axvspan(lo, hi, color=colour, alpha=0.30, lw=0, zorder=0)


def style(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.90", zorder=1)
    ax.set_axisbelow(True)


def panel_areas(ax, df):
    shade_epochs(ax)
    for _, name, colour in CLASSES:
        y = df[name].to_numpy(float)
        sd = np.nanstd(np.diff(y)) if len(y) > 2 else 0.0
        ax.fill_between(df["year"], y - sd, y + sd, color=colour, alpha=0.14,
                        lw=0, zorder=2)
        ax.plot(df["year"], y, color=colour, lw=1.5, zorder=3, label=name)
        print(f"[info] {name:26s} {y[0]:8.0f} -> {y[-1]:8.0f} ha, "
              f"year-to-year sd {sd:6.0f} ha")
    ax.set_ylabel("Mapped area (ha)")
    style(ax)


def panel_drivers(ax, df):
    shade_epochs(ax)
    ax.plot(df["year"], df["water_ha"], color="#2c6fa8", lw=1.5,
            label="surface water")
    ax.set_ylabel("Surface water (ha)")
    style(ax)
    ax2 = ax.twinx()
    ax2.plot(df["year"], df["balance_mm"], color="#7a5c3a", lw=1.2, ls="--",
             label="water balance")
    ax2.set_ylabel("P - PET (mm)", fontsize=9)
    ax2.tick_params(labelsize=8.5)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    return ax2


def panel_scenes(ax, df):
    shade_epochs(ax)
    ax.bar(df["year"], df["n_scenes"], width=0.75, color="#9aa7ad",
           edgecolor="0.35", linewidth=0.4, zorder=3)
    ax.set_ylabel("Usable scenes")
    ax.set_xlabel("Year")
    style(ax)


def build(csv_path, prefix):
    df = load(csv_path)
    print(f"[info] {len(df)} years, {df['year'].min():.0f}-{df['year'].max():.0f}")

    fig = plt.figure(figsize=(6.89, 6.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.03, hspace=0.05)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.0, 0.7])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))

    panel_areas(ax_a, df)
    ax_b2 = panel_drivers(ax_b, df)
    panel_scenes(ax_c, df)
    for ax in (ax_a, ax_b):
        ax.set_xticklabels([])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlim(df["year"].min() - 0.5, df["year"].max() + 0.5)

    # Does the class series simply track the number of observations?
    for _, name, _c in CLASSES:
        r = np.corrcoef(df[name], df["n_scenes"])[0, 1]
        flag = "  <-- tracks scene count" if abs(r) > 0.6 else ""
        print(f"[check] corr({name}, n_scenes) = {r:+.2f}{flag}")

    fig.canvas.draw()
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()
    fig_h = fig.get_size_inches()[1] * fig.dpi
    top = max(a.get_position().y1 for a in (ax_a, ax_b, ax_c))
    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.052, pos.y1 + 0.004, tag, fontsize=11,
                 fontweight="bold", va="bottom", ha="left")

    handles, labels = ax_a.get_legend_handles_labels()
    h2, l2 = ax_b.get_legend_handles_labels()
    h3, l3 = ax_b2.get_legend_handles_labels()
    low = min(a.get_tightbbox(r).y0 for a in (ax_a, ax_b, ax_c)) / fig_h
    leg = fig.legend(handles + h2 + h3, labels + l2 + l3, loc="upper center",
                     bbox_to_anchor=(0.5, low - 0.012), ncol=3, frameon=False,
                     handlelength=1.4, columnspacing=1.5, borderaxespad=0.0)

    import textwrap
    fig_w = fig.get_size_inches()[0] * fig.dpi
    n_pred = 40
    text = CREDIT.format(p=n_pred, n="the plot", py=".".join(
        map(str, sys.version_info[:3])), mpl=matplotlib.__version__,
        pd=pd.__version__)
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
    t.set_position((0.0, leg.get_window_extent(r).y0 / fig_h - 0.014))

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf and {prefix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="annual_series.csv")
    ap.add_argument("--prefix", default="fig13_timeseries")
    a = ap.parse_args()
    build(a.csv, a.prefix)
