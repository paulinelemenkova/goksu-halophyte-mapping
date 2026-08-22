#!/usr/bin/env python3
"""
fig11_transects.py -- Figure 11 of the Goksu Delta halophyte manuscript.

Cross-shore transects showing the mapped halophyte habitat sequence together
with the salinity, moisture and elevation profiles that accompany it.

Each panel is one meridional transect through the delta. The coloured strip
along the top gives the mapped class at every pixel the transect crosses; below
it the canopy salinity response, the canopy moisture and the surface elevation
are plotted against the same distance axis. The purpose is to test whether the
mapped sequence follows the compound gradient of Section 2.1 in the direction
the plot record predicts, rather than merely reproducing the class proportions.

Two conventions matter for reading the figure. Pixels outside the classified
domain -- open water, ground above the delta plain, series too short to fit --
leave the class strip blank rather than being interpolated across. And pixels
whose class-probability entropy exceeds the threshold are drawn hatched, so a
sequence that looks orderly can be checked against the confidence behind it.

INPUT
-----
    fig09_map_baseline_class.tif     classified raster
    fig09_map_baseline_entropy.tif   per-pixel entropy, scaled by 1000
    feature_stack_baseline.tif       40-band predictor stack
    goksu_dem.tif                    elevation

USAGE
-----
    python3 fig11_transects.py
    python3 fig11_transects.py --prefix fig11_transects_present \
        --class-tif fig10_map_present_class.tif ...
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
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

# Class order is the classifier's own (alphabetical); colours match Figure 9.
CLASS_ORDER = ["Swamp", "Temporary floodplain", "Temporary ponds",
               "Terrestrial saline flats"]
CLASS_COLOUR = {"Swamp": "#FFFF00", "Temporary floodplain": "#FF4500",
                "Temporary ponds": "#00FFFF",
                "Terrestrial saline flats": "#FF00FF"}
SHORT = {"Swamp": "Swamp", "Temporary floodplain": "Floodplain",
         "Temporary ponds": "Ponds", "Terrestrial saline flats": "Saline flats"}

# Transects: meridional lines through the western, central and eastern delta.
# Longitudes were chosen for coverage of the classified domain, not by eye.
TRANSECTS = [
    (33.940, "Western sector"),
    (33.980, "Central sector"),
    (34.060, "Eastern sector"),
]

# Band positions in the feature stack, from the column order of features.csv.
BANDS = {"ndmi_med": 19, "crsi_med": 29}
GAIN = 10000.0

CREDIT = (
    "Transects are meridional lines at {lons} E through the western, central and "
    "eastern delta, sampled at the 30 m grid of the classified raster and plotted "
    "against distance north from the southern edge of the analysis window. The class "
    "strip shows the mapped habitat at each pixel; blanks are pixels outside the "
    "classified domain and hatching marks pixels whose normalised class-probability "
    "entropy exceeds {emax}. Canopy salinity response and canopy moisture are the "
    "median CRSI and NDMI of the epoch series at each pixel, taken from the predictor "
    "stack; elevation is Copernicus WorldDEM-30. Of the {n} classified pixels crossed "
    "by the three transects, {lowpct:.0f} per cent fall above the entropy threshold. "
    "Software used for plotting figure: Python {py} with Matplotlib {mpl} and rasterio "
    "{rio}. Source: authors"
)


def sample_at(path, lon, lat, band=1):
    """Sample a raster at given coordinates.

    The DEM sits on a 1 arcsec grid while the classified raster and the
    predictor stack sit on the 30 m export grid, so they have different row
    counts. Sampling by coordinate rather than by row index keeps every
    profile on one distance axis.
    """
    with rasterio.open(path) as src:
        pts = [(lon, la) for la in lat]
        return np.array([v[0] for v in src.sample(pts, indexes=band)],
                        dtype="float64")


def sample_column(path, lon, band=1):
    """All rows of the raster at the column nearest a given longitude."""
    with rasterio.open(path) as src:
        col = int(round((lon - src.bounds.left)
                        / (src.bounds.right - src.bounds.left) * src.width))
        col = min(max(col, 0), src.width - 1)
        arr = src.read(band)
        lat_top, lat_bot = src.bounds.top, src.bounds.bottom
    v = arr[:, col].astype("float64")
    lat = np.linspace(lat_top, lat_bot, len(v))
    return lat, v


def km_north(lat):
    """Distance north from the southern end of the profile, kilometres."""
    return (lat - lat.min()) * 111.32


def panel(ax_strip, ax_prof, lon, title, paths, entropy_max, names):
    lat, cls = sample_column(paths["cls"], lon)
    _, ent = sample_column(paths["ent"], lon)
    dem = sample_at(paths["dem"], lon, lat)
    _, ndmi = sample_column(paths["stack"], lon, BANDS["ndmi_med"])
    _, crsi = sample_column(paths["stack"], lon, BANDS["crsi_med"])
    ndmi, crsi = ndmi / GAIN, crsi / GAIN
    x = km_north(lat)
    ent = np.where(ent < 0, np.nan, ent / 1000.0)

    # ---- class strip -----------------------------------------------------
    for i, name in enumerate(names):
        sel = cls == i
        if not sel.any():
            continue
        ax_strip.fill_between(x, 0, 1, where=sel, step="mid",
                              color=CLASS_COLOUR[name], lw=0)
    low = np.isfinite(ent) & (ent > entropy_max) & (cls >= 0)
    if low.any():
        ax_strip.fill_between(x, 0, 1, where=low, step="mid", facecolor="none",
                              hatch="////", edgecolor="0.25", lw=0.0)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_yticks([])
    ax_strip.set_xlim(x.min(), x.max())
    ax_strip.set_xticklabels([])
    ax_strip.tick_params(which="both", length=0)
    ax_strip.set_ylabel("class", rotation=0, ha="right", va="center",
                        fontsize=8)
    ax_strip.set_title(f"{title}  ({lon:.3f} E)", fontsize=9.5, loc="left",
                       pad=3)

    # ---- profiles --------------------------------------------------------
    ax_prof.plot(x, crsi, color="#C1663A", lw=1.3, label="CRSI (salinity)")
    ax_prof.plot(x, ndmi, color="#3B7EA1", lw=1.3, label="NDMI (moisture)")
    ax_prof.set_ylabel("Index value")
    ax_prof.set_xlim(x.min(), x.max())
    ax_prof.xaxis.set_minor_locator(AutoMinorLocator())
    ax_prof.yaxis.set_minor_locator(AutoMinorLocator())
    ax_prof.tick_params(which="both", top=True, direction="in")
    ax_prof.tick_params(which="major", length=4.5)
    ax_prof.tick_params(which="minor", length=2.5)
    ax_prof.grid(which="major", lw=0.5, color="0.90", zorder=0)
    ax_prof.set_axisbelow(True)

    ax_dem = ax_prof.twinx()
    ax_dem.plot(x, dem, color="#4a3f30", lw=1.0, ls="--", label="elevation")
    ax_dem.set_ylabel("Elevation (m)", fontsize=9)
    ax_dem.tick_params(labelsize=8.5)
    ax_dem.yaxis.set_minor_locator(AutoMinorLocator())
    ax_dem.set_ylim(-1, max(12, np.nanmax(dem[np.isfinite(dem)]) * 0.4))

    n_cls = int((cls >= 0).sum())
    n_low = int(low.sum())
    return n_cls, n_low, ax_dem


def build(paths, entropy_max, prefix):
    import os
    for k, v in paths.items():
        if not os.path.exists(v):
            sys.exit(f"ERROR: {v} not found ({k}).")

    names = CLASS_ORDER
    fig = plt.figure(figsize=(6.89, 7.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.03)
    gs = fig.add_gridspec(6, 1, height_ratios=[0.22, 1, 0.22, 1, 0.22, 1])

    tot_cls = tot_low = 0
    axes_prof = []
    axes_strip = []
    for k, (lon, title) in enumerate(TRANSECTS):
        ax_s = fig.add_subplot(gs[2 * k])
        ax_p = fig.add_subplot(gs[2 * k + 1])
        axes_strip.append(ax_s)
        n_cls, n_low, ax_dem = panel(ax_s, ax_p, lon, title, paths,
                                     entropy_max, names)
        tot_cls += n_cls
        tot_low += n_low
        axes_prof.append((ax_p, ax_dem))
        print(f"[info] {title:15s} {lon:.3f} E: {n_cls:4d} classified pixels, "
              f"{n_low:4d} above the entropy threshold "
              f"({100 * n_low / max(n_cls, 1):.0f} %)")
    axes_prof[-1][0].set_xlabel("Distance north along transect (km)")

    # Shared legend: classes, then the three profile lines.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()

    # Panel letters above the upper-left corner of each class strip, outside
    # the axes, as in Figure 12. Placed in figure coordinates so all three sit
    # at the same offset from their panel rather than at three axes-relative
    # positions that drift with the strip heights.
    for (ax_s, tag) in zip(axes_strip, "abc"):
        pos = ax_s.get_position()
        fig.text(pos.x0 - 0.055, pos.y1 + 0.008, f"({tag})", fontsize=11,
                 fontweight="bold", va="bottom", ha="left")
    fig_h = fig.get_size_inches()[1] * fig.dpi
    low_edge = min(a.get_tightbbox(r).y0 for a, _ in axes_prof) / fig_h

    handles = [Patch(facecolor=CLASS_COLOUR[n], edgecolor="0.3",
                     label=SHORT[n]) for n in names]
    handles.append(Patch(facecolor="white", edgecolor="0.25", hatch="////",
                         label=f"entropy > {entropy_max}"))
    lines = [plt.Line2D([], [], color="#C1663A", lw=1.3, label="CRSI"),
             plt.Line2D([], [], color="#3B7EA1", lw=1.3, label="NDMI"),
             plt.Line2D([], [], color="#4a3f30", lw=1.0, ls="--",
                        label="elevation")]
    leg = fig.legend(handles=handles + lines, loc="upper center",
                     bbox_to_anchor=(0.5, low_edge - 0.010),
                     ncol=4, frameon=False, handlelength=1.4,
                     columnspacing=1.5, borderaxespad=0.0)

    fig.canvas.draw()
    lb = leg.get_window_extent(r)
    hits = 0
    for a, _ in axes_prof:
        for ln in a.get_lines():
            d = np.column_stack(ln.get_data())
            d = d[np.isfinite(d).all(axis=1)]
            if d.shape[0] > 2:
                p = a.transData.transform(d)
                hits += int(((p[:, 0] >= lb.x0) & (p[:, 0] <= lb.x1)
                             & (p[:, 1] >= lb.y0) & (p[:, 1] <= lb.y1)).sum())
    print(f"[check] legend covers {hits} data vertices (must be 0)")

    import textwrap
    fig_w = fig.get_size_inches()[0] * fig.dpi
    text = CREDIT.format(
        lons=", ".join(f"{l:.3f}" for l, _ in TRANSECTS), emax=entropy_max,
        n=tot_cls, lowpct=100 * tot_low / max(tot_cls, 1),
        py=".".join(map(str, sys.version_info[:3])),
        mpl=matplotlib.__version__, rio=rasterio.__version__)
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
    fig.canvas.draw()
    gap = leg.get_window_extent(r).y0 - t.get_window_extent(r).y1
    print(f"[check] gap between legend and credit: {gap:.0f} px "
          "(must be positive)")

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(f"{prefix}.tif", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf, {prefix}.png and {prefix}.tif")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class-tif", default="fig09_map_baseline_class.tif")
    ap.add_argument("--entropy-tif", default="fig09_map_baseline_entropy.tif")
    ap.add_argument("--stack", default="feature_stack_baseline.tif")
    ap.add_argument("--dem", default="goksu_dem.tif")
    ap.add_argument("--entropy-max", type=float, default=0.85)
    ap.add_argument("--prefix", default="fig11_transects")
    a = ap.parse_args()
    build({"cls": a.class_tif, "ent": a.entropy_tif, "stack": a.stack,
           "dem": a.dem}, a.entropy_max, a.prefix)
