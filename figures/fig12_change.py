#!/usr/bin/env python3
"""
fig12_change.py -- Figure 12 of the Goksu Delta halophyte manuscript.

Spatial pattern of halophyte habitat change between the two epochs.

(a) Change map. Every jointly classified pixel falls into one of three states:
    not assessable, persistent, or converted. A pixel is assessable only where
    the normalised class-probability entropy of Eq. (9) is below threshold in
    BOTH epochs, which is the filter specified in Section 2.11. Converted
    pixels are coloured by their present-epoch class, using the palette of
    Figures 9 and 10, so the map reads as "what it became".

(b) Transition matrix over the assessable subset, in hectares, with the
    diagonal giving persistence. The matrix is computed only on pixels that
    pass the filter; reporting it over all classified pixels would present
    classifier disagreement as ecological change.

The share of the delta that is not assessable is reported on the figure rather
than hidden, because it is the dominant fact about this comparison: change can
be stated only for the fraction of the mapped area where both classifications
are confident.

INPUT
-----
    fig09_map_baseline_class.tif    fig09_map_baseline_entropy.tif
    fig10_map_present_class.tif     fig10_map_present_entropy.tif

USAGE
-----
    python3 fig12_change.py --entropy-max 0.85
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
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

# Classifier class order (alphabetical), with the palette of Figures 9 and 10.
NAMES = ["Swamp", "Temporary floodplain", "Temporary ponds",
         "Terrestrial saline flats"]
SHORT = {"Swamp": "Swamp", "Temporary floodplain": "Floodplain",
         "Temporary ponds": "Ponds", "Terrestrial saline flats": "Saline flats"}
COLOUR = {"Swamp": "#FFFF00", "Temporary floodplain": "#FF4500",
          "Temporary ponds": "#00FFFF", "Terrestrial saline flats": "#FF00FF"}

NOT_ASSESSABLE = "#e9e9e9"
PERSISTENT = "#9aa7ad"
BACKGROUND = "#ffffff"

CREDIT = (
    "Change is evaluated only where the normalised class-probability entropy of "
    "Eq. (9) is below {emax} in both epochs, the filter of Sect. 2.11. Of the {njoint} "
    "hectares classified in both epochs, {nconf} hectares ({pct:.0f} per cent) meet that "
    "condition; the remainder is drawn as not assessable and carries no change "
    "statement. On the assessable subset {persist:.0f} per cent of pixels retain their "
    "class. Converted pixels in (a) are coloured by the class they became. The matrix in "
    "(b) is in hectares, rows giving the baseline class and columns the present class, "
    "with the diagonal shaded as persistence. Areas are pixel counts scaled by the grid "
    "cell area and are not the stratified estimator of Eq. (12), which requires the "
    "validation sample of Sect. 2.10. Software used for plotting figure: Python {py} "
    "with Matplotlib {mpl} and rasterio {rio}. Source: authors"
)


def cell_ha(src):
    """Grid cell area in hectares for a geographic raster at this latitude."""
    lat = (src.bounds.top + src.bounds.bottom) / 2.0
    dy = abs(src.transform.e) * 111320.0
    dx = abs(src.transform.a) * 111320.0 * np.cos(np.radians(lat))
    return dx * dy / 1e4


def load(paths, entropy_max):
    for k, v in paths.items():
        if not os.path.exists(v):
            sys.exit(f"ERROR: {v} not found ({k}).")
    with rasterio.open(paths["cls_b"]) as s:
        b, ha, bounds = s.read(1), cell_ha(s), s.bounds
    with rasterio.open(paths["cls_p"]) as s:
        p = s.read(1)
    with rasterio.open(paths["ent_b"]) as s:
        eb = s.read(1).astype("float32") / 1000.0
    with rasterio.open(paths["ent_p"]) as s:
        ep = s.read(1).astype("float32") / 1000.0
    if not (b.shape == p.shape == eb.shape == ep.shape):
        sys.exit("ERROR: the four rasters are not on a common grid.")
    joint = (b >= 0) & (p >= 0)
    conf = joint & (eb <= entropy_max) & (ep <= entropy_max)
    return b, p, joint, conf, ha, bounds


def panel_map(ax, b, p, joint, conf, bounds):
    """State raster: 0 background, 1 not assessable, 2 persistent, 3-6 became X."""
    state = np.zeros(b.shape, dtype="uint8")
    state[joint & ~conf] = 1
    state[conf & (b == p)] = 2
    for j, _ in enumerate(NAMES):
        state[conf & (b != p) & (p == j)] = 3 + j

    cmap = ListedColormap([BACKGROUND, NOT_ASSESSABLE, PERSISTENT]
                          + [COLOUR[n] for n in NAMES])
    ax.imshow(state, cmap=cmap, vmin=0, vmax=6, interpolation="nearest",
              extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    return state, (bounds.right - bounds.left) / (bounds.top - bounds.bottom)


def panel_matrix(ax, b, p, conf, ha):
    n = len(NAMES)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = ((b == i) & (p == j) & conf).sum() * ha

    off = M.copy()
    np.fill_diagonal(off, np.nan)
    im = ax.imshow(off, cmap="YlOrRd", origin="upper",
                   vmin=0, vmax=np.nanmax(off))
    # Diagonal shaded as persistence rather than sharing the transition scale.
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   facecolor=PERSISTENT, edgecolor="white",
                                   lw=1.0, zorder=2))
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            dark = (i == j) or (np.isfinite(off[i, j])
                                and off[i, j] > 0.55 * np.nanmax(off))
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                    zorder=3, color="white" if dark else "#1a1a1a")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([SHORT[c] for c in NAMES], rotation=30, ha="right")
    ax.set_yticklabels([SHORT[c] for c in NAMES])
    ax.set_xlabel("Present epoch")
    ax.set_ylabel("Baseline epoch")
    ax.tick_params(which="both", length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Converted area (ha)", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    return M, cb


def build(paths, entropy_max, prefix):
    b, p, joint, conf, ha, bounds = load(paths, entropy_max)
    n_joint, n_conf = joint.sum() * ha, conf.sum() * ha
    persist = 100.0 * (b[conf] == p[conf]).mean()
    print(f"[info] jointly classified {n_joint:.0f} ha; assessable "
          f"{n_conf:.0f} ha ({100 * conf.sum() / joint.sum():.1f} %)")
    print(f"[info] persistence on the assessable subset: {persist:.1f} %")

    fig = plt.figure(figsize=(6.89, 4.3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    _, geo_aspect = panel_map(ax_a, b, p, joint, conf, bounds)
    M, cb = panel_matrix(ax_b, b, p, conf, ha)

    off = [(M[i, j], NAMES[i], NAMES[j])
           for i in range(len(NAMES)) for j in range(len(NAMES)) if i != j]
    print("[info] principal transitions on the assessable subset:")
    for v, a_, c_ in sorted(off, reverse=True)[:4]:
        print(f"[info]   {SHORT[a_]:12s} -> {SHORT[c_]:12s} {v:7.0f} ha")

    fig.canvas.draw()
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()
    fig_w_in, fig_h_in = fig.get_size_inches()
    fig_h = fig_h_in * fig.dpi

    # Both panels to a common height, tops aligned. With aspect "equal" the
    # map box is set by the geographic aspect and the matrix box by its 4 x 4
    # shape, so the two ended up different heights; the aspect is released and
    # the boxes are sized explicitly instead, which preserves both shapes while
    # matching their heights.
    pa, pb = ax_a.get_position(), ax_b.get_position()
    H = min(pa.height, pb.height)
    top = max(pa.y1, pb.y1)
    k = fig_h_in / fig_w_in            # figure-fraction aspect correction
    ax_a.set_aspect("auto")
    ax_b.set_aspect("auto")
    ax_a.set_position([pa.x0, top - H, H * k * geo_aspect, H])
    wb = H * k                          # square box for the 4 x 4 matrix
    ax_b.set_position([pb.x0, top - H, wb, H])
    cb.ax.set_position([pb.x0 + wb + 0.012, top - H, 0.018, H])

    fig.canvas.draw()
    # Panel letters at one level, just above the shared top of both boxes.
    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
        fig.text(ax.get_position().x0, top + 0.018, tag, fontsize=11,
                 fontweight="bold", va="bottom", ha="left")

    fig.canvas.draw()
    low = min(ax_a.get_tightbbox(r).y0, ax_b.get_tightbbox(r).y0) / fig_h

    handles = [Patch(facecolor=NOT_ASSESSABLE, edgecolor="0.5",
                     label="not assessable"),
               Patch(facecolor=PERSISTENT, edgecolor="0.5",
                     label="persistent")]
    handles += [Patch(facecolor=COLOUR[n], edgecolor="0.4",
                      label=f"became {SHORT[n].lower()}") for n in NAMES]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, low - 0.012), ncol=3, frameon=False,
                     handlelength=1.3, columnspacing=1.4, borderaxespad=0.0)

    import textwrap
    fig_w = fig.get_size_inches()[0] * fig.dpi
    text = CREDIT.format(emax=entropy_max, njoint=f"{n_joint:.0f}",
                         nconf=f"{n_conf:.0f}",
                         pct=100 * conf.sum() / joint.sum(), persist=persist,
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
    t.set_position((0.0, leg.get_window_extent(r).y0 / fig_h - 0.016))
    fig.canvas.draw()
    gap = leg.get_window_extent(r).y0 - t.get_window_extent(r).y1
    print(f"[check] gap between legend and credit: {gap:.0f} px "
          "(must be positive)")

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf and {prefix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls-baseline", default="fig09_map_baseline_class.tif")
    ap.add_argument("--cls-present", default="fig10_map_present_class.tif")
    ap.add_argument("--ent-baseline", default="fig09_map_baseline_entropy.tif")
    ap.add_argument("--ent-present", default="fig10_map_present_entropy.tif")
    ap.add_argument("--entropy-max", type=float, default=0.85)
    ap.add_argument("--prefix", default="fig12_change")
    a = ap.parse_args()
    build({"cls_b": a.cls_baseline, "cls_p": a.cls_present,
           "ent_b": a.ent_baseline, "ent_p": a.ent_present},
          a.entropy_max, a.prefix)
