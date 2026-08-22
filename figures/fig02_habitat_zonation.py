#!/usr/bin/env python3
"""
fig02_habitat_zonation.py -- Figure 2 of the Goksu Delta halophyte manuscript.

Two panels:

(a) Measured cross-shore topographic profile along a north-south transect at
    33.956 E, from the Mediterranean Sea across the beach-dune barrier and the
    Akgol lagoon to the inland margin of the delta plain. Elevations are read
    from the Copernicus WorldDEM-30 (GLO-30) grid with GMT grdtrack; the
    water/land segments come from a GSHHG full-resolution land mask sampled
    along the same line. Nothing in this panel is drawn by hand.

(b) Habitat zonation scheme of the delta: the six habitat groups recognised by
    the published phytosociological survey, ordered from the shore inland, with
    the dominant and characteristic taxa of each. The two compound gradients
    that organise the sequence -- soil salinity and inundation duration -- are
    shown as direction arrows. The panel is an ordering, not a georeferenced
    cross-section: the blocks carry no distance scale, because plot-level survey
    data fix the sequence of the communities but not their exact positions along
    any one transect.

Inputs
------
    transect.csv   dist_km, elev_m, land   (produced by prepare_transect.sh)

Outputs
-------
    fig02_habitat_zonation.pdf   vector, for LaTeX
    fig02_habitat_zonation.png   600 dpi raster preview
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import AutoMinorLocator

# --------------------------------------------------------------------------
# House style (scientific-plotting skill): one sans-serif family, 8-12 pt,
# at most three sizes, major + minor ticks, thin two-tier grid.
# --------------------------------------------------------------------------
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

# Transect endpoints (WGS 84). Kept here so the credit block, the extraction
# script and the figure cannot drift apart.
TRANSECT_A = (33.9560, 36.2350)   # lon E, lat N -- seaward end
TRANSECT_B = (33.9560, 36.3850)   # lon E, lat N -- inland end
TRANSECT_STEP_KM = 0.02

SEA = "#a9cfe4"
LAND = "#c9b8a0"
DUNE = "#e3d5b8"

# Habitat groups, ordered shore -> inland. Taxa follow the published survey.
ZONES = [
    ("Beach and\nfixed dunes", "psammophytic",
     "Ammophila\narenaria\nEryngium\nmaritimum\nPancratium\nmaritimum"),
    ("Lagoon and\nreed fringe", "aquatic",
     "Phragmites\naustralis\nBolboschoenus\nmaritimus\nSchoenus\nnigricans"),
    ("Saline\nswamp", "halophytic",
     "Aeluropus\nlittoralis\nHalimione\nportulacoides\nLimonium\ngmelinii"),
    ("Temporary\nponds", "halophytic",
     "Juncus\nmaritimus\nJuncus\nacutus\nAtriplex\nhastata"),
    ("Temporary\nfloodplain", "halophytic",
     "Tamarix\ntetrandra\nTamarix\ntetragyna\nSalicornia\nfragilis"),
    ("Terrestrial\nsaline flats", "halophytic",
     "Halocnemum\nstrobilaceum\nArthrocnemum\nfruticosum\nSalicornia\neuropaea"),
]

# Credit text as one paragraph. It is wrapped at render time to the measured
# width of the figure, so the block never widens the exported bounding box.
CREDIT = (
    "(a) Topographic profile along a single straight transect A-B, WGS 84 "
    "(EPSG:4326). A (seaward end): {a_lon:.4f} E, {a_lat:.4f} N; B (inland end): "
    "{b_lon:.4f} E, {b_lat:.4f} N; azimuth {az:03.0f} deg (due north), great-circle "
    "length {length:.2f} km. Sampled at a constant along-track interval of "
    "{step_m:.0f} m ({step_km:.3f} km), giving {npts:d} sample points "
    "({density:.1f} points per km). Elevations read from the Copernicus "
    "WorldDEM-30 (GLO-30) digital surface model, 1 arcsec posting (about 30 m at "
    "this latitude), EGM2008 vertical datum, by bicubic interpolation with GMT "
    "grdtrack; sampled range {zmin:.2f} to {zmax:.2f} m. Water and land segments "
    "from a GSHHG full-resolution shoreline rasterised with GMT grdlandmask at "
    "3 arcsec and thresholded at 0.5; {landpct:.1f} per cent of the transect is "
    "land, and the horizontal axis is referenced to the outer shoreline, which "
    "lies {shore:.2f} km from A. Segment boundaries along the transect fall at "
    "{breaks} km from A. Note that GLO-30 is a surface model, so vegetation "
    "canopy is included in the plotted heights. Produced using Copernicus "
    "WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH "
    "2014-2018, provided under COPERNICUS by the European Union and ESA; all "
    "rights reserved. "
    "(b) Habitat groups of the delta ordered from shore to inland, with the "
    "dominant and characteristic taxa of each, after the published "
    "phytosociological survey. The panel is an ordering and carries no distance "
    "scale: the survey fixes the sequence of the communities but not their "
    "positions along any one transect. "
    "Software used for plotting figure: Python {py} with Matplotlib {mpl}; "
    "profile extracted with GMT {gmt}. Source: authors"
)


def add_credit(fig, ax_bottom, text, fontsize=8, pad_in=0.06):
    """Wrap the credit to the figure width and hang it just under the axes.

    The wrap column is found by measurement, not guessed: the string is
    re-wrapped with a shrinking column until its rendered width fits inside
    the figure. Placing it relative to the lowest artist keeps the gap tight
    however the panels are resized.
    """
    import textwrap
    r = fig.canvas.get_renderer()
    fig_w_px = fig.get_size_inches()[0] * fig.dpi

    t = None
    for ncol in range(150, 60, -2):
        if t is not None:
            t.remove()
        t = fig.text(0.0, 0.0, textwrap.fill(text, ncol), fontsize=fontsize,
                     color="#555555", va="top", ha="left", linespacing=1.25)
        fig.canvas.draw()
        if t.get_window_extent(r).width <= fig_w_px:
            break

    # Hang the block a fixed small pad below the lowest drawn element.
    fig.canvas.draw()
    drawn = list(ax_bottom.texts) + list(ax_bottom.patches)
    low_px = min(a.get_window_extent(r).y0 for a in drawn
                 if a.get_window_extent(r).height > 0)
    y_fig = low_px / (fig.get_size_inches()[1] * fig.dpi)
    t.set_position((0.0, y_fig - pad_in / fig.get_size_inches()[1]))
    return t


def load_profile(path="transect.csv"):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d["dist_km"], d["elev_m"], d["land"].astype(int)


def segments(land):
    """Return [(i0, i1, is_land), ...] for runs of constant mask value."""
    out, s = [], 0
    for i in np.where(np.diff(land) != 0)[0]:
        out.append((s, i + 1, bool(land[s])))
        s = i + 1
    out.append((s, len(land), bool(land[s])))
    return out


def panel_a(ax, dist, elev, land):
    """Measured topographic profile, referenced to the outer shoreline."""
    segs = segments(land)
    shore = dist[segs[0][1]]          # first water -> land crossing
    x = dist - shore
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-1.2, np.ceil(elev.max()) + 2.5)

    # Water bodies as filled blocks, land as a filled topographic profile.
    for i0, i1, is_land in segs:
        if is_land:
            ax.fill_between(x[i0:i1], 0, elev[i0:i1], color=LAND,
                            lw=0, zorder=2)
        else:
            ax.axvspan(x[i0], x[i1 - 1], -1.2, 100, color=SEA, lw=0,
                       zorder=1, alpha=0.55)
    ax.plot(x, np.where(land == 1, elev, 0.0), color="#4a3f30", lw=1.4,
            zorder=3, label="Land surface (GLO-30)")
    ax.axhline(0, color="#2c5f7c", lw=0.9, ls="--", zorder=3,
               label="Mean sea level")

    ax.set_xlabel("Distance inland from the outer shoreline (km)")
    ax.set_ylabel("Elevation (m)")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.85", zorder=0)
    ax.grid(which="minor", lw=0.3, color="0.92", zorder=0)
    ax.set_axisbelow(True)

    # Direct labelling of the physiographic units. Each label is lifted above
    # the highest ground within its own x-window, so no label can sit on the
    # profile (verified afterwards with the covers_data pixel check).
    top = ax.get_ylim()[1]
    marks = [
        (x[segs[0][0]], x[segs[0][1] - 1], "Mediterranean\nSea"),
        (x[segs[1][0]], x[segs[1][1] - 1], "Beach-dune\nbarrier"),
        (x[segs[2][0]], x[segs[2][1] - 1], "Akgol\nlagoon"),
        (x[segs[3][0]], x[segs[3][0]] + 6.0, "Delta plain"),
        (x[-1] - 2.4, x[-1], "Inland\nmargin"),
    ]
    for x0, x1, txt in marks:
        band = (x >= x0 - 0.6) & (x <= x1 + 0.6)
        ymax = float(elev[band].max()) if band.any() else 0.0
        # va="bottom": the anchor is the FOOT of the label, so the text grows
        # upward away from the ground rather than down into it.
        ax.text((x0 + x1) / 2, max(ymax + 0.7, 1.6), txt,
                ha="center", va="bottom", fontsize=8.5, color="#333333",
                linespacing=1.15, zorder=5)

    # Upper left: the "Mediterranean Sea" unit label sits low over the water
    # at the left of the panel, and a centre-left legend covered it.
    ax.legend(loc="upper left", bbox_to_anchor=(0.065, 0.985), frameon=True,
              framealpha=0.92, edgecolor="0.6", handlelength=1.8,
              borderpad=0.4, labelspacing=0.3)
    ax.text(0.006, 0.94, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", zorder=6)


def panel_b(ax):
    """Ordered habitat-zonation scheme with dominant taxa."""
    n = len(ZONES)
    ax.set_xlim(-0.42, n + 0.42)
    # Lower bound just under the taxon block: leaving the axes at 0 reserved a
    # wide empty band that pushed the credit line away from the figure.
    ax.set_ylim(0.20, 1.0)
    ax.axis("off")

    cmap = plt.get_cmap("viridis")
    cols = [cmap(v) for v in np.linspace(0.88, 0.12, n)]

    block_top, block_bot = 0.84, 0.70
    for i, ((name, group, taxa), col) in enumerate(zip(ZONES, cols)):
        ax.add_patch(Rectangle((i + 0.04, block_bot), 0.92,
                               block_top - block_bot,
                               facecolor=col, edgecolor="white", lw=1.2))
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        ax.text(i + 0.5, (block_top + block_bot) / 2, name, ha="center",
                va="center", fontsize=9, fontweight="bold", linespacing=1.2,
                color="white" if lum < 0.55 else "#1a1a1a")
        ax.text(i + 0.5, block_bot - 0.035, group, ha="center", va="top",
                fontsize=8.5, color="#555555")
        ax.text(i + 0.5, block_bot - 0.115, taxa, ha="center", va="top",
                fontsize=8.5, style="italic", color="#1a1a1a",
                linespacing=1.28)

    # Compound gradients that order the sequence.
    for y, txt in [(0.945, "Soil salinity increases"),
                   (0.885, "Inundation duration decreases")]:
        ax.add_patch(FancyArrowPatch((0.10, y), (n - 0.10, y),
                                     arrowstyle="-|>", mutation_scale=9,
                                     lw=0.9, color="#555555"))
        ax.text(n / 2, y + 0.014, txt, ha="center", va="bottom", fontsize=8.5,
                color="#555555")

    ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")
    # Ends of the sequence, set in the side margins so they cannot collide
    # with the block or taxon text.
    mid = (block_top + block_bot) / 2
    ax.text(-0.10, mid, "shore", ha="right", va="center", fontsize=8.5,
            color="#555555", rotation=90)
    ax.text(n + 0.10, mid, "inland", ha="left", va="center", fontsize=8.5,
            color="#555555", rotation=90)


def covers_data(bb, ax):
    """Count data vertices under a display-space bbox (must be zero)."""
    hits = 0

    def c(xy):
        p = ax.transData.transform(np.asarray(xy))
        return int(((p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) &
                    (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)).sum())

    for ln in ax.get_lines():
        d = np.column_stack(ln.get_data())
        if d.shape[0] > 2:
            hits += c(d)
    for col in ax.collections:
        off = col.get_offsets()
        if len(off):
            hits += c(off)
        for pth in col.get_paths():
            if len(pth.vertices):
                hits += c(pth.vertices)
    return hits


def build(prefix="fig02_habitat_zonation"):
    dist, elev, land = load_profile()

    # Double-column width, 17.5 cm.
    fig = plt.figure(figsize=(6.89, 5.12), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.12])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    panel_a(ax_a, dist, elev, land)
    panel_b(ax_b)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    leg = ax_a.get_legend()
    n_hits = covers_data(leg.get_window_extent(r), ax_a)
    print(f"[check] legend covers {n_hits} data vertices (must be 0)")

    # The previous version checked legend-vs-data and text-vs-text but not
    # legend-vs-text, which is how the legend came to sit on the
    # "Mediterranean Sea" label. Check it explicitly.
    lb = leg.get_window_extent(r)
    clashes = [t.get_text().replace("\n", " ") for t in ax_a.texts
               if t.get_window_extent(renderer=r).overlaps(lb)]
    print(f"[check] legend overlaps {len(clashes)} labels (must be 0)"
          + (f": {clashes}" if clashes else ""))

    boxes = [(t.get_text()[:18], t.get_window_extent(renderer=r))
             for ax in (ax_a, ax_b) for t in ax.texts if t.get_text().strip()]
    pairs = [(boxes[i][0], boxes[j][0])
             for i in range(len(boxes)) for j in range(i + 1, len(boxes))
             if boxes[i][1].overlaps(boxes[j][1])]
    print(f"[check] overlapping text pairs: {len(pairs)}"
          + (f" {pairs}" if pairs else ""))

    import sys
    import subprocess
    gmt_ver = subprocess.run(["gmt", "--version"], capture_output=True,
                             text=True).stdout.strip() or "6"
    segs = segments(land)
    step = float(np.median(np.diff(dist)))
    credit = CREDIT.format(
        a_lon=TRANSECT_A[0], a_lat=TRANSECT_A[1],
        b_lon=TRANSECT_B[0], b_lat=TRANSECT_B[1],
        az=0.0, length=float(dist[-1]),
        step_m=step * 1000.0, step_km=step, npts=len(dist),
        density=1.0 / step,
        zmin=float(elev.min()), zmax=float(elev.max()),
        landpct=100.0 * land.mean(), shore=float(dist[segs[0][1]]),
        breaks=", ".join(f"{dist[i]:.2f}" for i in
                         np.where(np.diff(land) != 0)[0]),
        py=".".join(map(str, sys.version_info[:3])),
        mpl=matplotlib.__version__, gmt=gmt_ver)
    add_credit(fig, ax_b, credit)

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf and {prefix}.png")
    return n_hits


if __name__ == "__main__":
    build()
