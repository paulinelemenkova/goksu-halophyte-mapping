#!/usr/bin/env python3
"""
fig06_workflow.py -- Figure 6 of the Goksu Delta halophyte manuscript.

Processing workflow, from the open satellite archives and the published releve
archive, through preprocessing, feature extraction, supervised classification
and validation, to the habitat maps and the change assessment.

The diagram encodes the epoch-symmetric design of Section 2.6: both epochs pass
through identical preprocessing, identical predictor definitions, an identical
class scheme and identically parameterised models, and diverge only at the
sensor that supplied the reflectances. Boxes carry the manuscript section,
equation or figure that documents them, so the figure doubles as a map of the
Methods.

Colour: each of the seven stages takes its own hue from the Spectral colormap,
and boxes within a stage are graded in intensity, lightest first, so that
membership of a stage is legible at a glance while individual boxes remain
distinguishable.

No data are required: this is a diagram, drawn entirely from the described
procedure.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
})

EDGE = "#33404a"

# --------------------------------------------------------------------------
# Stage hues, sampled from Spectral. Positions avoid the palest centre of the
# ramp, where two stages would be hard to tell apart, and are spread so that
# consecutive stages in the flow do not take neighbouring hues.
# --------------------------------------------------------------------------
STAGE_POS = {
    "input":   0.78,   # blue-green
    "prep":    0.90,   # blue
    "ref":     0.14,   # red-orange
    "feature": 0.66,   # green
    "model":   0.97,   # deep blue
    "valid":   0.04,   # red
    "output":  0.28,   # orange
}
STAGE_LABEL = {
    "input":   "Input archives",
    "prep":    "Preprocessing",
    "ref":     "Reference preparation",
    "feature": "Predictors",
    "model":   "Model",
    "valid":   "Validation, uncertainty",
    "output":  "Outputs",
}

# Grade within a stage: fraction blended toward white, lightest box first.
GRADE = [0.48, 0.32, 0.16, 0.00]

# --------------------------------------------------------------------------
# Layout. Uniform box height, with a reserved strip at the foot of each box for
# the cross-reference line, which previously collided with the detail text.
# --------------------------------------------------------------------------
H = 1.35
ROW = {0: 12.30, 1: 10.40, 2: 8.50, 3: 6.60, 4: 4.70, 5: 2.80, 6: 0.90}
COL = {0: 0.4, 1: 4.2, 2: 8.0, 3: 11.9}
W1, W3 = 3.5, 3.7

# key: (x, y, w, stage, title, detail, cross-reference)
BOXES = {
    "ls":   (COL[0], ROW[0], W1, "input", "Landsat C2 L2",
             "TM, ETM+, OLI, OLI-2\n30 m", "Table 1, Fig. 4"),
    "s2":   (COL[1], ROW[0], W1, "input", "Sentinel-2 L2A",
             "MSI, 10-20 m\n2015 onwards", "Table 1"),
    "rel":  (COL[2], ROW[0], 3.6, "input", "Releve archive",
             "279 plots, 2004-05\n149 halophytic", "Fig. 3"),
    "anc":  (COL[3], ROW[0], W3, "input", "Ancillary layers",
             "GLO-30, surface water,\nERA5-Land", "Table 1, Fig. 1"),

    "mask": (COL[0], ROW[1], W1, "prep", "Quality mask",
             "cloud, shadow, cirrus,\nsaturation", "Sect. 2.4"),
    "wat":  (COL[1], ROW[1], W1, "prep", "Water mask",
             "permanent open\nwater removed", "Sect. 2.4"),
    "harm": (COL[0], ROW[2], 7.3, "prep", "Cross-sensor harmonisation",
             "band-wise transfer functions; common grid and CRS",
             "Sect. 2.4, Listing 2"),

    "agg":  (COL[2], ROW[1], 3.6, "ref", "Class aggregation",
             "13 associations to\nmappable habitat classes", "Table 2"),
    "buf":  (COL[2], ROW[2], 3.6, "ref", "Rasterise plots",
             "buffer to plot radius;\ninterior pixels only", "Sect. 2.3"),

    "feat": (COL[0], ROW[3], 11.2, "feature", "Predictor stack",
             "greenness (Eq. 1)    moisture (Eq. 2)    salinity (Eq. 3, 4)    "
             "phenology (Eq. 5)    terrain", "Sect. 2.5, Fig. 5"),

    "mod":  (COL[0], ROW[4], 5.4, "model", "Tree ensemble",
             "random forest (Eq. 6, 7);\ngradient boosting benchmark",
             "Sect. 2.7"),
    "tune": (6.2, ROW[4], 5.4, "model", "Nested tuning",
             "search inside training folds only", "Table 3"),

    "fold": (COL[3], ROW[1], W3, "valid", "Blocked folds",
             "plot-level blocking;\nno neighbour leakage", "Sect. 2.8"),
    "val":  (COL[3], ROW[2], W3, "valid", "Validation",
             "blocked cross-validation;\nsyntaxonomic ordering", "Sect. 2.10"),
    "unc":  (COL[3], ROW[3], W3, "valid", "Confidence layer",
             "class-probability\nentropy (Eq. 9)", "Fig. 14"),

    "eb":   (COL[0], ROW[5], 5.4, "output", "Baseline epoch map",
             "Landsat, 2003-2007", "Fig. 9"),
    "ep":   (6.2, ROW[5], 5.4, "output", "Present epoch map",
             "Sentinel-2, 2021-2026", "Fig. 10"),
    "chg":  (COL[0], ROW[6], 11.2, "output", "Change assessment",
             "stratified area estimator (Eq. 11, 12); transition matrix; "
             "confidence and interval filters", "Sect. 2.11, Fig. 12, 13"),
}

# (from, to, style): 'v' vertical, 'h' horizontal, 'e' short offset descent,
# 'r' long descent routed down the right margin, clear of the legend.
ARROWS = [
    ("ls", "mask", "v"), ("s2", "wat", "v"),
    ("mask", "harm", "v"), ("wat", "harm", "e"),
    ("rel", "agg", "v"), ("anc", "fold", "v"),
    ("agg", "buf", "v"),
    ("harm", "feat", "v"), ("buf", "feat", "e"),
    ("feat", "mod", "v"), ("mod", "tune", "h"),
    ("fold", "val", "v"), ("val", "unc", "v"),
    ("mod", "eb", "v"), ("tune", "ep", "v"),
    ("eb", "chg", "v"), ("ep", "chg", "e"),
    ("unc", "chg", "r"),
]

LEGEND_BOX = (COL[3], 3.60, W3, 2.45)    # free space in the right column

CREDIT = (
    "Each stage takes its own hue from the Spectral colormap and boxes within a "
    "stage are graded in intensity; the legend gives the seven stages. Every box "
    "carries the manuscript section, equation, table or figure that documents it. "
    "The two epochs follow identical paths and differ only in the sensor supplying "
    "the reflectances, which is what licenses attributing the difference between "
    "the two maps to the landscape rather than to the analysis. Software used for "
    "plotting figure: Python {py} with Matplotlib {mpl}. Source: authors"
)


# --------------------------------------------------------------------------
def stage_colours():
    """One base hue per stage, graded in intensity across that stage's boxes."""
    cmap = plt.get_cmap("Spectral")
    order = {}
    for key, rec in BOXES.items():
        order.setdefault(rec[3], []).append(key)
    out = {}
    for stage, keys in order.items():
        base = cmap(STAGE_POS[stage])[:3]
        for i, key in enumerate(keys):
            f = GRADE[i] if len(keys) > 1 else GRADE[1]
            out[key] = tuple(c + (1.0 - c) * f for c in base)
    return out


def lum(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def text_colours(rgb):
    """Light text on dark fills, dark text on light fills."""
    if lum(rgb) < 0.52:
        return "#ffffff", "#f4f4f4", "#e6ecf0"
    return "#1a1a1a", "#2b2b2b", "#46555f"


def draw_box(ax, key, rec, fills):
    x, y, w, stage, title, detail, ref = rec
    fill = fills[key]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, H, boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=fill, edgecolor=EDGE, linewidth=0.9, zorder=2))
    c_title, c_body, c_ref = text_colours(fill)
    cx = x + w / 2
    # Fixed bands: title at the top, cross-reference in a reserved strip at the
    # foot, detail centred in what remains. The previous version anchored the
    # detail from the top and the reference from the bottom, so a longer detail
    # ran straight through the reference line.
    ax.text(cx, y + H - 0.16, title, ha="center", va="top", fontsize=8.6,
            fontweight="bold", color=c_title, zorder=3)
    ax.text(cx, y + 0.30 + (H - 0.72) / 2, detail, ha="center", va="center",
            fontsize=7.3, color=c_body, linespacing=1.30, zorder=3)
    ax.text(cx, y + 0.10, ref, ha="center", va="bottom", fontsize=6.7,
            style="italic", color=c_ref, zorder=3)


def anchors(rec, side):
    x, y, w = rec[0], rec[1], rec[2]
    return {"top": (x + w / 2, y + H), "bottom": (x + w / 2, y),
            "left": (x, y + H / 2), "right": (x + w, y + H / 2)}[side]


def draw_arrow(ax, a, b, style):
    ra, rb = BOXES[a], BOXES[b]
    if style == "v":
        p0, p1, conn = anchors(ra, "bottom"), anchors(rb, "top"), "arc3,rad=0"
    elif style == "h":
        p0, p1, conn = anchors(ra, "right"), anchors(rb, "left"), "arc3,rad=0"
    elif style == "r":
        # Down the right margin and into the right edge of the target, so the
        # curve clears the legend block.
        p0, p1 = anchors(ra, "right"), anchors(rb, "right")
        conn = "arc3,rad=-0.42"
    else:
        p0, p1 = anchors(ra, "bottom"), anchors(rb, "top")
        conn = f"arc3,rad={0.20 if p1[0] < p0[0] else -0.20}"
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=conn, arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9, color=EDGE, zorder=1, shrinkA=1.5, shrinkB=1.5))


def draw_legend(ax):
    x, y, w, h = LEGEND_BOX
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="white", edgecolor=EDGE, linewidth=0.9, zorder=2))
    ax.text(x + w / 2, y + h - 0.12, "Stage", ha="center", va="top",
            fontsize=7.4, fontweight="bold", color="#1a1a1a", zorder=3)
    cmap = plt.get_cmap("Spectral")
    order = ["input", "prep", "ref", "feature", "model", "valid", "output"]
    top, bot = y + h - 0.36, y + 0.08
    step = (top - bot) / len(order)
    for i, stage in enumerate(order):
        yy = top - (i + 0.5) * step
        ax.add_patch(FancyBboxPatch(
            (x + 0.16, yy - 0.045), 0.26, 0.09,
            boxstyle="square,pad=0.008",
            facecolor=cmap(STAGE_POS[stage])[:3], edgecolor=EDGE,
            linewidth=0.5, zorder=3))
        ax.text(x + 0.52, yy, STAGE_LABEL[stage], ha="left", va="center",
                fontsize=6.6, color="#1a1a1a", zorder=3)


# --------------------------------------------------------------------------
def check_routed_arrow(ax):
    """Sample the routed arrow's path; no point may fall inside the legend."""
    import numpy as np
    lx, ly, lw, lh = LEGEND_BOX
    hits = 0
    for pa in ax.patches:
        if not isinstance(pa, FancyArrowPatch):
            continue
        v = pa.get_path().vertices
        if len(v) < 3:
            continue
        inside = ((v[:, 0] >= lx) & (v[:, 0] <= lx + lw)
                  & (v[:, 1] >= ly) & (v[:, 1] <= ly + lh))
        hits += int(inside.sum())
    return hits


def check_box_overlaps():
    """Boxes, including the legend, must not intersect. Pure geometry."""
    rects = {k: (r[0], r[1], r[2], H) for k, r in BOXES.items()}
    rects["legend"] = LEGEND_BOX
    keys = list(rects)
    bad = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            xa, ya, wa, ha = rects[keys[i]]
            xb, yb, wb, hb = rects[keys[j]]
            if xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha:
                bad.append((keys[i], keys[j]))
    return bad


def check_text(fig, ax):
    """No label may overflow its box, and no two labels may overlap."""
    r = fig.canvas.get_renderer()
    spill = []
    for key, rec in BOXES.items():
        x, y, w = rec[0], rec[1], rec[2]
        bb = ax.transData.transform([[x, y], [x + w, y + H]])
        for t in ax.texts:
            tb = t.get_window_extent(renderer=r)
            cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
            if bb[0][0] <= cx <= bb[1][0] and bb[0][1] <= cy <= bb[1][1]:
                if (tb.x0 < bb[0][0] - 1 or tb.x1 > bb[1][0] + 1
                        or tb.y0 < bb[0][1] - 1 or tb.y1 > bb[1][1] + 1):
                    spill.append((key, t.get_text().split("\n")[0][:22]))
    boxes = [(t.get_text().split("\n")[0][:20],
              t.get_window_extent(renderer=r)) for t in ax.texts]
    clash = [(boxes[i][0], boxes[j][0])
             for i in range(len(boxes)) for j in range(i + 1, len(boxes))
             if boxes[i][1].overlaps(boxes[j][1])]
    return spill, clash


def add_credit(fig, ax, text, fontsize=8, pad_in=0.10):
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
    low = ax.get_tightbbox(r).y0
    t.set_position((0.0, low / (fig.get_size_inches()[1] * fig.dpi)
                    - pad_in / fig.get_size_inches()[1]))
    return t


def build(prefix="fig06_workflow"):
    clash = check_box_overlaps()
    print(f"[check] intersecting boxes: {len(clash)}"
          + (f" {clash}" if clash else " (must be 0)"))

    fills = stage_colours()
    fig = plt.figure(figsize=(6.89, 5.9), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0.7, 13.85)
    ax.axis("off")

    for a, b, st in ARROWS:
        draw_arrow(ax, a, b, st)
    for key, rec in BOXES.items():
        draw_box(ax, key, rec, fills)
    draw_legend(ax)

    fig.canvas.draw()
    n_route = check_routed_arrow(ax)
    print(f"[check] arrow vertices inside the legend: {n_route} (must be 0)")
    spill, overlap = check_text(fig, ax)
    print(f"[check] labels overflowing their box: {len(spill)}"
          + (f" {spill}" if spill else " (must be 0)"))
    print(f"[check] overlapping label pairs: {len(overlap)}"
          + (f" {overlap}" if overlap else " (must be 0)"))

    ls = sorted((lum(c), k) for k, c in fills.items())
    print(f"[check] fill luminance {ls[0][0]:.2f} ({ls[0][1]}) "
          f"to {ls[-1][0]:.2f} ({ls[-1][1]})")

    add_credit(fig, ax, CREDIT.format(
        py=".".join(map(str, sys.version_info[:3])),
        mpl=matplotlib.__version__))

    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{prefix}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"wrote {prefix}.pdf and {prefix}.png "
          f"({len(BOXES)} boxes, {len(ARROWS)} arrows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="fig06_workflow")
    a = ap.parse_args()
    build(a.prefix)
