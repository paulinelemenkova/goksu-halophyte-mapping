#!/usr/bin/env python3
"""
fig03_reference_plots.py -- Figure 3 of the Goksu Delta halophyte manuscript.

Distribution of the published phytosociological releve plots across the delta.
Halophytic plots are coloured by habitat subtype and are the plots used to
train and validate the classification; sand-dune and aquatic plots are drawn
as small open grey symbols and are excluded.

Inputs
------
    reference_plots.csv   from make_plots_csv.py (transcribed from the source
                          tables of the published survey)
    goksu_dem.tif         Copernicus WorldDEM-30 (GLO-30) cut to the window

Usage
-----
    python3 fig03_reference_plots.py --dem goksu_dem.tif
"""

import argparse
import csv
import sys
from collections import Counter

import pygmt

REGION = [33.86, 34.11, 36.24, 36.42]
PROJ = "M15c"
MID_LAT = 36.33

# Halophytic subtypes, ordered shore -> inland, matched to the ordering of
# Figure 2 so the two figures read together.
# Deliberately non-green, non-blue: the relief backdrop is green and tan, so
# blue and green symbols sank into it. yellow / cyan / orangered / magenta are
# maximally separated from the basemap and from each other.
HALO = [
    ("Swamp", "#FFFF00", "c"),          # yellow
    ("Temporary ponds", "#00FFFF", "t"),  # cyan
    ("Temporary floodplain", "#FF4500", "d"),  # orangered
    ("Terrestrial saline flats", "#FF00FF", "s"),  # magenta1
]
# Open (white-filled) symbols read as "not used" against the filled halophytic
# ones. Line-only symbols such as "x" and "+" are avoided deliberately: this
# GMT/Ghostscript combination writes PostScript for them that gs cannot parse,
# and psconvert then fails with a bare status-79 error.
EXCLUDED = [
    ("dune", "#5a5a5a", "c", 0.13, "Sand dune plots (excluded)"),
    ("aquatic", "#5a5a5a", "i", 0.15, "Aquatic plots (excluded)"),
]

CPT_BREAKS = "0,5,20,100,300,500,600"
CPT_NAME = "etopo1"
CPT_LIGHTEN = 0.45      # lighter than Figure 1: the relief is only a backdrop
                        # here and must not compete with the plot symbols.


def _lighten_cpt(src, dst, amount):
    """Blend every colour in a CPT toward white by `amount` (0-1)."""
    def blend(tok):
        if "/" not in tok:
            return tok
        try:
            rgb = [float(v) for v in tok.split("/")]
        except ValueError:
            return tok
        if len(rgb) != 3:
            return tok
        return "/".join(f"{v + (255 - v) * amount:.4g}" for v in rgb)

    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            fout.write("\t".join(blend(p) for p in
                                 line.rstrip("\n").split("\t")) + "\n")
    return dst


def load(path="reference_plots.csv"):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["lat"] = float(r["lat"])
        r["lon"] = float(r["lon"])
    return rows


def build(dem=None, prefix="fig03_reference_plots"):
    rows = load()
    n_by_sub = Counter(r["subtype"] for r in rows if r["group"] == "halophytic")
    n_halo = sum(n_by_sub.values())
    n_dune = sum(r["group"] == "dune" for r in rows)
    n_aq = sum(r["group"] == "aquatic" for r in rows)
    print(f"[info] {len(rows)} plots: {n_halo} halophytic, "
          f"{n_dune} dune, {n_aq} aquatic")

    fig = pygmt.Figure()
    pygmt.config(
        FONT_TITLE="13p,Helvetica-Bold",
        FONT_LABEL="9p,Helvetica",
        FONT_ANNOT_PRIMARY="8p,Helvetica",
        MAP_FRAME_TYPE="fancy",
        MAP_FRAME_PEN="0.9p,black",
        MAP_GRID_PEN_PRIMARY="0.2p,white",
        MAP_TITLE_OFFSET="4p",
        MAP_TICK_LENGTH_PRIMARY="4p",
        FORMAT_GEO_MAP="ddd:mmF",
    )
    fig.basemap(
        region=REGION, projection=PROJ,
        frame=["WSne+tPhytosociological releve plots, Goksu Delta",
               "xa5mf1mg5m", "ya3mf1mg3m"],
    )

    relief = False
    if dem:
        try:
            pygmt.makecpt(cmap=CPT_NAME, series=CPT_BREAKS, continuous=False,
                          overrule_bg=True, output="_f3_raw.cpt")
            cpt = _lighten_cpt("_f3_raw.cpt", "_f3.cpt", CPT_LIGHTEN)
            shade = pygmt.grdgradient(grid=dem, radiance=[315, 35],
                                      normalize="e0.25")
            fig.grdimage(grid=dem, cmap=cpt, shading=shade)
            relief = True
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] relief unavailable ({exc})", file=sys.stderr)

    fig.coast(resolution="f", area_thresh=0, shorelines="0.5p,black",
              rivers=["a/0.8p,#3a7ebf"], lakes="#bcdcf0", water="#bcdcf0",
              land=None if relief else "#eae4d6")

    # The graticule is drawn by the first basemap call, which grdimage then
    # paints over. Redraw it (frame only, no title) so the white grid lines
    # sit on top of the relief.
    fig.basemap(frame=["WSne", "xa5mf1mg5m", "ya3mf1mg3m"])

    # Excluded plots first, so halophytic symbols sit on top.
    for grp, colour, style, size, _ in EXCLUDED:
        sel = [r for r in rows if r["group"] == grp]
        fig.plot(x=[r["lon"] for r in sel], y=[r["lat"] for r in sel],
                 style=f"{style}{size}c", fill="white", pen=f"0.5p,{colour}")

    for name, colour, style in HALO:
        sel = [r for r in rows if r["subtype"] == name]
        fig.plot(x=[r["lon"] for r in sel], y=[r["lat"] for r in sel],
                 style=f"{style}0.24c", fill=colour, pen="0.4p,black")

    # Legend built from the same lists that drew the symbols, with the counts
    # taken from the data so the caption and the map cannot disagree.
    with open("_f3_legend.txt", "w") as fh:
        fh.write("H 9p,Helvetica-Bold Halophytic plots (used)\n")
        for name, colour, style in HALO:
            fh.write(f"S 0.30c {style} 0.24c {colour} 0.4p,black 0.75c "
                     f"{name} (n = {n_by_sub[name]})\n")
        fh.write("G 0.10c\n")
        fh.write("H 9p,Helvetica-Bold Other habitat groups (excluded)\n")
        for grp, colour, style, size, label in EXCLUDED:
            n = n_dune if grp == "dune" else n_aq
            fh.write(f"S 0.30c {style} {size}c white 0.5p,{colour} 0.75c "
                     f"{label.split(' (')[0]} (n = {n})\n")
    fig.legend(spec="_f3_legend.txt", position="jTR+o0.25c/0.25c+w5.6c",
               box="+gwhite@12+p0.6p,black")

    fig.basemap(map_scale=f"jBL+o0.7c/0.7c+w5k+c{MID_LAT}+f+u+lkm")
    fig.basemap(rose="jBR+o0.5c/0.5c+w1.1c+f2+lW,E,S,N")

    credit = [
        "Plot positions transcribed from the survey tables of the published phytosociological study of the delta (CC BY 4.0); published as degrees and decimal",
        "minutes and converted to decimal degrees (WGS 84). {n} of the 279 surveyed plots carry usable coordinates; one further plot was dropped because its",
        "published longitude places it outside the delta. Relief: Copernicus WorldDEM-30 (GLO-30), 1 arcsec. Produced using Copernicus WorldDEM-30 (c) DLR e.V.",
        "2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA; all rights reserved. Coastline,",
        "lakes and rivers: GSHHG full resolution. Rendered with GMT/PyGMT. Source: authors",
    ]
    credit = [c.replace("{n}", str(len(rows))) for c in credit]
    for i, line in enumerate(credit):
        fig.text(position="BL", offset=f"0c/{-0.75 - 0.28 * i}c", text=line,
                 font="6p,Helvetica,gray30", justify="LT", no_clip=True)

    fig.savefig(f"{prefix}.pdf")
    fig.savefig(f"{prefix}.png", dpi=200)
    print(f"wrote {prefix}.pdf and {prefix}.png (relief: {relief})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", default="goksu_dem.tif")
    ap.add_argument("--prefix", default="fig03_reference_plots")
    a = ap.parse_args()
    build(dem=a.dem, prefix=a.prefix)
