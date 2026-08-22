#!/usr/bin/env python3
"""
A4_mapping.py -- PyGMT rendering of the habitat and change maps.

Listing A4 of the manuscript. Draws a classified raster with the cartographic
conventions used throughout: common extent, graticule and scale bar across
epochs so the two maps can be compared without reprojection artefacts, a
qualitative palette for the habitat classes, and a sequential palette for the
continuous confidence layer.

One detail is load-bearing and easy to get wrong. A categorical colour palette
must list its slices in ascending order of the integer code stored in the
raster, which is the classifier's own class order, not the order in which the
classes are presented in a legend. Writing the slices in display order produces
a palette with gaps and overlaps that GMT refuses to read, and the failure
message names neither cause.

Usage
-----
    python3 A4_mapping.py --raster class.tif --kind habitat --out map
    python3 A4_mapping.py --raster entropy.tif --kind confidence --out conf
"""

from __future__ import annotations

import argparse
import sys

import pygmt

REGION = [33.86, 34.11, 36.24, 36.42]
PROJECTION = "M15c"
MID_LAT = 36.33

# Class order is the classifier's (alphabetical); colours are the identity used
# in every figure of the paper.
CLASSES = [
    ("Swamp", "#FFFF00"),
    ("Temporary floodplain", "#FF4500"),
    ("Temporary ponds", "#00FFFF"),
    ("Terrestrial saline flats", "#FF00FF"),
]


def categorical_cpt(path, classes):
    """One slice per class, written in ascending code order."""
    with open(path, "w") as fh:
        for code, (_, colour) in enumerate(classes):
            fh.write(f"{code} {colour} {code + 1} {colour}\n")
        fh.write("B white\nF white\nN 200/200/200\n")
    return path


def legend_spec(path, classes, title="Halophyte habitat class"):
    with open(path, "w") as fh:
        fh.write(f"H 9p,Helvetica-Bold {title}\n")
        for name, colour in classes:
            fh.write(f"S 0.30c s 0.26c {colour} 0.4p,black 0.75c {name}\n")
    return path


def base_config():
    pygmt.config(
        FONT_TITLE="13p,Helvetica-Bold", FONT_LABEL="9p,Helvetica",
        FONT_ANNOT_PRIMARY="8p,Helvetica", MAP_FRAME_TYPE="fancy",
        MAP_FRAME_PEN="0.9p,black", MAP_GRID_PEN_PRIMARY="0.2p,white",
        MAP_TITLE_OFFSET="4p", FORMAT_GEO_MAP="ddd:mmF")


def draw(raster, kind, title, out):
    fig = pygmt.Figure()
    base_config()
    fig.basemap(region=REGION, projection=PROJECTION,
                frame=[f"WSne+t{title}", "xa5mf1mg5m", "ya3mf1mg3m"])

    if kind == "habitat":
        cpt = categorical_cpt("_habitat.cpt", CLASSES)
        fig.grdimage(grid=raster, cmap=cpt, nan_transparent=True)
    else:
        pygmt.makecpt(cmap="magma", series=[0, 1, 0.05], reverse=True)
        fig.grdimage(grid=raster, cmap=True, nan_transparent=True)

    fig.coast(resolution="f", area_thresh=0, shorelines="0.5p,black",
              water="#bcdcf0", rivers=["a/0.8p,#3a7ebf"])
    # The graticule is drawn by the first basemap call and painted over by
    # grdimage; redraw the frame so the grid sits on top.
    fig.basemap(frame=["WSne", "xa5mf1mg5m", "ya3mf1mg3m"])

    if kind == "habitat":
        fig.legend(spec=legend_spec("_habitat_legend.txt", CLASSES),
                   position="jBR+o0.25c/0.25c+w5.2c",
                   box="+gwhite@12+p0.6p,black")
    else:
        fig.colorbar(position="JMR+o0.55c/0c+w7c/0.35c",
                     frame=['x+l"Normalised entropy"'])

    fig.basemap(map_scale=f"jBL+o0.7c/0.7c+w5k+c{MID_LAT}+f+u+lkm")
    fig.basemap(rose="jTL+o0.5c/0.5c+w1.1c+f2+lW,E,S,N")

    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=200)
    fig.savefig(f"{out}.tif", dpi=600)
    print(f"wrote {out}.pdf, {out}.png and {out}.tif")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raster", required=True)
    ap.add_argument("--kind", choices=["habitat", "confidence"],
                    default="habitat")
    ap.add_argument("--title", default="Halophyte habitats")
    ap.add_argument("--out", default="map")
    a = ap.parse_args()
    try:
        draw(a.raster, a.kind, a.title, a.out)
    except Exception as exc:                                   # noqa: BLE001
        sys.exit(f"ERROR: {exc}")
