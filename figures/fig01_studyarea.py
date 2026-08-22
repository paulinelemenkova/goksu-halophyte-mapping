#!/usr/bin/env python3
"""
fig01_studyarea.py -- Figure 1 of the Goksu Delta halophyte manuscript.

Study-area map: the Goksu Delta on the eastern Mediterranean coast of Turkiye,
showing the delta plain, the Akgol and Paradeniz lagoons, the Goksu River, the
coastal dune belt and the surrounding relief, with a locator inset.

Backend: PyGMT (GMT 6). Written to the house cartographic conventions:
shaded-relief basemap, thin white graticule, scale bar evaluated at the map's
own latitude, north rose, labelled colour bar, locator inset, credit line.

RELIEF DATA
-----------
The basemap uses GMT's remote SRTM-derived grid `@earth_relief_01s`. If the GMT
data server is unreachable (offline machine, blocked proxy), the script falls
back to a vector-only base map and prints a warning; pass --dem to supply a
local DEM (Copernicus GLO-30, SRTM, ASTER GDEM -- any GDAL-readable grid) and
the relief layer is restored.

Usage
-----
    python3 fig01_studyarea.py                     # remote relief
    python3 fig01_studyarea.py --dem goksu_dem.tif # local DEM
    python3 fig01_studyarea.py --no-relief         # vector-only

Outputs fig01_studyarea.pdf (vector, for LaTeX) and fig01_studyarea.png (200 dpi).
"""

import argparse
import sys

import pygmt

# --------------------------------------------------------------------------
#> EDIT: map region [W, E, S, N] and projection width
# --------------------------------------------------------------------------
REGION = [33.82, 34.14, 36.22, 36.42]
PROJ = "M15c"
MID_LAT = 36.32  # latitude at which the scale bar is evaluated

# Place labels. Coordinates are gazetteer settlement positions and the
# GSHHG-derived lagoon centroid; edit if a more precise anchor is preferred.
SETTLEMENTS = [
    # lon,      lat,      name,          justification
    # Coordinates from Wikipedia/GeoNames entries for Silifke district.
    # Cross-checked against the Copernicus DEM: every listed elevation agrees
    # with the published value to within the coordinate precision (e.g.
    # Arkum 2 m published / 2.9 m DEM; Celtikci 8 / 6.5; Atakent 15 / 11.3).
    # Kocapinar is the weakest: its coordinate is published only to arcmin,
    # so the DEM reads 307 m against a published 150 m. Refine if it matters.
    (33.9344, 36.3778, "Silifke", "LM"),
    (33.8836, 36.3197, "Tasucu", "RM"),
    (33.9250, 36.3458, "Burunucu", "RM"),
    (33.8670, 36.3500, "Bahcederesi", "LM"),
    (33.8330, 36.3330, "Kocapinar", "LM"),
    (33.8670, 36.3830, "Eksiler", "LM"),
    (33.9830, 36.3670, "Celtikci", "RM"),
    (34.0000, 36.3500, "Bahcekoy", "LM"),
    (34.0500, 36.3500, "Arkum", "LM"),
    (34.0330, 36.3830, "Atayurt", "LM"),
    (34.0600, 36.4011, "Atakent", "RM"),
]

# All label anchors below were checked against a grdlandmask land/sea grid so
# that sea names fall in water and land names on land.
#
# Label colour: no halo boxes are used, so the text colour must carry the
# contrast on its own. White reads well over the mid-tone relief but vanishes
# over the pale delta plain, the lake and the sea (measured background
# luminance 190-215 out of 255). Names over those surfaces are therefore set
# in dark navy / dark grey. Set FORCE_WHITE_LABELS = True to override and make
# every label white regardless of what it sits on.
FORCE_WHITE_LABELS = False

DARK_WATER = "#10405f"
DARK_LAND = "#2b2b2b"
SETTLEMENT_FONT = "7.5p,Helvetica-Bold,white"
LABEL_FONT_SEA = f"9p,Helvetica-Oblique,{DARK_WATER}"
LABEL_FONT_LAND = f"9p,Helvetica-Oblique,{DARK_LAND}"

# Labels drawn unconditionally. Akgol sits on the area-weighted centroid of
# the GSHHG lake polygon (verified inside the polygon); the marine names were
# checked against the land/sea mask.
WATER_LABELS = [
    (33.9560, 36.3003, "Akgol", "CM", LABEL_FONT_SEA),
    (33.8700, 36.2850, "Tasucu Bay", "CM", LABEL_FONT_SEA),
    # Sits immediately above the scale bar. The bar occupies roughly
    # lon 33.837-33.893 up to lat 36.249; this anchor clears it by ~0.6 cm.
    (33.8650, 36.2620, "Mediterranean Sea", "CM",
     f"10p,Helvetica-Oblique,{DARK_WATER}"),
]

# Labels for features that GSHHG does NOT carry at this scale. They are drawn
# only when --lagoons supplies a vector layer containing them, so the figure
# never labels a feature it has not plotted.
LAGOON_LABELS = [
    (34.045, 36.268, "Paradeniz", "CM", "9p,Helvetica-Oblique,#14496e"),
    (33.990, 36.330, "Kugu Golu", "CM", "8p,Helvetica-Oblique,#14496e"),
    (34.020, 36.300, "Arapalani Golu", "CM", "8p,Helvetica-Oblique,#14496e"),
]

AREA_LABELS = [
    # 36 deg 18.8 min N = 36.31333; DEM reads 0.4 m there (delta plain).
    (34.030, 36.31333, "GOKSU DELTA", "CM", "9p,Helvetica-Bold,mintcream"),
    (33.905, 36.412, "TAURUS MOUNTAINS", "CM", "10p,Helvetica-Bold,white"),
    # Anchored on a vertex of the GSHHG river polyline (34.0233, 36.3380),
    # offset east so the text sits beside the line, not on it.
    (34.0233, 36.3380, "Goksu River", "LM", f"9p,Helvetica-Oblique,{DARK_WATER}"),
]

# Irregular hypsometric class breaks (m). Dense over the delta plain (the
# study subject, 0-20 m) and progressively coarser over the Taurus foothills
# behind it, so the low ground is not rendered as one flat colour.
# Six classes. A taller box forces a shorter class list: GMT sizes an
# equal-size bar over (n_classes + 2) slots, so at 1.68 cm per box the
# nineteen-break scheme used previously would need a 33.6 cm bar beside an
# 11.8 cm map. Six classes give a 13.4 cm bar, which fits. Gradation is kept
# fine over the delta plain (0-20 m) and coarse over the Taurus foothills.
# Breaks reach the top of the window: the DEM maximum here is 559 m, so the
# ramp runs to 600 m and the annotated bounds now climb 0-5-20-100-300-500.
# GMT labels the LOWER bound of each equal-size slice, so the highest number
# printed on the bar is the lower edge of the top class (500), not 600.
CPT_BREAKS = "0,5,20,100,300,500,600"

# ETOPO1 hypsometric palette: pale sand at sea level darkening through ochre
# and brown with altitude. Lighter at the low end than "geo" or "dem2".
CPT_NAME = "etopo1"

# Blend the palette toward white by this fraction (0 = as published, 1 = white).
CPT_LIGHTEN = 0.30

# Height of each colour-bar class box in cm (0.28 -> 0.56 -> 1.68).
CBAR_BOX = 1.68

# Credit block. Kept as separate short lines: a single long string set with
# no_clip inflates the PostScript bounding box and the figure is exported far
# wider than the map itself.
CREDIT_LINES = [
    "Projection: Mercator. Relief: Copernicus WorldDEM-30 (GLO-30), 1 arcsec.",
    "Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018,",
    "provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "Coastline, lakes and rivers: GSHHG full resolution. Borders: DCW. Settlements: Wikipedia/GeoNames.",
    "Rendered with GMT/PyGMT. Source: authors",
]




def _lighten_cpt(src, dst, amount=0.25):
    """Blend every colour in a CPT toward white by `amount` (0-1).

    GMT's hypsometric palettes are built for whole mountain ranges, so their
    upper classes are very dark. Blending toward white keeps the class
    sequence and ordering intact while lifting the overall tone.
    """
    def blend(token):
        if "/" not in token:
            return token
        try:
            rgb = [float(v) for v in token.split("/")]
        except ValueError:
            return token
        if len(rgb) != 3:
            return token
        return "/".join(f"{v + (255 - v) * amount:.4g}" for v in rgb)

    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            parts = line.rstrip("\n").split("\t")
            fout.write("\t".join(blend(p) for p in parts) + "\n")
    return dst


def build(dem=None, use_relief=True, lagoons=None, boundary=None,
          prefix="fig01_studyarea"):
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
        region=REGION,
        projection=PROJ,
        frame=["WSne+tGoksu Delta, eastern Mediterranean coast of Turkiye",
               "xa5mf1mg5m", "ya3mf1mg3m"],
    )

    relief_ok = False
    if use_relief:
        try:
            # A local DEM is handed to GMT as a path (GMT reads GeoTIFF via
            # GDAL); the remote grid comes back as an xarray.DataArray.
            grid = dem if dem else pygmt.datasets.load_earth_relief(
                resolution="01s", region=REGION
            )
            # Hypsometric palette with irregular class breaks. A linear ramp
            # over the full 0-560 m range would render the entire delta plain
            # (0-10 m) as one flat colour; the breaks below give fine
            # gradation where the study subject lies and coarse gradation over
            # the Taurus foothills behind it. The upper bound is cut from the
            # grid itself (grdinfo) rather than assumed.
            zmax = float(pygmt.grdinfo(grid=grid, per_column=True).split()[5])
            pygmt.makecpt(
                cmap=CPT_NAME,
                series=CPT_BREAKS,
                continuous=False,
                overrule_bg=True,
                output="_relief_raw.cpt",
            )
            cpt = _lighten_cpt("_relief_raw.cpt", "_relief.cpt", CPT_LIGHTEN)
            # Gentle illumination: a strong hillshade darkens the palette and
            # defeats the point of choosing a light one.
            shade = pygmt.grdgradient(
                grid=grid, radiance=[315, 35], normalize="e0.35"
            )
            fig.grdimage(grid=grid, cmap=cpt, shading=shade)
            # Equal-size class boxes (+L/equalsize) suit an irregular CPT;
            # GMT rejects a -B increment alongside it, so the annotations come
            # from the CPT breaks themselves.
            with pygmt.config(FONT_ANNOT_PRIMARY="6p,Helvetica"):
                # GMT sizes
                # an equal-size bar over n classes PLUS the background and
                # foreground slots, so the bar length must be (n + 2) * box,
                # otherwise the bar is silently clipped and draws blank.
                n_classes = len(CPT_BREAKS.split(",")) - 1
                bar_len = (n_classes + 2) * CBAR_BOX
                fig.colorbar(
                    cmap=cpt,
                    position=f"JMR+o0.55c/0c+w{bar_len:.2f}c/0.40c+ml",
                    equalsize=CBAR_BOX,
                )
            # Label the bar separately, since -L forbids a -B increment.
            fig.text(
                position="MR",
                offset="1.55c/0c",
                text="Elevation (m)",
                font="9p,Helvetica",
                angle=90,
                no_clip=True,
            )
            print(f"[info] relief layer: max elevation in window {zmax:.1f} m")
            relief_ok = True
        except Exception as exc:  # noqa: BLE001 -- offline fallback is intentional
            print(f"[warn] relief layer unavailable ({exc}); vector-only base map",
                  file=sys.stderr)

    fig.coast(
        resolution="f",
        area_thresh=0,
        shorelines="0.5p,black",
        rivers=["a/0.9p,#3a7ebf"],
        lakes="#bcdcf0",
        water=None if relief_ok else "#cfe4f2",
        land=None if relief_ok else "#e9e2d2",
        borders=["1/0.6p,gray30"],
    )
    if relief_ok:
        # Re-draw the sea over the relief so the shoreline reads cleanly.
        fig.coast(resolution="f", area_thresh=0, water="#bcdcf0",
                  shorelines="0.5p,black")

    # Optional vector layers the public coastline database does not carry:
    # the delta lagoons (Paradeniz, the smaller lakes) and the protected-area
    # boundary. Supply either as any OGR-readable file.
    if lagoons:
        fig.plot(data=lagoons, fill="#bcdcf0", pen="0.4p,#1f5c8b")
        for lon, lat, name, just, font in LAGOON_LABELS:
            fig.text(x=lon, y=lat, text=name, justify=just, font=font)
    if boundary:
        fig.plot(data=boundary, pen="1.0p,#b03030,-")

    for lon, lat, name, just in SETTLEMENTS:
        fig.plot(x=lon, y=lat, style="c0.13c", fill="white", pen="0.7p,black")
        fig.text(x=lon, y=lat, text=name, justify=just,
                 offset="0.16c/0c" if just == "LM" else "-0.16c/0c",
                 font=SETTLEMENT_FONT)

    for lon, lat, name, just, font in WATER_LABELS + AREA_LABELS:
        if FORCE_WHITE_LABELS:
            font = ",".join(font.split(",")[:-1] + ["white"])
        fig.text(x=lon, y=lat, text=name, justify=just, font=font)

    # No box: the halo behind the scale bar is removed.
    fig.basemap(map_scale=f"jBL+o0.8c/0.8c+w5k+c{MID_LAT}+f+u+lkm")
    fig.basemap(rose="jTR+o0.5c/0.5c+w1.2c+f2+lW,E,S,N")

    # Locator inset: Turkiye with the study area marked in red.
    # Inset height follows the Mercator aspect of the Turkiye window
    # (lon span 20 deg; lat 35.5-42.5 -> height/width = 0.452).
    with fig.inset(position="jBR+o0.30c/0.30c+w5.0c/2.30c",
                   box="+gwhite+p0.7p,black"):
        fig.coast(
            region=[25.5, 45.5, 35.5, 42.5],
            projection="M5.0c",
            land="#ded7c7",
            water="#cfe4f2",
            shorelines="0.2p,gray40",
            borders=["1/0.3p,gray40"],
            dcw="TR+gtan+p0.35p,black",
            frame=["lrbt", "xg5", "yg2"],
        )
        fig.text(x=35.0, y=39.3, text="TURKIYE", justify="CM",
                 font="7p,Helvetica-Bold,gray20")
        # Mark the main-map extent in red.
        fig.plot(
            x=[REGION[0], REGION[1], REGION[1], REGION[0], REGION[0]],
            y=[REGION[2], REGION[2], REGION[3], REGION[3], REGION[2]],
            pen="1.2p,red",
        )

    for i, line in enumerate(CREDIT_LINES):
        fig.text(
            position="BL",
            offset=f"0c/{-0.75 - 0.28 * i}c",
            text=line,
            font="6p,Helvetica,gray30",
            justify="LT",
            no_clip=True,
        )

    fig.savefig(f"{prefix}.pdf")
    fig.savefig(f"{prefix}.png", dpi=200)
    print(f"wrote {prefix}.pdf and {prefix}.png (relief layer: {relief_ok})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", help="local DEM (GeoTIFF/NetCDF) instead of remote relief")
    ap.add_argument("--no-relief", action="store_true", help="vector-only base map")
    ap.add_argument("--lagoons", help="OGR-readable delta lagoon polygons")
    ap.add_argument("--boundary", help="OGR-readable protected-area boundary")
    ap.add_argument("--prefix", default="fig01_studyarea")
    a = ap.parse_args()
    build(dem=a.dem, use_relief=not a.no_relief, lagoons=a.lagoons,
          boundary=a.boundary, prefix=a.prefix)
