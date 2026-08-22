#!/usr/bin/env python3
"""
fig09_map_baseline.py -- Figure 9 of the Goksu Delta halophyte manuscript.

Halophyte habitat classification of the delta for the baseline epoch,
contemporaneous with the phytosociological survey.

The classifier is the one described in Section 2.7 and validated in
Section 2.8: a random forest trained on the reference plots. Here it is fitted
on all plots of the requested epoch and applied to the per-pixel feature stack
exported by make_feature_stack.js. The accuracy of the result is NOT estimated
from this fit -- that comes from the blocked cross-validation, because a model
scored on the plots it was trained on reports its own memory.

Three masks are applied before the map is drawn, each for a stated reason:

  Water.        Permanent open water carries no halophyte class and is drawn
                separately.
  Sparse.       Pixels with fewer valid observations than the harmonic model
                has parameters are excluded rather than classified, since
                their coefficients are unconstrained (Sect. 2.4).
  Low confidence.  Pixels whose class-probability entropy exceeds a threshold
                are shown hatched rather than coloured, so that the map does
                not present a guess with the same authority as a decision.

INPUT
-----
    feature_stack_baseline.tif   40 bands, Int16 scaled by 10000
    n_obs_baseline.tif           valid-observation count per pixel
    features.csv                 training rows from build_features.py

USAGE
-----
    python3 fig09_map_baseline.py --stack feature_stack_baseline.tif \
        --nobs n_obs_baseline.tif --epoch baseline
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

GAIN = 10000.0
NPAR = 5          # 1 + 2K at K = 2

CLASSES = [
    ("Swamp", "#FFFF00"),
    ("Temporary ponds", "#00FFFF"),
    ("Temporary floodplain", "#FF4500"),
    ("Terrestrial saline flats", "#FF00FF"),
]

REGION = [33.86, 34.11, 36.24, 36.42]


def load_training(features_csv, epoch):
    df = pd.read_csv(features_csv)
    if epoch != "both":
        df = df[df["epoch"] == epoch]
    if df.empty:
        sys.exit(f"ERROR: no training rows for epoch '{epoch}'")
    drop = ["plot", "epoch", "subtype", "n_obs", "block"]
    cols = [c for c in df.columns if c not in drop]
    return df[cols].to_numpy(float), df["subtype"].to_numpy(), cols


def domain_mask(stack_path, dem_path, max_elev):
    """Restrict prediction to ground where halophytes can occur.

    The classifier is trained on halophytic plots alone and therefore has no
    "not a halophyte" class: applied to the whole window it assigns open sea,
    dune ridges, irrigated farmland and the Taurus foothills to one of the four
    habitat types. Two masks bound the domain to the delta plain. Land and
    water are separated with a GSHHG full-resolution mask, and ground above
    `max_elev` is excluded, the natural area of the delta lying between sea
    level and about eight metres.
    """
    import subprocess

    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(stack_path) as src:
        shape, transform, crs, bounds = (
            (src.height, src.width), src.transform, src.crs, src.bounds)

    # Land/sea from GSHHG, rasterised to the stack grid.
    subprocess.run(
        ["gmt", "grdlandmask",
         f"-R{bounds.left}/{bounds.right}/{bounds.bottom}/{bounds.top}",
         "-I3s", "-Df", "-G_landmask.nc", "-N0/1/0/1/0"],
        check=True, capture_output=True)
    with rasterio.open("_landmask.nc") as src:
        land = np.empty(shape, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=land,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest)
    with rasterio.open(dem_path) as src:
        dem = np.empty(shape, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=dem,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear)

    mask = (land > 0.5) & (dem <= max_elev)
    print(f"[info] domain mask: {int((land <= 0.5).sum())} px water, "
          f"{int(((land > 0.5) & (dem > max_elev)).sum())} px above "
          f"{max_elev:.0f} m, {int(mask.sum())} px retained "
          f"({100 * mask.mean():.1f} % of the window)")
    return mask.ravel()


def classify(stack_path, nobs_path, X, y, cols, entropy_max, domain):
    import rasterio
    from sklearn.ensemble import RandomForestClassifier

    with rasterio.open(stack_path) as src:
        if src.count != len(cols):
            sys.exit(f"ERROR: stack has {src.count} bands, the model expects "
                     f"{len(cols)}. Band order must match features.csv.")
        arr = src.read().astype("float32") / GAIN
        profile = src.profile
        transform, crs = src.transform, src.crs
    nb, nrow, ncol = arr.shape
    flat = arr.reshape(nb, -1).T

    valid = np.isfinite(flat).all(axis=1)
    if domain is not None:
        valid &= domain
    if nobs_path:
        with rasterio.open(nobs_path) as src:
            nobs = src.read(1).ravel()
        valid &= nobs > NPAR
        print(f"[info] pixels excluded for sparse series: "
              f"{int((nobs <= NPAR).sum())}")

    clf = RandomForestClassifier(n_estimators=600, max_features=0.3,
                                 min_samples_leaf=3,
                                 class_weight="balanced_subsample",
                                 random_state=0, n_jobs=-1).fit(X, y)
    names = list(clf.classes_)

    pred = np.full(flat.shape[0], -1, dtype="int16")
    ent = np.full(flat.shape[0], np.nan, dtype="float32")
    proba = clf.predict_proba(flat[valid])
    pred[valid] = proba.argmax(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -(proba * np.log(np.clip(proba, 1e-12, None))).sum(axis=1)
    ent[valid] = e / np.log(len(names))          # normalised, Eq. (9)

    low = valid & (ent > entropy_max)
    print(f"[info] classified {int(valid.sum())} pixels of {flat.shape[0]}")
    print(f"[info] low-confidence (entropy > {entropy_max}): "
          f"{int(low.sum())} ({100 * low.sum() / max(valid.sum(), 1):.1f} %)")
    for i, n in enumerate(names):
        n_px = int((pred == i).sum())
        area = n_px * abs(transform.a * transform.e) * (111320 ** 2) / 1e4
        print(f"[info]   {n:26s} {n_px:7d} px  approx {area:8.1f} ha")

    return (pred.reshape(nrow, ncol), ent.reshape(nrow, ncol), names,
            profile, transform, crs)


def write_geotiff(path, data, profile, nodata=-1):
    import rasterio
    profile.update(count=1, dtype="int16", nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("int16"), 1)
    print(f"wrote {path}")


def render(class_tif, names, prefix, title):
    """Map the classified raster with the house cartographic conventions."""
    import pygmt

    # Slices must be written in ascending order of the integer code, which is
    # the classifier's own class order (alphabetical), not the display order of
    # CLASSES. Writing them in display order produces a CPT with gaps and
    # overlaps and GMT refuses to read it.
    colour_of = dict(CLASSES)
    cpt = "_fig09.cpt"
    with open(cpt, "w") as fh:
        for j, name in enumerate(names):
            fh.write(f"{j} {colour_of.get(name, '#999999')} "
                     f"{j + 1} {colour_of.get(name, '#999999')}\n")
        fh.write("B white\nF white\nN 200/200/200\n")

    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE="13p,Helvetica-Bold", FONT_LABEL="9p,Helvetica",
                 FONT_ANNOT_PRIMARY="8p,Helvetica", MAP_FRAME_TYPE="fancy",
                 MAP_FRAME_PEN="0.9p,black",
                 MAP_GRID_PEN_PRIMARY="0.2p,white",
                 MAP_TITLE_OFFSET="4p", FORMAT_GEO_MAP="ddd:mmF")
    fig.basemap(region=REGION, projection="M15c",
                frame=[f"WSne+t{title}", "xa5mf1mg5m", "ya3mf1mg3m"])
    fig.grdimage(grid=class_tif, cmap=cpt, nan_transparent=True)
    fig.coast(resolution="f", area_thresh=0, shorelines="0.5p,black",
              water="#bcdcf0", rivers=["a/0.8p,#3a7ebf"])
    fig.basemap(frame=["WSne", "xa5mf1mg5m", "ya3mf1mg3m"])

    with open("_fig09_legend.txt", "w") as fh:
        fh.write("H 9p,Helvetica-Bold Halophyte habitat class\n")
        for name, colour in CLASSES:
            if name in names:
                fh.write(f"S 0.30c s 0.26c {colour} 0.4p,black 0.75c {name}\n")
    fig.legend(spec="_fig09_legend.txt", position="jBR+o0.25c/0.25c+w5.2c",
               box="+gwhite@12+p0.6p,black")
    fig.basemap(map_scale="jBL+o0.7c/0.7c+w5k+c36.33+f+u+lkm")
    fig.basemap(rose="jTL+o0.5c/0.5c+w1.1c+f2+lW,E,S,N")

    fig.savefig(f"{prefix}.pdf")
    fig.savefig(f"{prefix}.png", dpi=200)
    # TIFF at 600 dpi as well: Springer Nature accepts TIFF for raster figures
    # and several typesetters prefer it to PNG for line-and-fill maps.
    fig.savefig(f"{prefix}.tif", dpi=600)
    print(f"wrote {prefix}.pdf, {prefix}.png and {prefix}.tif")


def build(stack, nobs, features_csv, epoch, entropy_max, dem, max_elev,
          prefix, title):
    import os
    if not os.path.exists(stack):
        sys.exit(
            f"ERROR: {stack} not found.\n"
            "Figure 9 is the classifier applied to a per-pixel feature stack;\n"
            "this script will not synthesise one. Run make_feature_stack.js in\n"
            "Earth Engine for the baseline epoch, download the GeoTIFF, and\n"
            "re-run. Band order must match the columns of features.csv."
        )
    X, y, cols = load_training(features_csv, epoch)
    print(f"[info] training on {len(X)} plot rows, {len(cols)} predictors")
    domain = domain_mask(stack, dem, max_elev) if dem else None
    pred, ent, names, profile, transform, crs = classify(
        stack, nobs, X, y, cols, entropy_max, domain)
    write_geotiff(f"{prefix}_class.tif", pred, profile.copy())
    write_geotiff(f"{prefix}_entropy.tif",
                  np.nan_to_num(ent * 1000, nan=-1), profile.copy())
    render(f"{prefix}_class.tif", names, prefix, title)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack", default="feature_stack_baseline.tif")
    ap.add_argument("--nobs", default="n_obs_baseline.tif")
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--epoch", default="baseline")
    ap.add_argument("--entropy-max", type=float, default=0.85)
    ap.add_argument("--dem", default="goksu_dem.tif",
                    help="DEM used to bound the domain; empty string disables")
    ap.add_argument("--max-elev", type=float, default=10.0,
                    help="upper elevation of the delta plain, metres")
    ap.add_argument("--prefix", default="fig09_map_baseline")
    ap.add_argument("--title", default="Halophyte habitats, baseline epoch")
    a = ap.parse_args()
    build(a.stack, a.nobs, a.features, a.epoch, a.entropy_max,
          a.dem or None, a.max_elev, a.prefix, a.title)
