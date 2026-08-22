#!/usr/bin/env python3
"""
A1_preprocessing.py -- scene selection, cloud screening and index computation.

Listing A1 of the manuscript. Reads Landsat Collection 2 Level-2 scenes over the
study window, applies the per-pixel quality mask, converts to surface
reflectance and computes the four spectral indices that the harmonic model in
A2_harmonic.py is fitted to.

Two points of substance.

Scaling. Collection 2 Level-2 stores reflectance as scaled integers. The
conversion DN * 0.0000275 - 0.2 must be applied before any index is formed;
CRSI is a ratio of products of reflectances and is meaningless on raw DN.

Band roles, not band numbers. TM and ETM+ place blue at SR_B1 while OLI places
it at SR_B2. Selecting by number across a merged collection silently reads red
where near-infrared was intended, which produces plausible values and wrong
results. Bands are therefore selected by role.

Cross-sensor harmonisation is provided but not exercised in the reported
analysis: both epochs are Landsat, so no transfer function is applied. The
coefficients are retained for the Sentinel-2 extension described in the
discussion, where they would be required.

Usage
-----
    python3 A1_preprocessing.py --stack scenes/ --out indices.tif
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import rasterio

# Collection 2 Level-2 reflectance scaling.
SCALE, OFFSET = 0.0000275, -0.2

# Band role to band name, per instrument.
ROLES = {
    "TM":  {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3",
            "nir": "SR_B4", "swir1": "SR_B5"},
    "OLI": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
            "nir": "SR_B5", "swir1": "SR_B6"},
}
OLI_PLATFORMS = {"LANDSAT_8", "LANDSAT_9"}

# QA_PIXEL bits marking unusable pixels.
QA_BITS = (1, 2, 3, 4)      # dilated cloud, cirrus, cloud, cloud shadow

# Roy et al. (2016) OLS transfer functions, Landsat 8 OLI to ETM+ equivalent.
# Retained for a Sentinel-2 or cross-sensor extension; unused when both epochs
# come from the same instrument family.
HARMONISATION = {
    "blue":  (0.0003, 0.8474), "green": (0.0088, 0.8483),
    "red":   (0.0061, 0.9047), "nir":   (0.0412, 0.8462),
    "swir1": (0.0254, 0.8937),
}


def quality_mask(qa):
    """True where the pixel is usable."""
    bad = np.zeros(qa.shape, dtype=bool)
    for b in QA_BITS:
        bad |= (qa.astype(np.uint16) & (1 << b)) != 0
    return ~bad


def to_reflectance(dn):
    return dn.astype("float32") * SCALE + OFFSET


def harmonise(refl, role):
    """Apply the transfer function for one band role. Not used for a
    single-instrument analysis; see the module docstring."""
    a, b = HARMONISATION[role]
    return a + b * refl


def indices(b):
    """NDVI, NDMI, CRSI and SI from a dict of reflectance arrays."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b["nir"] - b["red"]) / (b["nir"] + b["red"])
        ndmi = (b["nir"] - b["swir1"]) / (b["nir"] + b["swir1"])
        num = b["nir"] * b["red"] - b["green"] * b["blue"]
        den = b["nir"] * b["red"] + b["green"] * b["blue"]
        crsi = np.sqrt(np.clip(num / den, 0, None))
        si = np.sqrt(np.clip(b["blue"] * b["red"], 0, None))
    return {"ndvi": ndvi, "ndmi": ndmi, "crsi": crsi, "si": si}


def read_scene(directory, platform, apply_harmonisation=False):
    """One scene directory of per-band GeoTIFFs to masked index arrays."""
    role_map = ROLES["OLI" if platform in OLI_PLATFORMS else "TM"]
    bands, profile = {}, None
    for role, name in role_map.items():
        hits = glob.glob(os.path.join(directory, f"*{name}.TIF")) \
            or glob.glob(os.path.join(directory, f"*{name}.tif"))
        if not hits:
            raise FileNotFoundError(f"{name} missing in {directory}")
        with rasterio.open(hits[0]) as src:
            arr = to_reflectance(src.read(1))
            profile = profile or src.profile
        bands[role] = harmonise(arr, role) if apply_harmonisation else arr

    qa_hits = glob.glob(os.path.join(directory, "*QA_PIXEL.TIF")) \
        or glob.glob(os.path.join(directory, "*QA_PIXEL.tif"))
    with rasterio.open(qa_hits[0]) as src:
        good = quality_mask(src.read(1))

    out = indices(bands)
    for k in out:
        out[k] = np.where(good, out[k], np.nan).astype("float32")
    return out, profile, float(good.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="scene directory")
    ap.add_argument("--platform", default="LANDSAT_8")
    ap.add_argument("--harmonise", action="store_true")
    ap.add_argument("--out", default="indices.tif")
    a = ap.parse_args()

    idx, profile, usable = read_scene(a.scene, a.platform, a.harmonise)
    print(f"[info] usable fraction of the scene footprint: {100 * usable:.1f} %")
    profile.update(count=len(idx), dtype="float32", nodata=np.nan,
                   compress="deflate")
    with rasterio.open(a.out, "w", **profile) as dst:
        for i, (name, arr) in enumerate(idx.items(), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, name)
    print(f"wrote {a.out} with bands {list(idx)}")


if __name__ == "__main__":
    main()
