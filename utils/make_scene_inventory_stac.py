#!/usr/bin/env python3
"""
make_scene_inventory_stac.py -- scene inventory for Figure 4, without Earth Engine.

Queries two public STAC APIs. Neither needs an account, a login, an API key or
a billing profile:

  Sentinel-2 L2A   Earth Search (Element 84, AWS open data)
                   https://earth-search.aws.element84.com/v1
  Landsat C2 L2    Microsoft Planetary Computer
                   https://planetarycomputer.microsoft.com/api/stac/v1
                   (covers Landsat 4, 5, 7, 8 and 9, so the whole archive back
                   to the 1980s; Earth Search's Landsat collection does not go
                   as far back, which is why the two catalogues are mixed)

TWO MODES
---------
--mode metadata  (default, fast, no downloads)
    Records the scene-level cloud cover published in the STAC item
    (eo:cloud_cover). This is computed over the WHOLE scene or tile. The delta
    is small and can sit near a tile edge, so this number is only a proxy for
    how usable a scene actually is over the study area.

--mode footprint  (slower, downloads a few MB per scene)
    Additionally reads the quality band over the study-area window only --
    Landsat QA_PIXEL, Sentinel-2 SCL -- through a windowed COG read, and
    computes the true fraction of the study area that survives masking. This
    is what the Earth Engine version did. Expect a few seconds per scene.

Run metadata mode first to see how many scenes there are, then footprint mode
on the epochs you actually need.

INSTALL
-------
    pip install pystac-client planetary-computer rasterio pandas

USAGE
-----
    python3 make_scene_inventory_stac.py --start 2003-01-01 --end 2007-12-31
    python3 make_scene_inventory_stac.py --start 2021-01-01 --end 2026-12-31 \
            --mode footprint --out scene_inventory_present.csv

Then concatenate the CSVs (keeping one header) into scene_inventory.csv and
hand that to fig04_image_inventory.py.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

AOI = [33.86, 36.24, 34.11, 36.42]          # W, S, E, N -- the study window

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Platform strings -> the sensor labels fig04_image_inventory.py expects.
SENSOR = {
    "landsat-4": "Landsat 4 TM", "landsat-5": "Landsat 5 TM",
    "landsat-7": "Landsat 7 ETM+", "landsat-8": "Landsat 8 OLI",
    "landsat-9": "Landsat 9 OLI-2",
    "sentinel-2a": "Sentinel-2A", "sentinel-2b": "Sentinel-2B",
    "sentinel-2c": "Sentinel-2C",
}

# Landsat Collection 2 QA_PIXEL bits that mark unusable pixels.
LS_BAD_BITS = (1, 2, 3, 4)      # dilated cloud, cirrus, cloud, cloud shadow
# Sentinel-2 SCL classes that mark unusable pixels.
# 3 shadow, 8 cloud medium, 9 cloud high, 10 cirrus. Class 11 (snow) is also
# excluded: over this delta it fires almost exclusively on bright salt crust.
S2_BAD_CLASSES = (0, 1, 3, 8, 9, 10)


def label(item):
    plat = str(item.properties.get("platform", "")).lower().replace("_", "-")
    return SENSOR.get(plat, plat or "unknown")


def usable_from_qa(href, bad_test, aoi):
    """Windowed COG read of a quality band; returns per cent usable over AOI."""
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *aoi)
        win = from_bounds(left, bottom, right, top, src.transform)
        arr = src.read(1, window=win, boundless=True, fill_value=0)
    if arr.size == 0:
        return float("nan")
    return float((~bad_test(arr)).mean() * 100.0)


def ls_bad(a):
    import numpy as np
    bad = np.zeros(a.shape, bool)
    for b in LS_BAD_BITS:
        bad |= (a & (1 << b)) != 0
    return bad


def s2_bad(a):
    import numpy as np
    return np.isin(a, S2_BAD_CLASSES)


def collect(start, end, mode, limit):
    from pystac_client import Client
    import planetary_computer as pc

    rows = []

    # ---- Sentinel-2 via Earth Search -------------------------------------
    print("querying Earth Search for Sentinel-2 L2A ...", file=sys.stderr)
    es = Client.open(EARTH_SEARCH)
    s2 = es.search(collections=["sentinel-2-l2a"], bbox=AOI,
                   datetime=f"{start}/{end}", limit=100)
    for i, it in enumerate(s2.items()):
        if limit and i >= limit:
            break
        r = {"sensor": label(it),
             "date": it.datetime.date().isoformat(),
             "cloud_pct": it.properties.get("eo:cloud_cover"),
             "usable_pct": float("nan")}
        if mode == "footprint" and "scl" in it.assets:
            try:
                r["usable_pct"] = usable_from_qa(it.assets["scl"].href,
                                                 s2_bad, AOI)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {it.id}: {e}", file=sys.stderr)
        rows.append(r)
    print(f"  {sum(r['sensor'].startswith('Sentinel') for r in rows)} scenes",
          file=sys.stderr)

    # ---- Landsat via Planetary Computer ----------------------------------
    print("querying Planetary Computer for Landsat C2 L2 ...", file=sys.stderr)
    mpc = Client.open(PLANETARY, modifier=pc.sign_inplace)
    ls = mpc.search(collections=["landsat-c2-l2"], bbox=AOI,
                    datetime=f"{start}/{end}", limit=100)
    n0 = len(rows)
    for i, it in enumerate(ls.items()):
        if limit and i >= limit:
            break
        r = {"sensor": label(it),
             "date": it.datetime.date().isoformat(),
             "cloud_pct": it.properties.get("eo:cloud_cover"),
             "usable_pct": float("nan")}
        if mode == "footprint" and "qa_pixel" in it.assets:
            try:
                r["usable_pct"] = usable_from_qa(it.assets["qa_pixel"].href,
                                                 ls_bad, AOI)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {it.id}: {e}", file=sys.stderr)
        rows.append(r)
    print(f"  {len(rows) - n0} scenes", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--end", default="2007-12-31")
    ap.add_argument("--mode", choices=["metadata", "footprint"],
                    default="metadata")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N scenes per catalogue (0 = no limit)")
    ap.add_argument("--out", default="scene_inventory.csv")
    a = ap.parse_args()

    rows = collect(a.start, a.end, a.mode, a.limit)
    if not rows:
        sys.exit("No scenes returned. Check the dates and the bounding box.")

    df = pd.DataFrame(rows).sort_values(["date", "sensor"])
    if a.mode == "metadata":
        # The figure needs a usable_pct column. In metadata mode the only
        # honest value is the complement of published cloud cover, and the
        # figure credit must say so rather than implying a footprint-level
        # computation that was never done.
        df["usable_pct"] = 100.0 - df["cloud_pct"].astype(float)
        print("\nNOTE: metadata mode. usable_pct = 100 - scene cloud cover,",
              file=sys.stderr)
        print("      i.e. a whole-scene figure, not a study-area one. Say so",
              file=sys.stderr)
        print("      in the caption, or re-run with --mode footprint.",
              file=sys.stderr)

    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(df)} scenes, "
          f"{df['date'].min()} to {df['date'].max()}")
    print(df["sensor"].value_counts().to_string())


if __name__ == "__main__":
    main()
