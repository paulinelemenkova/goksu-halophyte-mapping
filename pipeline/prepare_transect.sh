#!/bin/sh
# prepare_transect.sh -- extract the Figure 2 cross-shore profile.
#
# Samples the Copernicus WorldDEM-30 grid and a GSHHG full-resolution land
# mask along a north-south line at 33.956 E, from open sea at 36.235 N to the
# inland margin of the delta plain at 36.385 N. Output: transect.csv with
# columns dist_km, elev_m, land (1 = land, 0 = water).
set -e
DEM=${1:-goksu_dem.tif}
gmt grdlandmask -R33.82/34.14/36.22/36.42 -I3s -Df -Gmask.nc -N0/1/0/1/0
gmt project -C33.956/36.235 -E33.956/36.385 -G0.02 -Q > tr.txt
gmt grdtrack tr.txt -G"$DEM" > tr_z.txt
gmt grdtrack tr.txt -Gmask.nc  > tr_m.txt
python3 - <<'PY'
import numpy as np
z = np.loadtxt("tr_z.txt"); m = np.loadtxt("tr_m.txt")
land = (m[:, 3] > 0.5).astype(int)          # mask is interpolated; threshold it
np.savetxt("transect.csv", np.column_stack([z[:, 2], z[:, 3], land]),
           delimiter=",", header="dist_km,elev_m,land", comments="",
           fmt="%.4f,%.3f,%d")
print("transect.csv:", len(land), "points, max elevation %.2f m" % z[:, 3].max())
PY
