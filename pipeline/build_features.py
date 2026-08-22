#!/usr/bin/env python3
"""
build_features.py -- assemble the phenological and salinity feature stack.

Reads the per-plot index time series produced by make_plot_timeseries.js and
returns one feature row per plot per epoch: the harmonic coefficients of
Eq. (5) for each index, together with the derived amplitude and phase of the
first harmonic and the plain seasonal statistics.

The harmonic fit is ordinary least squares on the design matrix
[1, cos(2 pi k t / T), sin(2 pi k t / T)] for k = 1..K, solved per plot per
index. Plots with fewer valid observations than the model has parameters are
dropped rather than fitted, because a fit with no residual degrees of freedom
reports coefficients that are pure noise.

Usage
-----
    python3 build_features.py --csv plot_timeseries.csv --order 2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

INDICES = ["ndvi", "ndmi", "crsi", "si"]
T = 365.25

# Analysis epochs, matching Section 2.2 of the manuscript.
EPOCHS = {"baseline": ("2003-01-01", "2007-12-31"),
          "present": ("2021-01-01", "2026-12-31")}


def design(doy, order):
    """Harmonic design matrix for Eq. (5): [1, cos k, sin k] for k = 1..K."""
    cols = [np.ones_like(doy, dtype=float)]
    for k in range(1, order + 1):
        w = 2.0 * np.pi * k * doy / T
        cols += [np.cos(w), np.sin(w)]
    return np.column_stack(cols)


def harmonic_fit(doy, y, order):
    """Least-squares harmonic coefficients, amplitude and phase of harmonic 1."""
    n_par = 1 + 2 * order
    ok = np.isfinite(y)
    if ok.sum() <= n_par:          # no residual degrees of freedom
        return None
    X = design(doy[ok], order)
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid = y[ok] - X @ beta
    rmse = float(np.sqrt((resid ** 2).mean()))
    a1, b1 = beta[1], beta[2]
    return {"a0": beta[0], "amp1": float(np.hypot(a1, b1)),
            "pha1": float(np.arctan2(b1, a1)), "rmse": rmse,
            **{f"c{i}": float(v) for i, v in enumerate(beta[1:], start=1)}}


def build(csv_path, order):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["epoch"] = pd.NA
    for name, (lo, hi) in EPOCHS.items():
        df.loc[(df["date"] >= lo) & (df["date"] <= hi), "epoch"] = name
    df = df.dropna(subset=["epoch"])

    rows = []
    for (plot, epoch), g in df.groupby(["plot", "epoch"], sort=True):
        rec = {"plot": plot, "epoch": epoch,
               "subtype": g["subtype"].iloc[0], "n_obs": len(g)}
        usable = True
        for idx in INDICES:
            fit = harmonic_fit(g["doy"].to_numpy(float),
                               g[idx].to_numpy(float), order)
            if fit is None:
                usable = False
                break
            rec.update({f"{idx}_{k}": v for k, v in fit.items()})
            # Plain seasonal statistics alongside the harmonic description.
            rec[f"{idx}_med"] = float(g[idx].median())
            rec[f"{idx}_iqr"] = float(g[idx].quantile(.75) - g[idx].quantile(.25))
        if usable:
            rows.append(rec)

    out = pd.DataFrame(rows)
    print(f"[info] {len(out)} plot-epoch feature rows from {len(df)} observations")
    print(f"[info] {out.shape[1] - 4} predictors per row (harmonic order {order})")
    dropped = df.groupby(["plot", "epoch"]).ngroups - len(out)
    print(f"[info] dropped for insufficient observations: {dropped}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="plot_timeseries.csv")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--out", default="features.csv")
    a = ap.parse_args()
    build(a.csv, a.order).to_csv(a.out, index=False)
    print(f"wrote {a.out}")
