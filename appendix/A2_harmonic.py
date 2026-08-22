#!/usr/bin/env python3
"""
A2_harmonic.py -- harmonic phenology fitting and coefficient extraction.

Listing A2 of the manuscript. Fits Eq. (5) to each index series and returns the
predictor table used for classification: for every plot and epoch, the harmonic
coefficients of each of the four indices together with the amplitude and phase
of the first harmonic, the residual of the fit, and the median and
interquartile range of the observed series.

The fit is ordinary least squares on the design matrix
[1, cos(2 pi k t / T), sin(2 pi k t / T)] for k = 1..K, solved independently
per plot per index. A series with no residual degrees of freedom is discarded
rather than fitted: with 1 + 2K parameters and 1 + 2K observations the fit is
exact and its coefficients carry no information.

Day of year is used as the time variable rather than absolute date, so
observations from different years of one epoch stack onto a single annual
cycle. This is what allows a five-year epoch with sparse winter coverage to
constrain a seasonal model at all.

Usage
-----
    python3 A2_harmonic.py --csv plot_timeseries.csv --order 2 --out features.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

INDICES = ["ndvi", "ndmi", "crsi", "si"]
T = 365.25

EPOCHS = {"baseline": ("2003-01-01", "2007-12-31"),
          "present": ("2021-01-01", "2026-12-31")}


def design(doy, order):
    """Design matrix of Eq. (5): [1, cos k, sin k] for k = 1..K."""
    cols = [np.ones_like(doy, dtype=float)]
    for k in range(1, order + 1):
        w = 2.0 * np.pi * k * doy / T
        cols += [np.cos(w), np.sin(w)]
    return np.column_stack(cols)


def harmonic_fit(doy, y, order):
    """Coefficients, amplitude and phase of harmonic 1, and the fit residual."""
    n_par = 1 + 2 * order
    ok = np.isfinite(y)
    if ok.sum() <= n_par:                      # no residual degrees of freedom
        return None
    X = design(doy[ok], order)
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid = y[ok] - X @ beta
    a1, b1 = beta[1], beta[2]
    out = {"a0": float(beta[0]),
           "amp1": float(np.hypot(a1, b1)),
           "pha1": float(np.arctan2(b1, a1)),
           "rmse": float(np.sqrt((resid ** 2).mean()))}
    out.update({f"c{i}": float(v) for i, v in enumerate(beta[1:], start=1)})
    return out


def assign_epoch(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["epoch"] = pd.NA
    for name, (lo, hi) in EPOCHS.items():
        df.loc[(df["date"] >= lo) & (df["date"] <= hi), "epoch"] = name
    return df.dropna(subset=["epoch"])


def build(csv_path, order):
    df = assign_epoch(pd.read_csv(csv_path))
    rows, dropped = [], 0
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
            rec[f"{idx}_med"] = float(g[idx].median())
            rec[f"{idx}_iqr"] = float(g[idx].quantile(.75)
                                      - g[idx].quantile(.25))
        if usable:
            rows.append(rec)
        else:
            dropped += 1

    out = pd.DataFrame(rows)
    n_pred = out.shape[1] - 4
    print(f"[info] {len(out)} plot-epoch rows, {n_pred} predictors "
          f"(order {order}); {dropped} dropped for insufficient observations")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="plot_timeseries.csv")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--out", default="features.csv")
    a = ap.parse_args()
    build(a.csv, a.order).to_csv(a.out, index=False)
    print(f"wrote {a.out}")
