# Göksu Delta halophyte mapping — analysis scripts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22059005.svg)](https://doi.org/10.5281/zenodo.22059005)

Scripts accompanying *Distribution of* Salicornietea fruticosae *and* Juncetea maritimi *communities in the
Göksu Delta (Türkiye) mapped from Landsat time series* (Lemenkova, submitted to *Hacquetia*).

Every script runs from open-access inputs. No fieldwork data and no restricted
datasets are required.

## Layout

```
appendix/      the four listings printed in Appendix A
earth_engine/  JavaScript run in the Earth Engine code editor
pipeline/      the analysis chain, in run order
figures/       one script per manuscript figure
utils/         supporting tools
```

## Run order

The chain is linear. Each step writes a file the next one reads.

| # | Where | Script | Produces |
|---|---|---|---|
| 1 | `earth_engine/` | `make_scene_inventory.js` | `scene_inventory_*.csv` — one row per archive scene |
| 2 | `pipeline/` | `make_plots_csv.py` | `reference_plots.csv` — 281 survey plots |
| 3 | `earth_engine/` | `make_plot_timeseries.js` | `plot_timeseries.csv` — index series at each plot |
| 4 | `pipeline/` | `build_features.py` | `features.csv` — 290 rows, 40 predictors |
| 5 | `pipeline/` | `train_classifier.py` | accuracy under spatially blocked folds |
| 6 | `earth_engine/` | `make_feature_stack.js` | `feature_stack_*.tif`, `n_obs_*.tif` |
| 7 | `figures/` | `fig09_map_baseline.py` | classified and entropy rasters, habitat map |

Steps 1, 3 and 6 run in the browser at `code.earthengine.google.com`; the rest
run locally.

`fig09_map_baseline.py` produces both epoch maps:

```bash
python3 fig09_map_baseline.py --stack feature_stack_baseline.tif \
    --nobs n_obs_baseline.tif
python3 fig09_map_baseline.py --stack feature_stack_present.tif \
    --nobs n_obs_present.tif --epoch present --prefix fig10_map_present \
    --title "Halophyte habitats, present epoch"
```

## Figures

Figures 1, 3, 9, 10 and 11 are PyGMT; the rest are Matplotlib. Each writes PDF
for LaTeX and PNG at 600 dpi (200 dpi for the PyGMT maps), and prints its own
layout checks — legend-over-data, overlapping labels, panel geometry — on every
run.

Two need inputs that were not produced for the submitted version:

- `fig13_timeseries.py` needs `annual_series.csv` from
  `earth_engine/make_annual_series.js`, which classifies each year in the cloud.
  The corresponding figure is not in the manuscript.
- `fig14_uncertainty.py` is Figure 13 in the manuscript; the file name kept its
  original number.

## Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib rasterio pygmt requests
```

GMT 6.6.0 and PyGMT 0.18.0 for the maps. `utils/verify_dois.py` additionally
needs `requests`; `utils/make_scene_inventory_stac.py` needs `pystac-client`
and `planetary-computer`.

## Utilities

`make_scene_inventory_stac.py` retrieves the scene inventory from public STAC
APIs (Earth Search, Planetary Computer) instead of Earth Engine, for anyone
without a registered Cloud project. It has not been run against the live
endpoints.

`verify_dois.py` fills and checks bibliography DOIs against CrossRef, scoring
each match on title similarity and reporting weak matches rather than inserting
them.

## Reproducibility notes

Three details are load-bearing and easy to break:

**Band order.** The per-pixel feature stack must present its 40 bands in the
same order as the columns of `features.csv`. A classifier stores positions, not
names, so a reordered stack is read without error and produces a plausible,
wrong map. `fig09_map_baseline.py` checks the band count but cannot detect
reordering.

**Band roles.** TM and ETM+ place blue at `SR_B1`, OLI at `SR_B2`. Selecting by
number across a merged collection reads the wrong band silently.

**Spatial blocking.** Plots of one association cluster within metres. A random
train/test split reports memorisation as accuracy. All reported figures use
2 km blocks assigned whole to folds.

## Requirements

Python 3.12 with the packages pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The Earth Engine steps (`earth_engine/*.js`) run in the browser at
`code.earthengine.google.com` and need no local installation.

## Data

`data/` holds the small intermediate tables the pipeline reads and writes
(scene inventories, plot time series, the 290-row feature table, the transect).
All primary inputs are open access and are listed in the paper: Landsat
Collection 2 Level-2 (USGS), Copernicus WorldDEM-30 (ESA/Airbus), GSHHG (NOAA),
Global Surface Water (EC JRC) and ERA5-Land (ECMWF/C3S). Large rasters (DEMs,
feature stacks, classified maps) are **not** stored in Git; they are reproduced
by the scripts and, where useful, archived on Zenodo alongside the release.

## License

Code is released under the MIT License (see `LICENSE`). The accompanying data
tables in `data/` are released under CC BY 4.0.

## How to cite

Please cite both the software (via the Zenodo DOI in the badge above, once
minted) and the article. Citation metadata is in `CITATION.cff`.

## Funding

Supported by TÜBİTAK, BİDEB Science Fellowships and Grant Programmes
(grant 2221).
