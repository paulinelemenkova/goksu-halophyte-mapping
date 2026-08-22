/**
 * make_scene_inventory.js -- Earth Engine export for Figure 4.
 *
 * Builds the scene inventory the figure needs: one row per archive scene
 * intersecting the Goksu Delta, with sensor, acquisition date, scene-level
 * cloud cover, and the fraction of the study-area footprint that survives
 * cloud/shadow/cirrus masking. That last column is the one that matters --
 * scene-level cloud cover is reported over the whole tile, which for a small
 * delta at the tile edge can be wildly pessimistic or optimistic.
 *
 * HOW TO RUN
 *   1. Open https://code.earthengine.google.com and paste this in.
 *   2. Run. Two tasks appear under the Tasks tab; run both.
 *   3. Download scene_inventory_landsat.csv and scene_inventory_sentinel2.csv
 *      from Drive, concatenate them, and hand the result to
 *      fig04_image_inventory.py as scene_inventory.csv.
 *
 * OUTPUT COLUMNS  (exactly what the figure script expects)
 *   sensor      string   'Landsat 5 TM' | 'Landsat 7 ETM+' | 'Landsat 8 OLI'
 *                        | 'Landsat 9 OLI-2' | 'Sentinel-2A' | 'Sentinel-2B'
 *   date        string   ISO yyyy-MM-dd
 *   cloud_pct   number   scene-level cloud cover, per cent
 *   usable_pct  number   per cent of the AOI passing the mask, 0-100
 */

// ---------------------------------------------------------------- study area
var AOI = ee.Geometry.Rectangle([33.86, 36.24, 34.11, 36.42]);
var START = '1984-01-01';
var END   = '2026-12-31';

// ------------------------------------------------------------------ Landsat
// Collection 2 Level-2 surface reflectance. QA_PIXEL bits: 1 dilated cloud,
// 2 cirrus, 3 cloud, 4 cloud shadow.
function landsatUsable(img) {
  var qa = img.select('QA_PIXEL');
  var bad = qa.bitwiseAnd(1 << 1).neq(0)
      .or(qa.bitwiseAnd(1 << 2).neq(0))
      .or(qa.bitwiseAnd(1 << 3).neq(0))
      .or(qa.bitwiseAnd(1 << 4).neq(0));
  var frac = bad.not().rename('u').reduceRegion({
    reducer: ee.Reducer.mean(), geometry: AOI, scale: 30, maxPixels: 1e9,
    bestEffort: true
  }).get('u');
  return img.set('usable_pct', ee.Number(frac).multiply(100));
}

var LS_NAMES = {
  'LANDSAT_5': 'Landsat 5 TM',
  'LANDSAT_7': 'Landsat 7 ETM+',
  'LANDSAT_8': 'Landsat 8 OLI',
  'LANDSAT_9': 'Landsat 9 OLI-2'
};

var landsat = ee.ImageCollection([])
  .merge(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'))
  .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'))
  .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'))
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(AOI).filterDate(START, END)
  .map(landsatUsable);

var landsatRows = landsat.map(function (img) {
  var sc = ee.String(img.get('SPACECRAFT_ID'));
  return ee.Feature(null, {
    sensor: ee.Dictionary(LS_NAMES).get(sc, sc),
    date: img.date().format('YYYY-MM-dd'),
    cloud_pct: img.get('CLOUD_COVER'),
    usable_pct: img.get('usable_pct')
  });
});

// --------------------------------------------------------------- Sentinel-2
// Harmonised L2A. Cloud Score+ is used rather than the SCL band: SCL badly
// over-flags bright saline crusts as cloud, which is exactly the surface this
// study cares about.
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var CS_BAND = 'cs_cdf';
var CS_THRESH = 0.60;

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI).filterDate('2015-06-23', END)
  .linkCollection(csPlus, [CS_BAND])
  .map(function (img) {
    var frac = img.select(CS_BAND).gte(CS_THRESH).rename('u').reduceRegion({
      reducer: ee.Reducer.mean(), geometry: AOI, scale: 20, maxPixels: 1e9,
      bestEffort: true
    }).get('u');
    return img.set('usable_pct', ee.Number(frac).multiply(100));
  });

var s2Rows = s2.map(function (img) {
  var plat = ee.String(img.get('SPACECRAFT_NAME'));  // e.g. 'Sentinel-2A'
  return ee.Feature(null, {
    sensor: plat,
    date: img.date().format('YYYY-MM-dd'),
    cloud_pct: img.get('CLOUDY_PIXEL_PERCENTAGE'),
    usable_pct: img.get('usable_pct')
  });
});

// ------------------------------------------------------------------ exports
var COLS = ['sensor', 'date', 'cloud_pct', 'usable_pct'];

Export.table.toDrive({
  collection: landsatRows,
  description: 'scene_inventory_landsat',
  fileFormat: 'CSV',
  selectors: COLS
});

Export.table.toDrive({
  collection: s2Rows,
  description: 'scene_inventory_sentinel2',
  fileFormat: 'CSV',
  selectors: COLS
});

print('Landsat scenes intersecting the AOI:', landsat.size());
print('Sentinel-2 scenes intersecting the AOI:', s2.size());
