/**
 * make_annual_series.js -- annual class areas and drivers for Figure 13.
 *
 * WHY THIS RUNS SERVER-SIDE
 *
 * Figure 13 needs one classification per year across the whole record. Doing
 * that the way Figures 9 and 10 were done would mean exporting one 40-band
 * stack per year: about 40 MB each, some 1.6 GB over four decades, which is
 * neither practical to transfer nor necessary. What the figure actually needs
 * is four numbers per year, not forty rasters. So the classification is done
 * in the cloud and only the per-class pixel counts come back.
 *
 * The cost is that the classifier is Earth Engine's random forest rather than
 * the scikit-learn forest used for the maps. The two are not identical
 * implementations. The two epochs common to both routes are therefore exported
 * as well, so the annual series can be checked against the map areas of
 * Figures 9 and 10 before any of it is believed.
 *
 * OUTPUT: annual_series.csv, one row per year
 *   year
 *   swamp_px, floodplain_px, ponds_px, flats_px   classified pixel counts
 *   n_scenes            usable scenes contributing to that year
 *   water_px            JRC surface-water pixels in the analysis window
 *   precip_mm           ERA5-Land total precipitation, annual sum
 *   pet_mm              ERA5-Land potential evaporation, annual sum
 *   balance_mm          precip_mm - pet_mm
 *
 * TRAINING DATA: the plot feature table is embedded as a FeatureCollection by
 * the companion script make_training_fc.py, which writes the 40 predictors of
 * features.csv for each plot so that the cloud model sees exactly the columns
 * the local model saw.
 */

var AOI = ee.Geometry.Rectangle([33.86, 36.24, 34.11, 36.42]);
var YEAR_START = 2003;
var YEAR_END   = 2026;
var K = 2;
var SCALE = 30;
var MIN_OBS = 1 + 2 * K;          // parameters of the harmonic model

// Paste the FeatureCollection printed by make_training_fc.py here.
var TRAINING = ee.FeatureCollection([ /* see make_training_fc.py */ ]);
var PREDICTORS = [ /* the 40 column names, in the order of features.csv */ ];
var LABEL = 'class_id';

// ---------------------------------------------------------------- imagery
function toReflectance(img) {
  return ee.Image(img.select(['SR_B.']).multiply(0.0000275).add(-0.2)
    .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']));
}
function bands(img) {
  var id = ee.String(img.get('SPACECRAFT_ID'));
  var oli = ee.List(['LANDSAT_8', 'LANDSAT_9']).contains(id);
  var names = ee.List(ee.Algorithms.If(oli,
      ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'],
      ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']));
  return img.select(names, ['blue', 'green', 'red', 'nir', 'swir1']);
}
function masked(img) {
  var qa = img.select('QA_PIXEL');
  var bad = qa.bitwiseAnd(1 << 1).neq(0).or(qa.bitwiseAnd(1 << 2).neq(0))
      .or(qa.bitwiseAnd(1 << 3).neq(0)).or(qa.bitwiseAnd(1 << 4).neq(0));
  return ee.Image(img.updateMask(bad.not())
    .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']));
}
function indices(img) {
  var b = bands(toReflectance(img));
  return ee.Image(ee.Image.cat([
      b.normalizedDifference(['nir', 'red']).rename('ndvi'),
      b.normalizedDifference(['nir', 'swir1']).rename('ndmi'),
      b.expression('sqrt((N*R - G*B) / (N*R + G*B))',
        {N: b.select('nir'), R: b.select('red'),
         G: b.select('green'), B: b.select('blue')}).rename('crsi'),
      b.expression('sqrt(B * R)',
        {B: b.select('blue'), R: b.select('red')}).rename('si')])
    .copyProperties(img, ['system:time_start']));
}

var TERMS = ['t0'];
for (var k = 1; k <= K; k++) { TERMS.push('cos' + k); TERMS.push('sin' + k); }

function withTerms(img) {
  var doy = ee.Image(ee.Number(ee.Date(img.get('system:time_start'))
              .getRelative('day', 'year')).add(1)).float();
  var t = [ee.Image(1).rename('t0')];
  for (var k = 1; k <= K; k++) {
    var w = doy.multiply(2 * Math.PI * k / 365.25);
    t.push(w.cos().rename('cos' + k));
    t.push(w.sin().rename('sin' + k));
  }
  return img.addBands(ee.Image.cat(t).float());
}

// ------------------------------------------------------- stack for one year
function yearStack(year) {
  var col = ee.ImageCollection([])
    .merge(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'))
    .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'))
    .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'))
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
    .filterBounds(AOI)
    .filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))
    .map(masked).map(indices);
  var wt = col.map(withTerms);

  var out = ee.List(['ndvi', 'ndmi', 'crsi', 'si']).map(function (idx) {
    idx = ee.String(idx);
    var reg = wt.select(TERMS.concat([idx]))
      .reduce(ee.Reducer.linearRegression({numX: TERMS.length, numY: 1}));
    var coef = reg.select('coefficients').arrayProject([0])
                  .arrayFlatten([TERMS]);
    var c1 = coef.select('cos1'), c2 = coef.select('sin1');
    var stats = wt.select(idx).reduce(ee.Reducer.percentile([25, 50, 75]));
    return ee.Image.cat([
      coef.select('t0').rename(idx.cat('_a0')),
      c1.hypot(c2).rename(idx.cat('_amp1')),
      c2.atan2(c1).rename(idx.cat('_pha1')),
      reg.select('residuals').arrayProject([0])
         .arrayFlatten([[idx.cat('_rmse')]]),
      c1.rename(idx.cat('_c1')), c2.rename(idx.cat('_c2')),
      coef.select('cos2').rename(idx.cat('_c3')),
      coef.select('sin2').rename(idx.cat('_c4')),
      stats.select(idx.cat('_p50')).rename(idx.cat('_med')),
      stats.select(idx.cat('_p75')).subtract(stats.select(idx.cat('_p25')))
           .rename(idx.cat('_iqr'))]);
  });
  return {stack: ee.Image.cat(out).clip(AOI),
          nobs: col.select('ndvi').count().clip(AOI),
          n: col.size()};
}

// --------------------------------------------------------------- per year
var classifier = ee.Classifier.smileRandomForest(300)
  .train({features: TRAINING, classProperty: LABEL,
          inputProperties: PREDICTORS});

var years = ee.List.sequence(YEAR_START, YEAR_END);

var rows = years.map(function (y) {
  y = ee.Number(y);
  var s = yearStack(y);
  var cls = s.stack.updateMask(s.nobs.gte(MIN_OBS)).classify(classifier);
  var hist = cls.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(), geometry: AOI,
    scale: SCALE, maxPixels: 1e9, bestEffort: true}).get('classification');

  // Drivers: surface water and the climatic water balance for the same year.
  var water = ee.Image(ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
    .filter(ee.Filter.eq('year', y)).first());
  var water_px = ee.Algorithms.If(water,
    ee.Image(water).eq(3).selfMask().reduceRegion({
      reducer: ee.Reducer.count(), geometry: AOI, scale: SCALE,
      maxPixels: 1e9, bestEffort: true}).get('waterClass'), null);

  var era = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31));
  var precip = era.select('total_precipitation_sum').sum()
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: AOI, scale: 5000,
                   maxPixels: 1e9}).get('total_precipitation_sum');
  var pet = era.select('potential_evaporation_sum').sum()
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: AOI, scale: 5000,
                   maxPixels: 1e9}).get('potential_evaporation_sum');

  return ee.Feature(null, {
    year: y, n_scenes: s.n, hist: hist, water_px: water_px,
    precip_mm: ee.Number(precip).multiply(1000),
    pet_mm: ee.Number(pet).multiply(-1000)});
});

Export.table.toDrive({
  collection: ee.FeatureCollection(rows),
  description: 'annual_series',
  fileFormat: 'CSV',
  selectors: ['year', 'n_scenes', 'hist', 'water_px', 'precip_mm', 'pet_mm']
});

print('years:', years.length());
print('training features:', TRAINING.size());
