/**
 * make_feature_stack.js -- per-pixel predictor stack for Figures 9 and 10.
 *
 * Computes, for every pixel of the study area and for one epoch, exactly the
 * 40 predictors that build_features.py computes for the reference plots, in
 * exactly the same order. The classifier trained on the plots can then be
 * applied to the stack without any re-mapping of band names.
 *
 * BAND ORDER (must match features.csv column order):
 *   for each index in [ndvi, ndmi, crsi, si]:
 *     <idx>_a0    mean level of the harmonic fit
 *     <idx>_amp1  amplitude of the first harmonic
 *     <idx>_pha1  phase of the first harmonic, radians
 *     <idx>_rmse  root-mean-square residual of the fit
 *     <idx>_c1..c4  the four harmonic coefficients at order K = 2
 *     <idx>_med   median of the observed series
 *     <idx>_iqr   interquartile range of the observed series
 *
 * The harmonic fit is the same ordinary least squares as Eq. (5), solved
 * per pixel with ee.Reducer.linearRegression rather than per plot with
 * numpy.linalg.lstsq. The two agree to numerical precision on the same input.
 *
 * HOW TO RUN
 *   Set START and END to one epoch, run, then run the single export task.
 *   Repeat for the other epoch. Output is Int16 scaled by 10000 to keep the
 *   file small; fig09_map_baseline.py divides it back out.
 *
 * SIZE
 *   The study window at 30 m is roughly 740 x 670 pixels. Forty Int16 bands
 *   is about 40 MB, which Drive handles comfortably.
 */

var AOI = ee.Geometry.Rectangle([33.86, 36.24, 34.11, 36.42]);

var START = '2003-01-01';
var END   = '2007-12-31';
var EPOCH = 'baseline';          // used only in the export filename

var K = 2;                       // harmonic order; 1 + 2K coefficients
var SCALE = 30;
var GAIN = 10000;                // Int16 scaling

// ------------------------------------------------------------- reflectance
function toReflectance(img) {
  return ee.Image(img.select(['SR_B.'])
            .multiply(0.0000275).add(-0.2)
            .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']));
}

function bands(img) {
  var id  = ee.String(img.get('SPACECRAFT_ID'));
  var oli = ee.List(['LANDSAT_8', 'LANDSAT_9']).contains(id);
  var names = ee.List(ee.Algorithms.If(oli,
      ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'],
      ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']));
  return img.select(names, ['blue', 'green', 'red', 'nir', 'swir1']);
}

function masked(img) {
  var qa = img.select('QA_PIXEL');
  var bad = qa.bitwiseAnd(1 << 1).neq(0)
      .or(qa.bitwiseAnd(1 << 2).neq(0))
      .or(qa.bitwiseAnd(1 << 3).neq(0))
      .or(qa.bitwiseAnd(1 << 4).neq(0));
  return ee.Image(img.updateMask(bad.not())
           .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']));
}

function indices(img) {
  var b = bands(toReflectance(img));
  var ndvi = b.normalizedDifference(['nir', 'red']).rename('ndvi');
  var ndmi = b.normalizedDifference(['nir', 'swir1']).rename('ndmi');
  var crsi = b.expression('sqrt((N*R - G*B) / (N*R + G*B))',
      {N: b.select('nir'), R: b.select('red'),
       G: b.select('green'), B: b.select('blue')}).rename('crsi');
  var si = b.expression('sqrt(B * R)',
      {B: b.select('blue'), R: b.select('red')}).rename('si');
  return ee.Image(ee.Image.cat([ndvi, ndmi, crsi, si])
           .copyProperties(img, ['system:time_start']));
}

var col = ee.ImageCollection([])
  .merge(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'))
  .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'))
  .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'))
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(AOI).filterDate(START, END)
  .map(masked).map(indices);

// ---------------------------------------------------------- harmonic terms
// Attach the design matrix of Eq. (5) as bands, so linearRegression solves
// the same system that numpy.linalg.lstsq solves per plot.
function withTerms(img) {
  var doy = ee.Image(ee.Number(ee.Date(img.get('system:time_start'))
                .getRelative('day', 'year')).add(1)).float();
  var terms = [ee.Image(1).rename('t0')];
  for (var k = 1; k <= K; k++) {
    var w = doy.multiply(2 * Math.PI * k / 365.25);
    terms.push(w.cos().rename('cos' + k));
    terms.push(w.sin().rename('sin' + k));
  }
  return img.addBands(ee.Image.cat(terms).float());
}

var withT = col.map(withTerms);

var TERMS = ['t0'];
for (var k = 1; k <= K; k++) { TERMS.push('cos' + k); TERMS.push('sin' + k); }
var NPAR = TERMS.length;

function fitIndex(idx) {
  var reg = withT.select(TERMS.concat([idx]))
      .reduce(ee.Reducer.linearRegression({numX: NPAR, numY: 1}));
  // Coefficients arrive as an NPAR x 1 array image.
  var coef = reg.select('coefficients').arrayProject([0])
                .arrayFlatten([TERMS]);
  var a0 = coef.select('t0').rename(idx + '_a0');
  var c1 = coef.select('cos1'), c2 = coef.select('sin1');
  var c3 = coef.select('cos2'), c4 = coef.select('sin2');
  var amp1 = c1.hypot(c2).rename(idx + '_amp1');
  var pha1 = c2.atan2(c1).rename(idx + '_pha1');
  // linearRegression returns root-mean-square residual directly.
  var rmse = reg.select('residuals').arrayProject([0])
                .arrayFlatten([[idx + '_rmse']]);
  var stats = withT.select(idx).reduce(
      ee.Reducer.percentile([25, 50, 75]));
  var med = stats.select(idx + '_p50').rename(idx + '_med');
  var iqr = stats.select(idx + '_p75')
                 .subtract(stats.select(idx + '_p25')).rename(idx + '_iqr');
  return ee.Image.cat([
      a0, amp1, pha1, rmse,
      c1.rename(idx + '_c1'), c2.rename(idx + '_c2'),
      c3.rename(idx + '_c3'), c4.rename(idx + '_c4'),
      med, iqr]);
}

var stack = ee.Image.cat(['ndvi', 'ndmi', 'crsi', 'si'].map(fitIndex))
              .clip(AOI);

// Count of valid observations, so pixels too sparse to constrain the fit can
// be excluded rather than classified from noise (Sect. 2.4).
var nobs = col.select('ndvi').count().rename('n_obs').clip(AOI);

Export.image.toDrive({
  image: stack.multiply(GAIN).round().toInt16(),
  description: 'feature_stack_' + EPOCH,
  region: AOI,
  scale: SCALE,
  crs: 'EPSG:4326',
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: nobs.toInt16(),
  description: 'n_obs_' + EPOCH,
  region: AOI,
  scale: SCALE,
  crs: 'EPSG:4326',
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});

print('scenes in epoch:', col.size());
print('stack bands:', stack.bandNames());
