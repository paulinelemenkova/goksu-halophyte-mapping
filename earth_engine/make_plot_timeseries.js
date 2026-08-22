/**
 * make_plot_timeseries.js -- index time series at the reference plots (Figure 5).
 *
 * Samples NDVI, NDMI and two salinity indices at each halophytic releve plot,
 * for every Landsat scene passing the cloud mask, and exports one row per
 * plot per date. This is the input fig05_phenology_curves.py needs.
 *
 * The 149 plot coordinates are embedded below, so nothing has to be uploaded
 * as an asset. They are the halophytic subset of the published survey, already
 * screened against the study-area bounding box.
 *
 * OUTPUT COLUMNS
 *   plot     integer   plot number from the source survey
 *   subtype  string    Swamp | Temporary ponds | Temporary floodplain
 *                      | Terrestrial saline flats
 *   sensor   string    spacecraft id
 *   date     string    ISO yyyy-MM-dd
 *   doy      integer   day of year, 1-366
 *   ndvi, ndmi, crsi, si   numbers
 *
 * Run once per epoch, as for the scene inventory, and concatenate the CSVs.
 */

var START = '2003-01-01';
var END   = '2007-12-31';

// ---------------------------------------------------------------- plots
function f(id, lon, lat, sub) {
  return ee.Feature(ee.Geometry.Point([lon, lat]),
                    {plot: id, subtype: sub});
}

var PLOTS = ee.FeatureCollection([
  f(102,33.93553,36.29375,'Swamp'),
  f(146,33.94215,36.28743,'Swamp'),
  f(121,33.93795,36.29233,'Swamp'),
  f(145,33.94273,36.28682,'Swamp'),
  f(446,33.96310,36.28558,'Swamp'),
  f(423,33.97470,36.28323,'Swamp'),
  f(246,33.95938,36.28502,'Swamp'),
  f(501,34.02830,36.31105,'Swamp'),
  f(411,33.97538,36.28200,'Swamp'),
  f(502,34.02878,36.31122,'Swamp'),
  f(250,33.96362,36.28598,'Swamp'),
  f(503,34.02892,36.31125,'Swamp'),
  f(444,33.97827,36.28512,'Swamp'),
  f(254,33.97462,36.27738,'Swamp'),
  f(447,33.96350,36.28578,'Swamp'),
  f(255,33.97460,36.31138,'Swamp'),
  f(267,33.97875,36.29595,'Swamp'),
  f(438,33.97783,36.28417,'Swamp'),
  f(259,33.99112,36.28292,'Swamp'),
  f(696,34.04188,36.30805,'Swamp'),
  f(420,33.97442,36.28262,'Swamp'),
  f(695,34.04265,36.30753,'Swamp'),
  f(384,33.95378,36.28268,'Swamp'),
  f(414,33.97597,36.28235,'Swamp'),
  f(450,33.96610,36.28685,'Swamp'),
  f(382,33.95393,36.28223,'Swamp'),
  f(399,33.95625,36.28325,'Swamp'),
  f(421,33.97448,36.28362,'Swamp'),
  f(425,33.97510,36.28372,'Swamp'),
  f(398,33.95687,36.28390,'Swamp'),
  f(426,33.97477,36.28357,'Swamp'),
  f(383,33.95385,36.28248,'Swamp'),
  f(430,33.97608,36.28422,'Swamp'),
  f(437,33.97763,36.28413,'Swamp'),
  f(662,34.04622,36.30013,'Temporary ponds'),
  f(150,33.94452,36.28493,'Temporary ponds'),
  f(132,33.93893,36.29007,'Temporary ponds'),
  f(10,33.92292,36.30035,'Temporary ponds'),
  f(144,33.94277,36.28705,'Temporary ponds'),
  f(515,34.02773,36.30342,'Temporary ponds'),
  f(288,34.07635,36.37297,'Temporary ponds'),
  f(108,33.91785,36.29117,'Temporary ponds'),
  f(470,34.00782,36.31333,'Temporary ponds'),
  f(47,33.92592,36.29872,'Temporary ponds'),
  f(557,34.05128,36.30162,'Temporary ponds'),
  f(569,34.05392,36.30580,'Temporary ponds'),
  f(596,34.07245,36.31975,'Temporary ponds'),
  f(527,34.07412,36.33122,'Temporary ponds'),
  f(663,34.04583,36.29833,'Temporary ponds'),
  f(622,33.97418,36.31158,'Temporary ponds'),
  f(588,34.06543,36.31220,'Temporary ponds'),
  f(580,34.06122,36.30952,'Temporary ponds'),
  f(550,34.04967,36.29957,'Temporary ponds'),
  f(597,34.06960,36.31958,'Temporary ponds'),
  f(603,33.95560,36.31827,'Temporary ponds'),
  f(469,34.00725,36.31338,'Temporary ponds'),
  f(618,33.96755,36.31748,'Temporary ponds'),
  f(534,34.07088,36.32925,'Temporary ponds'),
  f(504,34.00467,36.33157,'Temporary ponds'),
  f(468,34.00688,36.31370,'Temporary ponds'),
  f(626,34.06172,36.35393,'Temporary ponds'),
  f(615,33.96483,36.31620,'Temporary ponds'),
  f(487,34.02060,36.30195,'Temporary ponds'),
  f(691,34.04277,36.30983,'Temporary ponds'),
  f(676,34.03910,36.30347,'Temporary floodplain'),
  f(538,34.07175,36.32258,'Temporary floodplain'),
  f(525,34.07542,36.33017,'Temporary floodplain'),
  f(544,34.07610,36.32677,'Temporary floodplain'),
  f(520,34.07307,36.32777,'Temporary floodplain'),
  f(705,34.03798,36.31750,'Temporary floodplain'),
  f(25,33.92522,36.30180,'Temporary floodplain'),
  f(340,34.07310,36.33908,'Temporary floodplain'),
  f(633,33.96077,36.30202,'Temporary floodplain'),
  f(541,34.07542,36.32452,'Temporary floodplain'),
  f(354,34.07615,36.33952,'Temporary floodplain'),
  f(646,33.95768,36.30785,'Temporary floodplain'),
  f(298,34.07205,36.36407,'Temporary floodplain'),
  f(264,33.97508,36.29058,'Temporary floodplain'),
  f(364,33.93598,36.33820,'Temporary floodplain'),
  f(328,34.07865,36.33812,'Temporary floodplain'),
  f(555,34.05200,36.30110,'Temporary floodplain'),
  f(330,34.07752,36.33700,'Temporary floodplain'),
  f(266,33.97472,36.29407,'Temporary floodplain'),
  f(329,34.07853,36.33697,'Temporary floodplain'),
  f(631,33.96237,36.30225,'Temporary floodplain'),
  f(243,33.95917,36.28222,'Temporary floodplain'),
  f(702,34.03322,36.31402,'Temporary floodplain'),
  f(654,34.04462,36.30320,'Temporary floodplain'),
  f(608,33.95447,36.31448,'Temporary floodplain'),
  f(655,34.04502,36.30218,'Temporary floodplain'),
  f(698,34.02498,36.30945,'Temporary floodplain'),
  f(598,34.07015,36.32070,'Temporary floodplain'),
  f(703,34.03357,36.31453,'Temporary floodplain'),
  f(697,34.04235,36.30873,'Temporary floodplain'),
  f(341,34.07177,36.33863,'Temporary floodplain'),
  f(689,33.96443,36.28778,'Temporary floodplain'),
  f(353,34.07630,36.33890,'Temporary floodplain'),
  f(690,34.04253,36.30970,'Temporary floodplain'),
  f(66,33.92765,36.29678,'Temporary floodplain'),
  f(658,34.04453,36.30033,'Temporary floodplain'),
  f(154,33.94427,36.28290,'Temporary floodplain'),
  f(155,33.94742,36.28418,'Temporary floodplain'),
  f(653,34.04485,36.30385,'Temporary floodplain'),
  f(629,33.95122,36.30105,'Temporary floodplain'),
  f(209,33.95418,36.27445,'Temporary floodplain'),
  f(37,33.92583,36.29885,'Temporary floodplain'),
  f(314,34.07562,36.36515,'Temporary floodplain'),
  f(315,34.07512,36.36478,'Temporary floodplain'),
  f(159,33.94580,36.28228,'Temporary floodplain'),
  f(202,33.95355,36.27777,'Temporary floodplain'),
  f(110,33.93523,36.29192,'Temporary floodplain'),
  f(45,33.92668,36.29753,'Temporary floodplain'),
  f(77,33.92907,36.29767,'Temporary floodplain'),
  f(36,33.92488,36.29868,'Temporary floodplain'),
  f(86,33.93237,36.29260,'Temporary floodplain'),
  f(30,33.92448,36.29965,'Temporary floodplain'),
  f(127,33.93588,36.29035,'Temporary floodplain'),
  f(241,33.95860,36.28402,'Terrestrial saline flats'),
  f(532,34.07257,36.32438,'Terrestrial saline flats'),
  f(242,33.95808,36.28357,'Terrestrial saline flats'),
  f(638,33.96512,36.30538,'Terrestrial saline flats'),
  f(642,33.96188,36.30985,'Terrestrial saline flats'),
  f(251,33.96947,36.28883,'Terrestrial saline flats'),
  f(474,34.00625,36.31823,'Terrestrial saline flats'),
  f(326,34.07783,36.34118,'Terrestrial saline flats'),
  f(476,34.00672,36.31813,'Terrestrial saline flats'),
  f(324,34.07788,36.34468,'Terrestrial saline flats'),
  f(268,33.98005,36.29555,'Terrestrial saline flats'),
  f(256,33.97565,36.27755,'Terrestrial saline flats'),
  f(323,34.07752,36.34410,'Terrestrial saline flats'),
  f(496,34.01855,36.31182,'Terrestrial saline flats'),
  f(252,33.96993,36.28892,'Terrestrial saline flats'),
  f(213,33.96303,36.27238,'Terrestrial saline flats'),
  f(245,33.95948,36.28325,'Terrestrial saline flats'),
  f(620,33.97083,36.31525,'Terrestrial saline flats'),
  f(214,33.96330,36.27297,'Terrestrial saline flats'),
  f(673,34.04113,36.30053,'Terrestrial saline flats'),
  f(322,34.07863,36.34520,'Terrestrial saline flats'),
  f(216,33.96735,36.27360,'Terrestrial saline flats'),
  f(261,33.97800,36.28682,'Terrestrial saline flats'),
  f(248,33.95958,36.28427,'Terrestrial saline flats'),
  f(196,33.95707,36.27428,'Terrestrial saline flats'),
  f(498,34.01840,36.30938,'Terrestrial saline flats'),
  f(262,33.97720,36.29082,'Terrestrial saline flats'),
  f(239,33.95702,36.28012,'Terrestrial saline flats'),
  f(260,33.97913,36.28738,'Terrestrial saline flats'),
  f(478,34.00782,36.31885,'Terrestrial saline flats'),
  f(672,34.04105,36.30015,'Terrestrial saline flats'),
  f(229,33.98518,36.27465,'Terrestrial saline flats')
]);

var AOI = PLOTS.geometry().bounds().buffer(2000);

// ------------------------------------------------------------- indices
// Landsat Collection 2 Level-2 scaling: DN * 0.0000275 - 0.2.
// ee.Image(...) wraps the result because copyProperties returns an Element,
// which has no .select method.
function toReflectance(img) {
  return ee.Image(img.select(['SR_B.'])
            .multiply(0.0000275).add(-0.2)
            .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']));
}

// Band positions differ between TM/ETM+ and OLI, so select by role rather
// than by number. ee.String has no chainable .equals, hence List.contains.
function bands(img) {
  var id  = ee.String(img.get('SPACECRAFT_ID'));
  var oli = ee.List(['LANDSAT_8', 'LANDSAT_9']).contains(id);
  var names = ee.List(ee.Algorithms.If(oli,
      ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'],   // OLI / OLI-2
      ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'])); // TM / ETM+
  return img.select(names, ['blue', 'green', 'red', 'nir', 'swir1']);
}

function indices(img) {
  var b = bands(toReflectance(img));
  var ndvi = b.normalizedDifference(['nir', 'red']).rename('ndvi');
  var ndmi = b.normalizedDifference(['nir', 'swir1']).rename('ndmi');
  // Canopy response salinity index (Scudiero et al. 2015).
  var crsi = b.expression(
      'sqrt((N*R - G*B) / (N*R + G*B))',
      {N: b.select('nir'), R: b.select('red'),
       G: b.select('green'), B: b.select('blue')}).rename('crsi');
  // Bare-soil salinity index (Douaoui et al. 2006).
  var si = b.expression('sqrt(B * R)',
      {B: b.select('blue'), R: b.select('red')}).rename('si');
  return ee.Image(ee.Image.cat([ndvi, ndmi, crsi, si])
           .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']));
}

// ----------------------------------------------------------- cloud mask
function masked(img) {
  var qa = img.select('QA_PIXEL');
  var bad = qa.bitwiseAnd(1 << 1).neq(0)
      .or(qa.bitwiseAnd(1 << 2).neq(0))
      .or(qa.bitwiseAnd(1 << 3).neq(0))
      .or(qa.bitwiseAnd(1 << 4).neq(0));
  return ee.Image(img.updateMask(bad.not())
           .copyProperties(img, ['system:time_start', 'SPACECRAFT_ID']));
}

var col = ee.ImageCollection([])
  .merge(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2'))
  .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'))
  .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'))
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(AOI).filterDate(START, END)
  .map(masked).map(indices);

// -------------------------------------------------------------- sample
// One row per plot per scene. Plots masked out by cloud on a given date
// simply do not appear, which is the correct behaviour: they are missing
// observations, not zeros.
var samples = col.map(function (img) {
  var d = ee.Date(img.get('system:time_start'));
  return img.sampleRegions({
      collection: PLOTS, scale: 30, geometries: false, tileScale: 4
    }).map(function (ft) {
      return ft.set({
        sensor: img.get('SPACECRAFT_ID'),
        date: d.format('YYYY-MM-dd'),
        doy: d.getRelative('day', 'year').add(1)
      });
    });
}).flatten();

Export.table.toDrive({
  collection: samples,
  description: 'plot_timeseries',
  fileFormat: 'CSV',
  selectors: ['plot', 'subtype', 'sensor', 'date', 'doy',
              'ndvi', 'ndmi', 'crsi', 'si']
});

print('plots:', PLOTS.size());
print('scenes in window:', col.size());
