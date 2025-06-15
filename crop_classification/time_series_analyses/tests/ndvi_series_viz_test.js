
// Function to remove cloud and snow pixels
function maskCloudAndShadows(image) {
  var cloudProb = image.select('MSK_CLDPRB');
  var snowProb = image.select('MSK_SNWPRB');
  var cloud = cloudProb.lt(5);
  var snow = snowProb.lt(5);
  var scl = image.select('SCL'); 
  var shadow = scl.eq(3); // 3 = cloud shadow
  var cirrus = scl.eq(10); // 10 = cirrus
  // Cloud probability less than 5% or cloud shadow classification
  var mask = (cloud.and(snow)).and(cirrus.neq(1)).and(shadow.neq(1));
  return image.updateMask(mask);
}
// Adding a NDVI band
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
  return image.addBands([ndvi])
}
var startDate = '2020-01-01'
var endDate = '2024-12-31'
// Use Sentinel-2 L2A data - which has better cloud masking
var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .map(maskCloudAndShadows)
    .map(addNDVI)
    .filter(ee.Filter.bounds(geometry))
    
var chart = ui.Chart.image.series({
    imageCollection: collection.select('ndvi'),
    region: geometry,
    reducer: ee.Reducer.mean()
    }).setOptions({
      interpolateNulls: true,
      lineWidth: 1,
      pointSize: 3,
      title: 'NDVI over Time at a Single Location',
      vAxis: {title: 'NDVI'},
      hAxis: {title: 'Date', format: 'YYYY-MMM', gridlines: {count: 12}}
    })
print(chart)