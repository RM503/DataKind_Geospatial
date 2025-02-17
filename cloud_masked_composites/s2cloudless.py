''' 
This module contains a collection of functions necessary for performing cloud-masking using s2cloudless.
For reference: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
'''
import ee 

def s2cloudless(
                AOI: ee.Geometry.Point,
                START_DATE: str = '2024-01-01',
                END_DATE: str = '2025-01-01',
                CLOUD_FILTER: int = 60, 
                CLD_PRB_THRESH: int = 40, 
                NIR_DRK_THRESH: float = 0.15, 
                CLD_PRJ_DIST: int = 2, 
                BUFFER: int = 100
    ):

    def add_cloud_bands(img: ee.Image) -> ee.Image:
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

        is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def add_shadow_bands(img: ee.Image) -> ee.Image:
        # `SCL` is the scene classification map of S2 with value 6 representing water
        not_water = img.select('SCL').neq(6) 

        # Identify dark NIR pixels that are not water (potential cloud shadow pixels)
        # `B8` is the NIR band 
        SR_BAND_SCALE = 1e4
        dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection)
        # Solar azimuth is the angle between the Sun's rays and the N-S direction
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input
        cld_proj = (
            img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
            .reproject(**{
                'crs': img.select(0).projection(),
                'scale': 100
            })
            .select('distance')
            .mask()
            .rename('cloud_transform')
        )

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')

        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(img: ee.Image) -> ee.Image:
        # Add cloud component bands
        img_cloud = add_cloud_bands(img)

        # Add cloud shadow component bands.
        img_cloud_shadow = add_shadow_bands(img_cloud)

        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0
        is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision
        is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
            .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
            .rename('cloudmask'))

        # Add the final cloud-shadow mask to the image.
        return img_cloud_shadow.addBands(is_cld_shdw)

    def apply_cld_shdw_mask(img: ee.Image) -> ee.Image:
        # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
        not_cld_shdw = img.select('cloudmask').Not()

        # Subset reflectance bands and update their masks, return the result.
        return img.select('B.*').updateMask(not_cld_shdw)
    
    def img_scale(img: ee.Image) -> ee.Image:
        return img.divide(10000)

    
    #############################################################################################################
    # Function definitions end
    #############################################################################################################

    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(AOI)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    
    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(AOI)
        .filterDate(START_DATE, END_DATE))
    
    s2_sr_cld_col = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

    s2_sr_median = (s2_sr_cld_col.map(add_cld_shdw_mask)
                                 .map(apply_cld_shdw_mask)
                                 .map(img_scale)
                                 .median())
    
    return s2_sr_median