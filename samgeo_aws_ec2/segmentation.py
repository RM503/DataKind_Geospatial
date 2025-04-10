import os 
import time
import glob 
from typing import Tuple 
import torch 
from samgeo import SamGeo
from tqdm import tqdm
import pandas as pd 
import geopandas as gpd
import uuid
import boto3
from retrieve_from_S3 import download_s3_folder
import json
import logging
import argparse 

logging.basicConfig(
    level=logging.INFO,
    filename='samgeo_run_logs.txt', 
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def folder_exists(output_bucket: str, path: str) -> bool:
    if not path.endswith('/'):
        path = path + '/'
    r = s3_client.list_objects(Bucket=output_bucket, Prefix=path, Delimiter='/', MaxKeys=1)
    return 'Contents' in r 

def samgeo_unit(
        sam: SamGeo,
        input_img_path: str,
        output_folder: str,
        batch: bool=False,
        foreground: bool=True,
        erosion_kernel: Tuple[int, int]=(3, 3),
        mask_multiplier: int=255
    ) -> Tuple[pd.DataFrame, str]:
    ''' 
    This function applies segment-geospatial on a single unit of GeoTIFF file. It can be used as a 
    stand-alone function or in a loop for processing multiple image tiles.

    Args: (i) SamGeo - the instantiated class of segment-geospatial
          (ii) input_img_path - file path to input GeoTIFF file
          (iii) output_folder - path where the segmented image will be stored
          (iv) batch - batch processing; defaults to True
          (v) foreground - whether to generate foreground masks; defaults to True
          (vi) erosion_kernel - the erosion kernel for filtering object masks and extract borders; defaults to (3, 3)
          (vii) mask_multiplier - multiplication factor for generated binary masks; defaults to 255

    Returns: pandas dataframe containing segment vectors in wkt format and directory to csv file
    '''

    if not os.path.exists(output_folder):
        logging.info(f"Creating ouput folder: {output_folder}")
        os.makedirs(output_folder)

    img_name = input_img_path.split('/')[-1].split('.')[0]
    output_path = f"{output_folder}/{img_name}_delineation.tiff"

    # Generate automatic mask generator
    sam.generate(
        input_img_path, 
        output_path, 
        batch=batch, 
        foreground=foreground, 
        erosion_kernel=erosion_kernel, 
        mask_multiplier=mask_multiplier
    )

    output_gpkg_path = f"{output_path.split('.')[0]}.gpkg"
    output_csv_path = f"{output_path.split('.')[0]}.csv"

    sam.tiff_to_gpkg(output_path, output_gpkg_path, simplify_tolerance=None)

    # Read the geopackage file, covert geometry into wkt and upload to S3 bucket
    gdf = gpd.read_file(output_gpkg_path)
    gdf['uuid'] = gdf.apply(lambda row: uuid.uuid4(), axis=1) # attach a unique uuid to each tile

    df = gdf.to_wkt()
    df.to_csv(output_csv_path, index=False) # convert geopackage date into csv and save

    return df, output_csv_path 

def segment_tiles(
        input_img_tiles: list[str],
        output_folder: str,
        output_bucket: str
) -> None:
    ''' 
    This function uses the `semgeo_unit()` function and applies it iteratively
    on multiple tiles.

    Args: (i) input_img_tiles - list of image tile directories
          (ii) output_folder - path where the segmented images will be stored (same as the one in `samgeo_unit()`)

    Returns: None 
    ''' 

    if not isinstance(input_img_tiles, list):
        logging.error(f"{input_img_tiles} must be of type list")

    img_list = input_img_tiles 

    start = time.time()

    if not folder_exists(output_bucket, output_folder):
        # Check if upload directory exists in output bucket
        logging.info(f"{output_folder} does not exist in {output_bucket}")

        s3_client.put_object(Bucket=output_bucket, Key=output_folder + '/')

        logging.info(f"{output_folder} created in {output_bucket}")

    for idx, img, in enumerate(tqdm(img_list, desc="Processing images")):
        print(f"Working on tile {idx}")
        logging.info(f"Working on tile {idx}")

        df, output_csv_path = samgeo_unit(sam, img, output_folder)

        # Upload file to S3 bucket in a try-except block
        try:
            s3_client.upload_file(output_csv_path, output_bucket, output_csv_path)
            logging.info(f"Segmentation file for tile {idx} uploaded to output bucket")
        except Exception as e:
            logging.error(f"Error uploading segmentation file for tile {idx}: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    end = time.time()

    execution_time = end - start
    logging.info(f"Segmentation finished; execution time: {execution_time / 60} minutes")

    # Remove contents after code is finished running
    #os.remove(output_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_bucket',
        required=True,
        help='Name of the S3 bucket from which files will be retrieved.'
    )
    parser.add_argument(
        '--output_bucket',
        required=True,
        help='Name of the S3 bucket to which files will be uploaded.'
    )

    args = parser.parse_args()

    input_bucket = args.input_bucket
    output_bucket = args.output_bucket

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    """ 
    For certain regions, due to geographic and image resolution issues, a different
    parameter set must be used. 
    """
    if os.path.exists("params.json"):
        with open("params.json", "r") as f:
            params = json.load(f)
    else:
        logging.error("Parameter file not found")


    sam_kwargs = params["params"][0]

    model_path = '/root/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth'
    check_model = os.path.exists(model_path)

    if check_model:
        logging.info(f"SAM model found at: {model_path}")
    else:
        logging.info(f"SAM model not found at: {model_path}")

    # Instantiate SAM automatic mask generator class
    sam = SamGeo(
	    model_type='vit_h',
	    checkpoint=model_path,
        device=device,
	    sam_kwargs=sam_kwargs
	)

    s3_client = boto3.client('s3')

    regions = [
        'Bomet_1',
        'Kajiado_1',
        'Kajiado_2',
        'Laikipia_1',
        'Machakos_1',
        'Mashuru_1',
        'Nakuru_1',
        'Narok_1',
        'Trans_Nzoia_1',
        'Uasin_Gishu_1'
    ]
    for count, region in enumerate(regions):
        logging.info(f"Segmenting region {count + 1}: {region}")
        download_s3_folder(input_bucket, region)

        img_tiles_list = glob.glob(f"{region}/*.tiff")

        segment_tiles(img_tiles_list, region, output_bucket)