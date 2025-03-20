import os 
import glob 
from tqdm import tqdm
import boto3 

def get_subfolders(folder_path: str) -> list[str]:
    subfolders_path_list = glob.glob(os.path.join(folder_path, '*'))

    subfolders_list = []
    for path in subfolders_path_list:
        subfolder = path.split('/')[-1]
        subfolders_list.append(subfolder)

    return subfolders_list

def folder_exists(bucket: str, path: str) -> bool:
    if not path.endswith('/'):
        path = path + '/'
    r = s3_client.list_objects(Bucket=bucket, Prefix=path, Delimiter='/', MaxKeys=1)
    return 'Contents' in r 

if __name__ == '__main__':
    s3_client = boto3.client('s3')
    bucket_name = 'regenorganics-prioritydistributorbuffer-tiffs'

    subfolders_list = get_subfolders('cloud_masked_composites/sentinelhub/images')
    
    for subfolder in subfolders_list:
        print(f'Wokring on {subfolder}')

        if not folder_exists(bucket_name, subfolder):
            s3_client.put_object(Bucket=bucket_name, Key=(subfolder+'/'))

        PATH_TO_BASE = 'cloud_masked_composites/sentinelhub/images'
        img_file_paths = sorted(glob.glob(os.path.join(PATH_TO_BASE, subfolder, '*.tiff')))

        for img_file_path in tqdm(img_file_paths):
            print(img_file_path)
            #if object_name is None:
            object_name = img_file_path.split('/')[-1]

            try:
                with open(img_file_path, 'rb') as f:
                    s3_client.upload_fileobj(f, bucket_name, subfolder + '/' + object_name)
                    print(f'{img_file_path} uploaded successfully.')
                
            except Exception as e:
                print(f'Error uploading {img_file_path}: {e}')