import os
import boto3
import argparse 

s3 = boto3.resource('s3')

def download_s3_folder(bucket_name: str, s3_folder: str, local_folder: str=None) -> None:
    ''' 
    This function downloads all the contents from a speficied folder inside
    an S3 bucket.
    '''

    bucket = s3.Bucket(bucket_name)

    if not s3_folder.endswith('/'):
        s3_folder += '/'

    for object in bucket.objects.filter(Prefix=s3_folder):
        # Creates a local folder of the same name as S3 folder if none exists
        if local_folder is None:
            folder = object.key.split('/')[0]
            os.makedirs(folder, exist_ok=True)
        
        if object.key.endswith('/'):
            continue 
        bucket.download_file(object.key, object.key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bucket_name',
        required=True,
        help='Name of the S3 bucket from which files will be retrieved.'
    )

    parser.add_argument(
        '--s3_folder',
        required=True,
        help='Name of the S3 folder from which files will be retrieved.'
    )

    args = parser.parse_args()

    bucket_name = args.bucket_name
    s3_folder = args.s3_folder

    download_s3_folder(bucket_name, s3_folder)