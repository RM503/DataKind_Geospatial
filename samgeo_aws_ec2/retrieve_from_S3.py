import os
import boto3 

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
