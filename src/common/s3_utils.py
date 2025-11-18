"""
S3 utilities for file upload and download operations.
Handles S3 operations with proper error handling.
"""
import boto3
from botocore.exceptions import ClientError

def upload_file_to_s3(local_path, s3_bucket, s3_key):
    """
    Upload file to S3 with error handling.
    
    Args:
        local_path: Local file path to upload
        s3_bucket: S3 bucket name
        s3_key: S3 key (object path) where file will be stored
    
    Raises:
        ClientError: If upload fails
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_path, s3_bucket, s3_key)
        print(f"Successfully uploaded {local_path} to s3://{s3_bucket}/{s3_key}")
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        raise

def download_file_from_s3(s3_bucket, s3_key, local_path):
    """
    Download file from S3 with error handling.
    
    Args:
        s3_bucket: S3 bucket name
        s3_key: S3 key (object path) of the file to download
        local_path: Local file path where file will be saved
    
    Raises:
        ClientError: If download fails
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(s3_bucket, s3_key, local_path)
        print(f"Successfully downloaded s3://{s3_bucket}/{s3_key} to {local_path}")
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        raise

