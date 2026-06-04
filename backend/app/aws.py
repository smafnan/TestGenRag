"""
AWS S3 helper for raw PDF storage.

Gracefully no-ops if S3_BUCKET is not set, so the app runs fine without AWS
credentials during local development.
"""

import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def upload_to_s3(file_path: str, key: str) -> str | None:
    """
    Upload a file to S3 and return its URI, or None if S3 is not configured.

    Args:
        file_path: Local path to the file.
        key: S3 object key (e.g. 'uploads/spec.pdf').

    Returns:
        S3 URI like 's3://bucket/uploads/spec.pdf', or None.
    """
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return None

    try:
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "eu-central-1"))
        s3.upload_file(file_path, bucket, key)
        return f"s3://{bucket}/{key}"
    except (BotoCoreError, ClientError) as exc:
        print(f"[WARN] S3 upload failed (continuing without S3): {exc}")
        return None
