"""Resilient multipart download from RunPod S3 using boto3 transfer manager."""

import os
import sys
import time
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


S3_ENDPOINT = "https://s3api-us-ca-2.runpod.io"
REGION = "us-ca-2"
BUCKET = "57grx70opn"
KEY = "Kabyle-DS.tar.gz"


class ProgressCallback:
    def __init__(self, total):
        self.total = total
        self.downloaded = 0
        self.start = time.time()

    def __call__(self, bytes_amount):
        self.downloaded += bytes_amount
        pct = self.downloaded / self.total * 100
        elapsed = time.time() - self.start
        speed = self.downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
        remaining = (self.total - self.downloaded) / (self.downloaded / elapsed) if self.downloaded > 0 else 0
        gib_done = self.downloaded / 1024**3
        gib_total = self.total / 1024**3
        print(
            f"\r  {gib_done:.2f}/{gib_total:.2f} GiB ({pct:.1f}%) "
            f"@ {speed:.1f} MiB/s  ETA {remaining:.0f}s   ",
            end="", flush=True
        )


def download(dest_path: str) -> None:
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")

    if not access_key or not secret_key:
        raise ValueError("S3_ACCESS_KEY and S3_SECRET_KEY env vars required")

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=REGION,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(retries={"max_attempts": 10, "mode": "standard"}),
    )

    meta = s3.head_object(Bucket=BUCKET, Key=KEY)
    total_size = meta["ContentLength"]
    print(f"File size: {total_size / 1024**3:.2f} GiB")

    transfer_config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,   # 100 MB
        max_concurrency=4,
        multipart_chunksize=100 * 1024 * 1024,   # 100 MB chunks — each finishes in <2min
        use_threads=True,
    )

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    progress = ProgressCallback(total_size)

    print(f"Downloading to {dest_path} with 4 parallel 100MB chunks...")
    s3.download_file(
        BUCKET,
        KEY,
        dest_path,
        Config=transfer_config,
        Callback=progress,
    )
    print(f"\nDownload complete: {dest_path}")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Kabyle-DS.tar.gz"
    download(dest)
