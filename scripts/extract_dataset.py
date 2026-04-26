"""Resilient dataset extraction with per-file retry for MooseFS/network volumes."""

import os
import sys
import tarfile
import time
import argparse


def extract_with_retry(archive_path: str, dest_dir: str, max_retries: int = 10) -> None:
    print(f"Opening archive: {archive_path}")
    failed = []

    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        total = len(members)
        print(f"Total entries in archive: {total:,}")

        extracted = 0
        skipped = 0

        for i, member in enumerate(members):
            dest_path = os.path.join(dest_dir, member.name)

            # Skip already extracted files
            if member.isfile() and os.path.exists(dest_path):
                size = os.path.getsize(dest_path)
                if size == member.size:
                    skipped += 1
                    if skipped % 5000 == 0:
                        print(f"  [{i+1}/{total}] Skipped {skipped:,} already extracted")
                    continue

            # Create directories without retry needed
            if member.isdir():
                os.makedirs(dest_path, exist_ok=True)
                continue

            # Extract file with per-file retry
            for attempt in range(1, max_retries + 1):
                try:
                    tf.extract(member, dest_dir, set_attrs=False)
                    extracted += 1
                    break
                except OSError as e:
                    if attempt < max_retries:
                        wait = min(2 ** attempt, 30)
                        print(f"  I/O error on {member.name} (attempt {attempt}), retrying in {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        print(f"  FAILED after {max_retries} attempts: {member.name}")
                        failed.append(member.name)

            if (extracted + skipped) % 5000 == 0 and extracted > 0:
                pct = (i + 1) / total * 100
                print(f"  Progress: {i+1:,}/{total:,} ({pct:.1f}%) — extracted: {extracted:,} skipped: {skipped:,}")

    print(f"\n=== Done ===")
    print(f"  Extracted : {extracted:,}")
    print(f"  Skipped   : {skipped:,}")
    print(f"  Failed    : {len(failed):,}")
    if failed:
        print("  Failed files:")
        for f in failed[:20]:
            print(f"    {f}")
    else:
        print("  All files extracted successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="Path to .tar.gz archive")
    parser.add_argument("--dest", required=True, help="Destination directory")
    parser.add_argument("--max-retries", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    extract_with_retry(args.archive, args.dest, args.max_retries)
