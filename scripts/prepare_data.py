"""Extract and prepare the Common Voice Kabyle dataset from a local archive."""

import argparse
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract the Common Voice Kabyle archive")
    parser.add_argument(
        "--archive",
        type=str,
        default="data/raw/cv-corpus-24.0-2025-12-05-kab.tar.gz",
        help="Path to the .tar.gz archive",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to extract into (default: data/raw)",
    )
    return parser.parse_args()


def main() -> None:
    """Extract the archive and display dataset info."""
    args = parse_args()

    archive_path = Path(args.archive)
    output_dir = Path(args.output_dir)

    if not archive_path.exists():
        print(f"Error: Archive not found at {archive_path}")
        print("Download the Kabyle dataset from https://commonvoice.mozilla.org/en/datasets")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive_path.name} to {output_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir, filter="data")

    # Find the extracted directory
    extracted_dirs = list(output_dir.glob("cv-corpus-*/kab"))
    if not extracted_dirs:
        print("Error: Could not find extracted Kabyle data.")
        return

    kab_dir = extracted_dirs[0]
    print(f"\nExtracted to: {kab_dir}")

    # Show split info
    print("\nAvailable splits:")
    for tsv in sorted(kab_dir.glob("*.tsv")):
        line_count = sum(1 for _ in open(tsv, encoding="utf-8")) - 1  # exclude header
        print(f"  {tsv.stem:>20}: {line_count:,} entries")

    clips_dir = kab_dir / "clips"
    if clips_dir.exists():
        clip_count = sum(1 for _ in clips_dir.glob("*.mp3"))
        print(f"\n  Audio clips: {clip_count:,} MP3 files")

    print("\nData is ready! Update 'data_dir' in your config or use the default path.")


if __name__ == "__main__":
    main()
