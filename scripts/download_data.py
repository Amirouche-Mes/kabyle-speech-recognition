"""Download the Kabyle dataset from Mozilla Common Voice via HuggingFace."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data.dataset import KabyleDataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download the Common Voice Kabyle dataset")
    parser.add_argument(
        "--language",
        type=str,
        default="kab",
        help="Language code (default: kab for Kabyle)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="17.0",
        help="Common Voice dataset version (default: 17.0)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw",
        help="Directory to cache downloaded data (default: data/raw)",
    )
    return parser.parse_args()


def main() -> None:
    """Download and display statistics for the Kabyle dataset."""
    load_dotenv()
    args = parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set. You may need it for Common Voice access.")
        print("Set it in your .env file or export HF_TOKEN=your_token")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Common Voice {args.version} - {args.language}")
    print(f"Cache directory: {cache_dir.resolve()}")

    dataset = KabyleDataset(
        language=args.language,
        version=args.version,
        cache_dir=str(cache_dir),
        hf_token=hf_token,
    )

    data = dataset.load()
    stats = dataset.get_stats()

    print("\nDataset statistics:")
    print("-" * 40)
    for split, count in stats.items():
        print(f"  {split:>12}: {count:,} examples")
    print("-" * 40)
    print(f"  {'Total':>12}: {sum(stats.values()):,} examples")

    print(f"\nSample from train split:")
    sample = data["train"][0]
    print(f"  Sentence: {sample['sentence']}")
    print(f"  Audio sampling rate: {sample['audio']['sampling_rate']} Hz")
    print(f"  Audio duration: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.2f}s")

    print("\nDownload complete!")


if __name__ == "__main__":
    main()
