"""Kabyle dataset loading from local Common Voice files."""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict

logger = logging.getLogger("train")


@dataclass
class KabyleDataset:
    """Loader for the Mozilla Common Voice Kabyle dataset from local files.

    Reads TSV metadata and audio clips extracted from the Common Voice archive.

    Args:
        data_dir: Path to the extracted Common Voice directory
            (e.g., 'data/raw/cv-corpus-25.0-2026-03-09/kab').
        sampling_rate: Target audio sampling rate in Hz.
    """

    data_dir: str = "data/raw/cv-corpus-25.0-2026-03-09/kab"
    sampling_rate: int = 16000
    _dataset: Optional[DatasetDict] = field(default=None, init=False, repr=False)

    # Common Voice uses 'dev' for validation
    SPLIT_MAP: dict[str, str] = field(
        default_factory=lambda: {"train": "train", "validation": "dev", "test": "test"},
        init=False,
    )

    def _load_split(self, split: str) -> Dataset:
        """Load a single dataset split from its TSV file.

        Args:
            split: One of 'train', 'validation', or 'test'.

        Returns:
            A HuggingFace Dataset with 'audio' and 'sentence' columns.
        """
        tsv_name = self.SPLIT_MAP[split]
        tsv_path = Path(self.data_dir) / f"{tsv_name}.tsv"

        if not tsv_path.exists():
            raise FileNotFoundError(f"Split file not found: {tsv_path}")

        clips_dir = Path(self.data_dir) / "clips"
        paths = []
        sentences = []

        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                audio_path = clips_dir / row["path"]
                if audio_path.exists():
                    paths.append(str(audio_path))
                    sentences.append(row["sentence"])

        ds = Dataset.from_dict({"audio": paths, "sentence": sentences})
        return ds

    def load(self) -> DatasetDict:
        """Load all splits (train, validation, test) from local files.

        Returns:
            DatasetDict with train, validation, and test splits.
        """
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_path}. "
                "Run 'python scripts/prepare_data.py' to extract the archive first."
            )

        splits = {}
        for split in self.SPLIT_MAP:
            tsv_path = data_path / f"{self.SPLIT_MAP[split]}.tsv"
            if tsv_path.exists():
                logger.info(f"Loading {split} split...")
                splits[split] = self._load_split(split)
                logger.info(f"  {split}: {len(splits[split]):,} examples")

        self._dataset = DatasetDict(splits)
        return self._dataset

    @property
    def dataset(self) -> DatasetDict:
        """Access the loaded dataset, loading it if necessary."""
        if self._dataset is None:
            self.load()
        return self._dataset

    def get_split(self, split: str) -> Dataset:
        """Get a specific dataset split.

        Args:
            split: One of 'train', 'validation', or 'test'.

        Returns:
            The requested dataset split.

        Raises:
            ValueError: If the split name is invalid.
        """
        valid_splits = set(self.SPLIT_MAP.keys())
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
        return self.dataset[split]

    def get_stats(self) -> dict[str, int]:
        """Get basic statistics about the dataset.

        Returns:
            Dictionary with number of examples per split.
        """
        return {split: len(self.dataset[split]) for split in self.dataset}
