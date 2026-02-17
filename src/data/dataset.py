"""Kabyle dataset loading and management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets import Audio, DatasetDict, load_dataset


@dataclass
class KabyleDataset:
    """Wrapper for the Mozilla Common Voice Kabyle dataset.

    Loads the dataset from HuggingFace and prepares it for Whisper fine-tuning.

    Args:
        dataset_name: HuggingFace dataset identifier.
        language: Language code for Common Voice.
        version: Dataset version to use.
        cache_dir: Local directory for caching downloaded data.
        hf_token: HuggingFace API token for authentication.
        sampling_rate: Target audio sampling rate in Hz.
    """

    dataset_name: str = "mozilla-foundation/common_voice_17_0"
    language: str = "kab"
    version: str = "17.0"
    cache_dir: Optional[str] = None
    hf_token: Optional[str] = None
    sampling_rate: int = 16000
    _dataset: Optional[DatasetDict] = field(default=None, init=False, repr=False)

    def load(self) -> DatasetDict:
        """Load the dataset from HuggingFace.

        Returns:
            The loaded dataset with train, validation, and test splits.
        """
        self._dataset = load_dataset(
            self.dataset_name,
            self.language,
            token=self.hf_token,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self._dataset = self._dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        return self._dataset

    @property
    def dataset(self) -> DatasetDict:
        """Access the loaded dataset, loading it if necessary."""
        if self._dataset is None:
            self.load()
        return self._dataset

    def get_split(self, split: str) -> DatasetDict:
        """Get a specific dataset split.

        Args:
            split: One of 'train', 'validation', or 'test'.

        Returns:
            The requested dataset split.

        Raises:
            ValueError: If the split name is invalid.
        """
        valid_splits = {"train", "validation", "test"}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
        return self.dataset[split]

    def get_stats(self) -> dict[str, int]:
        """Get basic statistics about the dataset.

        Returns:
            Dictionary with number of examples per split.
        """
        return {split: len(self.dataset[split]) for split in self.dataset}
