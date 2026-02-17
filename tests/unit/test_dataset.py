"""Tests for the KabyleDataset class."""

import pytest

from src.data.dataset import KabyleDataset


class TestKabyleDataset:
    """Tests for KabyleDataset initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test that KabyleDataset initializes with correct defaults."""
        dataset = KabyleDataset()
        assert dataset.language == "kab"
        assert dataset.sampling_rate == 16000
        assert dataset.dataset_name == "mozilla-foundation/common_voice_17_0"

    def test_custom_initialization(self) -> None:
        """Test that KabyleDataset accepts custom parameters."""
        dataset = KabyleDataset(
            language="kab",
            version="16.0",
            cache_dir="/tmp/test_cache",
            sampling_rate=8000,
        )
        assert dataset.version == "16.0"
        assert dataset.cache_dir == "/tmp/test_cache"
        assert dataset.sampling_rate == 8000

    def test_invalid_split_raises_error(self) -> None:
        """Test that requesting an invalid split raises ValueError."""
        dataset = KabyleDataset()
        with pytest.raises(ValueError, match="Invalid split"):
            dataset.get_split("invalid")
