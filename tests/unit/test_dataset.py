"""Tests for the KabyleDataset class."""

import pytest

from src.data.dataset import KabyleDataset


class TestKabyleDataset:
    """Tests for KabyleDataset initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test that KabyleDataset initializes with correct defaults."""
        dataset = KabyleDataset()
        assert dataset.sampling_rate == 16000
        assert "cv-corpus-25.0" in dataset.data_dir

    def test_custom_initialization(self) -> None:
        """Test that KabyleDataset accepts custom parameters."""
        dataset = KabyleDataset(
            data_dir="/tmp/test_data/kab",
            sampling_rate=8000,
        )
        assert dataset.data_dir == "/tmp/test_data/kab"
        assert dataset.sampling_rate == 8000

    def test_invalid_split_raises_error(self) -> None:
        """Test that requesting an invalid split raises ValueError."""
        dataset = KabyleDataset()
        with pytest.raises(ValueError, match="Invalid split"):
            dataset.get_split("invalid")

    def test_missing_data_dir_raises_error(self) -> None:
        """Test that loading from nonexistent directory raises FileNotFoundError."""
        dataset = KabyleDataset(data_dir="/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            dataset.load()
