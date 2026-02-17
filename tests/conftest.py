"""Pytest fixtures for Kabyle Speech Recognition tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_audio() -> dict:
    """Create a mock audio sample for testing.

    Returns:
        Dictionary mimicking a Common Voice audio example.
    """
    sampling_rate = 16000
    duration = 3.0
    num_samples = int(sampling_rate * duration)
    return {
        "audio": {
            "array": np.random.randn(num_samples).astype(np.float32),
            "sampling_rate": sampling_rate,
        },
        "sentence": "Azul fellawen",
    }


@pytest.fixture
def sample_dataset_stats() -> dict:
    """Create mock dataset statistics for testing."""
    return {
        "train": 5000,
        "validation": 500,
        "test": 500,
    }
