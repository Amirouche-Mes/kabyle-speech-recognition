"""Audio preprocessing utilities for Whisper fine-tuning."""

from typing import Any

import numpy as np
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer


def get_processor(model_name: str = "openai/whisper-small") -> WhisperProcessor:
    """Load the Whisper processor for a given model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        WhisperProcessor with feature extractor and tokenizer.
    """
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="kabyle", task="transcribe")
    return WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def preprocess_audio(
    example: dict[str, Any],
    processor: WhisperProcessor,
) -> dict[str, Any]:
    """Preprocess a single audio example for Whisper.

    Extracts log-mel spectrogram features and tokenizes the transcription.

    Args:
        example: A dataset example with 'audio' and 'sentence' fields.
        processor: WhisperProcessor instance.

    Returns:
        Dictionary with 'input_features' and 'labels'.
    """
    audio = example["audio"]
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]

    labels = processor.tokenizer(example["sentence"]).input_ids

    return {"input_features": input_features, "labels": labels}


def prepare_dataset(
    dataset: Any,
    processor: WhisperProcessor,
    num_proc: int = 4,
) -> Any:
    """Apply preprocessing to an entire dataset split.

    Args:
        dataset: A HuggingFace dataset split.
        processor: WhisperProcessor instance.
        num_proc: Number of processes for parallel preprocessing.

    Returns:
        Preprocessed dataset with audio features and tokenized labels.
    """
    dataset = dataset.map(
        lambda example: preprocess_audio(example, processor),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
    return dataset
