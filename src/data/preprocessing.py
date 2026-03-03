"""Audio preprocessing and data collation for Whisper fine-tuning."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer


def get_processor(model_name: str = "openai/whisper-small") -> WhisperProcessor:
    """Load the Whisper processor for a given model.

    Kabyle is not natively supported by Whisper, so we don't set a language
    token. The model will learn the language during fine-tuning.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        WhisperProcessor with feature extractor and tokenizer.
    """
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, task="transcribe")
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


@dataclass
class WhisperDataCollator:
    """Data collator for Whisper fine-tuning.

    Pads input features and labels to the same length within a batch.
    Replaces padding token ids in labels with -100 so they are ignored
    by the cross-entropy loss.

    Args:
        processor: WhisperProcessor instance.
    """

    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of features into a batch.

        Args:
            features: List of preprocessed examples with 'input_features' and 'labels'.

        Returns:
            Batch dictionary with padded 'input_features', 'labels', and 'decoder_input_ids'.
        """
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if the model prepends it automatically
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics(pred: Any, processor: WhisperProcessor) -> dict[str, float]:
    """Compute WER and CER metrics from model predictions.

    Args:
        pred: EvalPrediction object with predictions and label_ids.
        processor: WhisperProcessor for decoding.

    Returns:
        Dictionary with 'wer' and 'cer' values as percentages.
    """
    from jiwer import cer, wer

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    word_error_rate = 100 * wer(label_str, pred_str)
    char_error_rate = 100 * cer(label_str, pred_str)

    return {"wer": word_error_rate, "cer": char_error_rate}
