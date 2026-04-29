"""Audio preprocessing and data collation for Whisper fine-tuning."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer


def get_processor(model_name: str = "openai/whisper-small") -> WhisperProcessor:
    """Load the Whisper processor for a given model."""
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, task="transcribe")
    return WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


TARGET_SAMPLE_RATE = 16000


class LazyWhisperDataset(TorchDataset):
    """Lazy dataset that preprocesses audio on-the-fly instead of upfront."""

    def __init__(self, hf_dataset, processor: WhisperProcessor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio_array, sampling_rate = sf.read(example["audio"], dtype="float32")

        if sampling_rate != TARGET_SAMPLE_RATE:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=TARGET_SAMPLE_RATE)

        input_features = self.processor.feature_extractor(
            audio_array,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="np",
        ).input_features[0]

        labels = self.processor.tokenizer(example["sentence"]).input_ids

        # Whisper max decoder length is 448 tokens — truncate long transcripts
        if len(labels) > 448:
            labels = labels[:448]

        return {"input_features": input_features, "labels": labels}


def prepare_dataset(
    dataset: Any,
    processor: WhisperProcessor,
    num_proc: int = 4,
) -> LazyWhisperDataset:
    """Wrap a HuggingFace dataset in a lazy preprocessor."""
    return LazyWhisperDataset(dataset, processor)


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
