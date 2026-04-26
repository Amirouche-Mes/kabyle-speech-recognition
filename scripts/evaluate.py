"""Evaluation script for trained Whisper models on Kabyle test data."""

import argparse
import json
from pathlib import Path

import librosa
import soundfile as sf
import torch
from jiwer import cer, wer
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.data.dataset import KabyleDataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Whisper model on Kabyle data")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/cv-corpus-25.0-2026-03-09/kab",
        help="Path to the extracted Common Voice Kabyle data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: <model-path>/eval_results.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (useful for quick tests)",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation on the test set."""
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model and processor
    print(f"Loading model from {args.model_path}")
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    model.eval()

    # Load test data
    dataset = KabyleDataset(data_dir=args.data_dir)
    test_data = dataset.get_split(args.split)

    if args.max_samples:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))

    # Run inference
    references = []
    predictions = []

    print(f"Evaluating on {len(test_data)} examples from '{args.split}' split...")
    for example in tqdm(test_data):
        audio_array, sampling_rate = sf.read(example["audio"], dtype="float32")
        if sampling_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000
        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(transcription)
        references.append(example["sentence"])

    # Calculate metrics
    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)

    results = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "split": args.split,
        "num_examples": len(test_data),
        "wer": round(word_error_rate * 100, 2),
        "cer": round(char_error_rate * 100, 2),
    }

    print(f"\nResults:")
    print(f"  WER: {results['wer']}%")
    print(f"  CER: {results['cer']}%")

    # Save results
    output_path = args.output or str(Path(args.model_path) / "eval_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
