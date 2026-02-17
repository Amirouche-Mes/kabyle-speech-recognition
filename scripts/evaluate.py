"""Evaluation script for trained Whisper models on Kabyle test data."""

import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
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
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: <model-path>/eval_results.json)",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation on the test set."""
    load_dotenv()
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model and processor
    print(f"Loading model from {args.model_path}")
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    model.eval()

    # Load test data
    hf_token = os.getenv("HF_TOKEN")
    dataset = KabyleDataset(hf_token=hf_token)
    test_data = dataset.get_split(args.split)

    # Run inference
    references = []
    predictions = []

    print(f"Evaluating on {len(test_data)} examples from '{args.split}' split...")
    for example in tqdm(test_data):
        audio = example["audio"]
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
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
