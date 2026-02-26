"""Evaluate pre-trained Whisper models on Kabyle before fine-tuning.

Run this before training to establish baseline WER/CER metrics.
Usage:
    python scripts/baseline_eval.py --model openai/whisper-small
    python scripts/baseline_eval.py --model openai/whisper-large-v3
    python scripts/baseline_eval.py --model openai/whisper-small --max-samples 200
"""

import argparse
import json
from pathlib import Path

import torch
from jiwer import cer, wer
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.data.dataset import KabyleDataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained Whisper on Kabyle (baseline, before fine-tuning)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        help="HuggingFace model name (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/cv-corpus-24.0-2025-12-05/kab",
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
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (useful for quick tests)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline",
        help="Directory to save results (default: results/baseline)",
    )
    return parser.parse_args()


def main() -> None:
    """Run baseline evaluation."""
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model}")

    # Load model and processor
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    # Force Kabyle language and transcribe task
    model.generation_config.language = "kabyle"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Load dataset
    dataset = KabyleDataset(data_dir=args.data_dir)
    eval_data = dataset.get_split(args.split)

    if args.max_samples:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    print(f"Evaluating on {len(eval_data)} examples from '{args.split}' split...\n")

    references = []
    predictions = []

    for example in tqdm(eval_data):
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

    # Model short name for results file
    model_short = args.model.replace("/", "_")

    results = {
        "type": "baseline",
        "model": args.model,
        "split": args.split,
        "num_examples": len(eval_data),
        "wer": round(word_error_rate * 100, 2),
        "cer": round(char_error_rate * 100, 2),
    }

    print(f"\n{'=' * 50}")
    print(f"BASELINE RESULTS: {args.model}")
    print(f"{'=' * 50}")
    print(f"  Split:    {args.split}")
    print(f"  Samples:  {len(eval_data)}")
    print(f"  WER:      {results['wer']}%")
    print(f"  CER:      {results['cer']}%")
    print(f"{'=' * 50}")

    # Show some example predictions
    print(f"\nSample predictions:")
    for i in range(min(5, len(references))):
        print(f"\n  Reference:  {references[i]}")
        print(f"  Prediction: {predictions[i]}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_short}_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save detailed predictions
    details_path = output_dir / f"{model_short}_{args.split}_predictions.json"
    details = [
        {"reference": ref, "prediction": pred}
        for ref, pred in zip(references, predictions)
    ]
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Detailed predictions saved to {details_path}")


if __name__ == "__main__":
    main()
