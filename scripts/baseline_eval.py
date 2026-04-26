"""Evaluate pre-trained Whisper models on Kabyle before fine-tuning.

Run this before training to establish baseline WER/CER metrics.
Usage:
    python scripts/baseline_eval.py --model openai/whisper-small
    python scripts/baseline_eval.py --model openai/whisper-large-v3
    python scripts/baseline_eval.py --model openai/whisper-small --max-samples 200
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
from jiwer import cer, wer
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.data.dataset import KabyleDataset


def setup_logger(log_file: Path) -> logging.Logger:
    """Set up logger that writes to both stdout and a log file."""
    logger = logging.getLogger("baseline_eval")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


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
    parser.add_argument(
        "--log-checkpoints",
        type=int,
        default=4,
        help="Number of checkpoints at which to log sample predictions (default: 4)",
    )
    return parser.parse_args()


def main() -> None:
    """Run baseline evaluation."""
    args = parse_args()

    model_short = args.model.replace("/", "_")
    output_dir = Path(args.output)
    log = setup_logger(output_dir / f"{model_short}_{args.split}.log")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    log.info("Model:  %s", args.model)

    # Load model and processor
    log.info("Loading model and processor...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.eval()

    # Kabyle is not in Whisper's language list, so we don't force a language.
    # The model will auto-detect or default to its closest match.
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Load dataset
    dataset = KabyleDataset(data_dir=args.data_dir)
    eval_data = dataset.get_split(args.split)

    if args.max_samples:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    total = len(eval_data)
    log.info("Evaluating on %d examples from '%s' split...", total, args.split)

    # Determine checkpoint indices at which to log sample predictions
    n_checkpoints = max(1, args.log_checkpoints)
    checkpoint_indices = {
        int(round((i / n_checkpoints) * total)) - 1
        for i in range(1, n_checkpoints + 1)
    }
    checkpoint_indices = {max(0, idx) for idx in checkpoint_indices}

    references = []
    predictions = []

    for step, example in enumerate(tqdm(eval_data, desc="Evaluating")):
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

        # Log rolling metrics + a sample at each checkpoint
        if step in checkpoint_indices:
            partial_wer = wer(references, predictions) * 100
            partial_cer = cer(references, predictions) * 100
            log.info(
                "Checkpoint [%d/%d] — WER: %.2f%%  CER: %.2f%%",
                step + 1, total, partial_wer, partial_cer,
            )
            log.info("  Sample ref : %s", references[-1])
            log.info("  Sample pred: %s", predictions[-1])

    # Final metrics
    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)

    results = {
        "type": "baseline",
        "model": args.model,
        "split": args.split,
        "num_examples": total,
        "wer": round(word_error_rate * 100, 2),
        "cer": round(char_error_rate * 100, 2),
    }

    log.info("=" * 50)
    log.info("BASELINE RESULTS: %s", args.model)
    log.info("=" * 50)
    log.info("  Split:    %s", args.split)
    log.info("  Samples:  %d", total)
    log.info("  WER:      %.2f%%", results["wer"])
    log.info("  CER:      %.2f%%", results["cer"])
    log.info("=" * 50)

    log.info("Sample predictions (first 5):")
    for i in range(min(5, len(references))):
        log.info("  [%d] ref : %s", i, references[i])
        log.info("  [%d] pred: %s", i, predictions[i])

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_short}_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", output_path)

    # Save detailed predictions
    details_path = output_dir / f"{model_short}_{args.split}_predictions.json"
    details = [
        {"reference": ref, "prediction": pred}
        for ref, pred in zip(references, predictions)
    ]
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    log.info("Detailed predictions saved to %s", details_path)


if __name__ == "__main__":
    main()
