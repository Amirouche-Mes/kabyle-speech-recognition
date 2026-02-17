"""Training script for Whisper fine-tuning on Kabyle speech data."""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

from src.data.dataset import KabyleDataset
from src.data.preprocessing import get_processor, prepare_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Kabyle speech data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run the training pipeline."""
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    model_name = config["model"]["name"]
    output_dir = args.output_dir or config["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")

    # Load processor and model
    processor = get_processor(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = "kabyle"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Load and preprocess dataset
    hf_token = os.getenv("HF_TOKEN")
    dataset = KabyleDataset(hf_token=hf_token)
    data = dataset.load()

    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    train_data = prepare_dataset(data["train"], processor, num_proc=num_workers)
    val_data = prepare_dataset(data["validation"], processor, num_proc=num_workers)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"].get("warmup_steps", 500),
        num_train_epochs=config["training"]["epochs"],
        eval_strategy="steps",
        eval_steps=config["training"].get("eval_steps", 500),
        save_steps=config["training"].get("save_steps", 500),
        logging_steps=config["training"].get("logging_steps", 100),
        fp16=config["training"].get("fp16", True),
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=config["training"].get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=config["training"].get("report_to", "tensorboard"),
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=processor,
    )

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save final model
    trainer.save_model(f"{output_dir}/final")
    processor.save_pretrained(f"{output_dir}/final")
    print(f"Training complete! Model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
