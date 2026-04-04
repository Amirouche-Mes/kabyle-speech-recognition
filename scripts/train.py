"""Training script for Whisper fine-tuning on Kabyle speech data."""

import argparse
import os
from functools import partial
from pathlib import Path

import yaml
from dotenv import load_dotenv
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

from src.data.dataset import KabyleDataset
from src.data.preprocessing import (
    WhisperDataCollator,
    compute_metrics,
    get_processor,
    prepare_dataset,
)
from src.utils.logging import get_logger


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
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config",
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
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_lora(model: WhisperForConditionalGeneration, lora_config: dict) -> WhisperForConditionalGeneration:
    """Apply LoRA adapters to the model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    """Run the training pipeline."""
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    logger = get_logger("train")

    model_name = config["model"]["name"]
    output_dir = args.output_dir or config["training"]["output_dir"]
    data_dir = args.data_dir or config["data"].get("data_dir", "data/raw/cv-corpus-25.0-2026-03-09/kab")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Whisper Kabyle Fine-tuning")
    logger.info("=" * 50)
    logger.info(f"Model:      {model_name}")
    logger.info(f"Data dir:   {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Config:     {args.config}")

    # Load processor and model
    logger.info("Loading processor and model...")
    processor = get_processor(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    logger.info("Model loaded.")

    # Apply LoRA if configured
    if "lora" in config:
        logger.info("Applying LoRA adapters...")
        model = setup_lora(model, config["lora"])
        logger.info("LoRA applied.")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = KabyleDataset(data_dir=data_dir)
    data = dataset.load()
    logger.info(f"Dataset loaded — train: {len(data['train']):,} | val: {len(data['validation']):,}")

    # Apply sample limits from environment (for local testing)
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    train_split = data["train"]
    val_split = data["validation"]

    max_train = os.getenv("MAX_TRAIN_SAMPLES")
    max_eval = os.getenv("MAX_EVAL_SAMPLES")
    if max_train:
        train_split = train_split.select(range(min(int(max_train), len(train_split))))
        logger.info(f"[local] Limiting train to {len(train_split)} samples")
    if max_eval:
        val_split = val_split.select(range(min(int(max_eval), len(val_split))))
        logger.info(f"[local] Limiting eval to {len(val_split)} samples")

    # Preprocess
    logger.info("Preprocessing audio features...")
    train_data = prepare_dataset(train_split, processor, num_proc=num_workers)
    val_data = prepare_dataset(val_split, processor, num_proc=num_workers)
    logger.info("Preprocessing done.")

    # Data collator and metrics
    data_collator = WhisperDataCollator(processor=processor)
    metrics_fn = partial(compute_metrics, processor=processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"].get("warmup_steps", 500),
        num_train_epochs=config["training"]["epochs"],
        eval_strategy=config["training"].get("eval_strategy", "steps"),
        eval_steps=config["training"].get("eval_steps", 500),
        save_steps=config["training"].get("save_steps", 500),
        logging_steps=config["training"].get("logging_steps", 100),
        fp16=config["training"].get("fp16", True),
        bf16=config["training"].get("bf16", False),
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=config["training"].get("save_total_limit", 3),
        load_best_model_at_end=config["training"].get("eval_strategy", "steps") != "no",
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=config["training"].get("report_to", "tensorboard"),
    )

    logger.info("Training arguments:")
    logger.info(f"  batch_size={config['training']['batch_size']}")
    logger.info(f"  epochs={config['training']['epochs']}")
    logger.info(f"  learning_rate={config['training']['learning_rate']}")
    logger.info(f"  fp16={config['training'].get('fp16', True)} | bf16={config['training'].get('bf16', False)}")

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        processing_class=processor,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)
    logger.info("Training complete.")

    # Save final model
    trainer.save_model(f"{output_dir}/final")
    processor.save_pretrained(f"{output_dir}/final")
    logger.info(f"Model saved to {output_dir}/final")


if __name__ == "__main__":
    import logging
    import traceback

    try:
        main()
    except Exception:
        logging.getLogger("train").error(
            "Training failed with exception:\n" + traceback.format_exc()
        )
        raise
