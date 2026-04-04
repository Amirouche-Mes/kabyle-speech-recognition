"""Full benchmark pipeline: baseline → train → evaluate → compare.

Runs all three benchmark steps in sequence and prints a comparison table.
Works for both local testing and RunPod full runs.

Usage:
    # Local smoke test (uses MAX_TRAIN_SAMPLES / MAX_EVAL_SAMPLES from .env)
    python scripts/run_pipeline.py --config configs/whisper_small_local.yaml

    # Full RunPod run
    python scripts/run_pipeline.py --config configs/whisper_small.yaml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full benchmark pipeline")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit eval samples for baseline and post-training eval (overrides .env)",
    )
    return parser.parse_args()


def run_step(cmd: list[str], logger, step_name: str) -> None:
    """Run a subprocess step and stream its output to the logger."""
    logger.info(f">>> STEP: {step_name}")
    logger.info(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Step '{step_name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    logger.info(f"<<< STEP DONE: {step_name}")


def load_results(path: str) -> dict | None:
    """Load a JSON results file, return None if not found."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def print_comparison(baseline: dict | None, post: dict | None, logger) -> None:
    """Print a side-by-side WER/CER comparison table."""
    logger.info("")
    logger.info("=" * 55)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 55)
    logger.info(f"{'Stage':<25} {'WER (%)':>10} {'CER (%)':>10}")
    logger.info("-" * 55)
    if baseline:
        logger.info(
            f"{'Baseline (no training)':<25} "
            f"{baseline['wer']:>10.2f} "
            f"{baseline['cer']:>10.2f}"
        )
    else:
        logger.info(f"{'Baseline (no training)':<25} {'N/A':>10} {'N/A':>10}")
    if post:
        logger.info(
            f"{'After fine-tuning':<25} "
            f"{post['wer']:>10.2f} "
            f"{post['cer']:>10.2f}"
        )
    else:
        logger.info(f"{'After fine-tuning':<25} {'N/A':>10} {'N/A':>10}")
    if baseline and post:
        wer_delta = post["wer"] - baseline["wer"]
        cer_delta = post["cer"] - baseline["cer"]
        sign = lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}"
        logger.info("-" * 55)
        logger.info(f"{'Improvement':<25} {sign(wer_delta):>10} {sign(cer_delta):>10}")
        logger.info("  (negative = better)")
    logger.info("=" * 55)


def main() -> None:
    load_dotenv()
    args = parse_args()
    logger = get_logger("pipeline")

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    output_dir = config["training"]["output_dir"]
    data_dir = args.data_dir or config["data"].get(
        "data_dir", "data/raw/cv-corpus-25.0-2026-03-09/kab"
    )
    model_short = model_name.replace("/", "_")
    baseline_result_path = f"results/baseline/{model_short}_test.json"
    post_result_path = f"results/post_training/{model_short}_test.json"

    logger.info("=" * 55)
    logger.info("KABYLE ASR BENCHMARK PIPELINE")
    logger.info("=" * 55)
    logger.info(f"Model:   {model_name}")
    logger.info(f"Config:  {args.config}")
    logger.info(f"Data:    {data_dir}")

    python = sys.executable

    # ── STEP 1: Baseline eval ──────────────────────────────────
    baseline_cmd = [
        python, "scripts/baseline_eval.py",
        "--model", model_name,
        "--data-dir", data_dir,
        "--output", "results/baseline",
    ]
    if args.max_samples:
        baseline_cmd += ["--max-samples", str(args.max_samples)]
        

    run_step(baseline_cmd, logger, "Baseline evaluation (before training)")

    # ── STEP 2: Training ───────────────────────────────────────
    train_cmd = [python, "scripts/train.py", "--config", args.config]
    if args.data_dir:
        train_cmd += ["--data-dir", args.data_dir]

    run_step(train_cmd, logger, "Fine-tuning")

    # ── STEP 3: Post-training eval ─────────────────────────────
    post_cmd = [
        python, "scripts/evaluate.py",
        "--model-path", f"{output_dir}/final",
        "--data-dir", data_dir,
        "--output", f"results/post_training/{model_short}_test.json",
    ]
    if args.max_samples:
        post_cmd += ["--max-samples", str(args.max_samples)]

    run_step(post_cmd, logger, "Post-training evaluation")

    # ── COMPARISON ─────────────────────────────────────────────
    baseline = load_results(baseline_result_path)
    post = load_results(post_result_path)
    print_comparison(baseline, post, logger)


if __name__ == "__main__":
    main()


# =============================================================================
# MANUAL STEP-BY-STEP EXECUTION (for running in a Python terminal)
# Copy/paste each block one at a time to step through the pipeline manually.
# =============================================================================

# -- SETUP (run this first) ---------------------------------------------------
# import argparse, yaml, sys
# from dotenv import load_dotenv
# from src.utils.logging import get_logger
#
# load_dotenv()
# logger = get_logger("pipeline")
#
# args = argparse.Namespace(
#     config="configs/whisper_small_local.yaml",
#     data_dir=None,
#     max_samples=50,
# )
#
# with open(args.config) as f:
#     config = yaml.safe_load(f)
#
# model_name  = config["model"]["name"]
# output_dir  = config["training"]["output_dir"]
# data_dir    = config["data"].get("data_dir", "data/raw/cv-corpus-25.0-2026-03-09/kab")
# model_short = model_name.replace("/", "_")
# python      = sys.executable
#
# logger.info(f"model_name  = {model_name}")
# logger.info(f"output_dir  = {output_dir}")
# logger.info(f"data_dir    = {data_dir}")
# logger.info(f"model_short = {model_short}")

# -- STEP 1: Baseline eval ----------------------------------------------------
# baseline_cmd = [
#     python, "scripts/baseline_eval.py",
#     "--model",    model_name,
#     "--data-dir", data_dir,
#     "--output",   "results/baseline",
#     "--max-samples", str(args.max_samples),
# ]
# logger.info(f"Running: {' '.join(baseline_cmd)}")
# run_step(baseline_cmd, logger, "Baseline evaluation (before training)")

# -- STEP 2: Training ---------------------------------------------------------
# train_cmd = [python, "scripts/train.py", "--config", args.config]
# logger.info(f"Running: {' '.join(train_cmd)}")
# run_step(train_cmd, logger, "Fine-tuning")

# -- STEP 3: Post-training eval -----------------------------------------------
# post_cmd = [
#     python, "scripts/evaluate.py",
#     "--model-path", f"{output_dir}/final",
#     "--data-dir",   data_dir,
#     "--output",     f"results/post_training/{model_short}_test.json",
#     "--max-samples", str(args.max_samples),
# ]
# logger.info(f"Running: {' '.join(post_cmd)}")
# run_step(post_cmd, logger, "Post-training evaluation")

# -- COMPARISON ---------------------------------------------------------------
# baseline = load_results(f"results/baseline/{model_short}_test.json")
# post     = load_results(f"results/post_training/{model_short}_test.json")
# print_comparison(baseline, post, logger)