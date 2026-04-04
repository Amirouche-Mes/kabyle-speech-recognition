# Project Progress

Tracking what has been done and what comes next.

## Completed

### Phase 1 - Repository Setup
- Initialized git, connected to GitHub remote
- Created full directory structure (`src/`, `scripts/`, `notebooks/`, `configs/`, `tests/`, `docs/`, `data/`)
- Added all `__init__.py` files for Python packages

### Phase 2 - Core Configuration
- `pyproject.toml` with UV-based dependency management (Python 3.12+)
- `.gitignore` covering Python, data, models, IDE, and env files
- `.env.example` template (HF_TOKEN, WANDB_API_KEY, CUDA config)
- `.pre-commit-config.yaml` (black, flake8, mypy)
- `.github/workflows/ci.yml` (CI on Python 3.12/3.13, lint, test, coverage)

### Phase 3 - Documentation
- Professional `README.md` with badges, structure, quick start, roadmap
- `LICENSE` (MIT)
- `CONTRIBUTING.md` with dev setup, code style, PR process
- `docs/SETUP.md` with detailed setup and troubleshooting

### Phase 4 - Initial Code (v2 - Local Dataset)
- `src/data/dataset.py` - KabyleDataset class loading from local Common Voice files (TSV + clips)
- `src/data/preprocessing.py` - Audio preprocessing, WhisperDataCollator, compute_metrics (WER/CER)
- `scripts/prepare_data.py` - Extract Common Voice archive locally
- `scripts/train.py` - Full training pipeline with data collator, WER metrics, and LoRA support
- `scripts/evaluate.py` - Evaluation script with local dataset loading
- `configs/whisper_small.yaml` - Full fine-tuning config (Common Voice v24.0)
- `configs/whisper_large_lora.yaml` - LoRA fine-tuning config (Common Voice v24.0)

### Phase 5 - Notebook
- `notebooks/01_data_exploration.ipynb` - Dataset exploration notebook

### Phase 6 - Testing
- `tests/conftest.py` - Pytest fixtures
- `tests/unit/test_dataset.py` - Dataset unit tests

## Key Decisions
- Using HuggingFace `transformers` Whisper (not `openai-whisper`) for training ecosystem integration
- Loading dataset from local Common Voice v24.0 archive (not HuggingFace Datasets)
- Python 3.12+ minimum
- Working on `dev` branch, merging to `main` via PRs
- RunPod for cloud GPU training

### Phase 7 - Local Pipeline Validation
- Updated all dataset/config paths from Common Voice v24.0 to **v25.0** (2026-03-09)
- Created `.env` with local testing settings (CPU/MPS, sample limits, no W&B auth)
- Created `configs/whisper_small_local.yaml` — Mac-safe config (fp16 off, batch 1, eval disabled)
- Fixed audio decoding: replaced `datasets.Audio` feature (requires torchcodec) with direct `soundfile` loading in `preprocessing.py`, `evaluate.py`, `baseline_eval.py`
- Added `src/utils/logging.py` — timestamped log files per run, routes transformers/datasets logs to file
- Wired logger into `train.py` (replaces all prints) and `src/data/dataset.py`
- Added `MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES` env var support in `train.py` for local slice testing
- Made `eval_strategy`, `bf16`, `load_best_model_at_end` configurable from YAML
- Added top-level `try/except` in `train.py` to capture main-process errors in log file
- Created `scripts/run_pipeline.py` — orchestrates baseline eval → train → post eval → comparison table
- Added `--max-samples` flag to `evaluate.py`
- **Validated full local pipeline**: 100 train samples, 1 epoch, MPS backend, loss 6.0 → 3.4, ~2 min total

## Key Decisions
- Using HuggingFace `transformers` Whisper (not `openai-whisper`) for training ecosystem integration
- Loading dataset from local Common Voice v25.0 archive (not HuggingFace Datasets)
- Audio decoded with `soundfile` directly — avoids `torchcodec` dependency in datasets 4.x
- Python 3.12+ minimum
- Working on `dev` branch, merging to `main` via PRs
- RunPod for cloud GPU training

## Next Steps
1. Run full benchmark pipeline locally: baseline WER → train → post WER
2. Set up RunPod instance and transfer dataset + code
3. Run first full training experiment (Whisper Small, 152k samples, 10 epochs)
4. Run LoRA fine-tuning experiment (Whisper Large v3)
5. Add comprehensive test coverage
6. Create demo/inference script
7. Hyperparameter tuning experiments
