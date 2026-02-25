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

## Next Steps
1. Extract the dataset archive on RunPod and verify loading
2. Run first training experiment (Whisper Small)
3. Add comprehensive test coverage
4. Build audio preprocessing pipeline (normalization, filtering)
5. Run LoRA fine-tuning experiment (Whisper Large v3)
6. Create demo/inference script
7. Add more exploration notebooks
8. Hyperparameter tuning experiments
