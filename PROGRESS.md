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

### Phase 4 - Initial Code
- `src/data/dataset.py` - KabyleDataset class wrapping Common Voice
- `src/data/preprocessing.py` - Audio preprocessing utilities
- `scripts/download_data.py` - CLI script to download dataset via HuggingFace
- `scripts/train.py` - Training script with Whisper + HF Trainer
- `scripts/evaluate.py` - Evaluation script (WER/CER metrics)
- `configs/whisper_small.yaml` and `configs/whisper_large_lora.yaml`

### Phase 5 - Notebook
- `notebooks/01_data_exploration.ipynb` - Dataset exploration notebook

### Phase 6 - Testing
- `tests/conftest.py` - Pytest fixtures
- `tests/unit/test_dataset.py` - Basic dataset tests

## Key Decisions
- Using HuggingFace `transformers` Whisper (not `openai-whisper`) for training ecosystem integration
- Dataset access via HuggingFace Datasets with `HF_TOKEN`
- Python 3.12+ minimum
- Working on `dev` branch, merging to `main` via PRs

## Next Steps
1. Implement complete training loop with full data pipeline
2. Add comprehensive test coverage
3. Build preprocessing pipeline (audio normalization, filtering)
4. Implement full evaluation suite with detailed metrics
5. Add more training configurations and hyperparameter experiments
6. Create demo/inference script
7. Add more exploration notebooks
8. Run first training experiments on cloud GPU
