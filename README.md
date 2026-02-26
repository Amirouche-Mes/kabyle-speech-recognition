# Kabyle Speech Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Fine-tuning OpenAI Whisper for Automatic Speech Recognition (ASR) on the **Kabyle language** using the Mozilla Common Voice dataset.

## Overview

Kabyle (Taqbaylit) is a Berber language spoken by millions of people, primarily in the Kabylie region of Algeria. This project aims to build a high-quality ASR system for Kabyle by fine-tuning pre-trained Whisper models on the Common Voice Kabyle dataset.

### Key Features

- Fine-tuning Whisper models (small, large) on Kabyle speech data
- LoRA-based efficient fine-tuning for resource-constrained training
- Comprehensive evaluation with WER and CER metrics
- Experiment tracking with Weights & Biases

## Dataset

This project uses the [Mozilla Common Voice](https://commonvoice.mozilla.org/) Kabyle dataset, accessed via HuggingFace Datasets.

| Split      | Hours | Sentences |
|------------|-------|-----------|
| Train      | TBD   | TBD       |
| Validation | TBD   | TBD       |
| Test       | TBD   | TBD       |

## Project Structure

```
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model architectures and configs
│   ├── training/      # Training utilities
│   └── evaluation/    # Evaluation metrics
├── scripts/           # Executable scripts (download, train, eval)
├── notebooks/         # Jupyter notebooks for exploration
├── configs/           # Training configuration files (YAML)
├── tests/             # Unit and integration tests
├── docs/              # Additional documentation
└── data/              # Data directory (git-ignored)
```

## Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager
- GPU recommended for training (NVIDIA with CUDA)

### Installation

```bash
# Clone the repository
git clone https://github.com/Amirouche-Mes/kabyle-speech-recognition.git
cd kabyle-speech-recognition

# Install with UV
uv pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your HuggingFace token

# Install pre-commit hooks
pre-commit install
```

### Usage

```bash
# Download the Kabyle dataset
python scripts/download_data.py --language kab

# Train a model
python scripts/train.py --config configs/whisper_small.yaml

# Evaluate a trained model
python scripts/evaluate.py --model-path checkpoints/best_model
```

## Results

### Baseline (pre-trained, no fine-tuning)

| Model              | WER (%) | CER (%) |
|--------------------|---------|---------|
| Whisper Small      | TBD     | TBD     |
| Whisper Large v3   | TBD     | TBD     |

### After Fine-tuning

| Model                  | WER (%) | CER (%) | Training Time |
|------------------------|---------|---------|---------------|
| Whisper Small          | TBD     | TBD     | TBD           |
| Whisper Large + LoRA   | TBD     | TBD     | TBD           |

## Roadmap

- [x] **Phase 1 - Foundation**: Project structure, dependencies, CI/CD
- [ ] **Phase 2 - Training**: Complete training pipeline
- [ ] **Phase 3 - Optimization**: Hyperparameter tuning, LoRA experiments
- [ ] **Phase 4 - Deployment**: Inference API, demo application

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## References

- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Mozilla Common Voice contributors for the Kabyle speech dataset
- OpenAI for the Whisper model
- HuggingFace for the Transformers and Datasets libraries

## Citation

```bibtex
@misc{kabyle-asr-2026,
  title={Kabyle Speech Recognition with Whisper Fine-tuning},
  author={Amirouche Mes},
  year={2026},
  url={https://github.com/Amirouche-Mes/kabyle-speech-recognition}
}
```
