# Setup Guide

Detailed setup instructions for the Kabyle Speech Recognition project.

## System Requirements

- **OS**: Linux (recommended for training), macOS (development)
- **Python**: 3.12 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **RAM**: 16GB+ recommended
- **Disk**: 20GB+ free space for dataset and models

## Installing UV

UV is a fast Python package manager. Install it with:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

Verify installation:

```bash
uv --version
```

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Amirouche-Mes/kabyle-speech-recognition.git
cd kabyle-speech-recognition
```

### 2. Install Dependencies

```bash
# Install all dependencies including dev tools
uv pip install -e ".[dev]"
```

### 3. Configure Environment

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` and add your tokens:

- **HF_TOKEN**: Get from [HuggingFace Settings](https://huggingface.co/settings/tokens). Required for downloading the Common Voice dataset. You must also accept the dataset terms at [Common Voice on HuggingFace](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0).
- **WANDB_API_KEY** (optional): Get from [W&B Settings](https://wandb.ai/settings) for experiment tracking.

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

### 5. Verify Setup

```bash
# Run tests
pytest tests/

# Check formatting
black --check src/ tests/ scripts/

# Run linter
flake8 src/ tests/ scripts/
```

## Troubleshooting

### CUDA not detected

Ensure you have the correct PyTorch version for your CUDA version:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with specific CUDA version (example for CUDA 12.1)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### HuggingFace authentication errors

1. Verify your token is valid at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept the Common Voice dataset terms on HuggingFace
3. Ensure `HF_TOKEN` is set in your `.env` file

### Out of memory during training

- Reduce batch size in the config file
- Use LoRA fine-tuning (`configs/whisper_large_lora.yaml`)
- Use gradient accumulation steps
- Consider using `bitsandbytes` for 8-bit training
