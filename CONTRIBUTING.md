# Contributing to Kabyle Speech Recognition

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions. We are committed to providing a welcoming experience for everyone.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Open a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU)

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Explain why it would benefit the project

### Submitting Changes

1. Fork the repository
2. Create a feature branch from `dev`:
   ```bash
   git checkout -b feature/your-feature dev
   ```
3. Make your changes following the code style guidelines
4. Write or update tests as needed
5. Submit a pull request to the `dev` branch

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/kabyle-speech-recognition.git
cd kabyle-speech-recognition

# Install with dev dependencies using UV
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Flake8
- **Type checking**: mypy
- **Docstrings**: Google style
- **Type hints**: Required for all public functions

Run checks locally before submitting:

```bash
black --check src/ tests/ scripts/
flake8 src/ tests/ scripts/
mypy src/
pytest tests/
```

## Testing

- Write tests for all new functionality
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Run tests with: `pytest tests/`

## Pull Request Process

1. Ensure all CI checks pass
2. Update documentation if needed
3. Add a clear description of changes
4. Request review from maintainers
5. PRs are merged into `dev`, then periodically into `main`
