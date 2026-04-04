"""Logging setup for the Kabyle Speech Recognition project."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Create a logger that writes to both console and a timestamped log file.

    Each call with a new run creates a file like:
        logs/train_2026-04-04_14-32-05.log

    Args:
        name: Logger name, also used as the log filename prefix (e.g. 'train', 'evaluate').
        log_dir: Directory where log files are saved.

    Returns:
        Configured logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if logger is reused
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # File handler — DEBUG and above (full detail)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    # Also route transformers/datasets library logs to the same file
    for lib in ("transformers", "datasets"):
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.INFO)
        lib_logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")
    return logger
