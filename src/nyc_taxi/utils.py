"""Cross-cutting utilities: logging, reproducibility and device selection."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Final, TextIO

import numpy as np
import torch

__all__ = [
    "count_parameters",
    "get_device",
    "get_logger",
    "seed_everything",
    "setup_logging",
]

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure the root logger once, for both scripts and notebooks.

    Existing handlers are replaced so that repeated calls (common when a
    notebook cell is re-executed) never duplicate log lines.

    Args:
        level: Logging level as an ``int`` or a name such as ``"DEBUG"``.
        log_file: Optional file to mirror the log stream into. Parent
            directories are created automatically.
        stream: Stream to log to. Defaults to :data:`sys.stdout` rather than
            :data:`sys.stderr`: informational logs are not errors, and Jupyter
            renders anything on stderr with an error-red background, which makes
            a successful run look like a failed one.

    Raises:
        OSError: If ``log_file`` is given and its directory cannot be created.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(stream or sys.stdout)]
    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    # Third-party libraries are chatty at INFO level.
    for noisy in ("matplotlib", "PIL", "urllib3", "botocore", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        The configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed every source of randomness used by the pipeline.

    Args:
        seed: Seed value applied to ``random``, ``numpy``, ``torch`` (CPU and
            all CUDA devices) and the ``PYTHONHASHSEED`` environment variable.
        deterministic: If ``True``, force cuDNN into deterministic mode. This
            costs throughput and is intended for exact-reproduction runs only.

    Returns:
        The seed that was applied, for convenient logging.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    return seed


def get_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device from a config string.

    Args:
        preference: ``"auto"`` (CUDA when available, else CPU), ``"cuda"``,
            ``"cuda:N"`` or ``"cpu"``.

    Returns:
        The resolved :class:`torch.device`.

    Raises:
        ValueError: If ``preference`` is not a recognised device string.
        RuntimeError: If CUDA is requested explicitly but is unavailable.
    """
    normalised = preference.strip().lower()

    if normalised == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalised == "cpu":
        return torch.device("cpu")
    if normalised.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device {preference!r} was requested but CUDA is not available. "
                "Use training.device=cpu or training.device=auto."
            )
        return torch.device(normalised)

    raise ValueError(f"Unsupported device preference: {preference!r}.")


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the parameters of a module.

    Args:
        module: Module to inspect.
        trainable_only: Count only parameters with ``requires_grad=True``.

    Returns:
        The number of parameters.
    """
    params = module.parameters()
    return sum(p.numel() for p in params if p.requires_grad or not trainable_only)
