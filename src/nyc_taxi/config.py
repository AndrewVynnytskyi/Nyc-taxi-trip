"""Configuration loading and validation.

Every hyperparameter, path and business rule lives in ``configs/*.yaml``. This
module is the only sanctioned way to read them, which guarantees that no module
in :mod:`nyc_taxi` -- nor any notebook -- carries a hardcoded value.

Example:
    >>> from nyc_taxi.config import load_config
    >>> cfg = load_config(overrides=["training.max_epochs=5"])
    >>> cfg.training.max_epochs
    5
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["ConfigError", "get_project_root", "load_config", "save_config"]

#: Marker files used to locate the repository root from an arbitrary CWD
#: (notebooks run from ``notebooks/``, scripts from the repo root, containers
#: from ``/app``).
_ROOT_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", ".git")

#: Default config file, relative to the project root.
DEFAULT_CONFIG_PATH: Final[str] = "configs/train.yaml"

#: Dotted config keys holding a filesystem path that must be absolute after
#: loading. Derived keys (e.g. ``training.checkpoint_dir``) are listed too: by
#: the time interpolation has run, they already contain the *resolved* relative
#: value, so fixing only their source key would leave them CWD-dependent -- which
#: silently writes artifacts next to whatever directory the process started in.
_PATH_KEYS: Final[tuple[str, ...]] = (
    "project.data_dir",
    "project.artifact_dir",
    "training.checkpoint_dir",
    "data.cache_path",
    "etl.source.download_dir",
)


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing, malformed or incomplete."""


def get_project_root(start: Path | str | None = None) -> Path:
    """Locate the repository root by walking upwards for a marker file.

    Args:
        start: Directory to start the upward search from. Defaults to the
            directory containing this module.

    Returns:
        Absolute path to the repository root.

    Raises:
        ConfigError: If no marker file is found in any parent directory.
    """
    env_root = os.environ.get("NYC_TAXI_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    origin = Path(start).resolve() if start is not None else Path(__file__).resolve()
    candidates: Iterable[Path] = (origin, *origin.parents)
    for candidate in candidates:
        if candidate.is_dir() and any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate

    raise ConfigError(
        f"Could not locate the project root from {origin!s}: none of {_ROOT_MARKERS} were found. "
        "Set NYC_TAXI_PROJECT_ROOT to override."
    )


def load_config(
    config_path: Path | str | None = None,
    overrides: Sequence[str] | None = None,
    resolve: bool = True,
) -> DictConfig:
    """Load, override and validate the project configuration.

    Args:
        config_path: Path to the YAML config. Relative paths are resolved
            against the project root. Defaults to ``configs/train.yaml``.
        overrides: OmegaConf dotlist overrides, as given on the command line, e.g.
            ``["training.max_epochs=50", "optimizer.lr=3e-4"]``.
        resolve: Whether to eagerly resolve interpolations (``${...}``). Set to
            ``False`` to keep the config lazy, e.g. before further composition.

    Returns:
        The validated, read-only-by-convention configuration tree.

    Raises:
        ConfigError: If the file does not exist, is not a mapping, contains an
            unresolved mandatory value, or fails structural validation.
    """
    root = get_project_root()
    path = Path(config_path) if config_path is not None else Path(DEFAULT_CONFIG_PATH)
    if not path.is_absolute():
        path = root / path

    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    cfg = _read_yaml(path)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    if resolve:
        _resolve_in_place(cfg, path)

    _normalise_env_sentinels(cfg)
    _absolutise_paths(cfg, root)
    _validate(cfg)
    return cfg


def save_config(cfg: DictConfig, destination: Path | str) -> Path:
    """Persist a resolved config next to a run's artifacts for reproducibility.

    Args:
        cfg: Configuration to serialise.
        destination: Target ``.yaml`` file. Parent directories are created.

    Returns:
        The absolute path the config was written to.

    Raises:
        ConfigError: If the file cannot be written.
    """
    target = Path(destination).expanduser().resolve()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to write configuration to {target}: {exc}") from exc
    return target


def _read_yaml(path: Path) -> DictConfig:
    """Parse a YAML file into a :class:`DictConfig`.

    Args:
        path: File to parse.

    Returns:
        The parsed config.

    Raises:
        ConfigError: If the file cannot be parsed or its root is not a mapping.
    """
    try:
        cfg = OmegaConf.load(path)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error.
        raise ConfigError(f"Failed to parse configuration file {path}: {exc}") from exc

    if not isinstance(cfg, DictConfig):
        raise ConfigError(f"Configuration root must be a mapping, got {type(cfg).__name__}.")

    return cfg


def _apply_overrides(cfg: DictConfig, overrides: Sequence[str]) -> DictConfig:
    """Merge OmegaConf command-line (dotlist) overrides into a config.

    Args:
        cfg: Base configuration.
        overrides: Dotlist entries such as ``"optimizer.lr=3e-4"``.

    Returns:
        The merged configuration.

    Raises:
        ConfigError: If an override is malformed.
    """
    try:
        merged = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error.
        raise ConfigError(f"Invalid override in {list(overrides)!r}: {exc}") from exc

    if not isinstance(merged, DictConfig):  # pragma: no cover - merge preserves the type.
        raise ConfigError("Applying overrides produced a non-mapping configuration.")
    return merged


def _resolve_in_place(cfg: DictConfig, path: Path) -> None:
    """Eagerly resolve every ``${...}`` interpolation.

    Args:
        cfg: Configuration tree, mutated in place.
        path: Source file, used for the error message.

    Raises:
        ConfigError: If an interpolation cannot be resolved.
    """
    try:
        OmegaConf.resolve(cfg)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error.
        raise ConfigError(f"Failed to resolve interpolations in {path}: {exc}") from exc


def _normalise_env_sentinels(cfg: DictConfig) -> None:
    """Convert the string sentinels produced by ``${oc.env:...}`` into Python values.

    Environment variables are always strings, so ``NYC_TAXI_ROW_LIMIT=null``
    and an unset default of ``null`` must both become :data:`None`.

    Args:
        cfg: Configuration tree, mutated in place.
    """
    sentinels = {"null": None, "none": None, "true": True, "false": False}

    def _coerce(value: Any) -> Any:
        """Return the Python value for a sentinel string, else the value itself."""
        if isinstance(value, str) and value.strip().lower() in sentinels:
            return sentinels[value.strip().lower()]
        return value

    def _walk(node: DictConfig | ListConfig) -> None:
        if isinstance(node, DictConfig):
            for key in list(node.keys()):
                value = node[key]
                if isinstance(value, (DictConfig, ListConfig)):
                    _walk(value)
                else:
                    node[key] = _coerce(value)
            return

        for index in range(len(node)):
            item = node[index]
            if isinstance(item, (DictConfig, ListConfig)):
                _walk(item)
            else:
                node[index] = _coerce(item)

    _walk(cfg)


def _absolutise_paths(cfg: DictConfig, root: Path) -> None:
    """Rewrite relative directory settings as absolute paths under ``root``.

    Args:
        cfg: Configuration tree, mutated in place.
        root: Repository root used as the base for relative paths.
    """
    for dotted_key in _PATH_KEYS:
        raw = OmegaConf.select(cfg, dotted_key)
        if isinstance(raw, str) and raw and not Path(raw).is_absolute():
            OmegaConf.update(cfg, dotted_key, str((root / raw).resolve()), merge=False)


def _validate(cfg: DictConfig) -> None:
    """Assert that mandatory sections and invariants are present and sane.

    Args:
        cfg: Fully resolved configuration tree.

    Raises:
        ConfigError: If a required section is missing or a value is out of range.
    """
    required_sections = (
        "project",
        "etl",
        "database",
        "data",
        "model",
        "training",
        "optimizer",
        "scheduler",
        "wandb",
    )
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ConfigError(f"Configuration is missing required section(s): {missing}")

    split_total = float(cfg.data.val_size) + float(cfg.data.test_size)
    if not 0.0 < split_total < 1.0:
        raise ConfigError(
            f"data.val_size + data.test_size must lie in (0, 1), got {split_total:.3f}."
        )

    if int(cfg.training.max_epochs) < 1:
        raise ConfigError(f"training.max_epochs must be >= 1, got {cfg.training.max_epochs}.")

    if float(cfg.optimizer.lr) <= 0.0:
        raise ConfigError(f"optimizer.lr must be > 0, got {cfg.optimizer.lr}.")

    if cfg.training.monitor_mode not in {"min", "max"}:
        raise ConfigError(
            f"training.monitor_mode must be 'min' or 'max', got {cfg.training.monitor_mode!r}."
        )

    if int(cfg.etl.chunk_size) < 1:
        raise ConfigError(f"etl.chunk_size must be >= 1, got {cfg.etl.chunk_size}.")
