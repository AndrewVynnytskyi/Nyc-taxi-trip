"""Dataset, feature engineering and DataLoader construction.

The clean table produced by :mod:`nyc_taxi.etl` is the single source of truth
for training data. :class:`TaxiDataModule` owns the whole path from SQL (or a
parquet cache) to three ready-to-consume :class:`~torch.utils.data.DataLoader`
objects, and it fits the feature/target scalers on the training split only, so
no validation or test statistic ever leaks into training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from nyc_taxi.database import build_engine, read_table
from nyc_taxi.utils import get_logger

__all__ = ["DataError", "FeatureSpec", "TaxiDataModule", "TaxiTripDataset"]

logger = get_logger(__name__)

#: Alias for the float32 arrays exchanged between pandas, numpy and torch.
FloatArray = npt.NDArray[np.float32]

#: Split names in a fixed order, used for logging and loader lookup.
_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")


class DataError(RuntimeError):
    """Raised when data cannot be loaded, is empty, or violates the contract."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable description of the model's input contract.

    Attributes:
        names: Ordered feature-matrix column names, after cyclical encoding.
        target: Name of the regression target column.
        log_transform_target: Whether the target was ``log1p``-transformed.
    """

    names: tuple[str, ...]
    target: str
    log_transform_target: bool

    @property
    def input_dim(self) -> int:
        """Number of input features the model must accept."""
        return len(self.names)


class TaxiTripDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory tensor dataset of taxi trips.

    The full clean table fits comfortably in RAM as ``float32``, so features and
    targets are materialised once as contiguous tensors; indexing is then a
    view, not a copy.

    Args:
        features: Feature matrix of shape ``(n_samples, n_features)``.
        targets: Target vector of shape ``(n_samples,)``.
        spec: The feature contract these arrays were built against.

    Raises:
        DataError: If the arrays are empty, mis-shaped, or contain non-finite
            values.
    """

    def __init__(self, features: FloatArray, targets: FloatArray, spec: FeatureSpec) -> None:
        if features.ndim != 2:
            raise DataError(f"features must be 2-D, got shape {features.shape}.")
        if len(features) != len(targets):
            raise DataError(
                f"features and targets disagree: {len(features)} vs {len(targets)} rows."
            )
        if len(features) == 0:
            raise DataError("Refusing to build an empty dataset.")
        if features.shape[1] != spec.input_dim:
            raise DataError(
                f"features has {features.shape[1]} columns but the spec declares {spec.input_dim}."
            )
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise DataError("features/targets contain NaN or infinite values.")

        self.spec = spec
        self.features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        self.targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32)).view(-1, 1)

    def __len__(self) -> int:
        """Return the number of samples."""
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(features, target)`` pair at ``index``."""
        return self.features[index], self.targets[index]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_samples={len(self):,}, "
            f"n_features={self.spec.input_dim}, target={self.spec.target!r})"
        )


class TaxiDataModule:
    """Owns the data lifecycle: load, engineer, split, scale and batch.

    Typical use::

        module = TaxiDataModule(cfg)
        module.setup()
        train_loader = module.train_dataloader()

    Args:
        cfg: Fully resolved project configuration.

    Attributes:
        cfg: The configuration driving loading and batching.
        spec: The feature contract; available after :meth:`setup`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._frame: pd.DataFrame | None = None
        self._datasets: dict[str, TaxiTripDataset] = {}
        self._feature_scaler: StandardScaler | None = None
        self._target_scaler: StandardScaler | None = None
        self._spec: FeatureSpec | None = None
        self._splits: dict[str, pd.DataFrame] | None = None

    # -- Properties ----------------------------------------------------------

    @property
    def spec(self) -> FeatureSpec:
        """The feature contract established during :meth:`setup`.

        Raises:
            DataError: If accessed before :meth:`setup` has run.
        """
        if self._spec is None:
            raise DataError("Feature spec is unavailable; call setup() first.")
        return self._spec

    @property
    def splits(self) -> dict[str, pd.DataFrame]:
        """The engineered train/val/test frames the datasets were built from.

        Each frame holds the ``spec.names`` features, the target and the raw
        cyclical columns (e.g. ``pickup_hour``), with the shuffled index left
        by the split. These are the exact rows the model sees, so baselines
        computed on them are directly comparable.

        Raises:
            DataError: If accessed before :meth:`setup` has run.
        """
        if self._splits is None:
            raise DataError("Data splits are unavailable; call setup() first.")
        return self._splits

    @property
    def input_dim(self) -> int:
        """Number of input features the model must accept."""
        return self.spec.input_dim

    @property
    def frame(self) -> pd.DataFrame:
        """The raw clean-table DataFrame backing the splits.

        Raises:
            DataError: If accessed before :meth:`setup` has run.
        """
        if self._frame is None:
            raise DataError("No DataFrame is loaded; call setup() first.")
        return self._frame

    # -- Public API ----------------------------------------------------------

    def setup(self, frame: pd.DataFrame | None = None) -> "TaxiDataModule":
        """Prepare the train/val/test datasets.

        Args:
            frame: Pre-loaded clean data, mainly for testing. When omitted the
                data is read from the parquet cache or from PostgreSQL.

        Returns:
            ``self``, so the call chains.

        Raises:
            DataError: If the data is empty or a configured column is missing.
        """
        self._frame = frame if frame is not None else self.load_frame()
        engineered, self._spec = self.engineer_features(self._frame)

        splits = self._split(engineered)
        self._splits = splits
        feature_names = list(self._spec.names)
        target = self._spec.target

        x_train = splits["train"][feature_names].to_numpy(dtype=np.float32)
        self._feature_scaler = self._fit_feature_scaler(x_train)
        self._target_scaler = self._fit_target_scaler(
            splits["train"][target].to_numpy(dtype=np.float32)
        )

        for split in _SPLITS:
            part = splits[split]
            features = self.transform_features(part[feature_names].to_numpy(dtype=np.float32))
            targets = self.transform_target(part[target].to_numpy(dtype=np.float32))
            self._datasets[split] = TaxiTripDataset(features, targets, self._spec)
            logger.info("Split %-5s -> %s samples", split, f"{len(self._datasets[split]):,}")

        logger.info("Feature contract: %d inputs %s", self._spec.input_dim, feature_names)
        return self

    def load_frame(self) -> pd.DataFrame:
        """Load the clean table, preferring a local parquet cache.

        Returns:
            The clean-trip DataFrame.

        Raises:
            DataError: If the resulting frame is empty.
        """
        cache_path = Path(str(self.cfg.data.cache_path))
        use_cache = bool(self.cfg.data.use_cache)

        if use_cache and cache_path.is_file():
            logger.info("Loading cached data from %s", cache_path)
            frame = pd.read_parquet(cache_path)
        else:
            row_limit = self.cfg.data.get("row_limit")
            engine = build_engine(self.cfg)
            try:
                frame = read_table(
                    engine,
                    str(self.cfg.data.table),
                    limit=int(row_limit) if row_limit is not None else None,
                )
            finally:
                engine.dispose()

            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(cache_path, index=False)
                logger.info("Cached %s rows to %s", f"{len(frame):,}", cache_path)

        if frame.empty:
            raise DataError(
                f"Table {self.cfg.data.table!r} returned no rows. Run `nyc-taxi-etl` first."
            )
        return frame

    def engineer_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
        """Build the model-ready feature frame and its contract.

        Cyclical columns (hour of day, day of week) are encoded as sine/cosine
        pairs so that midnight sits next to 23:00 in feature space. The raw
        cyclical columns are kept in the frame for bucket baselines; the model
        input is governed by ``spec.names`` alone, which excludes them.

        Args:
            frame: Clean-table DataFrame.

        Returns:
            A tuple of the engineered frame (features, raw cyclical columns and
            target) and the :class:`FeatureSpec` describing it.

        Raises:
            DataError: If a configured feature or the target column is missing,
                or if no finite rows remain.
        """
        data_cfg = self.cfg.data
        numerical = [str(name) for name in data_cfg.numerical_features]
        cyclical: Mapping[str, int] = {
            str(name): int(period) for name, period in data_cfg.cyclical_features.items()
        }
        target = str(data_cfg.target)

        required = [*numerical, *cyclical, target]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise DataError(
                f"Columns {missing} are configured but absent from the source data. "
                f"Available: {sorted(frame.columns)}"
            )

        engineered = frame.loc[:, required].copy()
        feature_names: list[str] = list(numerical)

        for column, period in cyclical.items():
            radians = 2.0 * np.pi * engineered[column].to_numpy(dtype=np.float32) / float(period)
            engineered[f"{column}_sin"] = np.sin(radians)
            engineered[f"{column}_cos"] = np.cos(radians)
            feature_names.extend((f"{column}_sin", f"{column}_cos"))

        before = len(engineered)
        engineered = engineered.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[*feature_names, target]
        )
        if len(engineered) < before:
            logger.warning("Dropped %s non-finite row(s)", f"{before - len(engineered):,}")
        if engineered.empty:
            raise DataError("No finite rows remain after feature engineering.")

        spec = FeatureSpec(
            names=tuple(feature_names),
            target=target,
            log_transform_target=bool(data_cfg.log_transform_target),
        )
        return engineered.reset_index(drop=True), spec

    def transform_features(self, features: FloatArray) -> FloatArray:
        """Apply the fitted feature scaler.

        Args:
            features: Raw feature matrix.

        Returns:
            The standardised matrix, or the input unchanged when
            ``data.standardize_features`` is disabled.
        """
        if self._feature_scaler is None:
            return features.astype(np.float32, copy=False)
        return np.asarray(self._feature_scaler.transform(features), dtype=np.float32)

    def transform_target(self, targets: FloatArray) -> FloatArray:
        """Apply the target transform: optional ``log1p`` then standardisation.

        Args:
            targets: Raw target vector in the target's original unit.

        Returns:
            The transformed target vector.
        """
        values = targets.astype(np.float32, copy=True)
        if self.spec.log_transform_target:
            values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
        if self._target_scaler is not None:
            values = self._target_scaler.transform(values.reshape(-1, 1)).ravel()
        return np.asarray(values, dtype=np.float32)

    def inverse_transform_target(self, targets: FloatArray | torch.Tensor) -> FloatArray:
        """Map model-space predictions back to the target's original unit.

        This is the exact inverse of :meth:`transform_target`, which is what
        makes reported RMSE/MAE interpretable in the target's own unit
        (USD for ``fare_amount``, minutes for ``trip_duration_minutes``).

        Args:
            targets: Predictions or targets in model space.

        Returns:
            The values in the target's original unit, as ``float32``.
        """
        values = (
            targets.detach().cpu().numpy()
            if isinstance(targets, torch.Tensor)
            else np.asarray(targets)
        )
        values = values.astype(np.float32, copy=True).reshape(-1)
        if self._target_scaler is not None:
            values = self._target_scaler.inverse_transform(values.reshape(-1, 1)).ravel()
        if self.spec.log_transform_target:
            values = np.expm1(values)
        return np.asarray(values, dtype=np.float32)

    def dataset(self, split: str) -> TaxiTripDataset:
        """Return the dataset for a split.

        Args:
            split: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            The requested dataset.

        Raises:
            DataError: If the split name is unknown or :meth:`setup` has not run.
        """
        if split not in self._datasets:
            raise DataError(
                f"Unknown or unprepared split {split!r}; expected one of {_SPLITS} after setup()."
            )
        return self._datasets[split]

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Return the shuffled training loader."""
        return self._build_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Return the deterministic validation loader."""
        return self._build_loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Return the deterministic test loader."""
        return self._build_loader("test", shuffle=False)

    # -- Internals -----------------------------------------------------------

    def _split(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Partition the frame into train/val/test with a reproducible seed.

        Args:
            frame: Engineered feature frame.

        Returns:
            Mapping of split name to DataFrame.

        Raises:
            DataError: If the dataset is too small to produce three splits.
        """
        seed = int(self.cfg.project.seed)
        val_size = float(self.cfg.data.val_size)
        test_size = float(self.cfg.data.test_size)

        if len(frame) < 3:
            raise DataError(f"Need at least 3 rows to split, got {len(frame)}.")

        train_frame, holdout = train_test_split(
            frame, test_size=val_size + test_size, random_state=seed, shuffle=True
        )
        # Re-normalise so the test share is taken from the holdout, not the whole.
        relative_test = test_size / (val_size + test_size)
        val_frame, test_frame = train_test_split(
            holdout, test_size=relative_test, random_state=seed, shuffle=True
        )
        return {"train": train_frame, "val": val_frame, "test": test_frame}

    def _fit_feature_scaler(self, features: FloatArray) -> StandardScaler | None:
        """Fit the feature scaler on the training split only.

        Args:
            features: Training feature matrix.

        Returns:
            The fitted scaler, or ``None`` when standardisation is disabled.
        """
        if not bool(self.cfg.data.standardize_features):
            return None
        scaler = StandardScaler()
        scaler.fit(features)
        return scaler

    def _fit_target_scaler(self, targets: FloatArray) -> StandardScaler:
        """Fit the target scaler on the (optionally log-transformed) train split.

        Args:
            targets: Raw training targets in the target's original unit.

        Returns:
            The fitted scaler.
        """
        values = targets.astype(np.float32, copy=True)
        if self.spec.log_transform_target:
            values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
        scaler = StandardScaler()
        scaler.fit(values.reshape(-1, 1))
        return scaler

    def _build_loader(
        self, split: str, shuffle: bool
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Construct a DataLoader for a split from the ``data.loader`` config.

        Args:
            split: Split name.
            shuffle: Whether to shuffle; only ``True`` for training.

        Returns:
            The configured DataLoader.
        """
        loader_cfg = self.cfg.data.loader
        num_workers = int(loader_cfg.num_workers)
        batch_size = int(loader_cfg.batch_size if shuffle else loader_cfg.eval_batch_size)

        kwargs: dict[str, object] = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": bool(loader_cfg.pin_memory) and torch.cuda.is_available(),
            "drop_last": bool(loader_cfg.drop_last) and shuffle,
        }
        # These options are invalid when workers run in the main process.
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(loader_cfg.persistent_workers)
            kwargs["prefetch_factor"] = int(loader_cfg.prefetch_factor)

        return DataLoader(self.dataset(split), **kwargs)  # type: ignore[arg-type]
