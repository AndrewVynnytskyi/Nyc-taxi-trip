"""Non-neural baselines the MLP must beat, fitted on train and scored on test.

Every baseline is fitted on the ``train`` split only and scored on the
``test`` split only -- the same frames :class:`~nyc_taxi.data.TaxiDataModule`
exposes via ``splits`` -- so the numbers are directly comparable with the
MLP's test metrics. All scoring goes through :func:`_metrics` on numpy arrays,
which aligns rows by position: the splits carry shuffled indices, so any
index-based alignment would silently pair the wrong rows.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression

from nyc_taxi.utils import get_logger

__all__ = ["bucket_mean", "fit_full_linear_regression", "fit_one_linear_regression"]

logger = get_logger(__name__)

#: Float array type used for actual/predicted values.
FloatArray = npt.NDArray[np.float64]


def _metrics(actual: FloatArray, predicted: FloatArray) -> tuple[float, float]:
    """Return ``(rmse, r2)`` for two positionally aligned float arrays.

    Args:
        actual: Ground-truth values.
        predicted: Predictions, in the same row order as ``actual``.

    Returns:
        Tuple of ``(rmse, r2)``. ``r2`` is ``0.0`` when ``actual`` is constant.
    """
    residual = actual - predicted
    variance = float(np.var(actual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    r2 = float(1.0 - np.mean(residual**2) / variance) if variance else 0.0
    return rmse, r2


def bucket_mean(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple[float, float]:
    """Score the ``(pickup_dayofweek, pickup_hour)`` bucket-mean baseline.

    Bucket means are computed on ``train``; ``test`` buckets never seen in
    training fall back to the overall training mean.

    Args:
        train: Training frame with the raw cyclical columns and the target.
        test: Test frame with the same columns.
        target: Target column to predict.

    Returns:
        Test-set ``(rmse, r2)``.
    """
    keys = ["pickup_dayofweek", "pickup_hour"]
    table = train.groupby(keys)[target].mean().rename("predicted").reset_index()
    merged = test[keys].merge(table, on=keys, how="left")
    predicted = merged["predicted"].fillna(train[target].mean()).to_numpy(dtype=np.float64)

    actual = test[target].to_numpy(dtype=np.float64)
    return _metrics(actual, predicted)


def fit_one_linear_regression(frames: dict[str, pd.DataFrame], target: str) -> tuple[float, float]:
    """Score a linear regression on ``trip_distance`` alone.

    Logs the fitted slope and intercept, and for trip duration the implied
    average speed (``60 / slope`` mph), as a sanity check against the tariff.

    Args:
        frames: Splits keyed ``"train"`` and ``"test"``.
        target: Target column to predict.

    Returns:
        Test-set ``(rmse, r2)``.
    """
    lin_reg = LinearRegression()
    lin_reg.fit(frames["train"][["trip_distance"]], frames["train"][target])
    predicted = lin_reg.predict(frames["test"][["trip_distance"]])
    slope = float(lin_reg.coef_[0])
    intercept = float(lin_reg.intercept_)
    logger.info("%s: slope %.3f %s per mile, intercept %.3f", target, slope, target, intercept)
    if target == "trip_duration_minutes" and slope > 0:
        logger.info("%s: implied average speed %.1f mph", target, 60.0 / slope)
    actual = frames["test"][target].to_numpy(dtype=np.float64)
    return _metrics(actual, np.asarray(predicted, dtype=np.float64))


def fit_full_linear_regression(
    frames: dict[str, pd.DataFrame], target: str, features: Sequence[str]
) -> tuple[float, float]:
    """Score a linear regression on the MLP's exact input features.

    Args:
        frames: Splits keyed ``"train"`` and ``"test"``.
        target: Target column to predict.
        features: Feature columns; pass ``datamodule.spec.names`` so the
            regression sees the same inputs as the MLP and nothing else.

    Returns:
        Test-set ``(rmse, r2)``.

    Raises:
        ValueError: If ``features`` is empty or contains the target.
    """
    columns = list(features)
    if not columns or target in columns:
        raise ValueError(f"features must be non-empty and exclude {target!r}, got {columns}.")
    lin_reg = LinearRegression()
    lin_reg.fit(frames["train"][columns], frames["train"][target])
    predicted = lin_reg.predict(frames["test"][columns])
    actual = frames["test"][target].to_numpy(dtype=np.float64)
    return _metrics(actual, np.asarray(predicted, dtype=np.float64))
