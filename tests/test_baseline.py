"""Tests for the bucket-mean and linear baselines."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nyc_taxi.baseline import bucket_mean, fit_full_linear_regression, fit_one_linear_regression
from nyc_taxi.config import load_config
from nyc_taxi.data import TaxiDataModule
from nyc_taxi.etl import TaxiETLPipeline


@pytest.fixture()
def bucket_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "pickup_dayofweek": [0, 0, 0, 5, 5, 5],
            "pickup_hour": [8, 8, 9, 23, 23, 23],
            "fare_amount": [10.0, 12.0, 20.0, 14.0, 16.0, 18.0],
        }
    )
    # A shuffled, non-default index, as train_test_split produces.
    test = pd.DataFrame(
        {
            "pickup_dayofweek": [5, 0, 3, 0],
            "pickup_hour": [23, 8, 4, 9],
            "fare_amount": [17.0, 13.0, 9.0, 19.0],
        },
        index=[40, 7, 1001, 3],
    )
    return train, test


def test_bucket_mean_matches_hand_computation(
    bucket_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # Predictions: 16, 11, 15 (fallback), 20 -> residuals 1, 2, -6, -1.
    train, test = bucket_frames
    rmse, r2 = bucket_mean(train, test, "fare_amount")

    assert rmse == pytest.approx(math.sqrt(10.5))
    assert r2 == pytest.approx(1.0 - 10.5 / 14.75)


def test_bucket_mean_falls_back_to_train_mean() -> None:
    train = pd.DataFrame(
        {"pickup_dayofweek": [0, 1], "pickup_hour": [8, 9], "fare_amount": [10.0, 20.0]}
    )
    unseen = pd.DataFrame(
        {"pickup_dayofweek": [6, 6], "pickup_hour": [3, 4], "fare_amount": [15.0, 15.0]}
    )
    rmse, _ = bucket_mean(train, unseen, "fare_amount")

    assert rmse == pytest.approx(0.0)


def _linear_frames(shift: float = 0.0, noise: float = 0.0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)

    def make(size: int) -> pd.DataFrame:
        distance = rng.uniform(0.5, 20.0, size)
        fare = 3.0 + 2.0 * distance + rng.normal(0.0, noise, size)
        return pd.DataFrame({"trip_distance": distance, "fare_amount": fare})

    train = make(500)
    test = make(200)
    test["fare_amount"] += shift
    test.index = rng.permutation(len(test)) + 10_000
    return {"train": train, "test": test}


def test_linear_recovers_an_exact_line() -> None:
    rmse, r2 = fit_one_linear_regression(_linear_frames(), "fare_amount")

    assert rmse == pytest.approx(0.0, abs=1e-9)
    assert r2 == pytest.approx(1.0)


def test_linear_never_sees_test_labels() -> None:
    # Shifting only the test labels must not move the fit: the error is the shift.
    rmse, _ = fit_one_linear_regression(_linear_frames(shift=5.0), "fare_amount")

    assert rmse == pytest.approx(5.0)


def test_linear_rmse_matches_the_noise_level() -> None:
    rmse, r2 = fit_one_linear_regression(_linear_frames(noise=1.0), "fare_amount")

    assert 0.8 < rmse < 1.2
    assert r2 > 0.99


def _module_splits() -> TaxiDataModule:
    cfg = load_config(overrides=["data.use_cache=false"])
    rng = np.random.default_rng(1)
    size = 2_000
    pickup = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 86_400, size), unit="s")
    distance = rng.uniform(0.5, 20.0, size)
    raw = pd.DataFrame(
        {
            "VendorID": rng.integers(1, 3, size),
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup
            + pd.to_timedelta(rng.integers(60, 3600, size), unit="s"),
            "passenger_count": rng.integers(1, 5, size),
            "trip_distance": distance,
            "fare_amount": 2.5 + 2.5 * distance + rng.normal(0.0, 3.0, size),
        }
    )
    return TaxiDataModule(cfg).setup(frame=TaxiETLPipeline(cfg).transform(raw))


def test_full_linear_uses_only_model_features() -> None:
    module = _module_splits()
    target = module.spec.target
    # A column equal to the target: any "all columns except target" fit would find it.
    frames = {name: part.assign(leak=part[target]) for name, part in module.splits.items()}

    _, r2 = fit_full_linear_regression(frames, target, module.spec.names)
    _, leaked_r2 = fit_full_linear_regression(frames, target, [*module.spec.names, "leak"])

    assert 0.0 < r2 < 0.99
    assert leaked_r2 == pytest.approx(1.0)


def test_full_linear_rejects_the_target_as_a_feature() -> None:
    frames = _linear_frames()
    with pytest.raises(ValueError, match="exclude"):
        fit_full_linear_regression(frames, "fare_amount", ["trip_distance", "fare_amount"])
