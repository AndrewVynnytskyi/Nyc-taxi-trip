"""Smoke tests covering the config, ETL transform, data module and model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import DictConfig

from nyc_taxi.config import ConfigError, load_config
from nyc_taxi.data import DataError, TaxiDataModule
from nyc_taxi.etl import TaxiETLPipeline
from nyc_taxi.model import ModelError, build_model


@pytest.fixture(scope="session")
def cfg() -> DictConfig:
    return load_config(overrides=["data.use_cache=false", "training.max_epochs=1"])


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    size = 512
    pickup = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 86_400, size), unit="s")
    return pd.DataFrame(
        {
            "VendorID": rng.integers(1, 3, size),
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup
            + pd.to_timedelta(rng.integers(60, 3600, size), unit="s"),
            "passenger_count": rng.integers(1, 5, size),
            "trip_distance": rng.uniform(0.5, 20.0, size),
            "fare_amount": rng.uniform(3.0, 80.0, size),
        }
    )


def test_config_sections_and_paths_are_absolute(cfg: DictConfig) -> None:
    assert cfg.project.seed >= 0
    assert cfg.project.data_dir.startswith("/")
    assert cfg.project.artifact_dir.startswith("/")


def test_config_rejects_invalid_override() -> None:
    with pytest.raises(ConfigError):
        load_config(overrides=["data.val_size=0.9", "data.test_size=0.9"])


def test_transform_derives_features_and_filters(cfg: DictConfig, raw_frame: pd.DataFrame) -> None:
    cleaned = TaxiETLPipeline(cfg).transform(raw_frame)

    assert list(cleaned.columns) == [str(column) for column in cfg.etl.output_columns]
    assert (cleaned["trip_duration_minutes"] > 0).all()
    assert cleaned["pickup_hour"].between(0, 23).all()
    assert cleaned["pickup_dayofweek"].between(0, 6).all()
    assert not cleaned.isna().any().any()


def test_datamodule_roundtrips_the_target(cfg: DictConfig, raw_frame: pd.DataFrame) -> None:
    cleaned = TaxiETLPipeline(cfg).transform(raw_frame)
    module = TaxiDataModule(cfg).setup(frame=cleaned)

    original = cleaned[cfg.data.target].to_numpy(dtype=np.float32)
    recovered = module.inverse_transform_target(module.transform_target(original))
    np.testing.assert_allclose(recovered, original, rtol=1e-3, atol=1e-2)


def test_datamodule_rejects_missing_columns(cfg: DictConfig, raw_frame: pd.DataFrame) -> None:
    cleaned = TaxiETLPipeline(cfg).transform(raw_frame).drop(columns=["trip_distance"])
    with pytest.raises(DataError, match="configured but absent"):
        TaxiDataModule(cfg).setup(frame=cleaned)


def test_model_forward_shape_and_contract(cfg: DictConfig) -> None:
    model = build_model(cfg, input_dim=7).eval()
    output = model(torch.randn(16, 7))

    assert output.shape == (16, 1)
    with pytest.raises(ModelError, match="Expected 7 input features"):
        model(torch.randn(16, 5))
