"""Smoke tests covering the config, ETL transform, data module and model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import DictConfig

from nyc_taxi.config import ConfigError, load_config
from nyc_taxi.data import DataError, TaxiDataModule
from nyc_taxi.etl import ETLError, TaxiETLPipeline
from nyc_taxi.model import ModelError, build_model
from nyc_taxi.train import Trainer


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


def test_datamodule_splits_match_the_datasets(cfg: DictConfig, raw_frame: pd.DataFrame) -> None:
    cleaned = TaxiETLPipeline(cfg).transform(raw_frame)
    module = TaxiDataModule(cfg).setup(frame=cleaned)

    assert sum(len(part) for part in module.splits.values()) == len(cleaned)
    for split in ("train", "val", "test"):
        part = module.splits[split]
        assert len(part) == len(module.dataset(split))
        assert {"pickup_hour", "pickup_dayofweek"} <= set(part.columns)
        # Raw cyclical columns are kept for baselines but never fed to the model.
        assert "pickup_hour" not in module.spec.names


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


def test_extract_reads_parquet_and_normalises_timestamps(
    cfg: DictConfig, raw_frame: pd.DataFrame, tmp_path: Path
) -> None:
    """Microsecond timestamps from parquet are coerced to the ns contract."""
    source = tmp_path / "trips.parquet"
    raw_frame.astype(
        {
            "tpep_pickup_datetime": "datetime64[us]",
            "tpep_dropoff_datetime": "datetime64[us]",
        }
    ).to_parquet(source, index=False)

    pipeline = TaxiETLPipeline(cfg)
    chunks = list(pipeline.extract(source))

    assert chunks, "expected at least one chunk"
    combined = pd.concat(chunks, ignore_index=True)
    assert len(combined) == len(raw_frame)
    assert str(combined["tpep_pickup_datetime"].dtype) == "datetime64[ns]"
    # The raw landing table must stay faithful: every source column survives.
    assert set(raw_frame.columns) <= set(combined.columns)


def test_extract_honours_max_rows(cfg: DictConfig, raw_frame: pd.DataFrame, tmp_path: Path) -> None:
    """`etl.max_rows` truncates the stream, including mid-chunk."""
    source = tmp_path / "trips.csv"
    raw_frame.to_csv(source, index=False)

    capped = load_config(overrides=["etl.max_rows=100", "etl.chunk_size=64"])
    total = sum(len(chunk) for chunk in TaxiETLPipeline(capped).extract(source))

    assert total == 100


def test_resolve_format_rejects_unknown_suffix(cfg: DictConfig, tmp_path: Path) -> None:
    """`auto` format detection fails loudly rather than guessing."""
    with pytest.raises(ETLError, match="Cannot infer a format"):
        TaxiETLPipeline(cfg)._resolve_format(tmp_path / "trips.orc")


def test_metrics_are_reported_in_the_configured_unit(cfg: DictConfig) -> None:
    """The target unit is carried through to the Trainer for logging."""
    assert cfg.data.target_unit
    assert cfg.data.target not in cfg.data.numerical_features
    # Neither candidate target may appear as a feature: whichever is not the
    # target is unknown at pickup time and would leak.
    assert "fare_amount" not in cfg.data.numerical_features
    assert "trip_duration_minutes" not in cfg.data.numerical_features


def test_derived_paths_are_absolute_regardless_of_cwd(
    cfg: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact paths must not depend on the process working directory.

    Regression test: `training.checkpoint_dir` interpolates `project.artifact_dir`,
    so absolutising only the source key left the derived key relative and wrote
    checkpoints next to whatever directory the run started in.
    """
    monkeypatch.chdir(tmp_path)
    reloaded = load_config()

    for key in ("training.checkpoint_dir", "data.cache_path", "project.artifact_dir"):
        value = reloaded
        for part in key.split("."):
            value = value[part]
        assert Path(str(value)).is_absolute(), f"{key} is relative: {value}"
        assert "notebooks" not in str(value)


def test_logging_defaults_to_stdout_not_stderr() -> None:
    """Logs must not land on stderr.

    Regression test: Jupyter renders any stderr stream on an error-red
    background, so INFO logs made a successful notebook run look like a failed
    one. Informational logs belong on stdout.
    """
    import logging

    from nyc_taxi.utils import setup_logging

    setup_logging()
    streams = [
        h.stream for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)
    ]
    assert streams, "expected at least one stream handler"
    assert all(s is sys.stdout for s in streams), f"handler(s) not on stdout: {streams}"


@pytest.mark.parametrize(
    ("setting", "expected"),
    [("on", True), ("off", False)],
)
def test_progress_bar_setting_is_explicit(setting: str, expected: bool) -> None:
    """`training.progress_bar` on/off override TTY detection."""
    cfg = load_config(overrides=[f"training.progress_bar={setting}"])
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    assert trainer._resolve_progress_bar() is expected
