"""NYC Taxi Trip Analytics: production ETL and deep-learning pipeline."""

from __future__ import annotations

from nyc_taxi.config import ConfigError, load_config
from nyc_taxi.data import TaxiDataModule, TaxiTripDataset
from nyc_taxi.model import TripDurationMLP, build_model
from nyc_taxi.train import Trainer

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "TaxiDataModule",
    "TaxiTripDataset",
    "Trainer",
    "TripDurationMLP",
    "build_model",
    "load_config",
    "__version__",
]
