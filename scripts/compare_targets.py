"""Benchmark the two candidate regression targets against a heuristic baseline.

Trains :class:`~nyc_taxi.model.TripDurationMLP` twice -- once for each target --
from an identical feature set, so the two runs are directly comparable. For each
target it also computes the ``(dayofweek, hour)`` bucket-mean baseline from
``sql/02_analytics.sql`` (query 6), which is the bar a learned model must clear.

Usage::

    python scripts/compare_targets.py
    python scripts/compare_targets.py --targets fare_amount --max-epochs 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from nyc_taxi.config import ConfigError, load_config
from nyc_taxi.data import DataError, TaxiDataModule
from nyc_taxi.model import build_model
from nyc_taxi.train import Trainer, TrainingError
from nyc_taxi.utils import get_logger, seed_everything, setup_logging

logger = get_logger(__name__)

#: Targets to benchmark, mapped to the unit their metrics are reported in.
DEFAULT_TARGETS: dict[str, str] = {
    "fare_amount": "USD",
    "trip_duration_minutes": "min",
}


@dataclass(frozen=True, slots=True)
class TargetResult:
    """Outcome of training and evaluating one target.

    Attributes:
        target: Name of the target column.
        unit: Unit its metrics are expressed in.
        model_rmse: Test-set RMSE of the learned model.
        model_mae: Test-set MAE of the learned model.
        model_r2: Test-set coefficient of determination.
        baseline_rmse: RMSE of the (dayofweek, hour) bucket-mean heuristic.
        baseline_r2: R^2 of that heuristic.
        target_std: Standard deviation of the target, for scale context.
        epochs: Number of epochs actually run before early stopping.
    """

    target: str
    unit: str
    model_rmse: float
    model_mae: float
    model_r2: float
    baseline_rmse: float
    baseline_r2: float
    target_std: float
    epochs: int

    @property
    def rmse_improvement(self) -> float:
        """Fractional RMSE reduction of the model over the heuristic baseline."""
        return 1.0 - self.model_rmse / self.baseline_rmse if self.baseline_rmse else 0.0


def heuristic_baseline(frame: pd.DataFrame, target: str) -> tuple[float, float]:
    """Compute the ``(dayofweek, hour)`` bucket-mean baseline.

    Args:
        frame: Clean-table DataFrame.
        target: Target column to predict.

    Returns:
        Tuple of ``(rmse, r2)`` for the heuristic.
    """
    predicted = frame.groupby(["pickup_dayofweek", "pickup_hour"])[target].transform("mean")
    residual = (frame[target] - predicted).to_numpy(dtype=np.float64)
    variance = float(np.var(frame[target].to_numpy(dtype=np.float64)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    r2 = float(1.0 - np.mean(residual**2) / variance) if variance else 0.0
    return rmse, r2


def run_target(target: str, unit: str, overrides: Sequence[str]) -> TargetResult:
    """Train and evaluate the model for a single target.

    Args:
        target: Target column name.
        unit: Unit label for reporting.
        overrides: Additional dotlist config overrides.

    Returns:
        The populated :class:`TargetResult`.

    Raises:
        ConfigError: If the configuration is invalid.
        DataError: If the clean data cannot be loaded.
        TrainingError: If training fails.
    """
    cfg = load_config(overrides=[f"data.target={target}", f"data.target_unit={unit}", *overrides])
    seed_everything(int(cfg.project.seed))

    datamodule = TaxiDataModule(cfg).setup()
    model = build_model(cfg, input_dim=datamodule.input_dim)
    trainer = Trainer(model, datamodule, cfg)

    history = trainer.fit()
    metrics = trainer.evaluate(split="test", restore_best=True)
    baseline_rmse, baseline_r2 = heuristic_baseline(datamodule.frame, target)

    return TargetResult(
        target=target,
        unit=unit,
        model_rmse=metrics["rmse"],
        model_mae=metrics["mae"],
        model_r2=metrics["r2"],
        baseline_rmse=baseline_rmse,
        baseline_r2=baseline_r2,
        target_std=float(datamodule.frame[target].std()),
        epochs=len(history),
    )


def render(results: Sequence[TargetResult]) -> str:
    """Format the benchmark results as a fixed-width table.

    Args:
        results: Per-target outcomes.

    Returns:
        The rendered table.
    """
    header = (
        f"{'target':<24}{'unit':<6}{'std':>9}{'baseline':>10}{'model':>9}"
        f"{'improve':>9}{'R2':>8}{'epochs':>8}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.target:<24}{result.unit:<6}{result.target_std:>9.2f}"
            f"{result.baseline_rmse:>10.3f}{result.model_rmse:>9.3f}"
            f"{result.rmse_improvement:>8.1%}{result.model_r2:>8.4f}{result.epochs:>8d}"
        )
    lines.append("")
    lines.append("baseline/model columns are RMSE in the target's own unit (lower is better).")
    lines.append("improve = RMSE reduction of the model over the (dayofweek, hour) mean.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the target benchmark.

    Args:
        argv: Command-line arguments; defaults to :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` on a handled failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args(argv)

    setup_logging()
    overrides = list(args.overrides)
    if args.max_epochs is not None:
        overrides.append(f"training.max_epochs={args.max_epochs}")

    results: list[TargetResult] = []
    try:
        for target in args.targets:
            unit = DEFAULT_TARGETS.get(target, "unit")
            logger.info("=== Benchmarking target: %s ===", target)
            results.append(run_target(target, unit, overrides))
    except (ConfigError, DataError, TrainingError) as exc:
        logger.error("%s", exc)
        return 1

    print("\n" + render(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
