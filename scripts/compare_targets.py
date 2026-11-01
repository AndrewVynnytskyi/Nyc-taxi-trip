"""Benchmark the two candidate regression targets against non-neural baselines.

Trains :class:`~nyc_taxi.model.TripDurationMLP` twice -- once for each target --
from an identical feature set, so the two runs are directly comparable. For each
target it also scores three baselines from :mod:`nyc_taxi.baseline`, all fitted
on the train split and scored on the same test split as the MLP:

* ``bucket``  -- the ``(dayofweek, hour)`` bucket mean (the train/test version
  of the in-sample figure in ``sql/02_analytics.sql`` query 6);
* ``lin_1``   -- linear regression on ``trip_distance`` alone;
* ``lin_all`` -- linear regression on the MLP's exact input features.

The headline ``improve`` column is the MLP's RMSE reduction over ``lin_1``,
the simplest model that captures the tariff's per-mile structure.

Usage::

    python scripts/compare_targets.py
    python scripts/compare_targets.py --targets fare_amount --max-epochs 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

from nyc_taxi.baseline import bucket_mean, fit_full_linear_regression, fit_one_linear_regression
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
        baseline_rmse: Test RMSE of the train-fitted (dayofweek, hour) bucket mean.
        baseline_r2: R^2 of that heuristic.
        linear_one_rmse: Test RMSE of a linear regression on ``trip_distance`` alone.
        linear_one_r2: R^2 of that regression.
        linear_full_rmse: Test RMSE of a linear regression on the MLP's input features.
        linear_full_r2: R^2 of that regression.
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
    linear_one_rmse: float
    linear_one_r2: float
    linear_full_rmse: float
    linear_full_r2: float
    target_std: float
    epochs: int

    @property
    def rmse_improvement(self) -> float:
        """Fractional RMSE reduction of the model over the distance-only linear baseline."""
        return 1.0 - self.model_rmse / self.linear_one_rmse if self.linear_one_rmse else 0.0


def run_target(target: str, unit: str, overrides: Sequence[str]) -> TargetResult:
    """Train and evaluate the model and its baselines for a single target.

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
    baseline_rmse, baseline_r2 = bucket_mean(
        datamodule.splits["train"], datamodule.splits["test"], target
    )
    linear_one_rmse, linear_one_r2 = fit_one_linear_regression(datamodule.splits, target)
    linear_full_rmse, linear_full_r2 = fit_full_linear_regression(
        datamodule.splits, target, datamodule.spec.names
    )

    return TargetResult(
        target=target,
        unit=unit,
        model_rmse=metrics["rmse"],
        model_mae=metrics["mae"],
        model_r2=metrics["r2"],
        baseline_rmse=baseline_rmse,
        baseline_r2=baseline_r2,
        linear_one_rmse=linear_one_rmse,
        linear_one_r2=linear_one_r2,
        linear_full_rmse=linear_full_rmse,
        linear_full_r2=linear_full_r2,
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
        f"{'target':<24}{'unit':<6}{'std':>9}{'bucket':>9}{'lin_1':>9}{'lin_all':>9}"
        f"{'mlp':>9}{'improve':>9}{'R2':>8}{'epochs':>8}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.target:<24}{result.unit:<6}{result.target_std:>9.2f}"
            f"{result.baseline_rmse:>9.3f}{result.linear_one_rmse:>9.3f}"
            f"{result.linear_full_rmse:>9.3f}{result.model_rmse:>9.3f}"
            f"{result.rmse_improvement:>9.1%}{result.model_r2:>8.4f}{result.epochs:>8d}"
        )
    lines.append("")
    lines.append(
        "bucket/lin_1/lin_all/mlp columns are test RMSE in the target's unit (lower is better)."
    )
    lines.append(
        "improve = RMSE reduction of the MLP over the trip_distance-only linear regression."
    )
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
