"""Reusable training loop with Weights & Biases instrumentation.

:class:`Trainer` owns the optimisation lifecycle -- epoch loop, AMP, gradient
clipping and accumulation, LR scheduling, early stopping, checkpointing and
experiment logging -- and exposes it behind a three-method surface:
:meth:`Trainer.fit`, :meth:`Trainer.evaluate` and :meth:`Trainer.predict`.

Notebooks are therefore "thin": they compose the config, the data module and
the model, then call ``trainer.fit()``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from nyc_taxi.config import ConfigError, load_config, save_config
from nyc_taxi.data import DataError, TaxiDataModule
from nyc_taxi.model import build_model
from nyc_taxi.utils import count_parameters, get_device, get_logger, seed_everything, setup_logging

__all__ = ["EpochResult", "Trainer", "TrainingError", "TrainingHistory", "main"]

logger = get_logger(__name__)

#: Alias for the float32 arrays returned by prediction and metric helpers.
FloatArray = npt.NDArray[np.float32]

#: Batch type yielded by the project's DataLoaders.
Batch = tuple[Tensor, Tensor]

_CRITERIA: Final[Mapping[str, str]] = {"mse": "MSELoss", "mae": "L1Loss", "huber": "HuberLoss"}


class TrainingError(RuntimeError):
    """Raised when training cannot start or a training step fails."""


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Metrics for one completed epoch.

    Attributes:
        epoch: 1-based epoch index.
        train_loss: Mean training loss in model space.
        val_loss: Mean validation loss in model space.
        val_rmse: Validation RMSE in minutes (original units).
        val_mae: Validation MAE in minutes (original units).
        val_r2: Validation coefficient of determination.
        learning_rate: LR in effect during the epoch.
        duration_seconds: Wall-clock duration of the epoch.
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_rmse: float
    val_mae: float
    val_r2: float
    learning_rate: float
    duration_seconds: float

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a flat mapping for logging."""
        return {
            "epoch": float(self.epoch),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_rmse": self.val_rmse,
            "val_mae": self.val_mae,
            "val_r2": self.val_r2,
            "learning_rate": self.learning_rate,
            "epoch_seconds": self.duration_seconds,
        }


@dataclass
class TrainingHistory:
    """Ordered record of every epoch, for plotting and post-hoc analysis.

    Attributes:
        epochs: The per-epoch results in chronological order.
    """

    epochs: list[EpochResult] = field(default_factory=list)

    def append(self, result: EpochResult) -> None:
        """Record one epoch's result."""
        self.epochs.append(result)

    def series(self, metric: str) -> list[float]:
        """Extract one metric across all epochs.

        Args:
            metric: Key from :meth:`EpochResult.as_dict`.

        Returns:
            The metric's values in epoch order.

        Raises:
            KeyError: If the metric is unknown.
        """
        return [result.as_dict()[metric] for result in self.epochs]

    def to_dict(self) -> dict[str, list[float]]:
        """Return every metric as a column-oriented mapping (DataFrame-ready)."""
        if not self.epochs:
            return {}
        return {key: self.series(key) for key in self.epochs[0].as_dict()}

    def __len__(self) -> int:
        """Return the number of recorded epochs."""
        return len(self.epochs)


class Trainer:
    """Trains and evaluates a model against a :class:`TaxiDataModule`.

    Args:
        model: The network to optimise.
        datamodule: Prepared data module; :meth:`TaxiDataModule.setup` must
            already have been called.
        cfg: Fully resolved project configuration.

    Raises:
        TrainingError: If the requested device, criterion, optimiser or
            scheduler cannot be constructed.
    """

    def __init__(self, model: nn.Module, datamodule: TaxiDataModule, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.datamodule = datamodule
        self.device = get_device(str(cfg.training.device))
        self.model = model.to(self.device)

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.use_amp = bool(cfg.training.use_amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        self.history = TrainingHistory()
        self.current_epoch = 0
        self.global_step = 0

        self._monitor = str(cfg.training.monitor_metric)
        self._mode = str(cfg.training.monitor_mode)
        self._best_score = math.inf if self._mode == "min" else -math.inf
        self._best_epoch = 0
        self._epochs_without_improvement = 0
        self._run: Any | None = None

        self.checkpoint_path = Path(str(cfg.training.checkpoint_dir)) / str(
            cfg.training.checkpoint_name
        )

        logger.info(
            "Trainer ready | device=%s | amp=%s | params=%s",
            self.device,
            self.use_amp,
            f"{count_parameters(self.model):,}",
        )

    # -- Public API ----------------------------------------------------------

    def fit(self, max_epochs: int | None = None) -> TrainingHistory:
        """Run the full training loop with early stopping and checkpointing.

        Args:
            max_epochs: Override for ``training.max_epochs``. Useful for smoke
                tests without touching the config file.

        Returns:
            The populated :class:`TrainingHistory`.

        Raises:
            TrainingError: If the data module is not set up or a step fails.
        """
        try:
            train_loader = self.datamodule.train_dataloader()
            val_loader = self.datamodule.val_dataloader()
        except DataError as exc:
            raise TrainingError(f"Data module is not ready for training: {exc}") from exc

        epochs = int(max_epochs if max_epochs is not None else self.cfg.training.max_epochs)
        patience = int(self.cfg.training.early_stopping_patience)

        self._start_run()
        logger.info("Training for up to %d epoch(s) on %s", epochs, self.device)

        try:
            for epoch in range(1, epochs + 1):
                self.current_epoch = epoch
                started = time.perf_counter()

                train_loss = self._train_epoch(train_loader)
                val_metrics = self._evaluate(val_loader, stage="val")

                result = EpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_metrics["loss"],
                    val_rmse=val_metrics["rmse"],
                    val_mae=val_metrics["mae"],
                    val_r2=val_metrics["r2"],
                    learning_rate=self._current_lr(),
                    duration_seconds=time.perf_counter() - started,
                )
                self.history.append(result)
                self._step_scheduler(result)
                self._log_metrics(result.as_dict(), step=epoch)

                logger.info(
                    "Epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | "
                    "val_rmse=%.3f min | val_mae=%.3f min | val_r2=%.4f | lr=%.2e | %.1fs",
                    epoch,
                    epochs,
                    result.train_loss,
                    result.val_loss,
                    result.val_rmse,
                    result.val_mae,
                    result.val_r2,
                    result.learning_rate,
                    result.duration_seconds,
                )

                if self._is_improvement(result):
                    self._best_epoch = epoch
                    self._epochs_without_improvement = 0
                    self.save_checkpoint(self.checkpoint_path)
                else:
                    self._epochs_without_improvement += 1
                    if patience > 0 and self._epochs_without_improvement >= patience:
                        logger.info(
                            "Early stopping at epoch %d: %s did not improve for %d epoch(s). "
                            "Best %s=%.4f at epoch %d.",
                            epoch,
                            self._monitor,
                            patience,
                            self._monitor,
                            self._best_score,
                            self._best_epoch,
                        )
                        break
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user at epoch %d.", self.current_epoch)
        finally:
            self._finish_run()

        return self.history

    def evaluate(self, split: str = "test", restore_best: bool = True) -> dict[str, float]:
        """Evaluate the model on a held-out split.

        Args:
            split: One of ``"train"``, ``"val"``, ``"test"``.
            restore_best: Load the best checkpoint before evaluating, so the
                reported numbers match the selected model rather than the last
                epoch.

        Returns:
            Mapping with ``loss``, ``rmse``, ``mae`` and ``r2``. Error metrics
            are in minutes.

        Raises:
            TrainingError: If the split is unknown.
        """
        if restore_best and self.checkpoint_path.is_file():
            self.load_checkpoint(self.checkpoint_path)

        try:
            loader = {
                "train": self.datamodule.train_dataloader,
                "val": self.datamodule.val_dataloader,
                "test": self.datamodule.test_dataloader,
            }[split]()
        except KeyError as exc:
            raise TrainingError(f"Unknown split {split!r}.") from exc

        metrics = self._evaluate(loader, stage=split)
        logger.info(
            "%s | loss=%.4f | rmse=%.3f min | mae=%.3f min | r2=%.4f",
            split.upper(),
            metrics["loss"],
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
        )
        return metrics

    @torch.inference_mode()
    def predict(self, loader: DataLoader[Batch]) -> tuple[FloatArray, FloatArray]:
        """Generate predictions in the original unit (minutes).

        Args:
            loader: Loader over the data to predict.

        Returns:
            Tuple of ``(predictions, targets)``, both 1-D ``float32`` arrays of
            trip duration in minutes.
        """
        self.model.eval()
        predictions: list[Tensor] = []
        targets: list[Tensor] = []

        for features, target in loader:
            features = features.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                output = self.model(features)
            predictions.append(output.float().cpu())
            targets.append(target.float().cpu())

        stacked_pred = torch.cat(predictions).view(-1)
        stacked_true = torch.cat(targets).view(-1)
        return (
            self.datamodule.inverse_transform_target(stacked_pred),
            self.datamodule.inverse_transform_target(stacked_true),
        )

    def save_checkpoint(self, path: Path | str) -> Path:
        """Persist model weights, optimiser state and the config.

        Args:
            path: Destination ``.pt`` file. Parent directories are created.

        Returns:
            The absolute path written to.

        Raises:
            TrainingError: If the checkpoint cannot be written.
        """
        target = Path(path).expanduser().resolve()
        payload = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_score": self._best_score,
            "monitor_metric": self._monitor,
            "feature_names": list(self.datamodule.spec.names),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, target)
        except OSError as exc:
            raise TrainingError(f"Failed to write checkpoint to {target}: {exc}") from exc

        logger.debug("Checkpoint saved to %s (epoch %d)", target, self.current_epoch)
        return target

    def load_checkpoint(self, path: Path | str, strict: bool = True) -> None:
        """Restore weights (and optimiser state when present) from a checkpoint.

        Args:
            path: Checkpoint file to load.
            strict: Whether the state dict must match the model exactly.

        Raises:
            TrainingError: If the file is missing or cannot be loaded.
        """
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise TrainingError(f"Checkpoint not found: {source}")

        try:
            payload = torch.load(source, map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["model_state_dict"], strict=strict)
            if "optimizer_state_dict" in payload:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        except (OSError, KeyError, RuntimeError) as exc:
            raise TrainingError(f"Failed to load checkpoint {source}: {exc}") from exc

        logger.info("Restored checkpoint from %s (epoch %s)", source, payload.get("epoch", "?"))

    # -- Epoch-level internals ----------------------------------------------

    def _train_epoch(self, loader: DataLoader[Batch]) -> float:
        """Run one optimisation pass over the training set.

        Args:
            loader: Shuffled training loader.

        Returns:
            The sample-weighted mean training loss in model space.

        Raises:
            TrainingError: If the loss becomes non-finite.
        """
        self.model.train()
        accumulation = max(1, int(self.cfg.training.accumulation_steps))
        clip_norm = float(self.cfg.training.gradient_clip_norm)
        log_every = int(self.cfg.training.log_every_n_steps)

        running_loss = 0.0
        seen = 0
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            loader,
            desc=f"Epoch {self.current_epoch:03d} [train]",
            leave=False,
            unit="batch",
        )
        for step, (features, targets) in enumerate(progress, start=1):
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.use_amp):
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)

            if not torch.isfinite(loss):
                raise TrainingError(
                    f"Non-finite loss ({loss.item()}) at epoch {self.current_epoch}, step {step}. "
                    "Lower optimizer.lr or enable training.gradient_clip_norm."
                )

            self.scaler.scale(loss / accumulation).backward()

            if step % accumulation == 0:
                if clip_norm > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            batch_size = features.shape[0]
            running_loss += loss.item() * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

            if log_every > 0 and self.global_step > 0 and self.global_step % log_every == 0:
                self._log_metrics(
                    {"batch_loss": loss.item(), "learning_rate": self._current_lr()},
                    step=self.global_step,
                    commit=False,
                )

        return running_loss / max(seen, 1)

    @torch.inference_mode()
    def _evaluate(self, loader: DataLoader[Batch], stage: str) -> dict[str, float]:
        """Evaluate the model over a loader without updating weights.

        Args:
            loader: Loader to evaluate.
            stage: Split name, used for the progress-bar label.

        Returns:
            Mapping with ``loss`` (model space) plus ``rmse``, ``mae`` and
            ``r2`` computed in minutes.
        """
        self.model.eval()
        running_loss = 0.0
        seen = 0
        predictions: list[Tensor] = []
        targets: list[Tensor] = []

        progress = tqdm(
            loader,
            desc=f"Epoch {self.current_epoch:03d} [{stage}]",
            leave=False,
            unit="batch",
        )
        for features, target in progress:
            features = features.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.use_amp):
                output = self.model(features)
                loss = self.criterion(output, target)

            batch_size = features.shape[0]
            running_loss += loss.item() * batch_size
            seen += batch_size
            predictions.append(output.float().cpu())
            targets.append(target.float().cpu())

        y_pred = self.datamodule.inverse_transform_target(torch.cat(predictions).view(-1))
        y_true = self.datamodule.inverse_transform_target(torch.cat(targets).view(-1))

        metrics = self._regression_metrics(y_true, y_pred)
        metrics["loss"] = running_loss / max(seen, 1)
        return metrics

    @staticmethod
    def _regression_metrics(y_true: FloatArray, y_pred: FloatArray) -> dict[str, float]:
        """Compute RMSE, MAE and R^2 in the original unit.

        Args:
            y_true: Ground-truth values in minutes.
            y_pred: Predicted values in minutes.

        Returns:
            Mapping with ``rmse``, ``mae`` and ``r2``.
        """
        residual = y_pred.astype(np.float64) - y_true.astype(np.float64)
        total_variance = float(np.var(y_true.astype(np.float64)))
        return {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(1.0 - np.mean(residual**2) / total_variance) if total_variance else 0.0,
        }

    # -- Construction internals ---------------------------------------------

    def _build_criterion(self) -> nn.Module:
        """Instantiate the loss function from ``training.criterion``.

        Returns:
            The configured loss module.

        Raises:
            TrainingError: If the criterion name is unsupported.
        """
        name = str(self.cfg.training.criterion).strip().lower()
        if name not in _CRITERIA:
            raise TrainingError(
                f"Unsupported training.criterion {name!r}. Choose one of {sorted(_CRITERIA)}."
            )
        if name == "huber":
            return nn.HuberLoss(delta=float(self.cfg.training.huber_delta))
        return cast(nn.Module, getattr(nn, _CRITERIA[name])())

    def _build_optimizer(self) -> Optimizer:
        """Instantiate the optimiser from the ``optimizer`` config section.

        Returns:
            The configured optimiser.

        Raises:
            TrainingError: If the optimiser name is unsupported.
        """
        opt_cfg = self.cfg.optimizer
        name = str(opt_cfg.name).strip().lower()
        params = self.model.parameters()
        lr = float(opt_cfg.lr)
        weight_decay = float(opt_cfg.weight_decay)

        if name == "adamw":
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay, betas=tuple(opt_cfg.betas)
            )
        if name == "adam":
            return torch.optim.Adam(
                params, lr=lr, weight_decay=weight_decay, betas=tuple(opt_cfg.betas)
            )
        if name == "sgd":
            return torch.optim.SGD(
                params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=float(opt_cfg.momentum),
                nesterov=True,
            )
        raise TrainingError(f"Unsupported optimizer.name {opt_cfg.name!r}.")

    def _build_scheduler(self) -> LRScheduler | None:
        """Instantiate the LR scheduler from the ``scheduler`` config section.

        Returns:
            The configured scheduler, or ``None`` when ``scheduler.name`` is
            ``"none"``.

        Raises:
            TrainingError: If the scheduler name is unsupported.
        """
        sched_cfg = self.cfg.scheduler
        name = str(sched_cfg.name).strip().lower()

        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(self.cfg.training.max_epochs) - int(sched_cfg.warmup_epochs)),
                eta_min=float(sched_cfg.min_lr),
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=int(sched_cfg.step_size), gamma=float(sched_cfg.gamma)
            )
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=cast(Literal["min", "max"], self._mode),
                factor=float(sched_cfg.gamma),
                patience=int(sched_cfg.patience),
                min_lr=float(sched_cfg.min_lr),
            )
        raise TrainingError(f"Unsupported scheduler.name {sched_cfg.name!r}.")

    # -- Loop helpers --------------------------------------------------------

    def _step_scheduler(self, result: EpochResult) -> None:
        """Advance the LR scheduler, feeding the monitored metric to plateau.

        Args:
            result: The just-completed epoch's metrics.
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(self._score(result))
        else:
            self.scheduler.step()

    def _current_lr(self) -> float:
        """Return the learning rate of the first parameter group."""
        return float(self.optimizer.param_groups[0]["lr"])

    def _score(self, result: EpochResult) -> float:
        """Extract the monitored metric from an epoch result.

        Args:
            result: Epoch metrics.

        Returns:
            The monitored metric's value.

        Raises:
            TrainingError: If ``training.monitor_metric`` is not a known metric.
        """
        metrics = result.as_dict()
        if self._monitor not in metrics:
            raise TrainingError(
                f"training.monitor_metric={self._monitor!r} is not produced by the loop. "
                f"Available: {sorted(metrics)}."
            )
        return metrics[self._monitor]

    def _is_improvement(self, result: EpochResult) -> bool:
        """Decide whether an epoch beat the best score by the min-delta margin.

        Args:
            result: Epoch metrics.

        Returns:
            ``True`` when the monitored metric improved; the best score is
            updated as a side effect.
        """
        score = self._score(result)
        min_delta = float(self.cfg.training.early_stopping_min_delta)
        improved = (
            score < self._best_score - min_delta
            if self._mode == "min"
            else score > self._best_score + min_delta
        )
        if improved:
            self._best_score = score
        return improved

    # -- Experiment tracking -------------------------------------------------

    def _start_run(self) -> None:
        """Initialise the W&B run, degrading gracefully when unavailable.

        A missing ``wandb`` install or a failed handshake is logged as a
        warning and disables tracking; it never aborts training.
        """
        wandb_cfg = self.cfg.wandb
        if not bool(wandb_cfg.enabled):
            logger.info("W&B logging is disabled (wandb.enabled=false).")
            return

        try:
            import wandb  # noqa: PLC0415 - optional dependency, resolved at run time.

            self._run = wandb.init(
                project=str(wandb_cfg.project),
                entity=wandb_cfg.get("entity"),
                mode=cast(Any, str(wandb_cfg.mode)),
                group=str(wandb_cfg.group),
                job_type=str(wandb_cfg.job_type),
                tags=[str(tag) for tag in wandb_cfg.tags],
                config=cast(dict[str, Any], OmegaConf.to_container(self.cfg, resolve=True)),
                reinit=True,
            )
            if bool(wandb_cfg.log_gradients):
                wandb.watch(self.model, log="all", log_freq=int(wandb_cfg.watch_log_freq))
            logger.info("W&B run started: %s", getattr(self._run, "url", "offline"))
        except Exception as exc:  # noqa: BLE001 - tracking must never break training.
            logger.warning("W&B initialisation failed (%s); continuing without tracking.", exc)
            self._run = None

    def _log_metrics(self, metrics: Mapping[str, float], step: int, commit: bool = True) -> None:
        """Send metrics to W&B when a run is active.

        Args:
            metrics: Scalar metrics to log.
            step: X-axis value (epoch or global step).
            commit: Whether to close the current logging step.
        """
        if self._run is None:
            return
        try:
            self._run.log(dict(metrics), step=step, commit=commit)
        except Exception as exc:  # noqa: BLE001 - tracking must never break training.
            logger.warning("Failed to log metrics to W&B: %s", exc)

    def _finish_run(self) -> None:
        """Upload the best checkpoint and close the W&B run."""
        if self._run is None:
            return
        try:
            if self.checkpoint_path.is_file():
                self._run.summary["best_epoch"] = self._best_epoch
                self._run.summary[f"best_{self._monitor}"] = self._best_score
                self._run.save(str(self.checkpoint_path), policy="now")
            self._run.finish()
        except Exception as exc:  # noqa: BLE001 - tracking must never break training.
            logger.warning("Failed to finalise the W&B run: %s", exc)
        finally:
            self._run = None


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for a full training run.

    Args:
        argv: Command-line arguments; defaults to :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` on a handled configuration/data/training failure.
    """
    parser = argparse.ArgumentParser(description="Train the NYC taxi trip-duration model.")
    parser.add_argument("--config", default=None, help="Path to the YAML config.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotlist config overrides, e.g. optimizer.lr=3e-4 training.max_epochs=50",
    )
    args = parser.parse_args(argv)

    setup_logging()
    try:
        cfg = load_config(args.config, overrides=args.overrides)
        seed_everything(int(cfg.project.seed))

        datamodule = TaxiDataModule(cfg).setup()
        model = build_model(cfg, input_dim=datamodule.input_dim)

        trainer = Trainer(model, datamodule, cfg)
        trainer.fit()
        trainer.evaluate(split="test")

        save_config(cfg, Path(str(cfg.training.checkpoint_dir)) / "resolved_config.yaml")
    except (ConfigError, DataError, TrainingError) as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
