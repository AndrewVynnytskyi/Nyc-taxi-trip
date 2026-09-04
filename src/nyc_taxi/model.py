"""PyTorch model definitions for trip-duration regression.

The baseline is a residual MLP: a stack of pre-activation blocks with optional
batch normalisation, dropout and skip connections, followed by a scalar head.
Architecture is fully driven by the ``model`` config section -- no layer size,
dropout rate or activation is hardcoded here.
"""

from __future__ import annotations

from typing import Callable, Final, Mapping, Sequence, cast

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from nyc_taxi.utils import count_parameters, get_logger

__all__ = ["MLPBlock", "ModelError", "TripDurationMLP", "build_model"]

logger = get_logger(__name__)

#: Supported activations, keyed by the value of ``model.activation``.
_ACTIVATIONS: Final[Mapping[str, Callable[[], nn.Module]]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}

#: Registry of model architectures, keyed by the value of ``model.name``.
_REGISTRY: Final[dict[str, type[nn.Module]]] = {}


class ModelError(RuntimeError):
    """Raised when a model cannot be constructed from the configuration."""


def _resolve_activation(name: str) -> nn.Module:
    """Instantiate an activation module by config name.

    Args:
        name: Activation identifier, case-insensitive.

    Returns:
        A freshly constructed activation module.

    Raises:
        ModelError: If the name is not supported.
    """
    key = name.strip().lower()
    if key not in _ACTIVATIONS:
        raise ModelError(f"Unsupported activation {name!r}. Choose one of {sorted(_ACTIVATIONS)}.")
    return _ACTIVATIONS[key]()


class MLPBlock(nn.Module):
    """A pre-activation feed-forward block with an optional residual path.

    The block computes ``dropout(act(norm(linear(x))))`` and, when the input and
    output widths match and ``residual`` is enabled, adds the identity.

    Args:
        in_features: Input width.
        out_features: Output width.
        activation: Activation name, e.g. ``"gelu"``.
        dropout: Dropout probability in ``[0, 1)``.
        use_batch_norm: Whether to apply :class:`~torch.nn.BatchNorm1d`.
        residual: Whether to add a skip connection when widths permit.

    Raises:
        ModelError: If a width is non-positive or ``dropout`` is out of range.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_batch_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ModelError(
                f"Layer widths must be positive, got in={in_features}, out={out_features}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ModelError(f"dropout must lie in [0, 1), got {dropout}.")

        self.use_residual = bool(residual) and in_features == out_features
        self.linear = nn.Linear(in_features, out_features, bias=not use_batch_norm)
        self.norm: nn.Module = nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity()
        self.activation = _resolve_activation(activation)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the block.

        Args:
            inputs: Tensor of shape ``(batch, in_features)``.

        Returns:
            Tensor of shape ``(batch, out_features)``.
        """
        hidden = cast(Tensor, self.dropout(self.activation(self.norm(self.linear(inputs)))))
        return hidden + inputs if self.use_residual else hidden


class TripDurationMLP(nn.Module):
    """Residual MLP that regresses a single continuous trip attribute.

    The head emits an unbounded scalar in *model space*; converting it back to
    minutes is :meth:`nyc_taxi.data.TaxiDataModule.inverse_transform_target`'s
    job, which keeps the network agnostic to the target transform.

    Args:
        input_dim: Number of input features.
        hidden_dims: Widths of the hidden blocks, outermost first.
        dropout: Dropout probability applied inside every block.
        activation: Activation name shared by all blocks.
        use_batch_norm: Whether blocks use batch normalisation.
        residual: Whether blocks add skip connections when widths match.
        output_dim: Width of the regression head. Defaults to ``1``.

    Raises:
        ModelError: If ``input_dim``/``output_dim`` is non-positive or
            ``hidden_dims`` is empty.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.15,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        residual: bool = True,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ModelError(f"input_dim must be >= 1, got {input_dim}.")
        if output_dim < 1:
            raise ModelError(f"output_dim must be >= 1, got {output_dim}.")

        dims = [int(dim) for dim in hidden_dims]
        if not dims:
            raise ModelError("hidden_dims must contain at least one layer width.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(dims)

        blocks: list[nn.Module] = []
        previous = input_dim
        for width in dims:
            blocks.append(
                MLPBlock(
                    in_features=previous,
                    out_features=width,
                    activation=activation,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    residual=residual,
                )
            )
            previous = width

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(previous, output_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Apply Kaiming-uniform init to linear layers and zero their biases.

        Args:
            module: Submodule visited by :meth:`torch.nn.Module.apply`.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Predict the target for a batch of feature vectors.

        Args:
            features: Tensor of shape ``(batch, input_dim)``.

        Returns:
            Tensor of shape ``(batch, output_dim)`` in model space.

        Raises:
            ModelError: If the input rank or width does not match the contract.
        """
        if features.ndim != 2:
            raise ModelError(f"Expected a 2-D input (batch, features), got shape {features.shape}.")
        if features.shape[1] != self.input_dim:
            raise ModelError(f"Expected {self.input_dim} input features, got {features.shape[1]}.")
        return cast(Tensor, self.head(self.backbone(features)))

    @torch.inference_mode()
    def predict(self, features: Tensor) -> Tensor:
        """Run a gradient-free forward pass in evaluation mode.

        The module's original train/eval state is restored before returning.

        Args:
            features: Tensor of shape ``(batch, input_dim)``.

        Returns:
            Predictions of shape ``(batch, output_dim)``, detached.
        """
        was_training = self.training
        self.eval()
        try:
            return self.forward(features)
        finally:
            self.train(was_training)

    def extra_repr(self) -> str:
        """Summarise the architecture for ``print(model)``."""
        return (
            f"input_dim={self.input_dim}, hidden_dims={list(self.hidden_dims)}, "
            f"output_dim={self.output_dim}, params={count_parameters(self):,}"
        )


_REGISTRY["tripdurationmlp"] = TripDurationMLP


def build_model(cfg: DictConfig, input_dim: int) -> nn.Module:
    """Instantiate the configured architecture.

    Args:
        cfg: Full project config (the ``model`` section is read).
        input_dim: Feature count, taken from
            :attr:`nyc_taxi.data.TaxiDataModule.input_dim` so that the model and
            the data can never silently disagree.

    Returns:
        The constructed model, on CPU and in training mode.

    Raises:
        ModelError: If ``model.name`` is not registered or the hyperparameters
            are invalid.
    """
    model_cfg = cfg.model
    key = str(model_cfg.name).strip().lower()
    if key not in _REGISTRY:
        raise ModelError(f"Unknown model.name {model_cfg.name!r}. Registered: {sorted(_REGISTRY)}.")

    model = _REGISTRY[key](
        input_dim=input_dim,
        hidden_dims=[int(dim) for dim in model_cfg.hidden_dims],
        dropout=float(model_cfg.dropout),
        activation=str(model_cfg.activation),
        use_batch_norm=bool(model_cfg.use_batch_norm),
        residual=bool(model_cfg.residual),
    )
    logger.info(
        "Built %s | input_dim=%d | trainable params=%s",
        model_cfg.name,
        input_dim,
        f"{count_parameters(model):,}",
    )
    return model
