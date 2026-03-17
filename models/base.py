"""
models/base.py
==============
Abstract base class that every model wrapper must implement.

All models in the unified pipeline share:
- A common ``__init__(config)`` constructor
- ``training_step(batch)`` → scalar loss tensor (or None for ARIMA)
- ``validation_step(batch)`` → scalar loss tensor (or None)
- ``predict(context, n_samples)`` → (B, n_future, n_samples) ndarray
- ``save(path)`` / ``load(path)`` for checkpoint I/O

The ``forward`` method is intentionally left abstract; for generative
models it is not always well-defined, but subclasses must implement it
to satisfy the interface (e.g. returning the model's best point prediction).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class BaseModel(ABC):
    """Abstract interface for all trajectory forecasting models."""

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    @abstractmethod
    def __init__(self, config: dict) -> None:
        """
        Initialise the model from a merged config dict.

        The config contains at minimum:
          config["model"]    — model-specific hyperparameters
          config["data"]     — n_past, n_future, seed, …
          config["training"] — device, learning_rate, …
        """

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic forward pass (e.g. median / mean prediction).

        Parameters
        ----------
        x : (B, n_past) context tensor.

        Returns
        -------
        (B, n_future) point prediction tensor.
        """

    # -----------------------------------------------------------------------
    # Training interface
    # -----------------------------------------------------------------------

    @abstractmethod
    def training_step(self, batch: tuple) -> torch.Tensor:
        """
        Compute the training loss for one mini-batch.

        Parameters
        ----------
        batch : ``(context, target)`` tuple of tensors.

        Returns
        -------
        Scalar loss tensor (differentiable).
        """

    @abstractmethod
    def validation_step(self, batch: tuple) -> torch.Tensor:
        """
        Compute the validation loss for one mini-batch (no gradients).

        Parameters
        ----------
        batch : ``(context, target)`` tuple of tensors.

        Returns
        -------
        Scalar loss tensor.
        """

    # -----------------------------------------------------------------------
    # Probabilistic inference
    # -----------------------------------------------------------------------

    @abstractmethod
    def predict(
        self,
        context: torch.Tensor,
        n_samples: int = 500,
    ) -> np.ndarray:
        """
        Draw ``n_samples`` forecast trajectories for each context in the batch.

        Parameters
        ----------
        context   : (B, n_past) tensor of observed history.
        n_samples : Number of sample paths to draw.

        Returns
        -------
        samples : (B, n_future, n_samples) float32 ndarray.
        """

    # -----------------------------------------------------------------------
    # Checkpoint I/O
    # -----------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save model checkpoint to ``path``.

        Default implementation saves ``state_dict()`` if the model has one.
        Override for models that use a different checkpoint format (e.g.
        PyTorch Lightning).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "model") and hasattr(self.model, "state_dict"):
            torch.save(self.model.state_dict(), path)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must override save()."
            )

    def load(self, path: Path) -> None:
        """
        Load model checkpoint from ``path``.

        Default implementation loads ``state_dict()`` into ``self.model``.
        Override for alternative checkpoint formats.
        """
        path = Path(path)
        if hasattr(self, "model") and hasattr(self.model, "load_state_dict"):
            state = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must override load()."
            )

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the model name from config (set by subclasses)."""
        return getattr(self, "_name", self.__class__.__name__.lower())
