"""The served model: a small PyTorch MLP regressor plus a sklearn-style scaler,
bundled together so feature scaling travels with the weights into the registry."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TripDurationNet(nn.Module):
    """Plain MLP. Deliberately small — the point of the project is the lifecycle
    around the model, not squeezing out the last RMSE point."""

    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class StandardScaler:
    """Minimal, serializable standardizer (avoids a hard sklearn dep at serve time)."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "StandardScaler":
        return cls(np.array(state["mean"]), np.array(state["std"]))
