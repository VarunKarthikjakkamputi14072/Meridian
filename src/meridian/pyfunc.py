"""MLflow ``pyfunc`` wrapper so the registry holds one self-contained artifact:
torch weights + scaler + the exact feature order. Serving just does
``mlflow.pyfunc.load_model(...)`` and never has to re-implement preprocessing."""
from __future__ import annotations

import json

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from .model import StandardScaler, TripDurationNet


class TripDurationModel:
    """Thin object used by the mlflow.pyfunc.PythonModel below and in tests."""

    def __init__(self, net: TripDurationNet, scaler: StandardScaler, feature_cols: list[str]):
        self.net = net.eval()
        self.scaler = scaler
        self.feature_cols = feature_cols

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_cols].to_numpy(dtype="float32")
        xs = self.scaler.transform(x).astype("float32")
        with torch.no_grad():
            return self.net(torch.from_numpy(xs)).cpu().numpy()


class MLflowTripModel(mlflow.pyfunc.PythonModel):
    """mlflow.pyfunc.PythonModel implementation wrapping the torch net + scaler."""

    def load_context(self, context):  # noqa: D401 - mlflow hook
        with open(context.artifacts["meta"]) as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        net = TripDurationNet(n_features=len(self.feature_cols), hidden=meta["hidden"])
        net.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu"))
        scaler = StandardScaler.from_state(meta["scaler"])
        self._model = TripDurationModel(net, scaler, self.feature_cols)

    def predict(self, context, model_input: pd.DataFrame):
        return self._model.predict_df(model_input)
