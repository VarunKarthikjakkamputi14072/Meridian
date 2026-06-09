"""FastAPI serving layer.

Responsibilities:
  * load the ``production`` model from the MLflow registry (and hot-reload it),
  * serve predictions,
  * expose Prometheus metrics at ``/metrics`` (latency, throughput, predicted value),
  * log every request's features to a shared CSV so the drift monitor has live data.
"""
from __future__ import annotations

import csv
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from ..config import settings
from .schemas import PredictionResponse, TripFeatures

# --- Prometheus metrics ---
PRED_COUNT = Counter("meridian_predictions_total", "Total predictions served")
PRED_LATENCY = Histogram("meridian_prediction_latency_seconds", "Inference latency")
PRED_VALUE = Histogram(
    "meridian_predicted_duration_minutes", "Distribution of predicted trip duration",
    buckets=[2, 5, 10, 15, 20, 30, 45, 60, 90],
)
MODEL_INFO = Gauge("meridian_model_loaded", "1 if a model is loaded", ["version"])


class ModelHolder:
    """Loads the production model and supports hot-reload after a retrain."""

    def __init__(self):
        self._model = None
        self._version = "none"
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return self._version

    def load(self) -> str:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
        model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}@{settings.model_alias}")
        with self._lock:
            self._model = model
            self._version = mv.version
        MODEL_INFO.clear()
        MODEL_INFO.labels(version=mv.version).set(1)
        return mv.version

    def predict(self, df: pd.DataFrame) -> float:
        with self._lock:
            if self._model is None:
                raise RuntimeError("model not loaded")
            return float(self._model.predict(df)[0])


holder = ModelHolder()


def _log_prediction(features: dict, prediction: float) -> None:
    """Append the served features to the shared prediction log (drift input)."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    path = settings.prediction_log_path
    new_file = not path.exists()
    row = {"ts": datetime.now(timezone.utc).isoformat(), **features, "prediction": prediction}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Retry: in docker-compose the model may not be registered the instant we boot.
    for attempt in range(30):
        try:
            v = holder.load()
            print(f"[serving] loaded model version {v}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[serving] waiting for model ({attempt}): {e}")
            time.sleep(2)
    yield


app = FastAPI(title="Meridian — Trip Duration Serving", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": holder.version}


@app.post("/reload")
def reload_model():
    """Called after a retrain to pick up the new production version without a restart."""
    try:
        return {"reloaded": True, "model_version": holder.load()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"reload failed: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TripFeatures):
    if holder.version == "none":
        raise HTTPException(503, "model not loaded yet")
    payload = features.model_dump()
    df = pd.DataFrame([payload])[list(settings.feature_cols)]
    start = time.perf_counter()
    try:
        pred = holder.predict(df)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"inference error: {e}")
    PRED_LATENCY.observe(time.perf_counter() - start)
    PRED_COUNT.inc()
    PRED_VALUE.observe(pred)
    _log_prediction(payload, pred)
    return PredictionResponse(trip_duration_min=round(pred, 2), model_version=holder.version)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
