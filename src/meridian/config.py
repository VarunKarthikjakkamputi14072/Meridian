"""Central configuration. Everything is overridable via environment variables
so the same code runs locally, in docker-compose, and in CI."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MERIDIAN_", env_file=".env", extra="ignore",
        protected_namespaces=(),  # we intentionally use model_name / model_alias
    )

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5500"
    model_name: str = "taxi-trip-duration"
    # Stage to serve from. Training promotes the best run to this alias.
    model_alias: str = "production"

    # --- paths (shared volume in docker, local dir otherwise) ---
    data_dir: Path = Path("data")
    reference_file: str = "reference.parquet"      # training distribution snapshot
    prediction_log: str = "predictions.csv"        # live serving log, read by drift monitor

    # --- model / features ---
    seed: int = 42
    feature_cols: tuple[str, ...] = (
        "passenger_count",
        "trip_distance_km",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
        "temp_c",
        "precip_mm",
    )
    target_col: str = "trip_duration_min"

    # --- drift ---
    # Share of features that must drift before we consider the dataset drifted.
    drift_dataset_threshold: float = 0.5
    drift_window: int = 500            # rows of recent traffic to evaluate
    drift_check_interval_s: int = 30   # how often the monitor wakes up
    drift_metrics_port: int = 8001     # prometheus scrape port for the monitor
    retrain_on_drift: bool = True

    @property
    def reference_path(self) -> Path:
        return self.data_dir / self.reference_file

    @property
    def prediction_log_path(self) -> Path:
        return self.data_dir / self.prediction_log


settings = Settings()
