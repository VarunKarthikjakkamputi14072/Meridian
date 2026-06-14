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

    # --- RAG drift (fed by Transit's telemetry tap) ---
    # The receiver appends query telemetry here; the RAG monitor tails it — the
    # same "writer logs a CSV, monitor reads it" contract as serving/predictions.
    rag_telemetry_log: str = "rag_telemetry.csv"
    rag_reference_file: str = "rag_reference.parquet"
    rag_feature_cols: tuple[str, ...] = (
        "query_len",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_ms",
    )
    rag_drift_threshold: float = 0.5
    rag_drift_window: int = 300
    rag_service_port: int = 8002       # telemetry receiver + RAG drift metrics
    # On a RAG-drift breach, ask Hermes to re-ingest/re-embed the corpus — the
    # act side of the loop (observe on the read path, act on the write path).
    hermes_ingest_url: str = "http://order-api:8080/api/ingest"
    reembed_on_drift: bool = True
    reembed_doc_id: str = "rag-corpus"
    reembed_chunk_count: int = 200

    @property
    def reference_path(self) -> Path:
        return self.data_dir / self.reference_file

    @property
    def prediction_log_path(self) -> Path:
        return self.data_dir / self.prediction_log

    @property
    def rag_telemetry_path(self) -> Path:
        return self.data_dir / self.rag_telemetry_log

    @property
    def rag_reference_path(self) -> Path:
        return self.data_dir / self.rag_reference_file


settings = Settings()
