"""RAG telemetry receiver + drift monitor, in one FastAPI app.

Transit POSTs a record per chat query to ``/telemetry/rag``; this service
appends it to the shared CSV (the same contract serving uses for predictions)
and a background thread runs the RAG drift loop against it, exposing
``/metrics`` for Prometheus. Kept in one process so the demo needs one extra
service, not two.
"""
from __future__ import annotations

import csv
import threading
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from ..config import settings
from . import monitor

TELEMETRY_RECEIVED = Counter("meridian_rag_telemetry_received_total", "RAG telemetry records received")

_FIELDS = ["ts", "app", "model", *settings.rag_feature_cols, "cache_hit"]


class RagTelemetry(BaseModel):
    ts: float | None = None
    app: str = "unknown"
    model: str = ""
    query_len: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    cache_hit: int = 0


def _append(record: RagTelemetry) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    path = settings.rag_telemetry_path
    new_file = not path.exists()
    row = record.model_dump()
    row["ts"] = row.get("ts") or datetime.now(UTC).timestamp()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in _FIELDS})


app = FastAPI(title="Meridian — RAG Drift Monitor", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/telemetry/rag")
def receive(record: RagTelemetry) -> dict:
    _append(record)
    TELEMETRY_RECEIVED.inc()
    return {"ok": True}


@app.post("/reference/build")
def build_reference() -> dict:
    """Snapshot the current telemetry as the drift reference — the on-corpus
    baseline. Call after a warm-up of normal traffic so drift is judged against
    observed production, not a synthetic distribution."""
    path = settings.rag_telemetry_path
    if not path.exists():
        return {"ok": False, "reason": "no telemetry yet"}
    df = pd.read_csv(path)[list(settings.rag_feature_cols)]
    settings.rag_reference_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.rag_reference_path)
    return {"ok": True, "rows": int(len(df))}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
def _start_monitor() -> None:
    # The monitor loops; it idles until a reference parquet exists (built from a
    # warm-up via /reference/build, or by meridian.rag.build_reference).
    threading.Thread(target=monitor.run_loop, name="rag-drift-monitor", daemon=True).start()
