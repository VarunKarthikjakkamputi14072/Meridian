"""RAG drift monitor.

The taxi monitor's loop, pointed at a live RAG app instead of a regressor:

  1. read the most recent query telemetry Transit tapped from the gateway,
  2. compare it to the corpus-query reference with Evidently,
  3. publish RAG drift metrics for Prometheus,
  4. if the share of drifted features crosses the threshold, ask Hermes to
     re-embed — the act side of the loop.

Observe on the read path (Transit -> here), act on the write path (here ->
Hermes). The re-embed is the only place this differs from the model monitor,
which retrains; here the "fix" is fresh embeddings, not new model weights.
"""
from __future__ import annotations

import time

import httpx
import pandas as pd
from prometheus_client import Counter, Gauge

from ..config import settings
from ..drift.monitor import evaluate as _evidently_evaluate

RAG_DRIFT_SHARE = Gauge("meridian_rag_drift_share", "Share of RAG features detected as drifted")
RAG_DRIFT_FLAG = Gauge("meridian_rag_dataset_drift", "1 if RAG drift threshold breached")
RAG_FEATURE_DRIFT = Gauge("meridian_rag_feature_drift", "Per-feature RAG drift flag", ["feature"])
RAG_ROWS = Gauge("meridian_rag_rows_evaluated", "Rows in the live RAG window")
REEMBEDS = Counter("meridian_reembeds_triggered_total", "Re-embeds triggered by RAG drift")


def load_reference() -> pd.DataFrame:
    return pd.read_parquet(settings.rag_reference_path)[list(settings.rag_feature_cols)]


def load_current() -> pd.DataFrame | None:
    path = settings.rag_telemetry_path
    if not path.exists():
        return None
    df = pd.read_csv(path).tail(settings.rag_drift_window)
    cols = [c for c in settings.rag_feature_cols if c in df.columns]
    if len(df) < 50:  # not enough live queries to judge yet
        return None
    return df[cols]


def evaluate(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Evidently drift on the RAG features, with the RAG dataset threshold."""
    summary = _evidently_evaluate(reference, current)
    summary["dataset_drift"] = summary["share"] >= settings.rag_drift_threshold
    return summary


def _publish(summary: dict, n_rows: int) -> None:
    RAG_DRIFT_SHARE.set(summary["share"])
    RAG_DRIFT_FLAG.set(1 if summary["dataset_drift"] else 0)
    RAG_ROWS.set(n_rows)
    for feature, drifted in summary["per_feature"].items():
        RAG_FEATURE_DRIFT.labels(feature=feature).set(1 if drifted else 0)


def trigger_reembed() -> bool:
    """Ask Hermes to re-ingest/re-embed the corpus. Failure-isolated."""
    REEMBEDS.inc()
    payload = {
        "docId": settings.reembed_doc_id,
        "source": "meridian-rag-drift",
        "chunkCount": settings.reembed_chunk_count,
    }
    try:
        resp = httpx.post(settings.hermes_ingest_url, json=payload, timeout=10)
        print(f"[rag-drift] re-embed requested -> Hermes {resp.status_code}")
        return resp.status_code < 400
    except Exception as e:  # noqa: BLE001 — never crash the monitor on a trigger
        print(f"[rag-drift] could not reach Hermes to re-embed: {e}")
        return False


def run_once(reference: pd.DataFrame, already_triggered: bool) -> bool:
    """One evaluation cycle. Returns the updated 'already_triggered' latch."""
    current = load_current()
    if current is None:
        return already_triggered
    summary = evaluate(reference, current)
    _publish(summary, len(current))
    print(f"[rag-drift] share={summary['share']:.2f} "
          f"drifted={summary['n_drifted']} dataset_drift={summary['dataset_drift']}")
    if summary["dataset_drift"] and settings.reembed_on_drift and not already_triggered:
        trigger_reembed()
        return True
    if not summary["dataset_drift"]:
        return False
    return already_triggered


def run_loop(stop=lambda: False) -> None:
    reference = load_reference()
    already_triggered = False
    while not stop():
        already_triggered = run_once(reference, already_triggered)
        time.sleep(settings.drift_check_interval_s)
