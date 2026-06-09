"""Data-drift monitor.

Loop:
  1. read the most recent N rows the serving layer logged,
  2. compare them against the training reference with Evidently,
  3. publish drift metrics on a Prometheus endpoint,
  4. if the share of drifted features crosses the threshold, fire a retrain
     (and tell the serving layer to hot-reload the new model).

Prometheus scrapes step 3 and raises the alert (see prometheus/alerts.yml); this
process also acts on the breach directly so the demo closes the loop end-to-end.
"""
from __future__ import annotations

import time

import httpx
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import Counter, Gauge, start_http_server

from ..config import settings

DRIFT_SHARE = Gauge("meridian_data_drift_share", "Share of features detected as drifted")
DRIFT_FLAG = Gauge("meridian_dataset_drift", "1 if dataset drift threshold breached")
DRIFTED_FEATURES = Gauge("meridian_feature_drift", "Per-feature drift flag", ["feature"])
ROWS_EVALUATED = Gauge("meridian_drift_rows_evaluated", "Rows in the live window")
RETRAINS = Counter("meridian_retrains_triggered_total", "Retrains triggered by drift")


def _load_reference() -> pd.DataFrame:
    return pd.read_parquet(settings.reference_path)[list(settings.feature_cols)]


def _load_current() -> pd.DataFrame | None:
    path = settings.prediction_log_path
    if not path.exists():
        return None
    df = pd.read_csv(path).tail(settings.drift_window)
    cols = [c for c in settings.feature_cols if c in df.columns]
    if len(df) < 50:  # not enough live traffic to judge yet
        return None
    return df[cols]


def evaluate(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run Evidently and return a compact, JSON-able summary."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    metrics = report.as_dict()["metrics"]
    # DataDriftPreset emits a summary metric (DatasetDriftMetric) and a per-column
    # table (DataDriftTable); pull the share from the first, the breakdown from the second.
    summary = metrics[0]["result"]
    by_column = metrics[1]["result"]["drift_by_columns"]
    per_feature = {
        name: bool(info["drift_detected"]) for name, info in by_column.items()
    }
    share = float(summary["share_of_drifted_columns"])
    return {
        "share": share,
        "n_drifted": int(summary["number_of_drifted_columns"]),
        "dataset_drift": share >= settings.drift_dataset_threshold,
        "per_feature": per_feature,
    }


def _publish(summary: dict, n_rows: int) -> None:
    DRIFT_SHARE.set(summary["share"])
    DRIFT_FLAG.set(1 if summary["dataset_drift"] else 0)
    ROWS_EVALUATED.set(n_rows)
    for feature, drifted in summary["per_feature"].items():
        DRIFTED_FEATURES.labels(feature=feature).set(1 if drifted else 0)


def _trigger_retrain() -> None:
    RETRAINS.inc()
    print("[drift] threshold breached -> retraining on latest distribution")
    from ..train import train_and_register

    result = train_and_register(reason="drift-retrain")
    if result["promoted"]:
        _notify_serving_reload()


def _notify_serving_reload() -> None:
    url = settings.__dict__.get("serving_reload_url") or "http://serving:8000/reload"
    try:
        httpx.post(url, timeout=10)
        print("[drift] serving asked to reload new model")
    except Exception as e:  # noqa: BLE001
        print(f"[drift] could not reach serving for reload: {e}")


def run_forever() -> None:
    start_http_server(settings.drift_metrics_port)
    print(f"[drift] metrics on :{settings.drift_metrics_port}, "
          f"threshold={settings.drift_dataset_threshold}")
    reference = _load_reference()
    already_retrained = False
    while True:
        current = _load_current()
        if current is not None:
            summary = evaluate(reference, current)
            _publish(summary, len(current))
            print(f"[drift] share={summary['share']:.2f} "
                  f"drifted={summary['n_drifted']} dataset_drift={summary['dataset_drift']}")
            if summary["dataset_drift"] and settings.retrain_on_drift and not already_retrained:
                _trigger_retrain()
                reference = _load_reference()   # new baseline after retrain
                already_retrained = True
            elif not summary["dataset_drift"]:
                already_retrained = False
        time.sleep(settings.drift_check_interval_s)


if __name__ == "__main__":
    run_forever()
