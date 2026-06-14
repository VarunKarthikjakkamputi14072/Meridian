"""RAG drift detection mirrors the taxi-drift contract: baseline-vs-baseline is
quiet, baseline-vs-off-corpus fires."""

from meridian.rag.data import generate_rag_baseline, generate_rag_skewed
from meridian.rag.monitor import evaluate, trigger_reembed


def test_no_rag_drift_against_same_distribution():
    ref = generate_rag_baseline(n=2000, seed=1)
    cur = generate_rag_baseline(n=800, seed=2)
    summary = evaluate(ref, cur)
    assert summary["dataset_drift"] is False


def test_rag_drift_detected_against_off_corpus_queries():
    ref = generate_rag_baseline(n=2000, seed=1)
    cur = generate_rag_skewed(n=800, seed=2)
    summary = evaluate(ref, cur)
    assert summary["share"] > 0
    assert summary["dataset_drift"] is True


def test_trigger_reembed_posts_to_hermes(monkeypatch):
    sent = {}

    class _Resp:
        status_code = 202

    def _fake_post(url, json, timeout):  # noqa: A002
        sent["url"] = url
        sent["json"] = json
        return _Resp()

    monkeypatch.setattr("meridian.rag.monitor.httpx.post", _fake_post)
    ok = trigger_reembed()

    assert ok is True
    assert "/api/ingest" in sent["url"]
    assert sent["json"]["docId"]
    assert sent["json"]["chunkCount"] > 0
