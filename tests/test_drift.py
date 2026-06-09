from meridian.data import generate_baseline, generate_skewed
from meridian.drift.monitor import evaluate


def test_no_drift_against_same_distribution():
    ref = generate_baseline(n=3000, seed=1)
    cur = generate_baseline(n=1000, seed=2)
    summary = evaluate(ref, cur)
    assert summary["dataset_drift"] is False


def test_drift_detected_against_skewed_traffic():
    ref = generate_baseline(n=3000, seed=1)
    cur = generate_skewed(n=1000, seed=2)
    summary = evaluate(ref, cur)
    assert summary["share"] > 0
    assert summary["dataset_drift"] is True
