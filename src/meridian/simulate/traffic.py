"""Live-traffic simulator.

Phase 1 ("normal"): replay baseline-distribution trips so the model behaves and
drift stays low. Phase 2 ("skew"): switch to weather-impacted / geo-shifted trips
so the drift monitor crosses its threshold and the retrain loop fires.

    python -m meridian.simulate.traffic --normal 600 --skew 600 --rps 25
"""
from __future__ import annotations

import argparse
import time

import httpx

from ..config import settings
from ..data import generate_baseline, generate_skewed

FEATURES = list(settings.feature_cols)


def run(base_url: str, n_normal: int, n_skew: int, rps: int) -> None:
    url = f"{base_url.rstrip('/')}/predict"
    delay = 1.0 / max(rps, 1)
    with httpx.Client() as client:
        for phase, n, gen in (("normal", n_normal, generate_baseline),
                              ("skew", n_skew, generate_skewed)):
            if n <= 0:
                continue
            print(f"--- phase: {phase} ({n} requests) ---")
            df = gen(n=n).reset_index(drop=True)
            sent = ok = err = 0
            for _, row in df.iterrows():
                payload = {c: (int(row[c]) if c in ("passenger_count", "pickup_hour", "pickup_dayofweek")
                               else float(row[c])) for c in FEATURES}
                try:
                    r = client.post(url, json=payload, timeout=5)
                    ok += r.status_code == 200
                    err += r.status_code != 200
                except Exception:  # noqa: BLE001
                    err += 1
                sent += 1
                if sent % 100 == 0:
                    print(f"  [{phase}] sent={sent} ok={ok} err={err}")
                time.sleep(delay)
            print(f"  [{phase}] done: ok={ok} err={err}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Drive live traffic at the serving layer.")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--normal", type=int, default=600, help="normal-distribution requests")
    p.add_argument("--skew", type=int, default=600, help="skewed requests to trigger drift")
    p.add_argument("--rps", type=int, default=25)
    args = p.parse_args()
    run(args.url, args.normal, args.skew, args.rps)
