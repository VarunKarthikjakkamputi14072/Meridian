"""NVIDIA NIM (build.nvidia.com) client for the drift dashboard.

Kept dependency-light (httpx only) and self-contained so the dashboard runs
without importing the heavy training stack (torch / mlflow / evidently).

The endpoint is OpenAI-compatible. Configure via env:
    NVIDIA_API_KEY   (required for live RCA)
    NVIDIA_BASE_URL  (default https://integrate.api.nvidia.com/v1)
    NVIDIA_MODEL     (default meta/llama-3.3-70b-instruct)
"""
from __future__ import annotations

import json
import os

import httpx

BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")

SYSTEM_PROMPT = (
    "You are Meridian's MLOps Incident Commander. A data-drift monitor (Evidently) "
    "has compared live prediction traffic against the model's training distribution "
    "and flagged drift. Given the drift summary, write a crisp root-cause analysis for "
    "an on-call ML engineer. Be concrete and decisive. Structure your answer as:\n"
    "1. WHAT DRIFTED — name the features and how far they moved.\n"
    "2. LIKELY CAUSE — a plausible real-world explanation.\n"
    "3. MODEL IMPACT — why predictions are now unreliable.\n"
    "4. RECOMMENDATION — retrain or not, on what data, and confidence.\n"
    "Keep it under 180 words. No preamble."
)


def has_key() -> bool:
    return bool(os.getenv("NVIDIA_API_KEY"))


def generate_rca(summary: dict, feature_stats: dict | None = None) -> str:
    """Return an AI-written root-cause analysis for a drift summary.

    `summary` is the dict produced by meridian.drift.monitor.evaluate:
        {share, n_drifted, dataset_drift, per_feature}
    `feature_stats` optionally maps feature -> {reference_mean, current_mean}.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set.")

    drifted = [f for f, d in summary.get("per_feature", {}).items() if d]
    context = {
        "share_of_drifted_features": round(summary.get("share", 0.0), 3),
        "threshold": 0.5,
        "n_drifted": summary.get("n_drifted", len(drifted)),
        "drifted_features": drifted,
        "feature_shift": feature_stats or {},
    }
    user = (
        "Drift summary (JSON):\n"
        + json.dumps(context, indent=2)
        + "\n\nWrite the root-cause analysis."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers)
    if resp.status_code >= 400:
        raise RuntimeError(f"NIM error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
