"""Synthetic RAG query telemetry, with the same baseline-vs-skewed split as the
taxi generator.

The features are exactly what Transit's tap emits per chat completion, so the
real telemetry log and this synthetic reference share a schema:

* ``generate_rag_baseline`` — on-corpus traffic: short, focused questions the
  index answers well, with modest retrieved context and latency.
* ``generate_rag_skewed``   — off-corpus traffic: longer, broader questions the
  corpus does not cover, which pull in more context, burn more tokens, and run
  slower. That covariate shift is what the RAG drift monitor is meant to catch —
  the signal to re-embed with fresh documents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ("query_len", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms")


def _synthesize(n: int, rng: np.random.Generator, *, skew: bool) -> pd.DataFrame:
    if skew:
        # Off-corpus: longer questions, more retrieved context, slower answers.
        query_len = rng.normal(150, 40, size=n).clip(10, 600)
        context_tokens = rng.normal(900, 200, size=n).clip(50, 4000)
        completion_tokens = rng.normal(220, 60, size=n).clip(10, 1200)
        latency_ms = rng.normal(1500, 400, size=n).clip(50, 8000)
    else:
        # On-corpus: short, focused questions the index answers well.
        query_len = rng.normal(60, 18, size=n).clip(5, 400)
        context_tokens = rng.normal(400, 120, size=n).clip(20, 3000)
        completion_tokens = rng.normal(120, 35, size=n).clip(5, 800)
        latency_ms = rng.normal(800, 180, size=n).clip(30, 6000)

    # prompt_tokens ~ the user's question turned to tokens plus the retrieved
    # context that was stuffed into the prompt.
    prompt_tokens = (query_len / 4.0) + context_tokens
    total_tokens = prompt_tokens + completion_tokens

    return pd.DataFrame(
        {
            "query_len": query_len.round().astype(int),
            "prompt_tokens": prompt_tokens.round().astype(int),
            "completion_tokens": completion_tokens.round().astype(int),
            "total_tokens": total_tokens.round().astype(int),
            "latency_ms": latency_ms.round().astype(int),
        }
    )


def generate_rag_baseline(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    return _synthesize(n, np.random.default_rng(seed), skew=False)


def generate_rag_skewed(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    return _synthesize(n, np.random.default_rng(seed), skew=True)
