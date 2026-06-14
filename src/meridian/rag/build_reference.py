"""Write the RAG drift reference — the on-corpus query distribution the live
telemetry is compared against. Run once at setup (or after a deliberate corpus
refresh): ``python -m meridian.rag.build_reference``."""
from __future__ import annotations

from ..config import settings
from .data import generate_rag_baseline


def main() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    ref = generate_rag_baseline(n=2000, seed=settings.seed)
    ref.to_parquet(settings.rag_reference_path)
    print(f"[rag-drift] wrote reference {settings.rag_reference_path} "
          f"({len(ref)} rows, cols={list(ref.columns)})")


if __name__ == "__main__":
    main()
