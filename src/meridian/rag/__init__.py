"""RAG drift monitoring — watches the query telemetry Transit taps from the
gateway and, when production queries drift away from what the corpus can answer,
asks Hermes to re-embed. The MLOps loop applied to a live RAG app instead of the
taxi regressor: observe on the read path, act on the write path."""
