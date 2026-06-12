# Meridian — Drift Control Dashboard

A Streamlit "god-mode" UI on top of Meridian's MLOps pipeline. It makes the
self-healing loop **visible**: predict an ETA, flip on a blizzard, watch drift
cross the 0.5 threshold, have an **NVIDIA NIM** model write the root-cause
analysis, then retrain and watch drift recover.

## Run with the full stack (recommended)

The dashboard is wired into the root `docker compose` stack — `docker compose up
--build` from the repo root builds this image and serves it at
**http://localhost:8501**, already pointed at the `serving` container. Put your
`NVIDIA_API_KEY` in the repo-root `.env` and compose passes it through.

## Run (demo mode — no cluster needed)

```bash
cd dashboard
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
echo "NVIDIA_API_KEY=nvapi-..." >> ../.env      # for the AI root-cause panel
streamlit run app.py
```

Demo mode is fully self-contained: the predictor uses a heuristic and the
blizzard switch drives a simulated drift summary. Only the AI root-cause call
hits the network (build.nvidia.com).

## Live mode

Pick **Live** in the sidebar and point *Serving URL* at a running `serving`
container (`docker compose up`). The predictor then calls the real FastAPI
`/predict` route; the blizzard switch fires skewed traffic so the actual
Evidently monitor trips.

## NVIDIA NIM

The AI root-cause analysis is produced by `nim.py`, an OpenAI-compatible client
for build.nvidia.com. Configure via `.env`:

```
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
```
