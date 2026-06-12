"""Meridian — God-Mode drift dashboard.

Left:  ETA predictor (hits the serving /predict route, or a heuristic in demo).
Right: Drift Control Panel with a "Simulate Blizzard" switch that pushes the live
       distribution off the training distribution until the drift share crosses
       0.5 — at which point an NVIDIA NIM model writes the root-cause analysis and
       a retrain "heals" the model.

Runs standalone (demo mode) with only: streamlit, httpx, pandas.
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import streamlit as st


def _load_dotenv() -> None:
    """Load NVIDIA_* / MERIDIAN_* from a local .env without adding a dependency.

    Checks dashboard/.env then the repo-root .env; existing env vars win.
    """
    for candidate in (Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

import nim  # noqa: E402  (imported after .env load so module-level config picks it up)

st.set_page_config(page_title="Meridian — Drift Control", layout="wide", page_icon="🛰️")

THRESHOLD = 0.5

# Reference (training) vs blizzard (live) feature snapshots — drives the demo
# narrative and the context handed to the LLM.
# Six of ten features shift in a blizzard (weather + behaviour + geo) so the
# drifted share is 0.6 — comfortably over the 0.5 dataset-drift threshold.
BLIZZARD_STATS = {
    "temp_c": {"reference_mean": 14.2, "current_mean": -6.8},
    "precip_mm": {"reference_mean": 0.4, "current_mean": 19.1},
    "trip_distance_km": {"reference_mean": 3.4, "current_mean": 5.9},
    "pickup_hour": {"reference_mean": 13.6, "current_mean": 18.7},
    "pickup_lat": {"reference_mean": 40.75, "current_mean": 40.69},
    "dropoff_lat": {"reference_mean": 40.77, "current_mean": 40.83},
}
BLIZZARD_DRIFTED = set(BLIZZARD_STATS.keys())
ALL_FEATURES = [
    "passenger_count", "trip_distance_km", "pickup_hour", "pickup_dayofweek",
    "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon", "temp_c", "precip_mm",
]

# --- session state ---
ss = st.session_state
ss.setdefault("blizzard", False)
ss.setdefault("retrained", False)
ss.setdefault("rca", "")
ss.setdefault("model_version", "7")
ss.setdefault("retrain_log", [])


def drift_summary() -> dict:
    """Compute the current drift summary for demo mode."""
    if ss.blizzard and not ss.retrained:
        per = {f: (f in BLIZZARD_DRIFTED) for f in ALL_FEATURES}
        n = len(BLIZZARD_DRIFTED)
    else:
        per = {f: False for f in ALL_FEATURES}
        n = 0
    share = n / len(ALL_FEATURES)
    return {
        "share": share,
        "n_drifted": n,
        "dataset_drift": share >= THRESHOLD,
        "per_feature": per,
    }


def predict_eta(payload: dict, base_url: str, mode: str) -> tuple[float, str]:
    if mode == "Live":
        r = httpx.post(f"{base_url.rstrip('/')}/predict", json=payload, timeout=10)
        r.raise_for_status()
        d = r.json()
        return float(d["trip_duration_min"]), str(d.get("model_version", "?"))
    # Demo heuristic
    dist = payload["trip_distance_km"]
    eta = 4.0 + dist * 2.4
    if payload["pickup_hour"] in (7, 8, 9, 16, 17, 18, 19):
        eta *= 1.3
    eta += dist * payload["precip_mm"] * 0.18
    if payload["temp_c"] < 0:
        eta *= 1.15
    # A stale (pre-blizzard) model ignores the snow signal and under-predicts.
    if ss.blizzard and not ss.retrained:
        eta *= 0.62
    return round(eta, 1), ss.model_version


# ============================ Sidebar ============================
with st.sidebar:
    st.markdown("### ⚙️ Control")
    mode = st.radio("Mode", ["Demo", "Live"], help="Live mode calls the serving API + drift monitor.")
    serving_url = st.text_input("Serving URL", os.getenv("MERIDIAN_SERVING_URL", "http://localhost:8000"))
    st.divider()
    if nim.has_key():
        st.success(f"NVIDIA NIM ready\n\n`{nim.MODEL}`")
    else:
        st.warning("Set NVIDIA_API_KEY to enable\nAI root-cause analysis.")

# ============================ Header ============================
st.markdown("## 🛰️ Meridian — Autonomous Drift Control")
st.caption(
    "A self-healing ETA model. When the live world stops looking like the training world, "
    "the drift monitor trips, an NVIDIA NIM model explains **why**, and the pipeline retrains itself."
)

left, right = st.columns(2, gap="large")

# ---------------------------- ETA Predictor ----------------------------
with left:
    st.markdown("### 🚕 ETA Predictor")
    c1, c2 = st.columns(2)
    distance = c1.slider("Trip distance (km)", 0.5, 30.0, 4.2, 0.1)
    hour = c2.slider("Pickup hour", 0, 23, 18)
    passengers = c1.slider("Passengers", 1, 6, 2)
    temp = c2.slider("Temp (°C)", -15.0, 38.0, 14.0, 0.5)
    precip = c1.slider("Precip (mm)", 0.0, 40.0, 0.0, 0.5)
    dow = c2.slider("Day of week", 0, 6, 2)

    if st.button("Predict ETA", type="primary", use_container_width=True):
        payload = {
            "passenger_count": passengers, "trip_distance_km": distance,
            "pickup_hour": hour, "pickup_dayofweek": dow,
            "pickup_lat": 40.75, "pickup_lon": -73.98,
            "dropoff_lat": 40.77, "dropoff_lon": -73.96,
            "temp_c": temp, "precip_mm": precip,
        }
        try:
            eta, ver = predict_eta(payload, serving_url, mode)
            ss["last_eta"] = (eta, ver)
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")

    if ss.get("last_eta"):
        eta, ver = ss["last_eta"]
        st.metric("Predicted trip duration", f"{eta} min", help=f"model v{ver}")
        if ss.blizzard and not ss.retrained:
            st.error(
                "⚠️ This prediction came from a model that has **never seen a blizzard**. "
                "It is almost certainly too optimistic."
            )

# ---------------------------- Drift Control Panel ----------------------------
with right:
    st.markdown("### 🌪️ Drift Control Panel")
    ss.blizzard = st.toggle(
        "❄️ Simulate Blizzard",
        value=ss.blizzard,
        help="Push severe-snow / geo-shifted traffic at the model.",
    )
    if ss.blizzard:
        ss.retrained = ss.retrained  # keep
    else:
        ss.retrained = False
        ss.rca = ""

    summary = drift_summary()
    share = summary["share"]
    breached = summary["dataset_drift"]

    pct = int(share * 100)
    bar_color = "#ef4444" if breached else "#22c55e"
    st.markdown(
        f"""
        <div style="margin:0.2rem 0 0.4rem;">
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#94a3b8;">
            <span>Share of drifted features</span>
            <span>threshold {int(THRESHOLD*100)}%</span>
          </div>
          <div style="background:#1f2937;border-radius:6px;height:16px;position:relative;overflow:hidden;">
            <div style="background:{bar_color};width:{pct}%;height:100%;transition:width .5s;"></div>
            <div style="position:absolute;left:{int(THRESHOLD*100)}%;top:0;bottom:0;border-left:2px dashed #cbd5e1;"></div>
          </div>
          <div style="text-align:right;font-size:1.1rem;font-weight:700;color:{bar_color};margin-top:2px;">
            {pct}% drifted ({summary['n_drifted']}/{len(ALL_FEATURES)})
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if breached:
        st.error(f"🚨 DATASET DRIFT — share {share:.0%} ≥ {THRESHOLD:.0%}. Model is flying blind.")
        st.markdown("**Drifted features:** " + ", ".join(f"`{f}`" for f in BLIZZARD_DRIFTED))
    elif ss.blizzard and ss.retrained:
        st.success("✅ Drift back under threshold — retrained model matches the new world.")
    else:
        st.info("Distribution nominal. Live traffic matches the training reference.")

# ============================ AI Root-Cause + Retrain ============================
st.divider()
if summary["dataset_drift"]:
    rc1, rc2 = st.columns([3, 1], gap="large")
    with rc1:
        st.markdown("### 🧠 AI Root-Cause Analysis · NVIDIA NIM")
        if st.button("Generate root-cause analysis", type="primary"):
            if not nim.has_key():
                st.warning("NVIDIA_API_KEY not set — cannot call the model.")
            else:
                with st.spinner("Asking NVIDIA NIM to diagnose the drift…"):
                    try:
                        ss.rca = nim.generate_rca(summary, BLIZZARD_STATS)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"NIM call failed: {e}")
        if ss.rca:
            st.markdown(ss.rca)
    with rc2:
        st.markdown("### 🔧 Heal")
        st.caption("Retrain on the recent snowy window and hot-reload.")
        if st.button("Trigger retrain", use_container_width=True):
            with st.spinner("Retraining on drifted window…"):
                time.sleep(1.2)
            ss.model_version = str(int(ss.model_version) + 1)
            ss.retrained = True
            ss.retrain_log.insert(0, f"Promoted model v{ss.model_version} (reason: drift-retrain)")
            st.rerun()

if ss.retrain_log:
    with st.expander("Retrain history", expanded=False):
        for line in ss.retrain_log:
            st.markdown(f"- {line}")

st.caption(
    "Demo mode is self-contained. In Live mode the predictor calls the FastAPI serving route and "
    "the blizzard switch fires skewed traffic so the real Evidently monitor trips."
)
