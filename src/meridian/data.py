"""Dataset loading and synthetic generation.

The real target is the NYC Taxi Trip Duration dataset. Downloading it is multi-GB
and needs network, which makes demos and CI fragile, so this module generates data
with the *same schema and plausible dynamics* by default. Point ``load_real_csv`` at
a Kaggle/TLC export to swap in the real thing — the rest of the pipeline is identical.

Two generators exist on purpose:

* ``generate_baseline`` — the world the model was trained on.
* ``generate_skewed``   — weather-impacted / geographically shifted trips that the
  drift monitor is supposed to catch. This is what we feed as "live" traffic to make
  the drift alarm fire on demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Rough Manhattan bounding box — enough to look like NYC pickups/dropoffs.
_LAT = (40.70, 40.82)
_LON = (-74.02, -73.93)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def _synthesize(n: int, rng: np.random.Generator, *, skew: bool) -> pd.DataFrame:
    """Build trips and a physically-motivated trip duration.

    skew=True shifts the input distribution (rain, longer outer-borough trips,
    rush-hour concentration) so duration relationships change — i.e. real drift,
    not just noise.
    """
    if skew:
        # Heavy rain, longer suburban trips, jammed into evening rush hour.
        pickup_hour = rng.choice(24, size=n, p=_rush_weighted())
        passenger_count = rng.integers(1, 6, size=n)
        precip_mm = rng.gamma(shape=4.0, scale=3.0, size=n)          # wet
        temp_c = rng.normal(2.0, 4.0, size=n)                        # cold
        lat_span, lon_span = 0.20, 0.18                              # wider geo
    else:
        pickup_hour = rng.integers(0, 24, size=n)
        passenger_count = rng.integers(1, 5, size=n)
        precip_mm = rng.gamma(shape=1.2, scale=0.6, size=n)          # mostly dry
        temp_c = rng.normal(15.0, 7.0, size=n)
        lat_span, lon_span = 0.12, 0.09

    pickup_dayofweek = rng.integers(0, 7, size=n)
    pickup_lat = rng.uniform(_LAT[0], _LAT[0] + lat_span, size=n)
    pickup_lon = rng.uniform(_LON[0], _LON[0] + lon_span, size=n)
    dropoff_lat = pickup_lat + rng.normal(0, 0.02, size=n)
    dropoff_lon = pickup_lon + rng.normal(0, 0.02, size=n)

    dist = _haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    dist = np.clip(dist, 0.2, None)

    # Duration model the network must learn: base speed degraded by rush hour,
    # rain, and cold. ~ minutes.
    rush = ((pickup_hour >= 7) & (pickup_hour <= 9)) | ((pickup_hour >= 16) & (pickup_hour <= 19))
    speed_kmh = 22.0 - 6.0 * rush - 0.25 * precip_mm - 0.03 * np.maximum(0, 10 - temp_c)
    speed_kmh = np.clip(speed_kmh, 5.0, 30.0)
    duration = dist / speed_kmh * 60.0
    duration += 1.5 + rng.normal(0, 1.2, size=n)            # fixed overhead + noise
    duration = np.clip(duration, 1.0, 180.0)

    return pd.DataFrame(
        {
            "passenger_count": passenger_count,
            "trip_distance_km": dist,
            "pickup_hour": pickup_hour,
            "pickup_dayofweek": pickup_dayofweek,
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "dropoff_lat": dropoff_lat,
            "dropoff_lon": dropoff_lon,
            "temp_c": temp_c,
            "precip_mm": precip_mm,
            "trip_duration_min": duration,
        }
    )


def _rush_weighted() -> np.ndarray:
    w = np.ones(24)
    w[16:20] = 5.0  # evening rush concentration for skewed traffic
    return w / w.sum()


def generate_baseline(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    return _synthesize(n, np.random.default_rng(seed), skew=False)


def generate_skewed(n: int = 50_000, seed: int = 7) -> pd.DataFrame:
    return _synthesize(n, np.random.default_rng(seed), skew=True)


def load_real_csv(path: str) -> pd.DataFrame:
    """Adapt a real NYC taxi export to Meridian's schema.

    Expected raw columns: pickup_datetime, dropoff_datetime, passenger_count,
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude.
    Weather columns (temp_c, precip_mm) are optional and default to mild/dry.
    """
    raw = pd.read_csv(path, parse_dates=["pickup_datetime", "dropoff_datetime"])
    out = pd.DataFrame()
    out["passenger_count"] = raw["passenger_count"]
    out["pickup_hour"] = raw["pickup_datetime"].dt.hour
    out["pickup_dayofweek"] = raw["pickup_datetime"].dt.dayofweek
    out["pickup_lat"] = raw["pickup_latitude"]
    out["pickup_lon"] = raw["pickup_longitude"]
    out["dropoff_lat"] = raw["dropoff_latitude"]
    out["dropoff_lon"] = raw["dropoff_longitude"]
    out["trip_distance_km"] = _haversine_km(
        out["pickup_lat"], out["pickup_lon"], out["dropoff_lat"], out["dropoff_lon"]
    ).clip(0.2)
    out["temp_c"] = raw.get("temp_c", 15.0)
    out["precip_mm"] = raw.get("precip_mm", 0.0)
    dur = (raw["dropoff_datetime"] - raw["pickup_datetime"]).dt.total_seconds() / 60.0
    out["trip_duration_min"] = dur.clip(1.0, 180.0)
    return out.dropna()
