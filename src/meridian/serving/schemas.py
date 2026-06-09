from __future__ import annotations

from pydantic import BaseModel, Field


class TripFeatures(BaseModel):
    passenger_count: int = Field(ge=0, le=8, examples=[2])
    trip_distance_km: float = Field(ge=0, examples=[3.4])
    pickup_hour: int = Field(ge=0, le=23, examples=[18])
    pickup_dayofweek: int = Field(ge=0, le=6, examples=[2])
    pickup_lat: float = Field(examples=[40.75])
    pickup_lon: float = Field(examples=[-73.98])
    dropoff_lat: float = Field(examples=[40.77])
    dropoff_lon: float = Field(examples=[-73.96])
    temp_c: float = Field(examples=[14.0])
    precip_mm: float = Field(ge=0, examples=[0.0])


class PredictionResponse(BaseModel):
    trip_duration_min: float
    model_version: str
