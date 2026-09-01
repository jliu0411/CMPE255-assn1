from __future__ import annotations
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
FEATURES = ["vendor_id","passenger_count","store_flag","pickup_hour","pickup_weekday","pickup_month","is_weekend","is_rush_hour","haversine_km","manhattan_km","bearing_sin","bearing_cos","pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*EARTH_RADIUS_KM*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    lat1, lon1 = df["pickup_latitude"].astype(float), df["pickup_longitude"].astype(float)
    lat2, lon2 = df["dropoff_latitude"].astype(float), df["dropoff_longitude"].astype(float)
    distance = haversine_km(lat1, lon1, lat2, lon2)
    ns = haversine_km(lat1, lon1, lat2, lon1); ew = haversine_km(lat1, lon1, lat1, lon2)
    y = np.sin(np.radians(lon2-lon1))*np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1))*np.sin(np.radians(lat2))-np.sin(np.radians(lat1))*np.cos(np.radians(lat2))*np.cos(np.radians(lon2-lon1))
    bearing = np.arctan2(y, x)
    hour = dt.dt.hour.fillna(12).astype(int)
    out = pd.DataFrame({
        "vendor_id": pd.to_numeric(df.get("vendor_id", 1), errors="coerce").fillna(1),
        "passenger_count": pd.to_numeric(df.get("passenger_count", 1), errors="coerce").fillna(1).clip(1, 8),
        "store_flag": df.get("store_and_fwd_flag", pd.Series("N", index=df.index)).astype(str).str.upper().eq("Y").astype(int),
        "pickup_hour": hour, "pickup_weekday": dt.dt.weekday.fillna(0), "pickup_month": dt.dt.month.fillna(1),
        "is_weekend": dt.dt.weekday.fillna(0).ge(5).astype(int),
        "is_rush_hour": hour.isin([7,8,9,16,17,18,19]).astype(int),
        "haversine_km": distance, "manhattan_km": ns+ew,
        "bearing_sin": np.sin(bearing), "bearing_cos": np.cos(bearing),
        "pickup_latitude": lat1, "pickup_longitude": lon1, "dropoff_latitude": lat2, "dropoff_longitude": lon2,
    })
    return out[FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0)

def validate_rows(df: pd.DataFrame) -> pd.Series:
    return (df.trip_duration.between(30, 21600) & df.pickup_latitude.between(40.45, 41.0) & df.dropoff_latitude.between(40.45, 41.0) & df.pickup_longitude.between(-74.3, -73.6) & df.dropoff_longitude.between(-74.3, -73.6) & df.passenger_count.between(1, 8))
