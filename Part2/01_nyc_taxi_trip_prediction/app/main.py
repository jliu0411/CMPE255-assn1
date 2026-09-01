from __future__ import annotations
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from src.features import make_features

ROOT=Path(__file__).resolve().parents[1]; MODEL_PATH=ROOT/"artifacts/taxi_duration_model.joblib"; STATIC=ROOT/"app/static"
app=FastAPI(title="RideCast NYC API",version="1.0.0",description="NYC taxi trip-duration inference service")
app.mount("/static",StaticFiles(directory=STATIC),name="static")

class Trip(BaseModel):
    pickup_latitude: float=Field(ge=40.45,le=41.0);pickup_longitude: float=Field(ge=-74.3,le=-73.6)
    dropoff_latitude: float=Field(ge=40.45,le=41.0);dropoff_longitude: float=Field(ge=-74.3,le=-73.6)
    pickup_datetime: datetime;passenger_count:int=Field(default=1,ge=1,le=8);vendor_id:int=Field(default=1,ge=1,le=2);store_and_fwd_flag:str="N"
    @field_validator("store_and_fwd_flag")
    @classmethod
    def flag(cls,v):
        if v.upper() not in {"Y","N"}: raise ValueError("must be Y or N")
        return v.upper()

def load_artifact():
    if not MODEL_PATH.exists(): raise HTTPException(503,"Model is not trained. Run `python scripts/generate_sample_data.py` and `python -m src.train`.")
    return joblib.load(MODEL_PATH)

@app.get("/",include_in_schema=False)
def home(): return FileResponse(STATIC/"index.html")
@app.get("/api/health")
def health(): return {"status":"ok","model_ready":MODEL_PATH.exists()}
@app.get("/api/model")
def model_info():
    artifact=load_artifact();return {"version":artifact["version"],**artifact["metadata"]}
@app.post("/api/predict")
def predict(trip:Trip):
    artifact=load_artifact(); frame=pd.DataFrame([trip.model_dump()]); features=make_features(frame)
    seconds=float(np.clip(np.expm1(artifact["model"].predict(features)[0]),30,21600)); distance=float(features.haversine_km.iloc[0])
    return {"duration_seconds":round(seconds),"duration_minutes":round(seconds/60,1),"distance_km":round(distance,2),"distance_miles":round(distance*.621371,2),"range_minutes":[round(seconds*.82/60,1),round(seconds*1.22/60,1)],"model_version":artifact["version"]}
