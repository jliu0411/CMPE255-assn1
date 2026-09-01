from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score
from .features import FEATURES, make_features, validate_rows

ROOT = Path(__file__).resolve().parents[1]
def metrics(y, pred):
    pred=np.clip(pred,1,None)
    return {"rmsle":round(float(np.sqrt(mean_squared_log_error(y,pred))),4),"mae_seconds":round(float(mean_absolute_error(y,pred)),1),"r2":round(float(r2_score(y,pred)),4)}

def train(data_path: Path, max_rows: int|None=None):
    df=pd.read_csv(data_path,nrows=max_rows,parse_dates=["pickup_datetime"])
    required={"pickup_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count","trip_duration"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    raw_rows=len(df); df=df.loc[validate_rows(df)].sort_values("pickup_datetime").reset_index(drop=True)
    if len(df)<100: raise ValueError("At least 100 valid rows are required.")
    split=int(len(df)*.8); train_df,val_df=df.iloc[:split],df.iloc[split:]
    X_train,X_val=make_features(train_df),make_features(val_df); y_train,y_val=train_df.trip_duration,val_df.trip_duration
    baseline=DummyRegressor(strategy="median").fit(X_train,np.log1p(y_train))
    model=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.07,max_iter=260,max_leaf_nodes=31,l2_regularization=1.0,random_state=42).fit(X_train,np.log1p(y_train))
    baseline_pred=np.expm1(baseline.predict(X_val)); pred=np.expm1(model.predict(X_val))
    report={"trained_at":datetime.now(timezone.utc).isoformat(),"source":data_path.name,"raw_rows":raw_rows,"training_rows":len(train_df),"validation_rows":len(val_df),"split":"chronological 80/20","features":FEATURES,"baseline":metrics(y_val,baseline_pred),"model":metrics(y_val,pred),"duration_median_seconds":round(float(df.trip_duration.median()),1),"distance_median_km":round(float(make_features(df).haversine_km.median()),2)}
    artifact={"model":model,"metadata":report,"version":"1.0.0"}
    (ROOT/"artifacts").mkdir(exist_ok=True); joblib.dump(artifact,ROOT/"artifacts"/"taxi_duration_model.joblib")
    (ROOT/"artifacts"/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); return report

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--data",type=Path,default=ROOT/"data/raw/sample_train.csv");parser.add_argument("--max-rows",type=int);args=parser.parse_args()
    try: train(args.data,args.max_rows)
    except Exception as exc: print(f"Training failed: {exc}",file=sys.stderr);raise
