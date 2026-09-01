"""Reproducible, scriptable EDA. Writes charts to reports/figures."""
from pathlib import Path
import matplotlib.pyplot as plt, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; df=pd.read_csv(ROOT/"data/raw/sample_train.csv",parse_dates=["pickup_datetime"]); out=ROOT/"reports/figures";out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(1,2,figsize=(12,4));df.trip_duration.clip(upper=df.trip_duration.quantile(.99)).div(60).hist(bins=50,ax=ax[0],color="#1c665f");ax[0].set(title="Trip duration (p99 clipped)",xlabel="Minutes")
df.assign(hour=df.pickup_datetime.dt.hour).groupby("hour").trip_duration.median().div(60).plot(ax=ax[1],color="#ff765f",marker="o");ax[1].set(title="Median duration by pickup hour",ylabel="Minutes");fig.tight_layout();fig.savefig(out/"duration_overview.png",dpi=160);print(df.describe(include="all").to_string())
