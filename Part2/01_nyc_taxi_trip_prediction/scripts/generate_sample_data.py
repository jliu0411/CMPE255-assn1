from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; rng=np.random.default_rng(42); n=12000
pickup_lat=rng.normal(40.758,0.035,n).clip(40.60,40.90);pickup_lon=rng.normal(-73.982,0.045,n).clip(-74.15,-73.75)
angle=rng.uniform(0,2*np.pi,n);distance=rng.gamma(2.1,1.45,n).clip(.15,25);drop_lat=(pickup_lat+np.sin(angle)*distance/111).clip(40.50,40.95);drop_lon=(pickup_lon+np.cos(angle)*distance/(111*np.cos(np.radians(pickup_lat)))).clip(-74.2,-73.7)
start=pd.Timestamp("2016-01-01"); pickup=start+pd.to_timedelta(rng.integers(0,181*86400,n),unit="s");rush=pd.Series(pickup).dt.hour.isin([7,8,9,16,17,18,19]).to_numpy();weekend=pd.Series(pickup).dt.weekday.ge(5).to_numpy()
speed=np.where(rush,15,22)*np.where(weekend,1.12,1);duration=(distance/speed*3600+120+rng.lognormal(4.4,.48,n)).clip(35,21600).astype(int)
df=pd.DataFrame({"id":[f"sample{i:07d}" for i in range(n)],"vendor_id":rng.integers(1,3,n),"pickup_datetime":pickup,"dropoff_datetime":pickup+pd.to_timedelta(duration,unit="s"),"passenger_count":rng.choice([1,2,3,4,5,6],n,p=[.7,.14,.05,.04,.04,.03]),"pickup_longitude":pickup_lon,"pickup_latitude":pickup_lat,"dropoff_longitude":drop_lon,"dropoff_latitude":drop_lat,"store_and_fwd_flag":rng.choice(["N","Y"],n,p=[.995,.005]),"trip_duration":duration})
out=ROOT/"data/raw/sample_train.csv";out.parent.mkdir(parents=True,exist_ok=True);df.to_csv(out,index=False);print(f"Wrote {len(df):,} reproducible sample trips to {out}")
