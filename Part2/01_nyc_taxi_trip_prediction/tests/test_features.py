import pandas as pd
from src.features import haversine_km,make_features
def test_zero_distance(): assert haversine_km(40.7,-74,40.7,-74)==0
def test_features_are_finite():
    df=pd.DataFrame([{"pickup_datetime":"2016-01-01 08:00","pickup_latitude":40.75,"pickup_longitude":-73.99,"dropoff_latitude":40.76,"dropoff_longitude":-73.97,"passenger_count":2,"vendor_id":1,"store_and_fwd_flag":"N"}]);x=make_features(df);assert x.shape==(1,16);assert x.notna().all().all();assert x.iloc[0].is_rush_hour==1
