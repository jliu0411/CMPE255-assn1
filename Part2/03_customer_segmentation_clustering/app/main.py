from pathlib import Path
import joblib,pandas as pd
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from src.features import feature_frame,prepare
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/segments.joblib';STATIC=ROOT/'app/static';app=FastAPI(title='Mosaic Segmentation API',version='1.0.0');app.mount('/static',StaticFiles(directory=STATIC),name='static')
def artifact():
    if not ART.exists():raise HTTPException(503,'Model not trained. Run the sample-data and training commands.')
    return joblib.load(ART)
class Customer(BaseModel):
    Year_Birth:int=Field(ge=1915,le=1997);Income:float=Field(gt=0,le=250000);Kidhome:int=Field(ge=0,le=5);Teenhome:int=Field(ge=0,le=5);Recency:int=Field(ge=0,le=365);MntWines:float=Field(ge=0);MntFruits:float=Field(ge=0);MntMeatProducts:float=Field(ge=0);MntFishProducts:float=Field(ge=0);MntSweetProducts:float=Field(ge=0);MntGoldProds:float=Field(ge=0);NumDealsPurchases:int=Field(ge=0);NumWebPurchases:int=Field(ge=0);NumCatalogPurchases:int=Field(ge=0);NumStorePurchases:int=Field(ge=0);NumWebVisitsMonth:int=Field(ge=0);AcceptedCmp1:int=Field(default=0,ge=0,le=1);AcceptedCmp2:int=Field(default=0,ge=0,le=1);AcceptedCmp3:int=Field(default=0,ge=0,le=1);AcceptedCmp4:int=Field(default=0,ge=0,le=1);AcceptedCmp5:int=Field(default=0,ge=0,le=1);Response:int=Field(default=0,ge=0,le=1);Complain:int=Field(default=0,ge=0,le=1);Dt_Customer:str='01-01-2013';Marital_Status:str='Single';ID:int=999999
@app.get('/',include_in_schema=False)
def home():return FileResponse(STATIC/'index.html')
@app.get('/api/health')
def health():return {'status':'ok','model_ready':ART.exists()}
@app.get('/api/dashboard')
def dashboard():return artifact()['metadata']
@app.post('/api/segment')
def segment(customer:Customer):
    a=artifact();df=pd.DataFrame([customer.model_dump()]);x=feature_frame(prepare(df));z=a['scaler'].transform(x);cluster=int(a['model'].predict(z)[0]);distance=float(min(a['model'].transform(z)[0]));persona=next(p for p in a['metadata']['personas'] if p['cluster']==cluster);return {'cluster':cluster,'segment':persona['name'],'description':persona['description'],'recommended_action':persona['action'],'distance_to_centroid':round(distance,3),'model_version':a['version']}
