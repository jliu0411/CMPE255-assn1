import pandas as pd
from src.features import prepare,feature_frame
def test_engineering():
    row={'ID':1,'Year_Birth':1980,'Income':60000,'Kidhome':1,'Teenhome':0,'Recency':20,'Dt_Customer':'01-01-2013','Marital_Status':'Married','NumDealsPurchases':2,'NumWebVisitsMonth':5,'Complain':0,'Response':0}
    for c in ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']:row[c]=100
    for c in ['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']:row[c]=3
    for c in ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5']:row[c]=0
    x=prepare(pd.DataFrame([row]));assert x.iloc[0].TotalSpend==600;assert feature_frame(x).shape==(1,15)
