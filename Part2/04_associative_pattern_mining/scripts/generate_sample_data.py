from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];rng=np.random.default_rng(42);products={'TEA':['VINTAGE TEA SET','CERAMIC TEA CUP','TEA TIME NAPKINS','GLASS SUGAR BOWL'],'BAKE':['REGENCY CAKESTAND 3 TIER','CAKE TINS PANTRY DESIGN','BAKING SET SPACEBOY','PAPER BUNTING'],'GIFT':['JUMBO BAG RED RETROSPOT','RED RETROSPOT GIFT WRAP','PINK GIFT BAG','RIBBON REEL HEARTS'],'HOME':['WHITE HANGING HEART T-LIGHT HOLDER','SCENTED CANDLE SET','WOODEN PICTURE FRAME','LANTERN CREAM'],'PARTY':['PARTY BUNTING','PAPER CUPS STARS','PAPER PLATES STARS','BALLOONS PASTEL'],'LUNCH':['LUNCH BAG RED RETROSPOT','LUNCH BOX SPACEBOY','SET OF 3 SNACK BOXES','WATER BOTTLE RETROSPOT']};all_items=sum(products.values(),[]);prices={p:round(float(rng.uniform(1.2,18)),2) for p in all_items};rows=[];start=pd.Timestamp('2010-12-01')
for i in range(5000):
    themes=rng.choice(list(products),size=1 if rng.random()<.72 else 2,replace=False);basket=set();
    for t in themes:
        picks=rng.choice(products[t],size=rng.integers(2,5),replace=False);basket.update(picks)
    if rng.random()<.2:basket.add(rng.choice(all_items))
    invoice=str(536000+i);date=start+pd.to_timedelta(i*95+rng.integers(3600),unit='m');country=rng.choice(['United Kingdom','Germany','France','Netherlands','Spain'],p=[.82,.06,.05,.04,.03]);customer=rng.integers(12000,18500)
    for item in basket:rows.append({'InvoiceNo':invoice,'StockCode':f'S{all_items.index(item):04d}','Description':item,'Quantity':int(rng.integers(1,13)),'InvoiceDate':date,'UnitPrice':prices[item],'CustomerID':customer,'Country':country})
df=pd.DataFrame(rows);bad=df.sample(50,random_state=42).copy();bad['InvoiceNo']='C'+bad.InvoiceNo;bad['Quantity']=-bad.Quantity;df=pd.concat([df,bad],ignore_index=True);out=ROOT/'data/raw/sample_online_retail.csv';out.parent.mkdir(parents=True,exist_ok=True);df.to_csv(out,index=False);print(f'Wrote {len(df):,} lines across 5,000 synthetic invoices to {out}')
