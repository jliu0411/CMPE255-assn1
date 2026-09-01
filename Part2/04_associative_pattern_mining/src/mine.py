from __future__ import annotations
import argparse,json,re
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from .mining import association_rules,frequent_itemsets,rule_confidence
ROOT=Path(__file__).resolve().parents[1]
def read(path):return pd.read_excel(path) if path.suffix.lower() in {'.xlsx','.xls'} else pd.read_csv(path,encoding_errors='replace')
def clean(df):
    required={'InvoiceNo','Description','Quantity','UnitPrice','InvoiceDate'};missing=required-set(df.columns)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    x=df.copy();x['InvoiceNo']=x.InvoiceNo.astype(str);x['Description']=x.Description.astype('string').str.upper().str.replace(r'\s+',' ',regex=True).str.strip();x['Quantity']=pd.to_numeric(x.Quantity,errors='coerce');x['UnitPrice']=pd.to_numeric(x.UnitPrice,errors='coerce');x['InvoiceDate']=pd.to_datetime(x.InvoiceDate,errors='coerce')
    valid=~x.InvoiceNo.str.upper().str.startswith('C')&x.Quantity.gt(0)&x.UnitPrice.gt(0)&x.Description.notna()&x.InvoiceDate.notna();x=x.loc[valid].drop_duplicates(['InvoiceNo','Description']);x=x[~x.Description.str.contains(r'POSTAGE|CARRIAGE|MANUAL|DOTCOM|AMAZON FEE|BANK CHARGES',regex=True,na=False)];x['Revenue']=x.Quantity*x.UnitPrice;return x
def main(path,min_support=.02,min_confidence=.3,min_lift=1.15):
    raw=read(path);df=clean(raw);invoice_dates=df.groupby('InvoiceNo').InvoiceDate.min().sort_values();split=max(1,int(len(invoice_dates)*.8));train_ids=set(invoice_dates.index[:split]);test_ids=set(invoice_dates.index[split:]);to_baskets=lambda ids:[set(x) for x in df[df.InvoiceNo.isin(ids)].groupby('InvoiceNo').Description.apply(list) if len(set(x))>=2];train,test=to_baskets(train_ids),to_baskets(test_ids)
    itemsets,n=frequent_itemsets(train,min_support);rules=association_rules(itemsets,n,min_confidence,min_lift)[:500]
    for r in rules:
        hold,eligible=rule_confidence(r,test);r['holdout_confidence']=round(hold,4);r['holdout_eligible']=eligible;r['confidence_drift']=round(hold-r['confidence'],4);r['stable']=eligible>=10 and abs(hold-r['confidence'])<=.2
    products=df.groupby('Description').agg(quantity=('Quantity','sum'),revenue=('Revenue','sum'),invoices=('InvoiceNo','nunique')).sort_values('invoices',ascending=False).head(50).reset_index();top_items=[{'name':r.Description,'invoices':int(r.invoices),'quantity':int(r.quantity),'revenue':round(float(r.revenue),2)} for r in products.itertuples()]
    countries=df.groupby('Country' if 'Country' in df else df.assign(Country='Unknown').Country).agg(invoices=('InvoiceNo','nunique'),revenue=('Revenue','sum')).sort_values('revenue',ascending=False).head(10).reset_index().to_dict('records')
    meta={'trained_at':datetime.now(timezone.utc).isoformat(),'source':path.name,'raw_lines':len(raw),'clean_lines':len(df),'removed_lines':len(raw)-len(df),'transactions':int(df.InvoiceNo.nunique()),'products':int(df.Description.nunique()),'train_baskets':len(train),'holdout_baskets':len(test),'avg_basket_size':round(sum(map(len,train))/len(train),2),'min_support':min_support,'min_confidence':min_confidence,'min_lift':min_lift,'frequent_itemsets':len(itemsets),'rules_count':len(rules),'stable_rules':sum(r['stable'] for r in rules),'rules':rules,'top_items':top_items,'countries':countries}
    (ROOT/'artifacts').mkdir(exist_ok=True);(ROOT/'artifacts/rules.json').write_text(json.dumps(meta,indent=2,default=str));print(json.dumps({k:v for k,v in meta.items() if k not in ['rules','top_items','countries']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/raw/sample_online_retail.csv');p.add_argument('--min-support',type=float,default=.02);p.add_argument('--min-confidence',type=float,default=.3);p.add_argument('--min-lift',type=float,default=1.15);a=p.parse_args();main(a.data,a.min_support,a.min_confidence,a.min_lift)
