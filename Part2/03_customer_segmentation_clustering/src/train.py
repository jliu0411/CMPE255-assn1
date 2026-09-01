from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score,adjusted_rand_score
from sklearn.preprocessing import RobustScaler
from .features import FEATURES,feature_frame,prepare,read_data
ROOT=Path(__file__).resolve().parents[1]
NAMES=['High-Value Loyalists','Digital Enthusiasts','Family-Focused Savers','Occasional Traditionalists','Campaign-Ready Explorers','At-Risk Customers','Steady Omnichannel','Emerging Customers']
ACTIONS=['Reward loyalty with early access and premium bundles.','Use personalized digital launches and replenishment journeys.','Lead with value packs, family relevance, and deal clarity.','Use low-frequency reminders and simple store-led offers.','Test cross-category campaigns and measure incremental lift.','Prioritize win-back journeys and service recovery.','Coordinate consistent offers across web, catalog, and store.','Build trust with onboarding and second-purchase incentives.']
def main(path):
    raw=read_data(path);df=prepare(raw);X=feature_frame(df);scaler=RobustScaler().fit(X);Z=scaler.transform(X);runs=[]
    for k in range(2,9):
        labels=[];inertias=[]
        for seed in [11,29,47]:
            m=KMeans(k,n_init=20,random_state=seed).fit(Z);labels.append(m.labels_);inertias.append(m.inertia_)
        sil=silhouette_score(Z,labels[0]);stability=np.mean([adjusted_rand_score(labels[0],q) for q in labels[1:]])
        runs.append({'k':k,'silhouette':round(float(sil),4),'calinski_harabasz':round(float(calinski_harabasz_score(Z,labels[0])),1),'davies_bouldin':round(float(davies_bouldin_score(Z,labels[0])),4),'inertia':round(float(inertias[0]),1),'stability':round(float(stability),4)})
    sil=np.array([r['silhouette'] for r in runs]);db=np.array([r['davies_bouldin'] for r in runs]);scores=(sil-sil.min())/(np.ptp(sil)+1e-9)+(db.max()-db)/(np.ptp(db)+1e-9)+np.array([r['stability'] for r in runs])*.35
    best=runs[int(scores.argmax())]['k'];model=KMeans(best,n_init=30,random_state=42).fit(Z);df['cluster']=model.labels_;pca=PCA(2,random_state=42).fit(Z);xy=pca.transform(Z);df['pc1'],df['pc2']=xy[:,0],xy[:,1]
    means=df.groupby('cluster')[FEATURES].mean();overall=df[FEATURES].mean();persona=[]
    order=means.assign(value=means.TotalSpend.rank()+means.Income.rank()+means.TotalPurchases.rank()).sort_values('value',ascending=False).index
    for rank,c in enumerate(order):
        row=means.loc[c];name=NAMES[rank];persona.append({'cluster':int(c),'name':name,'action':ACTIONS[rank],'size':int((df.cluster==c).sum()),'share':round(float((df.cluster==c).mean()*100),1),'income':round(float(row.Income)),'spend':round(float(row.TotalSpend)),'purchases':round(float(row.TotalPurchases),1),'recency':round(float(row.Recency),1),'campaign_rate':round(float(row.CampaignAcceptance/6*100),1),'description':f"{('Above' if row.TotalSpend>overall.TotalSpend else 'Below')} average spend with {('strong' if row.TotalPurchases>overall.TotalPurchases else 'selective')} purchase activity."})
    mapping={p['cluster']:p['name'] for p in persona};df['segment']=df.cluster.map(mapping)
    sample=df[['ID','segment','pc1','pc2','Income','TotalSpend','Recency','TotalPurchases']].sample(min(1000,len(df)),random_state=42).round(3).to_dict('records')
    meta={'trained_at':datetime.now(timezone.utc).isoformat(),'source':path.name,'raw_rows':len(raw),'model_rows':len(df),'removed_rows':len(raw)-len(df),'selected_k':best,'features':FEATURES,'selection_runs':runs,'pca_explained_variance':round(float(pca.explained_variance_ratio_.sum()),4),'personas':persona,'sample':sample,'quality':{'missing_income':int(raw.Income.isna().sum()),'duplicate_ids':int(raw.ID.duplicated().sum()),'retained_pct':round(len(df)/len(raw)*100,1)}}
    artifact={'scaler':scaler,'model':model,'pca':pca,'metadata':meta,'version':'1.0.0'};(ROOT/'artifacts').mkdir(exist_ok=True);joblib.dump(artifact,ROOT/'artifacts/segments.joblib');(ROOT/'artifacts/metrics.json').write_text(json.dumps(meta,indent=2));df.to_csv(ROOT/'artifacts/customer_segments.csv',index=False);print(json.dumps({k:v for k,v in meta.items() if k not in ['sample']},indent=2))
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/raw/sample_marketing_campaign.csv');main(p.parse_args().data)
