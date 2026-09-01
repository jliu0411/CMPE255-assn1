import json
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/rules.json';STATIC=ROOT/'app/static';app=FastAPI(title='Affinity Basket Intelligence API',version='1.0.0');app.mount('/static',StaticFiles(directory=STATIC),name='static')
def load():
    if not ART.exists():raise HTTPException(503,'Rules not mined. Run data generation and mining first.')
    return json.loads(ART.read_text())
class Basket(BaseModel):items:list[str]=Field(min_length=1,max_length=30)
@app.get('/',include_in_schema=False)
def home():return FileResponse(STATIC/'index.html')
@app.get('/api/health')
def health():return {'status':'ok','artifact_ready':ART.exists()}
@app.get('/api/dashboard')
def dashboard():return load()
@app.post('/api/recommend')
def recommend(basket:Basket):
    selected={x.upper().strip() for x in basket.items};candidates={}
    for r in load()['rules']:
        if set(r['antecedents'])<=selected:
            for item in set(r['consequents'])-selected:
                score=r['confidence']*r['lift']*(1+r['support']);old=candidates.get(item)
                row={'item':item,'score':round(score,4),'confidence':r['confidence'],'lift':r['lift'],'support':r['support'],'stable':r['stable'],'because':r['antecedents']}
                if not old or row['score']>old['score']:candidates[item]=row
    return {'basket':sorted(selected),'recommendations':sorted(candidates.values(),key=lambda x:x['score'],reverse=True)[:8],'rules_evaluated':len(load()['rules'])}
