import json
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
ROOT=Path(__file__).resolve().parents[1];STATIC=ROOT/'app/static';RESULTS=ROOT/'artifacts/results.json';app=FastAPI(title='Atlas Skills Lab',version='1.0.0');app.mount('/static',StaticFiles(directory=STATIC),name='static')
@app.get('/',include_in_schema=False)
def home():return FileResponse(STATIC/'index.html')
@app.get('/api/health')
def health():return {'status':'ok','results_ready':RESULTS.exists()}
@app.get('/api/results')
def results():
    if not RESULTS.exists():raise HTTPException(503,'Run scripts/run_lab.py first.')
    return json.loads(RESULTS.read_text())
