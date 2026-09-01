from __future__ import annotations
import json,time,uuid
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
ROOT=Path(__file__).resolve().parents[1];STATIC=ROOT/"server/static";ART=ROOT/"artifacts/model.pt";app=FastAPI(title="Lumen NanoLM",version="1.0.0");app.mount('/static',StaticFiles(directory=STATIC),name='static');sessions={};runtime={"model":None,"tokenizer":None,"torch":None,"device":"unavailable","loaded":False}
def load_model():
    if runtime["loaded"]:return
    if not ART.exists():return
    try:
        import torch
        from core.config import ModelConfig
        from core.model import NanoLM
        from core.tokenizer import ByteTokenizer
        artifact=torch.load(ART,map_location='cpu',weights_only=False);device='cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu';model=NanoLM(ModelConfig(**artifact['model_config']));model.load_state_dict(artifact['model']);model.to(device).eval();runtime.update(model=model,tokenizer=ByteTokenizer(),torch=torch,device=device,loaded=True)
    except Exception as exc: runtime["error"]=str(exc)
@app.on_event('startup')
def startup():load_model()
@app.get('/',include_in_schema=False)
def root():return FileResponse(STATIC/'index.html')
@app.get('/api/health')
def health():load_model();return {"status":"ok","model_ready":runtime['loaded'],"mode":"trained" if runtime['loaded'] else "demo","device":runtime['device'],"sessions":len(sessions)}
@app.get('/api/metrics')
def metrics():
    mp=ROOT/'artifacts/metrics.json';dp=ROOT/'data/processed/stats.json';base={"history":[],"parameters":0,"device":"not trained","model_config":{"d_model":192,"n_layers":4,"n_heads":6,"n_kv_heads":2,"block_size":256},"demo":True}
    if mp.exists():base.update(json.loads(mp.read_text()));base['demo']=False
    base['dataset']=json.loads(dp.read_text()) if dp.exists() else {};return base
class ChatRequest(BaseModel):
    message:str=Field(min_length=1,max_length=1000);session_id:str|None=None;temperature:float=Field(.8,ge=.05,le=2);top_k:int=Field(40,ge=0,le=256);top_p:float=Field(.9,gt=0,le=1);max_tokens:int=Field(120,ge=8,le=400);repetition_penalty:float=Field(1.08,ge=1,le=2)
def demo_reply(text):
    q=text.lower();answers=[(('transformer','attention'),'A transformer predicts tokens using attention, which lets each position combine relevant earlier context. This project adds RoPE, grouped-query attention, RMSNorm, and SwiGLU.'),(('crisp','data science'),'CRISP-DM iterates through business understanding, data understanding, preparation, modeling, evaluation, and deployment.'),(('model','lumen'),'Lumen is a small educational decoder-only transformer. Train the checkpoint to enable neural generation; until then I use transparent demo responses.'),(('hello','hi'),'Hello! I’m Lumen. Ask me about this model, transformers, training, or CRISP-DM.')]
    for keys,a in answers:
        if any(k in q for k in keys):return a
    return 'I am in demo mode because no trained checkpoint is loaded. I can still explain this project, but run the training command to enable original neural generation.'
def answer(req):
    sid=req.session_id or str(uuid.uuid4());history=sessions.setdefault(sid,[])[-8:];started=time.time()
    if runtime['loaded']:
        prompt='<|system|> You are Lumen, concise and honest.\n'+''.join(f"<|{m['role']}|> {m['content']}\n" for m in history)+f'<|user|> {req.message}\n<|assistant|> '
        tok,torch=runtime['tokenizer'],runtime['torch'];ids=tok.encode(prompt);x=torch.tensor([ids],dtype=torch.long,device=runtime['device']);out=runtime['model'].generate(x,req.max_tokens,req.temperature,req.top_k,req.top_p,req.repetition_penalty);text=tok.decode(out[0,len(ids):].tolist()).split('<|')[0].strip();mode='trained'
    else:text=demo_reply(req.message);mode='demo'
    history.extend([{"role":"user","content":req.message},{"role":"assistant","content":text}]);sessions[sid]=history[-10:];return sid,text,mode,round((time.time()-started)*1000,1)
@app.post('/api/chat')
def chat(req:ChatRequest):
    load_model();sid,text,mode,latency=answer(req);return {"session_id":sid,"response":text,"mode":mode,"latency_ms":latency}
@app.post('/api/chat/stream')
def chat_stream(req:ChatRequest):
    load_model();sid,text,mode,latency=answer(req)
    def events():
        for word in text.split(' '):yield f"data: {json.dumps({'token':word+' '})}\n\n"
        yield f"data: {json.dumps({'done':True,'session_id':sid,'mode':mode,'latency_ms':latency})}\n\n"
    return StreamingResponse(events(),media_type='text/event-stream')
@app.delete('/api/sessions/{session_id}')
def clear(session_id:str):sessions.pop(session_id,None);return {"cleared":True}
