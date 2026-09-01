from __future__ import annotations
import argparse,json,math,random,time
from dataclasses import asdict
from pathlib import Path
import numpy as np,torch
from .config import ModelConfig,TrainConfig
from .model import NanoLM
ROOT=Path(__file__).resolve().parents[1]
def device(): return "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"
def main(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);dev=device()
    mc=ModelConfig(256,args.block_size,args.d_model,args.layers,args.heads,args.kv_heads,args.dropout);tc=TrainConfig(args.batch_size,args.grad_accum,args.max_steps,args.lr,args.lr*.1,min(80,args.max_steps//10),args.eval_interval,10,.1,1.0,args.seed)
    train=np.fromfile(ROOT/"data/processed/train.bin",dtype=np.uint8);val=np.fromfile(ROOT/"data/processed/val.bin",dtype=np.uint8)
    model=NanoLM(mc).to(dev);optimizer=torch.optim.AdamW(model.parameters(),lr=tc.learning_rate,betas=(.9,.95),weight_decay=tc.weight_decay);use_amp=dev=="cuda";scaler=torch.amp.GradScaler("cuda",enabled=use_amp);history=[];started=time.time()
    def batch(data):
        ix=np.random.randint(0,len(data)-mc.block_size-1,tc.batch_size);x=np.stack([data[i:i+mc.block_size] for i in ix]);y=np.stack([data[i+1:i+mc.block_size+1] for i in ix]);return torch.from_numpy(x.astype(np.int64)).to(dev),torch.from_numpy(y.astype(np.int64)).to(dev)
    @torch.inference_mode()
    def evaluate():
        model.eval();losses=[]
        for _ in range(tc.eval_batches):
            x,y=batch(val)
            with torch.autocast(device_type=dev,dtype=torch.float16,enabled=use_amp): _,loss=model(x,y)
            losses.append(loss.item())
        model.train();return float(np.mean(losses))
    model.train();optimizer.zero_grad(set_to_none=True)
    for step in range(1,tc.max_steps+1):
        lr=tc.learning_rate*min(step/max(tc.warmup_steps,1),1) if step<=tc.warmup_steps else tc.min_lr+.5*(tc.learning_rate-tc.min_lr)*(1+math.cos(math.pi*(step-tc.warmup_steps)/max(1,tc.max_steps-tc.warmup_steps)))
        for g in optimizer.param_groups:g["lr"]=lr
        running=0
        for _ in range(tc.grad_accum):
            x,y=batch(train)
            with torch.autocast(device_type=dev,dtype=torch.float16,enabled=use_amp): _,loss=model(x,y);loss=loss/tc.grad_accum
            scaler.scale(loss).backward();running+=loss.item()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),tc.grad_clip);scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        if step==1 or step%tc.eval_interval==0 or step==tc.max_steps:
            vl=evaluate();row={"step":step,"train_loss":round(running,4),"val_loss":round(vl,4),"perplexity":round(math.exp(min(vl,10)),2),"lr":lr,"tokens_seen":step*tc.batch_size*tc.grad_accum*mc.block_size,"elapsed_seconds":round(time.time()-started,1)};history.append(row);print(row)
    artifact={"model":model.state_dict(),"model_config":mc.to_dict(),"train_config":asdict(tc),"history":history,"parameters":model.parameter_count,"device":dev,"trained_at":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(ROOT/"artifacts").mkdir(exist_ok=True);torch.save(artifact,ROOT/"artifacts/model.pt");(ROOT/"artifacts/metrics.json").write_text(json.dumps({k:v for k,v in artifact.items() if k!="model"},indent=2));print(f"Saved {model.parameter_count/1e6:.2f}M parameter model")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument('--max-steps',type=int,default=1200);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--grad-accum',type=int,default=4);p.add_argument('--block-size',type=int,default=256);p.add_argument('--d-model',type=int,default=192);p.add_argument('--layers',type=int,default=4);p.add_argument('--heads',type=int,default=6);p.add_argument('--kv-heads',type=int,default=2);p.add_argument('--dropout',type=float,default=.1);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--eval-interval',type=int,default=50);p.add_argument('--seed',type=int,default=42);main(p.parse_args())
