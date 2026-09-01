from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6): super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps
    def forward(self,x): return self.weight*x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

def apply_rope(x,cos,sin):
    a,b=x[...,::2],x[...,1::2];out=torch.stack((a*cos-b*sin,a*sin+b*cos),dim=-1);return out.flatten(-2)

class CausalGQA(nn.Module):
    def __init__(self,c:ModelConfig):
        super().__init__();self.h=c.n_heads;self.kv=c.n_kv_heads;self.d=c.d_model//c.n_heads;self.dropout=c.dropout
        self.q=nn.Linear(c.d_model,self.h*self.d,bias=False);self.k=nn.Linear(c.d_model,self.kv*self.d,bias=False);self.v=nn.Linear(c.d_model,self.kv*self.d,bias=False);self.out=nn.Linear(c.d_model,c.d_model,bias=False)
        inv=1/(c.rope_base**(torch.arange(0,self.d,2).float()/self.d));self.register_buffer("inv_freq",inv,persistent=False)
    def forward(self,x):
        b,t,_=x.shape;q=self.q(x).view(b,t,self.h,self.d).transpose(1,2);k=self.k(x).view(b,t,self.kv,self.d).transpose(1,2);v=self.v(x).view(b,t,self.kv,self.d).transpose(1,2)
        pos=torch.arange(t,device=x.device,dtype=self.inv_freq.dtype);freq=torch.outer(pos,self.inv_freq);cos,sin=freq.cos()[None,None],freq.sin()[None,None]
        q,k=apply_rope(q,cos,sin),apply_rope(k,cos,sin);repeat=self.h//self.kv;k=k.repeat_interleave(repeat,dim=1);v=v.repeat_interleave(repeat,dim=1)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.dropout if self.training else 0.0);return self.out(y.transpose(1,2).contiguous().view(b,t,-1))

class SwiGLU(nn.Module):
    def __init__(self,c): super().__init__();hidden=int(8*c.d_model/3);hidden=64*((hidden+63)//64);self.gate=nn.Linear(c.d_model,hidden,bias=False);self.up=nn.Linear(c.d_model,hidden,bias=False);self.down=nn.Linear(hidden,c.d_model,bias=False);self.dropout=nn.Dropout(c.dropout)
    def forward(self,x): return self.dropout(self.down(F.silu(self.gate(x))*self.up(x)))
class Block(nn.Module):
    def __init__(self,c): super().__init__();self.n1=RMSNorm(c.d_model);self.attn=CausalGQA(c);self.n2=RMSNorm(c.d_model);self.ff=SwiGLU(c)
    def forward(self,x): x=x+self.attn(self.n1(x));return x+self.ff(self.n2(x))

class NanoLM(nn.Module):
    def __init__(self,c:ModelConfig):
        super().__init__();c.validate();self.config=c;self.embed=nn.Embedding(c.vocab_size,c.d_model);self.drop=nn.Dropout(c.dropout);self.blocks=nn.ModuleList([Block(c) for _ in range(c.n_layers)]);self.norm=RMSNorm(c.d_model);self.head=nn.Linear(c.d_model,c.vocab_size,bias=False);self.head.weight=self.embed.weight;self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,std=.02)
        if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=.02)
    def forward(self,ids,targets=None):
        x=self.drop(self.embed(ids));
        for block in self.blocks:x=block(x)
        logits=self.head(self.norm(x));loss=None if targets is None else F.cross_entropy(logits.flatten(0,1),targets.flatten())
        return logits,loss
    @torch.inference_mode()
    def generate(self,ids,max_new_tokens=120,temperature=.8,top_k=40,top_p=.9,repetition_penalty=1.08):
        self.eval()
        for _ in range(max_new_tokens):
            context=ids[:,-self.config.block_size:];logits,_=self(context);logits=logits[:,-1,:]/max(temperature,.05)
            for token in set(ids[0,-128:].tolist()): logits[:,token]=torch.where(logits[:,token]<0,logits[:,token]*repetition_penalty,logits[:,token]/repetition_penalty)
            if top_k: threshold=torch.topk(logits,min(top_k,logits.size(-1))).values[:,-1,None];logits[logits<threshold]=-float("inf")
            if top_p<1:
                sorted_logits,idx=torch.sort(logits,descending=True);probs=torch.softmax(sorted_logits,-1);mask=probs.cumsum(-1)-probs>top_p;sorted_logits[mask]=-float("inf");logits.scatter_(1,idx,sorted_logits)
            ids=torch.cat((ids,torch.multinomial(torch.softmax(logits,-1),1)),dim=1)
        return ids
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
