from dataclasses import asdict, dataclass

@dataclass
class ModelConfig:
    vocab_size:int=256; block_size:int=256; d_model:int=192; n_layers:int=4; n_heads:int=6; n_kv_heads:int=2; dropout:float=.1; rope_base:float=10000.0
    def validate(self):
        assert self.d_model%self.n_heads==0,"d_model must divide evenly across heads"
        assert self.n_heads%self.n_kv_heads==0,"query heads must divide evenly across KV heads"
    def to_dict(self): return asdict(self)

@dataclass
class TrainConfig:
    batch_size:int=8; grad_accum:int=4; max_steps:int=1200; learning_rate:float=3e-4; min_lr:float=3e-5; warmup_steps:int=80; eval_interval:int=50; eval_batches:int=20; weight_decay:float=.1; grad_clip:float=1.0; seed:int=42
