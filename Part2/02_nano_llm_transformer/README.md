# Lumen — a laptop-scale language model

Lumen is an end-to-end educational language-model project: a decoder-only transformer built from first principles, a reproducible training pipeline, a stateful streaming chatbot, and an experiment dashboard. It follows CRISP-DM from use-case definition through monitoring.

## Modern primitives

- Pre-normalized decoder blocks with **RMSNorm**
- **Rotary position embeddings (RoPE)**
- **Grouped-query attention (GQA)**
- PyTorch scaled-dot-product attention, using Flash Attention kernels when supported
- **SwiGLU** feed-forward layers
- Tied token embeddings, AdamW, cosine decay, warmup, gradient clipping
- Automatic mixed precision, gradient accumulation, deterministic splits, checkpoints
- Character tokenizer with explicit system/user/assistant turns—small and inspectable
- Temperature, top-k, top-p and repetition-penalty generation

The default model is about 10M parameters and is designed for 4–8 GB laptop GPUs. It is a learning model trained on a deliberately small corpus, not a substitute for a production assistant.

## Quick start

```powershell
cd CMPE255-assn1/Part2/02_nano_llm_transformer
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python -m core.train
python -m uvicorn server.main:app --reload --port 8002
```

Open http://127.0.0.1:8002. Training defaults can be overridden:

```powershell
python -m core.train --max-steps 2000 --batch-size 8 --grad-accum 4 --d-model 256 --layers 6
```

For a smaller CPU smoke run:

```powershell
python -m core.train --max-steps 30 --batch-size 4 --block-size 128 --d-model 128 --layers 2 --heads 4 --kv-heads 2
```

## Workflow

1. Review [CRISP-DM.md](docs/CRISP-DM.md) and the data card.
2. Add or replace permissively licensed text in `data/raw/corpus.txt`.
3. Run `scripts/prepare_data.py` to validate, deduplicate, split, and tokenize.
4. Train and inspect validation loss/perplexity in the dashboard.
5. Start the API and test `/api/chat`; interactive docs are at `/docs`.
6. Use the dashboard to inspect throughput, configuration, run history, and operational health.

## Tests

```powershell
python -m pytest -q
```

## Honest expectations

The bundled corpus proves the complete pipeline and teaches architecture behavior. Coherent open-domain chat requires substantially more high-quality data and compute. The UI labels untrained/demo mode, generated output can be incorrect, and conversation state is held in memory only.
