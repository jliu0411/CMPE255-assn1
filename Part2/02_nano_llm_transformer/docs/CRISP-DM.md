# CRISP-DM Report

## 1. Business understanding

Goal: teach and demonstrate the complete lifecycle of a language model that a student can inspect and train locally. Success means the system trains within laptop constraints, validation loss improves over an untrained model, the API generates text, and the dashboard exposes reproducible evidence. It is not intended to compete with foundation models or support high-stakes decisions.

## 2. Data understanding

The bundled corpus contains small, authored instruction/response examples about machine learning and software practice. It is intentionally transparent and permissively reusable. This scale validates the pipeline but cannot provide broad knowledge. Data risks include duplication, narrow topic coverage, style imbalance, unsafe additions, personal information, and licensing ambiguity.

## 3. Data preparation

`prepare_data.py` normalizes newlines and whitespace, removes exact case-insensitive duplicate blocks, drops tiny fragments, creates a deterministic 90/10 document-boundary split, saves byte tokens, and records counts plus a SHA-256 fingerprint. Byte tokenization guarantees coverage and avoids an external tokenizer dependency, at the cost of longer sequences.

## 4. Modeling

The decoder-only transformer uses RMSNorm, RoPE, grouped-query causal attention through PyTorch SDPA, SwiGLU, residual connections, and tied embeddings. The default approximately 10M-parameter configuration balances learning value with 4–8 GB laptop GPUs. Training uses next-token cross entropy, AdamW, warmup plus cosine decay, gradient clipping, mixed precision on CUDA, and gradient accumulation.

## 5. Evaluation

Primary offline measures are validation cross-entropy and perplexity. Compare with the initial checkpoint and inspect the train-validation gap for overfitting. Qualitative prompts test instruction boundaries, repetition, UTF-8 behavior, and uncertainty language. A useful classroom promotion gate is decreasing validation loss plus successful API, determinism, and safety smoke tests; low perplexity alone does not imply factual or safe behavior.

## 6. Deployment

FastAPI exposes health, model telemetry, stateful chat, streaming chat, and session deletion. The service uses GPU automatically when available. Conversations reside only in process memory and are bounded to recent turns. The UI presents demo mode honestly when a checkpoint is absent. Production hardening would require authentication, rate limits, moderation, durable consent-aware storage, batching, a KV cache, model registry, evaluation suites, drift monitoring, and rollback.

## Iteration plan

Expand only with licensed, documented sources; create held-out prompt suites; add BPE after establishing the byte baseline; measure tokens/second and memory; compare model sizes; conduct safety review; then package a quantized checkpoint for deployment.
