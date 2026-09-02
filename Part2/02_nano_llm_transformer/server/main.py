from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.assistant import GroundedAssistant

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "server/static"
ARTIFACT = ROOT / "artifacts/model.pt"
CORPUS = ROOT / "data/raw/corpus.txt"

app = FastAPI(title="Lumen NanoLM", version="1.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
sessions: dict[str, list[dict]] = {}
grounded = GroundedAssistant(CORPUS)
runtime = {
    "model": None,
    "tokenizer": None,
    "torch": None,
    "device": "unavailable",
    "loaded": False,
}


def load_model() -> None:
    if runtime["loaded"] or not ARTIFACT.exists():
        return
    try:
        import torch
        from core.config import ModelConfig
        from core.model import NanoLM
        from core.tokenizer import ByteTokenizer

        artifact = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        model = NanoLM(ModelConfig(**artifact["model_config"]))
        model.load_state_dict(artifact["model"])
        model.to(device).eval()
        runtime.update(
            model=model,
            tokenizer=ByteTokenizer(),
            torch=torch,
            device=device,
            loaded=True,
            error=None,
        )
    except Exception as exc:
        runtime["error"] = str(exc)


@app.on_event("startup")
def startup() -> None:
    load_model()


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    load_model()
    return {
        "status": "ok",
        "model_ready": runtime["loaded"],
        "mode": "hybrid",
        "default_chat_mode": "grounded",
        "neural_mode_available": runtime["loaded"],
        "device": runtime["device"],
        "sessions": len(sessions),
        "model_error": runtime.get("error"),
    }


@app.get("/api/metrics")
def metrics():
    metrics_path = ROOT / "artifacts/metrics.json"
    data_path = ROOT / "data/processed/stats.json"
    base = {
        "history": [],
        "parameters": 0,
        "device": "not trained",
        "model_config": {
            "d_model": 192,
            "n_layers": 4,
            "n_heads": 6,
            "n_kv_heads": 2,
            "block_size": 256,
        },
        "demo": True,
    }
    if metrics_path.exists():
        base.update(json.loads(metrics_path.read_text(encoding="utf-8")))
        base["demo"] = False
    base["dataset"] = (
        json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else {}
    )
    return base


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = None
    temperature: float = Field(0.8, ge=0.05, le=2)
    top_k: int = Field(40, ge=0, le=256)
    top_p: float = Field(0.9, gt=0, le=1)
    max_tokens: int = Field(120, ge=8, le=400)
    repetition_penalty: float = Field(1.08, ge=1, le=2)


def neural_reply(req: ChatRequest, prompt_text: str, history: list[dict]) -> str:
    if not runtime["loaded"]:
        return "Neural mode is unavailable because no checkpoint could be loaded."
    prompt = (
        "<|system|> You are Lumen, concise and honest.\n"
        + "".join(
            f"<|{item['role']}|> {item['content']}\n" for item in history[-4:]
        )
        + f"<|user|> {prompt_text}\n<|assistant|> "
    )
    tokenizer, torch = runtime["tokenizer"], runtime["torch"]
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=runtime["device"])
    output = runtime["model"].generate(
        x,
        req.max_tokens,
        req.temperature,
        req.top_k,
        req.top_p,
        req.repetition_penalty,
    )
    text = tokenizer.decode(output[0, len(ids) :].tolist()).split("<|")[0].strip()
    return text or "The nano-transformer did not produce a readable response."


def answer(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.setdefault(session_id, [])[-10:]
    started = time.time()
    raw_neural = req.message.strip().lower().startswith("/neural")
    if raw_neural:
        prompt_text = req.message.strip()[len("/neural") :].strip()
        if not prompt_text:
            text = "Add text after /neural. Example: /neural Machine learning is"
            mode = "grounded"
        else:
            text = neural_reply(req, prompt_text, history)
            mode = "neural" if runtime["loaded"] else "grounded"
    else:
        text = grounded.reply(req.message, history)
        mode = "grounded"

    history.extend(
        [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": text},
        ]
    )
    sessions[session_id] = history[-10:]
    latency = round((time.time() - started) * 1000, 1)
    return session_id, text, mode, latency


@app.post("/api/chat")
def chat(req: ChatRequest):
    load_model()
    session_id, text, mode, latency = answer(req)
    return {
        "session_id": session_id,
        "response": text,
        "mode": mode,
        "latency_ms": latency,
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    load_model()
    session_id, text, mode, latency = answer(req)

    def events():
        for word in text.split(" "):
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        yield (
            f"data: {json.dumps({'done': True, 'session_id': session_id, 'mode': mode, 'latency_ms': latency})}\n\n"
        )

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/api/sessions/{session_id}")
def clear(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": True}

