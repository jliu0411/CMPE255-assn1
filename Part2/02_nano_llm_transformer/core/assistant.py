from __future__ import annotations

import re
from pathlib import Path

WORD_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in",
    "is", "it", "me", "of", "on", "the", "to", "what", "why", "will", "you",
}
PAIR_RE = re.compile(
    r"<\|user\|>\s*(.*?)\s*<\|assistant\|>\s*(.*?)(?=\n\s*<\|user\|>|\Z)",
    re.DOTALL,
)
FOLLOW_UPS = {"tell me more", "explain more", "can you explain more", "why", "how so"}


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def tokens(text: str) -> set[str]:
    result = set()
    for word in WORD_RE.findall(text.lower()):
        if word in STOP_WORDS:
            continue
        # Enough stemming for simple prompts such as transformer/transformers.
        result.add(word[:-1] if len(word) > 4 and word.endswith("s") else word)
    return result


class GroundedAssistant:
    """Small, deterministic assistant grounded in the bundled project corpus."""

    def __init__(self, corpus_path: Path):
        corpus = corpus_path.read_text(encoding="utf-8")
        self.examples = [
            (question.strip(), answer.strip(), tokens(question))
            for question, answer in PAIR_RE.findall(corpus)
        ]

    def _match(self, message: str) -> tuple[str | None, int]:
        query = tokens(message)
        normalized = normalize(message)
        best_answer, best_score = None, 0
        for question, answer, question_tokens in self.examples:
            overlap = len(query & question_tokens)
            score = overlap * 3
            if normalize(question) in normalized or normalized in normalize(question):
                score += 5
            if score > best_score:
                best_answer, best_score = answer, score
        return best_answer, best_score

    def reply(self, message: str, history: list[dict] | None = None) -> str:
        clean = normalize(message)
        history = history or []

        if not clean:
            return "Please enter a question."
        if clean in {"hello", "hi", "hey", "hello lumen", "good morning", "good afternoon"}:
            return "Hello! I am Lumen. Ask me about machine learning, transformers, CRISP-DM, model training, data quality, or this project."
        if "speak english" in clean or "can you talk" in clean:
            return "Yes. I answer in clear English by default."
        if clean in {"help", "what should i ask", "show me examples"}:
            return (
                "Try asking: What is a transformer? What is attention? What is CRISP-DM? "
                "Why is data quality important? How was Lumen built?"
            )
        if "how were you built" in clean or "how are you built" in clean:
            return (
                "Lumen has a small decoder-only transformer built in PyTorch with RMSNorm, "
                "rotary position embeddings, grouped-query attention, and SwiGLU layers. "
                "Normal chat uses grounded answers from the project corpus because the tiny "
                "neural checkpoint is too small for reliable open-domain conversation."
            )
        if "raw model" in clean or "neural mode" in clean:
            return (
                "Start a message with /neural to sample directly from the trained nano-transformer. "
                "For example: /neural Machine learning is. Raw output may be incomplete or nonsensical."
            )
        if clean in FOLLOW_UPS and history:
            previous_users = [item["content"] for item in history if item.get("role") == "user"]
            if previous_users:
                message = previous_users[-1] + " " + message

        answer, score = self._match(message)
        if answer and score >= 3:
            return answer

        return (
            "I do not have enough grounded information to answer that reliably. "
            "I am a small educational assistant, not a general-purpose LLM. "
            "Ask me about transformers, machine learning, CRISP-DM, training, evaluation, "
            "data quality, or the Lumen project."
        )
