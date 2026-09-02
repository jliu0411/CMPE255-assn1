from pathlib import Path

from core.assistant import GroundedAssistant

ROOT = Path(__file__).resolve().parents[1]
assistant = GroundedAssistant(ROOT / "data/raw/corpus.txt")


def test_greeting_is_readable():
    assert assistant.reply("hello").startswith("Hello!")


def test_known_question_is_grounded():
    response = assistant.reply("What is a transformer?")
    assert "attention" in response.lower()
    assert "token" in response.lower()


def test_project_question_explains_hybrid_design():
    response = assistant.reply("How were you built?")
    assert "PyTorch" in response
    assert "grounded" in response


def test_unknown_question_is_honest():
    response = assistant.reply("Who will win an election in 2050?")
    assert "do not have enough grounded information" in response

