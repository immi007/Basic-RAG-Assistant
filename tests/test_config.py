import pytest

from basic_rag.config import Settings


def test_defaults_are_valid():
    settings = Settings()
    assert settings.chunk_overlap < settings.chunk_words
    assert settings.top_k > 0


def test_invalid_values_are_rejected():
    bad = [
        {"chunk_words": 0},
        {"chunk_words": 10, "chunk_overlap": 10},
        {"chunk_overlap": -1},
        {"top_k": 0},
        {"min_score": 2.0},
        {"llm_timeout": 0},
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            Settings(**kwargs)


def test_from_env_overrides_defaults():
    settings = Settings.from_env({"RAG_LLM_MODEL": "llama3.2:1b", "RAG_TOP_K": "5"})
    assert settings.llm_model == "llama3.2:1b"
    assert settings.top_k == 5
    assert settings.embed_model == Settings().embed_model


def test_from_env_ignores_empty_values():
    assert Settings.from_env({"RAG_TOP_K": ""}).top_k == Settings().top_k


def test_from_env_reports_bad_numbers():
    with pytest.raises(ValueError):
        Settings.from_env({"RAG_TOP_K": "many"})
