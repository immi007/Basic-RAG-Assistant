import pytest

from basic_rag.generate import NO_ANSWER_REPLY, GenerationError, OllamaGenerator, build_messages
from basic_rag.index import Hit


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.chat_kwargs = None

    def list(self):
        if self.error:
            raise self.error
        return {"models": []}

    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        if self.error:
            raise self.error
        return {"message": {"content": self.reply}}


def make_hits(chunks):
    return [Hit(chunks[0], 0.5), Hit(chunks[1], 0.4)]


def test_messages_number_passages_in_hit_order(chunks):
    messages = build_messages("What is RAG?", make_hits(chunks))

    assert messages[0]["role"] == "system"
    assert NO_ANSWER_REPLY in messages[0]["content"]
    user = messages[1]["content"]
    assert user.index("[1] " + chunks[0].text) < user.index("[2] " + chunks[1].text)
    assert user.endswith("Question: What is RAG?")


def test_generate_returns_stripped_text_with_deterministic_options(chunks):
    client = FakeClient(reply="  RAG adds context. [1]  ")
    generator = OllamaGenerator("llama3.2", client=client)

    answer = generator.generate("What is RAG?", make_hits(chunks))

    assert answer == "RAG adds context. [1]"
    assert client.chat_kwargs["model"] == "llama3.2"
    assert client.chat_kwargs["options"] == {"temperature": 0}


def test_client_errors_become_generation_errors_with_a_hint(chunks):
    generator = OllamaGenerator("llama3.2", client=FakeClient(error=ConnectionError("refused")))
    with pytest.raises(GenerationError) as info:
        generator.generate("q", make_hits(chunks))
    assert "ollama pull llama3.2" in str(info.value)


def test_empty_model_output_is_an_error(chunks):
    generator = OllamaGenerator("llama3.2", client=FakeClient(reply="   "))
    with pytest.raises(GenerationError):
        generator.generate("q", make_hits(chunks))


def test_is_available_reflects_server_state():
    assert OllamaGenerator("m", client=FakeClient(reply="x")).is_available() is True
    down = OllamaGenerator("m", client=FakeClient(error=ConnectionError("down")))
    assert down.is_available() is False
