import sys
from types import SimpleNamespace

import pytest

from matrix.llm import GeminiClient, LLMError


class FakeGenerateContent:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeGenAIClient:
    def __init__(self, response):
        self.models = FakeGenerateContent(response)
        self.api_key = None


def install_fake_sdk(monkeypatch, response):
    fake_client = FakeGenAIClient(response)
    fake_genai = SimpleNamespace(
        Client=lambda api_key: _client_with_key(fake_client, api_key)
    )
    fake_types = SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs)
    )

    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(
        sys.modules,
        "google.genai",
        SimpleNamespace(types=fake_types),
    )

    return fake_client


def _client_with_key(client, api_key):
    client.api_key = api_key
    return client


def test_gemini_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        GeminiClient("test-model")


def test_gemini_sends_prompt_and_returns_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    fake = install_fake_sdk(
        monkeypatch,
        SimpleNamespace(text='{"distribution": {"H1": 0.25}}'),
    )

    client = GeminiClient("test-model", temperature=0.2, max_tokens=123)
    result = client.complete("system instructions", "user prompt")

    assert result == '{"distribution": {"H1": 0.25}}'
    assert fake.api_key == "test-key"
    assert fake.models.kwargs["model"] == "test-model"
    assert fake.models.kwargs["contents"] == "user prompt"

    config = fake.models.kwargs["config"]
    assert config.system_instruction == "system instructions"
    assert config.temperature == 0.2
    assert config.max_output_tokens == 123
    assert config.response_mime_type == "application/json"


def test_gemini_rejects_empty_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    install_fake_sdk(monkeypatch, SimpleNamespace(text=""))

    client = GeminiClient("test-model")

    with pytest.raises(LLMError, match="empty response"):
        client.complete("system", "user")
