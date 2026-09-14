import pytest

from app.investigation.gemini_client import (
    GeminiReasoningClient,
)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv(
        "GEMINI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="GEMINI_API_KEY",
    ):
        GeminiReasoningClient()


def test_generate_returns_response(monkeypatch):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    client = GeminiReasoningClient.__new__(
        GeminiReasoningClient
    )

    class FakeResponse:
        text = '{"claims": []}'

    class FakeModels:
        def generate_content(
            self,
            model,
            contents,
            config,
        ):
            assert model == "gemini-test"
            assert contents == "test prompt"
            assert config.response_mime_type == "application/json"
            assert config.temperature == 0.1

            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    client.client = FakeClient()
    client.model = "gemini-test"
    client.max_retries = 0
    client.base_delay_seconds = 0

    result = client.generate("test prompt")

    assert result == '{"claims": []}'


def test_empty_response_raises(monkeypatch):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    client = GeminiReasoningClient.__new__(
        GeminiReasoningClient
    )

    class FakeResponse:
        text = ""

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    client.client = FakeClient()
    client.model = "gemini-test"
    client.max_retries = 0
    client.base_delay_seconds = 0

    with pytest.raises(
        RuntimeError,
        match="empty response",
    ):
        client.generate("test prompt")