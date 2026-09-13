"""
Verifies GeminiEmbeddingProvider retries transient (429/5xx) failures
and gives up on non-retryable ones, WITHOUT making a live API call or
actually sleeping between retries.
"""

import pytest
from google.genai import errors

from app.retrieval import gemini_embeddings as gemini_embeddings_module


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedContentResponse:
    def __init__(self, values_list):
        self.embeddings = [_FakeEmbedding(values) for values in values_list]


def _make_api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "boom"}})


class _FlakyThenSucceedsModels:
    """Fails with a retryable error a fixed number of times, then
    succeeds."""

    def __init__(self, failures_before_success: int, failure_code: int = 429):
        self.failures_before_success = failures_before_success
        self.failure_code = failure_code
        self.call_count = 0

    def embed_content(self, model, contents, config=None):
        self.call_count += 1

        if self.call_count <= self.failures_before_success:
            raise _make_api_error(self.failure_code)

        return _FakeEmbedContentResponse([[0.1, 0.2, 0.3]])


class _AlwaysFailsModels:
    def __init__(self, failure_code: int):
        self.failure_code = failure_code
        self.call_count = 0

    def embed_content(self, model, contents, config=None):
        self.call_count += 1
        raise _make_api_error(self.failure_code)


def _build_provider(monkeypatch, models):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key-not-a-real-secret")

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = models

    monkeypatch.setattr(gemini_embeddings_module.genai, "Client", _FakeClient)
    monkeypatch.setattr(gemini_embeddings_module.time, "sleep", lambda _: None)

    return gemini_embeddings_module.GeminiEmbeddingProvider()


def test_transient_failure_is_retried_and_eventually_succeeds(monkeypatch):
    models = _FlakyThenSucceedsModels(failures_before_success=2, failure_code=429)
    provider = _build_provider(monkeypatch, models)

    embedding = provider.embed_query("some question")

    assert embedding == [0.1, 0.2, 0.3]
    assert models.call_count == 3  # 2 failures + 1 success


def test_retries_are_exhausted_and_error_is_raised(monkeypatch):
    models = _AlwaysFailsModels(failure_code=503)
    provider = _build_provider(monkeypatch, models)

    with pytest.raises(errors.APIError):
        provider.embed_query("some question")

    assert models.call_count == gemini_embeddings_module.DEFAULT_MAX_RETRIES + 1


def test_non_retryable_error_raises_immediately(monkeypatch):
    models = _AlwaysFailsModels(failure_code=400)
    provider = _build_provider(monkeypatch, models)

    with pytest.raises(errors.APIError):
        provider.embed_query("some question")

    assert models.call_count == 1
