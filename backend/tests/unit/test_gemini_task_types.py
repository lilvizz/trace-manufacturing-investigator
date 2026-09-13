"""
Verifies GeminiEmbeddingProvider requests the correct embedding
task_type for documents vs. queries, WITHOUT making a live API call.

This mocks google.genai.Client so it never touches the network, and
sets a fake GEMINI_API_KEY only to satisfy the provider's own
"is a key configured" check (the real Gemini API is never invoked).
"""

import pytest

from app.retrieval import gemini_embeddings as gemini_embeddings_module


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedContentResponse:
    def __init__(self, values_list):
        self.embeddings = [_FakeEmbedding(values) for values in values_list]


class _FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, model, contents, config=None):
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )

        if isinstance(contents, str):
            return _FakeEmbedContentResponse([[0.1, 0.2, 0.3]])

        return _FakeEmbedContentResponse([[0.1, 0.2, 0.3] for _ in contents])


class _FakeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.models = _FakeModels()


@pytest.fixture
def fake_gemini_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key-not-a-real-secret")
    monkeypatch.setattr(gemini_embeddings_module.genai, "Client", _FakeClient)


def test_document_embeddings_use_retrieval_document_task_type(
    fake_gemini_client,
):
    provider = gemini_embeddings_module.GeminiEmbeddingProvider()

    provider.embed_documents(["a chunk of corpus text"])

    call = provider.client.models.calls[-1]
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_query_embeddings_use_retrieval_query_task_type(fake_gemini_client):
    provider = gemini_embeddings_module.GeminiEmbeddingProvider()

    provider.embed_query("a user question")

    call = provider.client.models.calls[-1]
    assert call["config"].task_type == "RETRIEVAL_QUERY"


def test_batched_document_embeddings_all_use_retrieval_document(
    fake_gemini_client,
):
    provider = gemini_embeddings_module.GeminiEmbeddingProvider()

    provider.embed_documents_batched(
        ["chunk one", "chunk two", "chunk three"],
        batch_size=2,
    )

    assert len(provider.client.models.calls) == 2
    for call in provider.client.models.calls:
        assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
