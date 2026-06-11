# tests/test_vectorstore.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from ingestion.chunker import Chunk


# ── Mock embedder so tests never hit OpenAI API ───────────────────

def make_mock_embedder():
    """Returns a deterministic fake embedder (1536-dim vectors)."""
    class MockEmbedder:
        def embed(self, text: str) -> list[float]:
            # deterministic: hash text to a seed
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**31)
            rng  = np.random.default_rng(seed)
            vec  = rng.random(1536).astype("float32")
            return (vec / np.linalg.norm(vec)).tolist()

        def embed_batch(self, texts, batch_size=100):
            return [self.embed(t) for t in texts]

    return MockEmbedder()


def make_store():
    from vectorstore.store import VectorStore
    store = VectorStore()
    store.embedder = make_mock_embedder()
    return store


def make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(text=t, source="test.pdf", chunk_index=i,
              metadata={"filename": "test.pdf"})
        for i, t in enumerate(texts)
    ]


# ── Tests ─────────────────────────────────────────────────────────

def test_store_starts_empty():
    store = make_store()
    assert store.total == 0

def test_add_chunks_increases_total():
    store  = make_store()
    chunks = make_chunks(["Revenue grew 12% YoY.", "Net margin improved to 22%."])
    store.add_chunks(chunks)
    assert store.total == 2

def test_search_returns_results():
    store  = make_store()
    chunks = make_chunks([
        "Apple revenue reached $500B in FY2024.",
        "Operating costs increased due to R&D spend.",
        "Net profit margin was 24% for the quarter.",
    ])
    store.add_chunks(chunks)
    results = store.search("revenue profit", top_k=2)
    assert len(results) == 2
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)

def test_search_scores_between_0_and_1():
    store  = make_store()
    store.add_chunks(make_chunks(["Q3 results exceeded analyst expectations."]))
    results = store.search("quarterly results")
    assert all(0.0 <= r["score"] <= 1.0 for r in results)

def test_search_empty_store():
    store   = make_store()
    results = store.search("anything")
    assert results == []

def test_search_top_k_respected():
    store  = make_store()
    store.add_chunks(make_chunks([f"Document chunk number {i}" for i in range(10)]))
    results = store.search("document", top_k=3)
    assert len(results) <= 3

def test_metadata_preserved():
    store  = make_store()
    chunks = make_chunks(["EBITDA margin expanded in Q4."])
    store.add_chunks(chunks)
    results = store.search("EBITDA")
    assert results[0]["source"]   == "test.pdf"
    assert results[0]["filename"] == "test.pdf"

def test_save_and_load(tmp_path, monkeypatch):
    from vectorstore.store import VectorStore
    from config import get_settings

    # Point index path to temp dir
    monkeypatch.setattr("vectorstore.store.get_settings", lambda: type("S", (), {
        "faiss_index_path": str(tmp_path / "idx"),
        "openai_api_key": "",
        "openai_embedding_model": "text-embedding-3-small",
        "vector_store": "faiss",
    })())

    store = VectorStore()
    store.embedder = make_mock_embedder()
    store.add_chunks(make_chunks(["Quarterly revenue up 15%.", "Costs rose due to inflation."]))
    store.save()

    store2 = VectorStore()
    store2.embedder = make_mock_embedder()
    store2.load()
    assert store2.total == 2

def test_add_empty_chunks():
    store = make_store()
    store.add_chunks([])   # should not raise
    assert store.total == 0