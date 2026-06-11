# vectorstore/store.py
import json
import numpy as np
from pathlib import Path
from loguru import logger
from ingestion.chunker import Chunk
from .embedder import Embedder
from config import get_settings


class VectorStore:
    """
    FAISS-based vector store with save/load support.
    Swap VECTOR_STORE=pinecone in .env to switch to Pinecone in production.
    """

    def __init__(self):
        self.settings = get_settings()
        self.embedder = Embedder()
        self._index   = None      # FAISS index
        self._meta    = []        # parallel metadata list

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and index a list of Chunk objects."""
        if not chunks:
            return
        texts  = [c.text for c in chunks]
        metas  = [{"text": c.text, "source": c.source,
                   "chunk_index": c.chunk_index, **c.metadata}
                  for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        self._faiss_add(embeddings, metas)
        logger.info(f"Indexed {len(chunks)} chunks — total: {self.total}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k most relevant chunks for a query string."""
        if self._index is None or self.total == 0:
            logger.warning("Vector store is empty — ingest documents first.")
            return []
        q_emb = self.embedder.embed(query)
        return self._faiss_search(q_emb, top_k)

    def save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss
        path = Path(self.settings.faiss_index_path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w") as f:
            json.dump(self._meta, f)
        logger.info(f"Saved {self.total} vectors → {path}")

    def load(self) -> None:
        """Load index and metadata from disk if they exist."""
        import faiss
        path      = Path(self.settings.faiss_index_path)
        idx_file  = path / "index.faiss"
        meta_file = path / "metadata.json"
        if idx_file.exists() and meta_file.exists():
            self._index = faiss.read_index(str(idx_file))
            with open(meta_file) as f:
                self._meta = json.load(f)
            logger.info(f"Loaded {self.total} vectors from {path}")
        else:
            logger.info("No existing index found — starting fresh.")

    @property
    def total(self) -> int:
        return self._index.ntotal if self._index else 0

    # ── FAISS internals ───────────────────────────────────────────────────────

    def _faiss_add(self, embeddings: list[list[float]], metas: list[dict]):
        import faiss
        vectors = np.array(embeddings, dtype="float32")
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        if self._index is None:
            dim = vectors.shape[1]
            self._index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after L2 norm
        self._index.add(vectors)
        self._meta.extend(metas)

    def _faiss_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        import faiss
        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)
        k = min(top_k, self.total)
        scores, indices = self._index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = self._meta[idx].copy()
            entry["score"] = round(float(score), 4)
            results.append(entry)
        return results