"""FAISS vector search over fastembed (ONNX) embeddings, cached on disk."""
from __future__ import annotations

import json

import faiss
import numpy as np
from fastembed import TextEmbedding

import config
from rag.chunker import Chunk, kb_fingerprint


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


class VectorIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embedder = TextEmbedding(config.EMBED_MODEL, cache_dir=config.MODEL_CACHE_DIR)
        self.index = self._load_or_build()

    def _load_or_build(self) -> faiss.Index:
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index_path = config.INDEX_DIR / "kb.faiss"
        meta_path = config.INDEX_DIR / "kb_meta.json"
        fingerprint = kb_fingerprint()
        if index_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") == fingerprint and meta.get("count") == len(self.chunks):
                return faiss.read_index(str(index_path))

        vectors = _normalize(np.array(list(self.embedder.passage_embed([c.text for c in self.chunks])), dtype="float32"))
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine
        index.add(vectors)
        faiss.write_index(index, str(index_path))
        meta_path.write_text(json.dumps({"fingerprint": fingerprint, "count": len(self.chunks), "model": config.EMBED_MODEL}))
        return index

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = _normalize(np.array(list(self.embedder.query_embed(query)), dtype="float32"))
        scores, ids = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
