"""Cross-encoder reranker: re-scores (query, passage) pairs for precise ordering."""
from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

import config


class Reranker:
    def __init__(self):
        self.model = TextCrossEncoder(config.RERANK_MODEL, cache_dir=config.MODEL_CACHE_DIR)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.rerank(query, passages)]
