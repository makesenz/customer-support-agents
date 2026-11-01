"""BM25 keyword search: catches exact terms such as "COD", "₹300", "IFSC", product names."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rag.chunker import Chunk

_STOP = set(
    "a an the is are was were be been to of in on for and or with my i me we you your our it this that "
    "do does did can could how what when where why which who will would should please hi hello".split()
)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9₹@.]+", text.lower())
    return [t.strip(".") for t in tokens if t.strip(".") and t not in _STOP]


class KeywordIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:k] if scores[i] > 0]
