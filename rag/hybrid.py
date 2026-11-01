"""Hybrid retrieval pipeline.

    query ─┬─ BM25 keyword search ─┐
           ├─ FAISS vector search ─┴─ Reciprocal Rank Fusion ─ cross-encoder rerank ─ top-k passages
           └─ Neo4j knowledge graph (entity linking + 1-hop) ──────────────────────── structured facts
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import config
from core.tracing import traceable
from rag.chunker import Chunk, load_chunks
from rag.graph import KnowledgeGraph
from rag.keyword_index import KeywordIndex
from rag.reranker import Reranker
from rag.vector_index import VectorIndex


@dataclass
class Passage:
    source: str
    section: str
    text: str
    rerank_score: float
    bm25_rank: int | None
    vector_rank: int | None


@dataclass
class RetrievalResult:
    query: str
    passages: list[Passage]
    graph_facts: list[str]
    found: bool
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(p.source for p in self.passages))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = config.RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self):
        self.chunks: list[Chunk] = load_chunks()
        self.keyword = KeywordIndex(self.chunks)
        self.vector = VectorIndex(self.chunks)
        self.reranker = Reranker()
        self.graph = KnowledgeGraph()

    @traceable(name="bm25_search", run_type="retriever")
    def _keyword(self, query: str) -> list[int]:
        return [i for i, _ in self.keyword.search(query, config.TOP_K_EACH)]

    @traceable(name="faiss_search", run_type="retriever")
    def _vector(self, query: str) -> list[int]:
        return [i for i, _ in self.vector.search(query, config.TOP_K_EACH)]

    @traceable(name="rerank")
    def _rerank(self, query: str, candidates: list[int]) -> list[tuple[int, float]]:
        scores = self.reranker.score(query, [self.chunks[i].text for i in candidates])
        return sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)

    @traceable(name="knowledge_graph", run_type="retriever")
    def _graph(self, query: str) -> list[str]:
        return self.graph.query(query)

    @traceable(name="hybrid_retrieve", run_type="retriever")
    def retrieve(self, query: str, top_k: int = config.TOP_K_FINAL) -> RetrievalResult:
        t = {}
        start = time.perf_counter()
        kw = self._keyword(query)
        t["bm25"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        vec = self._vector(query)
        t["faiss"] = (time.perf_counter() - start) * 1000

        fused = reciprocal_rank_fusion([kw, vec])[: config.TOP_K_EACH + 4]
        start = time.perf_counter()
        reranked = self._rerank(query, fused)
        t["rerank"] = (time.perf_counter() - start) * 1000

        passages = [
            Passage(
                source=self.chunks[i].source,
                section=self.chunks[i].section,
                text=self.chunks[i].text,
                rerank_score=round(score, 3),
                bm25_rank=kw.index(i) + 1 if i in kw else None,
                vector_rank=vec.index(i) + 1 if i in vec else None,
            )
            for i, score in reranked
            if score >= config.RERANK_MIN_SCORE
        ][:top_k]

        start = time.perf_counter()
        facts = self._graph(query)
        t["graph"] = (time.perf_counter() - start) * 1000
        return RetrievalResult(query=query, passages=passages, graph_facts=facts,
                               found=bool(passages) or bool(facts), timings_ms=t)


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
