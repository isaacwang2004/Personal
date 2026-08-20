from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import PipelineConfig
from .schemas import RetrievalResult, TranslationMemoryEntry


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class BM25Index:
    """Small dependency-free BM25 implementation for the portfolio demo."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.tokenized = [_tokenize(doc) for doc in documents]
        self.k1 = k1
        self.b = b
        self.doc_lengths = np.array([len(doc) for doc in self.tokenized], dtype=float)
        self.avg_doc_length = float(self.doc_lengths.mean()) if documents else 0.0
        self.term_freqs = [Counter(doc) for doc in self.tokenized]
        self.doc_freqs: dict[str, int] = defaultdict(int)
        for doc in self.tokenized:
            for term in set(doc):
                self.doc_freqs[term] += 1

    def score(self, query: str) -> np.ndarray:
        if not self.documents:
            return np.array([], dtype=float)

        n_docs = len(self.documents)
        scores = np.zeros(n_docs, dtype=float)
        for term in _tokenize(query):
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for i, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                norm = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[i] / max(self.avg_doc_length, 1.0)
                )
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return scores


class LocalVectorIndex:
    """
    Lightweight local vector baseline using TF-IDF n-gram vectors.

    For a production-style dense setup, replace this component with a sentence
    embedding model or an external vector store. The rest of the pipeline stays
    unchanged.
    """

    def __init__(self, documents: list[str]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.matrix = self.vectorizer.fit_transform(documents) if documents else None

    def score(self, query: str) -> np.ndarray:
        if self.matrix is None:
            return np.array([], dtype=float)
        query_vec = self.vectorizer.transform([query])
        return cosine_similarity(query_vec, self.matrix)[0]


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if math.isclose(lo, hi):
        return np.ones_like(values) if hi > 0 else np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _rank_positions(scores: np.ndarray) -> dict[int, int]:
    order = np.argsort(-scores)
    return {int(idx): rank + 1 for rank, idx in enumerate(order)}


@dataclass
class HybridRetriever:
    entries: list[TranslationMemoryEntry]
    config: PipelineConfig = PipelineConfig()

    def __post_init__(self):
        docs = [entry.source for entry in self.entries]
        self.bm25 = BM25Index(docs)
        self.vector = LocalVectorIndex(docs)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if not self.entries:
            return []

        top_k = top_k or self.config.top_k
        bm25_raw = self.bm25.score(query)
        vector_scores = self.vector.score(query)
        bm25_norm = _minmax(bm25_raw)

        hybrid = (
            self.config.lexical_weight * bm25_norm
            + self.config.vector_weight * vector_scores
        )

        bm25_ranks = _rank_positions(bm25_raw)
        vector_ranks = _rank_positions(vector_scores)
        rrf = np.array([
            1 / (self.config.rrf_k + bm25_ranks[i])
            + 1 / (self.config.rrf_k + vector_ranks[i])
            for i in range(len(self.entries))
        ])

        # Exact normalized text match should be treated as maximally trusted.
        normalized_query = " ".join(_tokenize(query))
        for i, entry in enumerate(self.entries):
            if " ".join(_tokenize(entry.source)) == normalized_query:
                vector_scores[i] = 1.0
                hybrid[i] = 1.0

        order = np.lexsort((-hybrid, -rrf))[::-1]
        order = sorted(range(len(self.entries)), key=lambda i: (rrf[i], hybrid[i]), reverse=True)

        results = []
        for i in order[:top_k]:
            results.append(
                RetrievalResult(
                    entry=self.entries[i],
                    bm25_score=float(bm25_raw[i]),
                    vector_score=float(vector_scores[i]),
                    hybrid_score=float(hybrid[i]),
                    rrf_score=float(rrf[i]),
                )
            )
        return results
