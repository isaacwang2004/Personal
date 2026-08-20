from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ElasticsearchHit:
    source: str
    target: str
    lexical_rank: int | None
    vector_rank: int | None
    rrf_score: float


class ElasticsearchHybridBackend:
    """
    Optional Elasticsearch backend for a translation-memory index.

    The module intentionally receives embeddings from the caller so that the
    embedding provider is decoupled from storage and no provider credentials
    are required here.
    """

    def __init__(self, client, index_name: str, rrf_k: int = 60):
        self.client = client
        self.index_name = index_name
        self.rrf_k = rrf_k

    @classmethod
    def from_env(cls, index_name: str | None = None) -> "ElasticsearchHybridBackend":
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise ImportError(
                "Install the optional dependency with: pip install -e '.[elasticsearch]'"
            ) from exc

        url = os.getenv("ELASTICSEARCH_URL", "")
        username = os.getenv("ELASTICSEARCH_USERNAME", "")
        password = os.getenv("ELASTICSEARCH_PASSWORD", "")
        if not url:
            raise ValueError("Set ELASTICSEARCH_URL in the environment.")

        auth = (username, password) if username and password else None
        client = Elasticsearch(url, basic_auth=auth)
        return cls(client, index_name or os.getenv("ELASTICSEARCH_INDEX", "translation-memory-demo"))

    def create_index(self, embedding_dims: int) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        self.client.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "source": {"type": "text"},
                    "target": {"type": "text"},
                    "source_language": {"type": "keyword"},
                    "target_language": {"type": "keyword"},
                    "source_vector": {
                        "type": "dense_vector",
                        "dims": embedding_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        )

    def index_entries(self, entries: Iterable[dict]) -> None:
        from elasticsearch.helpers import bulk

        actions = [
            {
                "_index": self.index_name,
                "_source": entry,
            }
            for entry in entries
        ]
        bulk(self.client, actions)
        self.client.indices.refresh(index=self.index_name)

    def search(
        self,
        query: str,
        query_embedding: list[float],
        source_language: str,
        target_language: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[ElasticsearchHit]:
        filters = [
            {"term": {"source_language": source_language}},
            {"term": {"target_language": target_language}},
        ]

        lexical = self.client.search(
            index=self.index_name,
            size=candidate_k,
            query={
                "bool": {
                    "must": [{"match": {"source": {"query": query, "fuzziness": "AUTO"}}}],
                    "filter": filters,
                }
            },
        )["hits"]["hits"]

        vector = self.client.search(
            index=self.index_name,
            size=candidate_k,
            knn={
                "field": "source_vector",
                "query_vector": query_embedding,
                "k": candidate_k,
                "num_candidates": max(candidate_k * 3, 50),
                "filter": {"bool": {"filter": filters}},
            },
        )["hits"]["hits"]

        lexical_rank = {hit["_id"]: rank + 1 for rank, hit in enumerate(lexical)}
        vector_rank = {hit["_id"]: rank + 1 for rank, hit in enumerate(vector)}
        payload = {hit["_id"]: hit["_source"] for hit in lexical + vector}

        ids = set(lexical_rank) | set(vector_rank)
        fused = []
        for doc_id in ids:
            score = 0.0
            if doc_id in lexical_rank:
                score += 1.0 / (self.rrf_k + lexical_rank[doc_id])
            if doc_id in vector_rank:
                score += 1.0 / (self.rrf_k + vector_rank[doc_id])
            source = payload[doc_id]
            fused.append(
                ElasticsearchHit(
                    source=source["source"],
                    target=source["target"],
                    lexical_rank=lexical_rank.get(doc_id),
                    vector_rank=vector_rank.get(doc_id),
                    rrf_score=float(score),
                )
            )

        return sorted(fused, key=lambda x: x.rrf_score, reverse=True)[:top_k]
