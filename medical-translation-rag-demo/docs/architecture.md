# Architecture Notes

## 1. Translation memory retrieval

The public demo separates retrieval into two signals:

- **BM25** for lexical overlap;
- **vector similarity** for semantic/phrase similarity.

The two rankings are fused with reciprocal-rank fusion (RRF), while the vector score is retained separately for high-confidence routing.

## 2. Confidence routing

A configurable threshold determines whether the pipeline can reuse a trusted TM translation directly. Lower-confidence inputs instead use retrieved TM entries as context for generation.

This avoids forcing the LLM to regenerate text when an approved historical translation already provides a strong match.

## 3. Terminology injection

The glossary matcher:

1. filters terms by language pair;
2. prioritizes longer terms to reduce nested-term collisions;
3. returns approved translations and optional notes;
4. passes them to the prompt as explicit constraints.

## 4. LLM layer

The LLM client is intentionally provider-neutral and environment-driven. No credentials or internal endpoints are stored in the repository.

## 5. Production extensions

A production implementation could replace the local vector baseline with:

- sentence-transformer embeddings;
- managed embedding APIs;
- Elasticsearch/OpenSearch kNN;
- a dedicated vector database.

The retrieval and pipeline interfaces are intentionally small so the backend can be swapped without changing the routing logic.
