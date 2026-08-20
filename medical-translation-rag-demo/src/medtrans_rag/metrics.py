from __future__ import annotations

from difflib import SequenceMatcher


def normalized_edit_similarity(reference: str, prediction: str) -> float:
    """Character-level similarity in [0, 1]; useful as a lightweight demo metric."""
    return SequenceMatcher(None, reference, prediction).ratio()


def retrieval_hit_at_k(retrieved_sources: list[str], expected_source: str, k: int = 5) -> float:
    return float(expected_source in retrieved_sources[:k])


def terminology_coverage(translation: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    translation_lower = translation.lower()
    hits = sum(term.lower() in translation_lower for term in expected_terms)
    return hits / len(expected_terms)
