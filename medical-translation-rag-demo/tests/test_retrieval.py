from medtrans_rag.retrieval import HybridRetriever
from medtrans_rag.schemas import TranslationMemoryEntry


def test_exact_match_ranks_first():
    entries = [
        TranslationMemoryEntry("Alpha beta.", "甲乙。", "English", "Chinese"),
        TranslationMemoryEntry("Completely different sentence.", "不同句子。", "English", "Chinese"),
    ]
    results = HybridRetriever(entries).search("Alpha beta.", top_k=2)
    assert results[0].entry.source == "Alpha beta."
    assert results[0].vector_score == 1.0
