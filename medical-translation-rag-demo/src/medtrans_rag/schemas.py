from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TranslationMemoryEntry:
    source: str
    target: str
    source_language: str
    target_language: str
    origin: str = "tm"


@dataclass(frozen=True)
class RetrievalResult:
    entry: TranslationMemoryEntry
    bm25_score: float
    vector_score: float
    hybrid_score: float
    rrf_score: float


@dataclass(frozen=True)
class GlossaryMatch:
    term_in: str
    term_out: str
    note: str = ""


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    route: str
    matched_terms: list[GlossaryMatch] = field(default_factory=list)
    retrieved_examples: list[RetrievalResult] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
