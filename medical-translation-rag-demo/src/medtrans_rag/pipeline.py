from __future__ import annotations

import pandas as pd

from .config import PipelineConfig
from .glossary import Glossary
from .llm import Translator
from .retrieval import HybridRetriever
from .schemas import TranslationMemoryEntry, TranslationResult


def load_translation_memory(path: str) -> list[TranslationMemoryEntry]:
    df = pd.read_csv(path).fillna("")
    required = {"source", "target", "source_language", "target_language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing TM columns: {sorted(missing)}")

    return [
        TranslationMemoryEntry(
            source=str(row["source"]),
            target=str(row["target"]),
            source_language=str(row["source_language"]),
            target_language=str(row["target_language"]),
            origin=str(row.get("origin", "tm")),
        )
        for _, row in df.iterrows()
    ]


def build_prompt(
    text: str,
    source_language: str,
    target_language: str,
    retrieved,
    matched_terms,
) -> str:
    examples = "\n".join(
        f"- Source: {r.entry.source}\n  Target: {r.entry.target}"
        for r in retrieved
    ) or "- None"

    terms = "\n".join(
        f"- {t.term_in} -> {t.term_out}" + (f" ({t.note})" if t.note else "")
        for t in matched_terms
    ) or "- None"

    return f"""Translate the source text from {source_language} to {target_language}.

Requirements:
1. Preserve meaning and factual content.
2. Use the approved terminology when applicable.
3. Use retrieved bilingual examples only as contextual guidance.
4. Return only the translated text.

Approved terminology:
{terms}

Retrieved translation-memory examples:
{examples}

Source text:
{text}
""".strip()


class TranslationPipeline:
    def __init__(
        self,
        translation_memory: list[TranslationMemoryEntry],
        glossary: Glossary,
        translator: Translator | None = None,
        config: PipelineConfig | None = None,
    ):
        self.config = config or PipelineConfig()
        self.glossary = glossary
        self.translator = translator
        self.translation_memory = translation_memory

    def _filter_tm(self, source_language: str, target_language: str):
        return [
            item
            for item in self.translation_memory
            if item.source_language.lower() == source_language.lower()
            and item.target_language.lower() == target_language.lower()
        ]

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        dry_run: bool = True,
    ) -> TranslationResult:
        tm_entries = self._filter_tm(source_language, target_language)
        retriever = HybridRetriever(tm_entries, self.config)
        retrieved = retriever.search(text, self.config.top_k)
        terms = self.glossary.find_terms(text, source_language, target_language)

        best = retrieved[0] if retrieved else None
        if best and best.vector_score >= self.config.reuse_threshold:
            return TranslationResult(
                source_text=text,
                translated_text=best.entry.target,
                route="tm_reuse",
                matched_terms=terms,
                retrieved_examples=retrieved,
                debug={"reuse_threshold": self.config.reuse_threshold},
            )

        prompt = build_prompt(
            text=text,
            source_language=source_language,
            target_language=target_language,
            retrieved=retrieved,
            matched_terms=terms,
        )

        if dry_run:
            translated = ""
            route = "dry_run_llm_with_context"
        else:
            if self.translator is None:
                raise ValueError("A translator is required when dry_run=False.")
            translated = self.translator.translate(prompt)
            route = "llm_with_context"

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            route=route,
            matched_terms=terms,
            retrieved_examples=retrieved,
            debug={"prompt": prompt, "reuse_threshold": self.config.reuse_threshold},
        )
