from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medtrans_rag.glossary import Glossary
from medtrans_rag.pipeline import TranslationPipeline, load_translation_memory


def show(result):
    print(f"route: {result.route}")
    print("matched terms:", [f"{t.term_in} -> {t.term_out}" for t in result.matched_terms])
    print("top TM matches:")
    for item in result.retrieved_examples[:3]:
        print(
            f"  vector={item.vector_score:.3f} hybrid={item.hybrid_score:.3f} "
            f"source={item.entry.source}"
        )
    if result.translated_text:
        print("translation:", result.translated_text)
    if "prompt" in result.debug:
        print("\n--- prompt preview ---")
        print(result.debug["prompt"])


def main():
    tm = load_translation_memory(ROOT / "data" / "sample_translation_memory.csv")
    glossary = Glossary.from_csv(ROOT / "data" / "sample_glossary.csv")
    pipeline = TranslationPipeline(tm, glossary)

    print("=== Example 1: high-confidence TM reuse ===")
    result = pipeline.translate(
        "The participant signed the informed consent form.",
        source_language="English",
        target_language="Chinese",
        dry_run=True,
    )
    show(result)

    print("\n=== Example 2: retrieval + glossary + LLM context ===")
    result = pipeline.translate(
        "Please record any adverse event and contact the study site if symptoms become worse.",
        source_language="English",
        target_language="Chinese",
        dry_run=True,
    )
    show(result)


if __name__ == "__main__":
    main()
