import pandas as pd

from medtrans_rag.glossary import Glossary
from medtrans_rag.pipeline import TranslationPipeline
from medtrans_rag.schemas import TranslationMemoryEntry


def test_tm_reuse_route():
    tm = [
        TranslationMemoryEntry(
            source="The participant signed the informed consent form.",
            target="受试者签署了知情同意书。",
            source_language="English",
            target_language="Chinese",
        )
    ]
    glossary = Glossary(pd.DataFrame([
        {
            "term_in": "informed consent",
            "term_out": "知情同意",
            "source_language": "English",
            "target_language": "Chinese",
            "note": "",
        }
    ]))
    pipeline = TranslationPipeline(tm, glossary)
    result = pipeline.translate(
        "The participant signed the informed consent form.", "English", "Chinese"
    )
    assert result.route == "tm_reuse"
    assert result.translated_text == "受试者签署了知情同意书。"


def test_dry_run_builds_context_prompt():
    tm = [
        TranslationMemoryEntry(
            source="Please report any adverse event after the visit.",
            target="请在访视后报告任何不良事件。",
            source_language="English",
            target_language="Chinese",
        )
    ]
    glossary = Glossary(pd.DataFrame([
        {
            "term_in": "adverse event",
            "term_out": "不良事件",
            "source_language": "English",
            "target_language": "Chinese",
            "note": "",
        }
    ]))
    pipeline = TranslationPipeline(tm, glossary)
    result = pipeline.translate(
        "Record the adverse event in the report.", "English", "Chinese", dry_run=True
    )
    assert result.route == "dry_run_llm_with_context"
    assert "adverse event -> 不良事件" in result.debug["prompt"]
