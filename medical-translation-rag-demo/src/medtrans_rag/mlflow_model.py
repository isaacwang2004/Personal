from __future__ import annotations

"""Optional MLflow pyfunc wrapper for portfolio demonstration."""

import pandas as pd


def build_mlflow_model_class():
    try:
        import mlflow.pyfunc
    except ImportError as exc:
        raise ImportError("Install MLflow with: pip install -e '.[mlflow]'") from exc

    from .glossary import Glossary
    from .pipeline import TranslationPipeline, load_translation_memory

    class TranslationPyfuncModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            tm = load_translation_memory(context.artifacts["translation_memory"])
            glossary = Glossary.from_csv(context.artifacts["glossary"])
            self.pipeline = TranslationPipeline(tm, glossary)

        def predict(self, context, model_input, params=None):
            rows = []
            for _, row in model_input.iterrows():
                result = self.pipeline.translate(
                    text=str(row["text"]),
                    source_language=str(row["source_language"]),
                    target_language=str(row["target_language"]),
                    dry_run=True,
                )
                rows.append(
                    {
                        "route": result.route,
                        "translated_text": result.translated_text,
                        "matched_terms": ", ".join(t.term_in for t in result.matched_terms),
                    }
                )
            return pd.DataFrame(rows)

    return TranslationPyfuncModel
