"""Example only. Logs the dry-run portfolio pipeline as an MLflow pyfunc model."""
from pathlib import Path

import mlflow

from medtrans_rag.mlflow_model import build_mlflow_model_class

ROOT = Path(__file__).resolve().parents[1]
TranslationPyfuncModel = build_mlflow_model_class()

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="translation_pipeline",
        python_model=TranslationPyfuncModel(),
        artifacts={
            "translation_memory": str(ROOT / "data" / "sample_translation_memory.csv"),
            "glossary": str(ROOT / "data" / "sample_glossary.csv"),
        },
        input_example={
            "text": "The participant signed the informed consent form.",
            "source_language": "English",
            "target_language": "Chinese",
        },
    )
