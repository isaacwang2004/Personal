from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int = 5
    reuse_threshold: float = 0.90
    lexical_weight: float = 0.45
    vector_weight: float = 0.55
    rrf_k: int = 60
