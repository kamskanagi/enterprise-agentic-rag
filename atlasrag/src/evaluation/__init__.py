"""Evaluation Suite

Quality metrics, evaluation runner, and baseline comparison for AtlasRAG.
"""

from atlasrag.src.evaluation.baseline import load_baseline, save_baseline
from atlasrag.src.evaluation.metrics import (
    compute_citation_coverage,
    compute_faithfulness_proxy,
)
from atlasrag.src.evaluation.models import (
    Baseline,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
    MetricScores,
)
from atlasrag.src.evaluation.runner import EvaluationRunner, load_dataset

__all__ = [
    "compute_citation_coverage",
    "compute_faithfulness_proxy",
    "EvaluationSample",
    "MetricScores",
    "EvaluationResult",
    "EvaluationReport",
    "Baseline",
    "EvaluationRunner",
    "load_dataset",
    "save_baseline",
    "load_baseline",
]
