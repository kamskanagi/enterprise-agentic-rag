"""Evaluation Data Models

Pydantic models for evaluation samples, metric scores, results, and baselines.
"""

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EvaluationSample(BaseModel):
    """A single row from the golden dataset."""

    query: str
    reference_answer: str
    source_documents: list[str] = Field(default_factory=list)


class MetricScores(BaseModel):
    """Computed metric scores for a single evaluation or aggregate."""

    faithfulness: float = 0.0
    citation_coverage: float = 0.0
    answer_relevancy: float = 0.0
    latency_p95: float = 0.0


class EvaluationResult(BaseModel):
    """Per-sample evaluation result."""

    sample: EvaluationSample
    generated_answer: str = ""
    scores: MetricScores = Field(default_factory=MetricScores)
    latency_seconds: float = 0.0


class EvaluationReport(BaseModel):
    """Aggregate evaluation report across all samples."""

    timestamp: datetime = Field(default_factory=_utcnow)
    mean_scores: MetricScores = Field(default_factory=MetricScores)
    pass_: bool = Field(default=True, alias="pass")
    results: list[EvaluationResult] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class Baseline(BaseModel):
    """Stored baseline metric scores for regression comparison."""

    timestamp: datetime = Field(default_factory=_utcnow)
    scores: MetricScores = Field(default_factory=MetricScores)
    description: Optional[str] = None
