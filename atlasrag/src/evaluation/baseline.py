"""Baseline I/O

Save and load evaluation baselines for regression comparison.
"""

import json
from pathlib import Path

from atlasrag.src.evaluation.models import Baseline, EvaluationReport, MetricScores


def save_baseline(report: EvaluationReport, path: str | Path) -> None:
    """Save an evaluation report's mean scores as a baseline.

    Args:
        report: The evaluation report to extract scores from.
        path: File path to write the baseline JSON.
    """
    baseline = Baseline(
        timestamp=report.timestamp,
        scores=report.mean_scores,
    )
    Path(path).write_text(baseline.model_dump_json(indent=2))


def load_baseline(path: str | Path) -> Baseline:
    """Load a baseline from a JSON file.

    Args:
        path: File path to the baseline JSON.

    Returns:
        Baseline object.
    """
    data = json.loads(Path(path).read_text())
    return Baseline(**data)
