"""Quality Gates

Pass/fail threshold logic for CI/CD quality gates.
"""

from dataclasses import dataclass, field

from atlasrag.src.evaluation.models import Baseline, EvaluationReport


@dataclass(frozen=True)
class QualityGate:
    """A single quality gate definition.

    Args:
        name: Human-readable gate name.
        metric_key: Attribute name on MetricScores.
        threshold: Maximum allowed change (as a fraction, e.g. 0.03 = 3%).
        direction: "min" means lower is worse (e.g. faithfulness),
                   "max" means higher is worse (e.g. latency).
    """

    name: str
    metric_key: str
    threshold: float
    direction: str = "min"  # "min" = drop is bad, "max" = increase is bad


DEFAULT_GATES: list[QualityGate] = [
    QualityGate(
        name="Faithfulness drop",
        metric_key="faithfulness",
        threshold=0.03,
        direction="min",
    ),
    QualityGate(
        name="Citation coverage drop",
        metric_key="citation_coverage",
        threshold=0.05,
        direction="min",
    ),
    QualityGate(
        name="Latency increase",
        metric_key="latency_p95",
        threshold=0.20,
        direction="max",
    ),
]


def check_gates(
    report: EvaluationReport,
    baseline: Baseline | None,
    gates: list[QualityGate] | None = None,
) -> list[dict]:
    """Check quality gates against a baseline.

    Args:
        report: Current evaluation report.
        baseline: Stored baseline. If None, gates are skipped.
        gates: List of gates to check. Defaults to DEFAULT_GATES.

    Returns:
        List of violation dicts with gate name, metric values, and delta.
    """
    if baseline is None:
        return []

    if gates is None:
        gates = DEFAULT_GATES

    violations = []
    for gate in gates:
        current = getattr(report.mean_scores, gate.metric_key)
        base = getattr(baseline.scores, gate.metric_key)

        if base == 0:
            continue

        if gate.direction == "min":
            # Lower current is bad — check if drop exceeds threshold
            drop = base - current
            if drop > gate.threshold:
                violations.append({
                    "gate": gate.name,
                    "metric": gate.metric_key,
                    "current": current,
                    "baseline": base,
                    "delta": drop,
                    "threshold": gate.threshold,
                })
        else:
            # Higher current is bad — check if fractional increase exceeds threshold
            increase = (current - base) / base
            if increase > gate.threshold:
                violations.append({
                    "gate": gate.name,
                    "metric": gate.metric_key,
                    "current": current,
                    "baseline": base,
                    "delta": increase,
                    "threshold": gate.threshold,
                })

    return violations
