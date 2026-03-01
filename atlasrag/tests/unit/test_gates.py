"""Tests for Phase 12: Quality Gates

Tests gate pass/fail logic, thresholds, and edge cases.
"""

import pytest

from atlasrag.src.evaluation.gates import DEFAULT_GATES, QualityGate, check_gates
from atlasrag.src.evaluation.models import (
    Baseline,
    EvaluationReport,
    MetricScores,
)


class TestQualityGates:
    def test_gate_pass(self):
        report = EvaluationReport(
            mean_scores=MetricScores(
                faithfulness=0.92,
                citation_coverage=0.88,
                latency_p95=1.2,
            ),
        )
        baseline = Baseline(
            scores=MetricScores(
                faithfulness=0.93,
                citation_coverage=0.87,
                latency_p95=1.1,
            ),
        )
        violations = check_gates(report, baseline)
        assert len(violations) == 0

    def test_gate_fail_faithfulness(self):
        report = EvaluationReport(
            mean_scores=MetricScores(
                faithfulness=0.85,
                citation_coverage=0.90,
                latency_p95=1.0,
            ),
        )
        baseline = Baseline(
            scores=MetricScores(
                faithfulness=0.95,
                citation_coverage=0.90,
                latency_p95=1.0,
            ),
        )
        violations = check_gates(report, baseline)
        faith_violations = [v for v in violations if v["metric"] == "faithfulness"]
        assert len(faith_violations) == 1
        assert faith_violations[0]["delta"] == pytest.approx(0.10)

    def test_gate_fail_latency(self):
        report = EvaluationReport(
            mean_scores=MetricScores(
                faithfulness=0.95,
                citation_coverage=0.90,
                latency_p95=2.0,
            ),
        )
        baseline = Baseline(
            scores=MetricScores(
                faithfulness=0.95,
                citation_coverage=0.90,
                latency_p95=1.0,
            ),
        )
        violations = check_gates(report, baseline)
        latency_violations = [v for v in violations if v["metric"] == "latency_p95"]
        assert len(latency_violations) == 1

    def test_gate_no_baseline_skips(self):
        report = EvaluationReport(
            mean_scores=MetricScores(faithfulness=0.50),
        )
        violations = check_gates(report, None)
        assert len(violations) == 0

    def test_multiple_violations(self):
        report = EvaluationReport(
            mean_scores=MetricScores(
                faithfulness=0.80,
                citation_coverage=0.75,
                latency_p95=3.0,
            ),
        )
        baseline = Baseline(
            scores=MetricScores(
                faithfulness=0.95,
                citation_coverage=0.90,
                latency_p95=1.0,
            ),
        )
        violations = check_gates(report, baseline)
        metrics_violated = {v["metric"] for v in violations}
        assert "faithfulness" in metrics_violated
        assert "citation_coverage" in metrics_violated
        assert "latency_p95" in metrics_violated

    def test_custom_gates(self):
        custom_gates = [
            QualityGate(
                name="Strict faithfulness",
                metric_key="faithfulness",
                threshold=0.01,
                direction="min",
            ),
        ]
        report = EvaluationReport(
            mean_scores=MetricScores(faithfulness=0.93),
        )
        baseline = Baseline(
            scores=MetricScores(faithfulness=0.95),
        )
        violations = check_gates(report, baseline, gates=custom_gates)
        assert len(violations) == 1
