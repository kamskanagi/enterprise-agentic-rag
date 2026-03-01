"""Tests for Phase 11: Evaluation Suite

Tests dataset loading, metric computation, runner, baseline I/O, and comparison.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


# ============================================================================
# Dataset Loading
# ============================================================================


class TestLoadDataset:
    def test_load_dataset(self, tmp_path):
        dataset_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps({
                "query": "What is X?",
                "reference_answer": "X is Y.",
                "source_documents": ["doc1.pdf"],
            }),
            json.dumps({
                "query": "What is Z?",
                "reference_answer": "Z is W.",
                "source_documents": ["doc2.pdf"],
            }),
        ]
        dataset_file.write_text("\n".join(lines))

        samples = load_dataset(dataset_file)
        assert len(samples) == 2
        assert samples[0].query == "What is X?"
        assert samples[1].source_documents == ["doc2.pdf"]

    def test_load_dataset_skips_blank_lines(self, tmp_path):
        dataset_file = tmp_path / "test.jsonl"
        content = json.dumps({
            "query": "Q", "reference_answer": "A", "source_documents": []
        })
        dataset_file.write_text(f"{content}\n\n\n{content}\n")

        samples = load_dataset(dataset_file)
        assert len(samples) == 2


# ============================================================================
# Citation Coverage
# ============================================================================


class TestCitationCoverage:
    def test_citation_coverage_full(self):
        answer = "The policy allows remote work [1]. Approval is needed [2]."
        assert compute_citation_coverage(answer) == 1.0

    def test_citation_coverage_partial(self):
        answer = "The policy allows remote work [1]. No citation here."
        assert compute_citation_coverage(answer) == 0.5

    def test_citation_coverage_none(self):
        answer = "No citations anywhere. Just plain text."
        assert compute_citation_coverage(answer) == 0.0

    def test_citation_coverage_empty(self):
        assert compute_citation_coverage("") == 0.0


# ============================================================================
# Faithfulness Proxy
# ============================================================================


class TestFaithfulnessProxy:
    def test_faithfulness_proxy_high_overlap(self):
        answer = "The company allows remote work three days per week."
        context = ["The company allows employees to work remotely up to three days per week."]
        score = compute_faithfulness_proxy(answer, context)
        assert score > 0.5

    def test_faithfulness_proxy_no_overlap(self):
        answer = "Purple elephants dance gracefully."
        context = ["The company policy covers remote work arrangements."]
        score = compute_faithfulness_proxy(answer, context)
        assert score < 0.3

    def test_faithfulness_proxy_empty(self):
        assert compute_faithfulness_proxy("", []) == 0.0
        assert compute_faithfulness_proxy("hello", []) == 0.0
        assert compute_faithfulness_proxy("", ["some context"]) == 0.0


# ============================================================================
# Evaluation Runner
# ============================================================================


class TestEvaluationRunner:
    def test_evaluation_runner_with_mock_graph(self):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "The policy allows remote work [1]. Approval required [2].",
            "context": [
                {"content": "Policy allows remote work arrangements."},
                {"content": "Manager approval required for remote work."},
            ],
        }

        samples = [
            EvaluationSample(
                query="What is the remote work policy?",
                reference_answer="Employees can work remotely.",
                source_documents=["policy.pdf"],
            ),
        ]

        runner = EvaluationRunner()
        report = runner.evaluate(samples, mock_graph)

        assert len(report.results) == 1
        assert report.results[0].generated_answer != ""
        assert report.mean_scores.citation_coverage > 0
        assert report.mean_scores.faithfulness > 0

    def test_evaluation_runner_handles_graph_error(self):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("LLM unavailable")

        samples = [
            EvaluationSample(
                query="Test?",
                reference_answer="Answer.",
                source_documents=[],
            ),
        ]

        runner = EvaluationRunner()
        report = runner.evaluate(samples, mock_graph)

        assert len(report.results) == 1
        assert report.results[0].generated_answer == ""

    def test_report_aggregation(self):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = [
            {
                "answer": "Answer one [1].",
                "context": [{"content": "Answer one source."}],
            },
            {
                "answer": "Answer two. No citations.",
                "context": [{"content": "Unrelated context."}],
            },
        ]

        samples = [
            EvaluationSample(query="Q1", reference_answer="A1", source_documents=[]),
            EvaluationSample(query="Q2", reference_answer="A2", source_documents=[]),
        ]

        runner = EvaluationRunner()
        report = runner.evaluate(samples, mock_graph)

        assert len(report.results) == 2
        # Mean citation coverage: (1.0 + 0.0) / 2 = 0.5
        assert report.mean_scores.citation_coverage == pytest.approx(0.5)


# ============================================================================
# Baseline Save/Load
# ============================================================================


class TestBaseline:
    def test_baseline_save_load_roundtrip(self, tmp_path):
        report = EvaluationReport(
            mean_scores=MetricScores(
                faithfulness=0.92,
                citation_coverage=0.85,
                answer_relevancy=0.78,
                latency_p95=1.5,
            ),
            results=[],
        )

        path = tmp_path / "baseline.json"
        save_baseline(report, path)
        loaded = load_baseline(path)

        assert loaded.scores.faithfulness == pytest.approx(0.92)
        assert loaded.scores.citation_coverage == pytest.approx(0.85)
        assert loaded.scores.latency_p95 == pytest.approx(1.5)


# ============================================================================
# Baseline Comparison
# ============================================================================


class TestBaselineComparison:
    def test_compare_to_baseline_no_regression(self):
        report = EvaluationReport(
            mean_scores=MetricScores(
                faithfulness=0.95,
                citation_coverage=0.90,
                latency_p95=1.0,
            ),
        )
        baseline = Baseline(
            scores=MetricScores(
                faithfulness=0.90,
                citation_coverage=0.85,
                latency_p95=1.5,
            ),
        )

        runner = EvaluationRunner()
        regressions = runner.compare_to_baseline(report, baseline)
        assert len(regressions) == 0

    def test_compare_to_baseline_with_regression(self):
        report = EvaluationReport(
            mean_scores=MetricScores(
                faithfulness=0.80,
                citation_coverage=0.90,
                latency_p95=1.0,
            ),
        )
        baseline = Baseline(
            scores=MetricScores(
                faithfulness=0.95,
                citation_coverage=0.85,
                latency_p95=1.5,
            ),
        )

        runner = EvaluationRunner()
        regressions = runner.compare_to_baseline(report, baseline)
        assert len(regressions) == 1
        assert regressions[0]["metric"] == "faithfulness"

    def test_compare_to_baseline_latency_regression(self):
        report = EvaluationReport(
            mean_scores=MetricScores(latency_p95=3.0),
        )
        baseline = Baseline(
            scores=MetricScores(latency_p95=1.5),
        )

        runner = EvaluationRunner()
        regressions = runner.compare_to_baseline(report, baseline)
        assert any(r["metric"] == "latency_p95" for r in regressions)
