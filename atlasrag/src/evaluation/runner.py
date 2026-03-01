"""Evaluation Runner

Orchestrates running golden dataset samples through the agent graph
and computing aggregate metrics.
"""

import json
import logging
import time
from pathlib import Path

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

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> list[EvaluationSample]:
    """Load evaluation samples from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of EvaluationSample objects.
    """
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                samples.append(EvaluationSample(**data))
    return samples


class EvaluationRunner:
    """Runs evaluation samples through the agent pipeline and collects metrics."""

    def evaluate(
        self, samples: list[EvaluationSample], graph
    ) -> EvaluationReport:
        """Run each sample through the graph and compute metrics.

        Args:
            samples: List of evaluation samples.
            graph: Compiled LangGraph agent graph (or any callable
                   accepting ``{"query": str}`` and returning state dict).

        Returns:
            EvaluationReport with per-sample and aggregate results.
        """
        results: list[EvaluationResult] = []
        latencies: list[float] = []

        for sample in samples:
            start = time.perf_counter()
            try:
                state = graph.invoke({"query": sample.query})
            except Exception as e:
                logger.warning("Sample failed: %s — %s", sample.query[:50], e)
                results.append(EvaluationResult(sample=sample))
                latencies.append(0.0)
                continue
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            answer = state.get("answer", "")
            context_chunks = state.get("context", [])
            context_texts = [
                c.get("content", "") if isinstance(c, dict) else str(c)
                for c in context_chunks
            ]

            citation_cov = compute_citation_coverage(answer)
            faithfulness = compute_faithfulness_proxy(answer, context_texts)

            result = EvaluationResult(
                sample=sample,
                generated_answer=answer,
                scores=MetricScores(
                    faithfulness=faithfulness,
                    citation_coverage=citation_cov,
                    answer_relevancy=0.0,
                    latency_p95=0.0,
                ),
                latency_seconds=elapsed,
            )
            results.append(result)

        # Aggregate
        mean_scores = self._aggregate_scores(results, latencies)
        return EvaluationReport(
            mean_scores=mean_scores,
            results=results,
        )

    def compare_to_baseline(
        self, report: EvaluationReport, baseline: Baseline
    ) -> list[dict]:
        """Compare report to baseline and return list of regressions.

        Args:
            report: Current evaluation report.
            baseline: Stored baseline to compare against.

        Returns:
            List of dicts describing regressions (metric, current, baseline, delta).
        """
        regressions = []
        for metric in ("faithfulness", "citation_coverage", "answer_relevancy"):
            current = getattr(report.mean_scores, metric)
            base = getattr(baseline.scores, metric)
            if base > 0 and current < base:
                delta = base - current
                regressions.append({
                    "metric": metric,
                    "current": current,
                    "baseline": base,
                    "delta": delta,
                })

        # Latency: regression if current is higher
        if (
            baseline.scores.latency_p95 > 0
            and report.mean_scores.latency_p95 > baseline.scores.latency_p95
        ):
            regressions.append({
                "metric": "latency_p95",
                "current": report.mean_scores.latency_p95,
                "baseline": baseline.scores.latency_p95,
                "delta": report.mean_scores.latency_p95
                - baseline.scores.latency_p95,
            })

        return regressions

    @staticmethod
    def _aggregate_scores(
        results: list[EvaluationResult], latencies: list[float]
    ) -> MetricScores:
        """Compute mean scores across all results."""
        if not results:
            return MetricScores()

        n = len(results)
        mean_faith = sum(r.scores.faithfulness for r in results) / n
        mean_citation = sum(r.scores.citation_coverage for r in results) / n
        mean_relevancy = sum(r.scores.answer_relevancy for r in results) / n

        # p95 latency
        valid_latencies = sorted(lat for lat in latencies if lat > 0)
        if valid_latencies:
            idx = int(len(valid_latencies) * 0.95)
            idx = min(idx, len(valid_latencies) - 1)
            p95 = valid_latencies[idx]
        else:
            p95 = 0.0

        return MetricScores(
            faithfulness=mean_faith,
            citation_coverage=mean_citation,
            answer_relevancy=mean_relevancy,
            latency_p95=p95,
        )
