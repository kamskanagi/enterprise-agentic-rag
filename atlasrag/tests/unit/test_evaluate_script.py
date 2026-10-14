"""Tests for the evaluation CLI (atlasrag.scripts.evaluate)

The script is the CI quality gate, so it must fail closed: any condition that
prevents a trustworthy comparison has to produce a non-zero exit code.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from atlasrag.scripts import evaluate

GOOD_STATE = {
    "answer": "Employees get 20 days of vacation [1].",
    "context": [{"content": "Employees get 20 days of paid vacation.", "source": "hr.txt"}],
}


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "How many vacation days?",
                "reference_answer": "20 days.",
                "source_documents": ["hr.txt"],
                "key_facts": ["20 days"],
            }
        )
    )
    return str(path)


def _graph_returning(state):
    graph = MagicMock()
    graph.invoke.return_value = state
    return graph


def _run(args, graph):
    with patch("atlasrag.src.agents.graph.create_agent_graph", return_value=graph):
        return evaluate.main(args)


def _save_baseline(dataset, path):
    assert (
        _run(["--dataset", dataset, "--save-baseline", str(path)], _graph_returning(GOOD_STATE))
        == 0
    )


class TestEvaluateScript:
    def test_graph_creation_failure_exits_nonzero(self, dataset):
        with patch(
            "atlasrag.src.agents.graph.create_agent_graph", side_effect=RuntimeError("no LLM")
        ):
            assert evaluate.main(["--dataset", dataset]) == 1

    def test_sample_error_exits_nonzero(self, dataset):
        graph = MagicMock()
        graph.invoke.side_effect = RuntimeError("timeout")
        assert _run(["--dataset", dataset], graph) == 1

    def test_missing_baseline_exits_nonzero(self, dataset, tmp_path):
        missing = tmp_path / "nope.json"
        assert (
            _run(["--dataset", dataset, "--compare-to", str(missing)], _graph_returning(GOOD_STATE))
            == 2
        )

    def test_no_regression_passes(self, dataset, tmp_path):
        baseline = tmp_path / "baseline.json"
        _save_baseline(dataset, baseline)
        assert baseline.exists()
        args = ["--dataset", dataset, "--compare-to", str(baseline), "--ignore-latency"]
        assert _run(args, _graph_returning(GOOD_STATE)) == 0

    def test_correctness_regression_fails(self, dataset, tmp_path):
        baseline = tmp_path / "baseline.json"
        _save_baseline(dataset, baseline)
        wrong = {**GOOD_STATE, "answer": "Employees get 15 days of vacation [1]."}
        args = ["--dataset", dataset, "--compare-to", str(baseline), "--ignore-latency"]
        assert _run(args, _graph_returning(wrong)) == 1

    def test_ignore_latency_skips_latency_gate(self, dataset, tmp_path):
        baseline = tmp_path / "baseline.json"
        _save_baseline(dataset, baseline)
        data = json.loads(baseline.read_text())
        data["scores"]["latency_p95"] = 1e-9  # Any real run is "slower" than this
        baseline.write_text(json.dumps(data))

        args = ["--dataset", dataset, "--compare-to", str(baseline)]
        assert _run(args, _graph_returning(GOOD_STATE)) == 1
        assert _run([*args, "--ignore-latency"], _graph_returning(GOOD_STATE)) == 0
