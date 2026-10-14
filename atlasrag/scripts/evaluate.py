#!/usr/bin/env python3
"""Evaluation Script

Runs the golden dataset evaluation suite and optionally compares to a baseline.

This is the CI quality gate, so it fails closed: if the agent graph cannot be
built, any sample errors, the baseline is missing, or a quality gate is
violated, it exits non-zero.

Usage:
    python -m atlasrag.scripts.evaluate
    python -m atlasrag.scripts.evaluate --dataset path/to/dataset.jsonl
    python -m atlasrag.scripts.evaluate --save-baseline baseline.json
    python -m atlasrag.scripts.evaluate --compare-to baseline.json --ignore-latency

Exit codes:
    0  evaluation ran and no gate was violated
    1  graph unavailable, a sample errored, or a gate was violated
    2  --compare-to baseline file not found
"""

import argparse
import sys
from pathlib import Path

from atlasrag.src.evaluation.baseline import load_baseline, save_baseline
from atlasrag.src.evaluation.gates import DEFAULT_GATES, check_gates
from atlasrag.src.evaluation.runner import EvaluationRunner, load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AtlasRAG Evaluation Suite")
    parser.add_argument(
        "--dataset",
        default="atlasrag/eval/golden_dataset.jsonl",
        help="Path to golden dataset JSONL file",
    )
    parser.add_argument(
        "--save-baseline",
        default=None,
        help="Save evaluation results as a baseline to this path",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Fail if results regress past the quality gates vs this baseline",
    )
    parser.add_argument(
        "--ignore-latency",
        action="store_true",
        help="Skip the latency gate (baselines from another machine are not comparable)",
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("AtlasRAG Evaluation Suite")
    print("=" * 70)

    if args.compare_to and not Path(args.compare_to).exists():
        print(f"\nBaseline not found: {args.compare_to}")
        print("Create one with --save-baseline before gating on it.")
        return 2

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    samples = load_dataset(args.dataset)
    print(f"Loaded {len(samples)} evaluation samples")

    # Create graph
    print("\nInitializing agent graph...")
    try:
        from atlasrag.src.agents.graph import create_agent_graph

        graph = create_agent_graph()
    except Exception as e:
        print(f"ERROR: Could not create agent graph ({e})")
        return 1

    # Run evaluation
    runner = EvaluationRunner()
    print(f"\nRunning evaluation on {len(samples)} samples...")
    report = runner.evaluate(samples, graph)

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Answer Correctness: {report.mean_scores.answer_correctness:.2%}")
    print(f"  Retrieval Hit Rate: {report.mean_scores.retrieval_hit_rate:.2%}")
    print(f"  Faithfulness:       {report.mean_scores.faithfulness:.2%}")
    print(f"  Citation Coverage:  {report.mean_scores.citation_coverage:.2%}")
    print(f"  Latency (p95):      {report.mean_scores.latency_p95:.3f}s")

    errors = [r for r in report.results if r.error]
    if errors:
        print(f"\nERROR: {len(errors)} of {len(samples)} samples failed to run:")
        for r in errors:
            print(f"  - {r.sample.query[:60]}: {r.error}")
        return 1

    # Save baseline
    if args.save_baseline:
        save_baseline(report, args.save_baseline)
        print(f"\nBaseline saved to: {args.save_baseline}")

    # Compare to baseline
    if args.compare_to:
        print(f"\nChecking quality gates against baseline: {args.compare_to}")
        gates = DEFAULT_GATES
        if args.ignore_latency:
            gates = [g for g in gates if g.metric_key != "latency_p95"]
        violations = check_gates(report, load_baseline(args.compare_to), gates=gates)

        if violations:
            print("\nQUALITY GATE FAILED:")
            for v in violations:
                print(
                    f"  {v['gate']}: {v['current']:.4f} vs baseline {v['baseline']:.4f} "
                    f"(change {v['delta']:.4f}, allowed {v['threshold']:.4f})"
                )
            return 1
        print("\nAll quality gates passed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
