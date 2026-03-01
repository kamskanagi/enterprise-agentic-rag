#!/usr/bin/env python3
"""Evaluation Script

Runs the golden dataset evaluation suite and optionally compares to a baseline.

Usage:
    python -m atlasrag.scripts.evaluate
    python -m atlasrag.scripts.evaluate --dataset path/to/dataset.jsonl
    python -m atlasrag.scripts.evaluate --save-baseline baseline.json
    python -m atlasrag.scripts.evaluate --compare-to baseline.json
"""

import argparse
import sys

from atlasrag.src.evaluation.baseline import load_baseline, save_baseline
from atlasrag.src.evaluation.runner import EvaluationRunner, load_dataset


def main():
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
        help="Compare results against a stored baseline",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AtlasRAG Evaluation Suite")
    print("=" * 70)

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
        print(f"Warning: Could not create agent graph ({e})")
        print("Running in dry-run mode with no graph execution")
        graph = None

    if graph is None:
        print("\nNo graph available. Exiting.")
        sys.exit(0)

    # Run evaluation
    runner = EvaluationRunner()
    print(f"\nRunning evaluation on {len(samples)} samples...")
    report = runner.evaluate(samples, graph)

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Faithfulness:       {report.mean_scores.faithfulness:.2%}")
    print(f"  Citation Coverage:  {report.mean_scores.citation_coverage:.2%}")
    print(f"  Answer Relevancy:   {report.mean_scores.answer_relevancy:.2%}")
    print(f"  Latency (p95):      {report.mean_scores.latency_p95:.3f}s")

    # Save baseline
    if args.save_baseline:
        save_baseline(report, args.save_baseline)
        print(f"\nBaseline saved to: {args.save_baseline}")

    # Compare to baseline
    exit_code = 0
    if args.compare_to:
        print(f"\nComparing to baseline: {args.compare_to}")
        baseline = load_baseline(args.compare_to)
        regressions = runner.compare_to_baseline(report, baseline)

        if regressions:
            print("\nREGRESSIONS DETECTED:")
            for reg in regressions:
                print(
                    f"  {reg['metric']}: {reg['current']:.4f} "
                    f"(baseline: {reg['baseline']:.4f}, delta: -{reg['delta']:.4f})"
                )
            exit_code = 1
        else:
            print("\nNo regressions found.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
