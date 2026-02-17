# Evaluation Suite (Phase 11)

**Status:** Planning phase - to be implemented in Phase 11

**Purpose:** Measure and track RAG system quality using RAGAS metrics.

## RAGAS Metrics

RAGAS (Retrieval-Augmented Generation Assessment) provides standardized metrics:

### Faithfulness (Target: >90%)
Measures if the answer is grounded in retrieved sources.

- How many claims in the answer are supported by documents?
- Range: 0.0 (unsupported) to 1.0 (fully grounded)

Example:
- Query: "What's our vacation policy?"
- Answer: "We provide 20 days vacation (Policy p.3) and free coffee (no source)"
- Faithfulness: ~50% (one claim unsupported)

### Answer Relevancy (Target: >85%)
Measures if the answer actually answers the question.

- Does the answer address the user's question?
- Range: 0.0 (irrelevant) to 1.0 (perfectly relevant)

Example:
- Query: "What's our vacation policy?"
- Answer: "Our company was founded in 1995"
- Relevancy: 0% (off-topic)

### Context Precision (Target: >80%)
Measures if retrieved documents are relevant.

- What % of retrieved chunks are relevant to the query?
- Range: 0.0 (all irrelevant) to 1.0 (all relevant)

Example:
- Query: "vacation policy"
- Retrieved: [chunk1 about vacation, chunk2 about vacation, chunk3 about parking]
- Precision: 66% (2 out of 3 relevant)

### Context Recall (Target: >80%)
Measures if retrieval found all relevant documents.

- What % of relevant documents were retrieved?
- Range: 0.0 (missed all) to 1.0 (found all)

Example:
- Query: "vacation policy"
- Relevant docs: 5 total
- Retrieved: 3 out of 5
- Recall: 60%

## Custom Metrics

### Citation Coverage (Target: >95%)
Percentage of sentences that have citations.

```
Answer: "We provide 20 days annually (Policy p.3).
         Additional leave may be approved (Benefits p.5).
         Please contact HR."

Cited: 2/3 sentences = 66%
```

### Latency Metrics
- p50 (median): 50th percentile response time
- p95 (tail): 95th percentile (worst-case performance)
- p99 (very worst): 99th percentile

Target: p95 < 5 seconds

## Key Files (to be implemented in Phase 11)

- `ragas_evaluator.py` - RAGAS integration
- `citation_metrics.py` - Custom citation analysis
- `latency_tracker.py` - Performance monitoring
- `runner.py` - Evaluation orchestration
- `baseline.py` - Baseline tracking and comparison
- `models.py` - Type definitions (EvaluationResult, etc.)

## Golden Dataset Format

Location: `eval/golden_dataset.jsonl`

Format (JSON Lines):
```json
{
  "query": "What's our vacation policy?",
  "reference_answer": "Employees receive 20 days of paid vacation annually",
  "source_documents": [
    {"id": "doc1", "content": "Our vacation policy provides 20 days annually..."},
    {"id": "doc2", "content": "Additional unpaid leave may be requested..."}
  ],
  "expected_citations": ["doc1"]
}
```

## Usage Examples (Phase 11+)

### Run Full Evaluation
```python
from src.evaluation.runner import EvaluationRunner

runner = EvaluationRunner()
results = runner.evaluate(dataset_path="eval/golden_dataset.jsonl")

print(f"Faithfulness: {results.faithfulness:.1%}")
print(f"Relevancy: {results.answer_relevancy:.1%}")
print(f"Precision: {results.context_precision:.1%}")
print(f"Recall: {results.context_recall:.1%}")
print(f"Citation Coverage: {results.citation_coverage:.1%}")
print(f"p95 Latency: {results.latency_p95:.2f}s")
```

### Compare to Baseline
```python
# Compare current metrics to saved baseline
baseline = runner.load_baseline()
current = runner.evaluate(dataset_path="eval/golden_dataset.jsonl")

# Check for regressions
for metric in ['faithfulness', 'citation_coverage']:
    baseline_val = getattr(baseline, metric)
    current_val = getattr(current, metric)
    drop = baseline_val - current_val

    if drop > 0.03:  # More than 3% drop
        print(f"⚠️  {metric} regressed by {drop:.1%}")
```

### Track Metrics Over Time
```python
# Save results for historical tracking
runner.save_results(results, tag="release-2024-02-16")

# Plot trends
runner.plot_metrics_over_time()
```

## Integration with CI/CD

### GitHub Actions (Phase 12)
```yaml
name: Evaluation
on: [pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run evaluation
        run: python scripts/evaluate.py
      - name: Check regressions
        run: python scripts/check_baseline.py
        continue-on-error: false
```

### Failure Conditions
CI fails if any of these happen:
- Faithfulness drops >3%
- Citation coverage drops >5%
- Recall drops >5%
- p95 latency increases >20%

## Configuration Dependencies

Relies on Phase 2 configuration:
- `settings.eval_dataset_path` - Path to golden dataset
- `settings.ragas_metrics` - Which metrics to compute
- `settings.eval_batch_size` - Batch size for evaluation

## Testing (Phase 11+)

```python
import pytest
from src.evaluation.runner import EvaluationRunner

@pytest.fixture
def runner():
    return EvaluationRunner()

def test_faithfulness_metric(runner):
    """Test faithfulness calculation"""
    # Grounded answer
    score = runner.compute_faithfulness(
        answer="20 days (Policy p.3)",
        source="Employees get 20 days annually"
    )
    assert score > 0.9

    # Ungrounded answer
    score = runner.compute_faithfulness(
        answer="Free coffee (no source)",
        source="Employees get 20 days vacation"
    )
    assert score < 0.5

def test_citation_coverage(runner):
    """Test citation extraction"""
    answer = "20 days (Policy). Also unpaid (Benefits)."
    coverage = runner.compute_citation_coverage(answer)
    assert coverage == 1.0  # All sentences cited
```

## Database Schema (to be created in Phase 11)

```sql
CREATE TABLE evaluation_results (
    id UUID PRIMARY KEY,
    evaluation_timestamp TIMESTAMP,
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_precision FLOAT,
    context_recall FLOAT,
    citation_coverage FLOAT,
    latency_p50 FLOAT,
    latency_p95 FLOAT,
    latency_p99 FLOAT,
    metadata JSONB
);

CREATE TABLE baselines (
    id UUID PRIMARY KEY,
    tag VARCHAR(255),  -- e.g., "v1.0", "release-2024-02"
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_precision FLOAT,
    context_recall FLOAT,
    citation_coverage FLOAT,
    latency_p95 FLOAT,
    created_at TIMESTAMP
);
```

## Future Enhancements

- [ ] Human evaluation integration
- [ ] A/B testing framework
- [ ] Prompt optimization evaluation
- [ ] Model comparison (Ollama vs OpenAI vs Gemini)
- [ ] Domain-specific metrics
- [ ] Cost tracking
- [ ] Multi-language evaluation
