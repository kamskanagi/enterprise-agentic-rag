# Evaluation Dataset

This directory contains the golden evaluation dataset and baseline metrics for AtlasRAG.

## Golden Dataset

**File:** `golden_dataset.jsonl` (to be created in Phase 11)

**Format:** JSON Lines (one JSON object per line)

```json
{
  "query": "What is our vacation policy?",
  "reference_answer": "Employees receive 20 days of paid vacation annually",
  "source_documents": [
    {
      "id": "doc1",
      "content": "Our vacation policy provides 20 days of paid vacation annually for all full-time employees..."
    },
    {
      "id": "doc2",
      "content": "Additional unpaid leave may be requested with manager approval..."
    }
  ],
  "expected_citations": ["doc1"],
  "difficulty": "easy"
}
```

**Fields:**
- `query`: User's question
- `reference_answer`: Expected answer (for relevancy evaluation)
- `source_documents`: Documents that should be retrieved
- `expected_citations`: Which documents should be cited
- `difficulty`: easy/medium/hard (for benchmarking)

## Dataset Statistics

**To be populated in Phase 11:**
- Total Q&A pairs: [target: 50-150]
- Easy questions: [x]
- Medium questions: [x]
- Hard questions: [x]
- Average query length: [x] tokens
- Average document length: [x] tokens

## Evaluation Baselines

**File:** `baselines.json` (to be created in Phase 11)

Tracks target metrics:
- Faithfulness: >90%
- Answer Relevancy: >85%
- Context Precision: >80%
- Context Recall: >80%
- Citation Coverage: >95%
- p95 Latency: <5s

## How to Create Dataset

### Step 1: Collect Questions
Start with 10-20 questions across different categories:
- Vacation/leave policies
- Remote work
- Health benefits
- Equipment/tools
- Compensation
- Career development

### Step 2: Source Documents
Use actual company documents (PDFs, word docs) that answer these questions.

### Step 3: Write Reference Answers
Create ground-truth answers based on the documents.

### Step 4: Format as JSON Lines
Convert to the JSONL format above.

### Step 5: Validate
Ensure all reference answers are grounded in source documents.

## Testing the Dataset

Run evaluation:
```bash
python scripts/evaluate.py --dataset atlasrag/eval/golden_dataset.jsonl
```

Check coverage:
```bash
python -c "
import jsonlines
with jsonlines.open('eval/golden_dataset.jsonl') as reader:
    data = list(reader)
    print(f'Total Q&A pairs: {len(data)}')
    difficulties = {}
    for item in data:
        d = item.get('difficulty', 'unknown')
        difficulties[d] = difficulties.get(d, 0) + 1
    print(f'Difficulty distribution: {difficulties}')
"
```

## Updates Over Time

- **Phase 11:** Initial dataset (50 Q&A pairs)
- **Phase 12:** Expand to 100+ Q&A pairs
- **Phase 15:** Domain-specific datasets for different use cases

## CI/CD Integration

The evaluation script (Phase 11) will:
1. Load this dataset
2. Run queries through AtlasRAG
3. Compute metrics
4. Compare to baselines
5. Fail PR if regressions detected

See [../src/evaluation/README.md](../src/evaluation/README.md) for details.
