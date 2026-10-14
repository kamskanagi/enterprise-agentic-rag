# Evaluation Dataset

Golden dataset and baseline used by the CI evaluation gate (`evaluate` job in
`.github/workflows/ci.yml`).

## Golden Dataset

**File:** `golden_dataset.jsonl` — 12 questions answerable from `data/samples/`
(HR handbook, security policy, engineering onboarding). One question spans two documents.

```json
{
  "query": "How many days of paid vacation do full-time employees receive per year?",
  "reference_answer": "Full-time employees receive 20 days of paid vacation per year, accruing at 1.67 days per month.",
  "source_documents": ["hr-handbook.txt"],
  "key_facts": ["20 days"]
}
```

**Fields:**
- `query`: User question sent to the agent graph
- `reference_answer`: Ground-truth answer, written from the source document
- `source_documents`: Filenames in `data/samples/` the answer must come from (retrieval hit rate)
- `key_facts`: Short phrases a correct answer must contain (answer correctness). Matching is
  case-insensitive and treats number words as digits ("three days" matches "3 days").

`test_evaluation.py::TestGoldenDataset` checks that every source document exists and that each
reference answer contains all of its key facts.

## Baseline

**File:** `baseline.json` — mean scores the gate compares against. Thresholds live in
`atlasrag/src/evaluation/gates.py`.

To re-baseline after an intentional change, take the `candidate-baseline` artifact (also printed in
the CI job summary) from a run of the change and commit it here, so the new scores are reviewed in
the PR.

## Adding Questions

1. Add or reuse a document in `data/samples/`.
2. Write the question, a reference answer quoting the document's facts, and `key_facts` that would
   be absent from a wrong answer (prefer exact figures: "14 characters", "$2,000").
3. Run `pytest atlasrag/tests/unit/test_evaluation.py`, then re-baseline.
