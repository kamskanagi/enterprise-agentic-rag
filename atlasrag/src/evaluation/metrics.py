"""Evaluation Metrics

Custom metric computation for citation coverage and faithfulness proxy.
"""

import re


def _count_sentences(text: str) -> int:
    """Count sentences by splitting on sentence-ending punctuation."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def _count_cited_sentences(text: str) -> int:
    """Count sentences that contain at least one citation reference."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip() and re.search(r"\[\d+\]", s)])


def compute_citation_coverage(answer: str) -> float:
    """Compute the fraction of sentences that have citation references.

    Args:
        answer: The generated answer text.

    Returns:
        Float between 0.0 and 1.0 representing citation coverage.
    """
    total = _count_sentences(answer)
    if total == 0:
        return 0.0
    cited = _count_cited_sentences(answer)
    return cited / total


def compute_faithfulness_proxy(answer: str, context: list[str]) -> float:
    """Compute a simple token-overlap faithfulness proxy.

    For each sentence in the answer, we check what fraction of its
    non-trivial tokens appear in the combined context. The final score
    is the mean overlap across all sentences.

    Args:
        answer: The generated answer text.
        context: List of context passage strings.

    Returns:
        Float between 0.0 and 1.0 representing faithfulness.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    if not sentences:
        return 0.0

    # Build token set from all context
    context_tokens = set()
    for passage in context:
        context_tokens.update(word.lower() for word in re.findall(r"\b\w{3,}\b", passage))

    if not context_tokens:
        return 0.0

    sentence_scores = []
    for sentence in sentences:
        # Remove citation markers before tokenizing
        clean = re.sub(r"\[\d+\]", "", sentence)
        tokens = [word.lower() for word in re.findall(r"\b\w{3,}\b", clean)]
        if not tokens:
            sentence_scores.append(0.0)
            continue
        overlap = sum(1 for t in tokens if t in context_tokens)
        sentence_scores.append(overlap / len(tokens))

    return sum(sentence_scores) / len(sentence_scores)


_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b")


def _normalize(text: str) -> str:
    """Lowercase, spell numbers as digits, and collapse whitespace."""
    text = _NUMBER_WORD_RE.sub(lambda m: _NUMBER_WORDS[m.group(1)], text.lower())
    return " ".join(text.split())


def compute_answer_correctness(answer: str, key_facts: list[str]) -> float | None:
    """Compute the fraction of a sample's key facts that appear in the answer.

    Unlike the faithfulness proxy, this checks the answer against the golden
    reference, so a fluent answer copied from the wrong passage scores zero.

    Args:
        answer: The generated answer text.
        key_facts: Short phrases a correct answer must contain (e.g. "20 days").

    Returns:
        Float between 0.0 and 1.0, or None when the sample defines no key facts.
    """
    if not key_facts:
        return None
    normalized = _normalize(answer)
    found = sum(1 for fact in key_facts if _normalize(fact) in normalized)
    return found / len(key_facts)


def compute_retrieval_hit_rate(
    retrieved_sources: list[str], expected_sources: list[str]
) -> float | None:
    """Compute the fraction of expected source documents present in the context.

    Args:
        retrieved_sources: Source filenames of the retrieved context chunks.
        expected_sources: Source filenames the answer should be grounded in.

    Returns:
        Float between 0.0 and 1.0, or None when the sample defines no sources.
    """
    if not expected_sources:
        return None
    retrieved = set(retrieved_sources)
    return sum(1 for source in expected_sources if source in retrieved) / len(expected_sources)
