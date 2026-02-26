"""Verifier Agent Node + Router

Checks that the answer has adequate citation coverage. The router
function decides whether to repair or complete based on verification
result and remaining repair iterations.

Phase 8 enhancements:
- Per-paragraph citation checks
- Citation reference validation (cited [N] must exist in context)
- Unsupported claim flagging
- LLM-based contradiction detection (strict mode only)
- Weighted confidence scoring
"""

import logging
import re
from typing import Dict, List, Optional

from atlasrag.src.agents.state import AgentState
from atlasrag.src.config import get_settings

logger = logging.getLogger(__name__)


# ============================================================================
# Sentence-level helpers (Phase 7 — kept for backward compat)
# ============================================================================


def _count_sentences(text: str) -> int:
    """Count sentences by splitting on sentence-ending punctuation."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def _count_cited_sentences(text: str) -> int:
    """Count sentences that contain at least one citation reference."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip() and re.search(r"\[\d+\]", s)])


# ============================================================================
# Phase 8 helpers
# ============================================================================


def _split_paragraphs(text: str) -> List[str]:
    """Split text on double newlines into non-empty paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _check_paragraph_citations(text: str) -> List[Dict]:
    """Check each paragraph has at least one citation reference.

    Returns a list of issue dicts for paragraphs missing citations.
    """
    paragraphs = _split_paragraphs(text)
    issues: List[Dict] = []
    for i, paragraph in enumerate(paragraphs):
        if not re.search(r"\[\d+\]", paragraph):
            issues.append({
                "type": "paragraph_missing_citation",
                "message": f"Paragraph {i + 1} has no citation references",
                "severity": "warning",
            })
    return issues


def _validate_citation_references(
    citations: List[str], context: List[Dict]
) -> List[Dict]:
    """Ensure each cited [N] maps to an existing context source.

    Citations are 1-indexed references like [1], [2], etc.
    Context is a list of chunks — citation [N] maps to context[N-1].
    """
    issues: List[Dict] = []
    num_sources = len(context)
    for citation in citations:
        match = re.match(r"\[(\d+)\]", citation)
        if match:
            ref_num = int(match.group(1))
            if ref_num < 1 or ref_num > num_sources:
                issues.append({
                    "type": "invalid_citation_reference",
                    "message": f"Citation {citation} references source {ref_num} but only {num_sources} sources available",
                    "severity": "error",
                })
    return issues


def _find_unsupported_claims(text: str) -> List[str]:
    """Extract sentences that have no citation reference."""
    sentences = re.split(r"[.!?]+", text)
    unsupported = []
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped and not re.search(r"\[\d+\]", stripped):
            unsupported.append(stripped)
    return unsupported


def _detect_contradictions(
    context: List[Dict], llm=None
) -> List[Dict]:
    """Use LLM to check if context sources contradict each other.

    Only called in strict mode when enable_contradiction_detection is True.
    Returns a list of contradiction issue dicts.
    """
    if llm is None or len(context) < 2:
        return []

    source_texts = []
    for i, chunk in enumerate(context):
        source_texts.append(f"Source [{i + 1}]: {chunk.get('content', '')}")
    sources_block = "\n\n".join(source_texts)

    prompt = (
        "Analyze the following sources for contradictions. "
        "If any two sources directly contradict each other, list each contradiction. "
        "If there are no contradictions, respond with exactly: NO_CONTRADICTIONS\n\n"
        f"{sources_block}\n\n"
        "Response:"
    )

    try:
        response = llm.generate(prompt)
        content = response.content.strip()
        if content == "NO_CONTRADICTIONS":
            return []
        return [{
            "type": "source_contradiction",
            "message": content,
            "severity": "warning",
        }]
    except Exception as e:
        logger.warning("Contradiction detection failed: %s", e)
        return []


# ============================================================================
# Main verifier node
# ============================================================================


def verifier_node(state: AgentState, llm=None) -> dict:
    """Verify citation coverage of the generated answer.

    Args:
        state: Current agent state with ``answer``, ``citations``, and ``context``.
        llm: Optional LLM provider for contradiction detection.

    Returns:
        Partial state update with ``verification_passed``, ``confidence``,
        ``verification_issues``, and ``unsupported_claims``.
    """
    settings = get_settings()
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    context = state.get("context", [])

    total_sentences = _count_sentences(answer)

    if total_sentences == 0:
        logger.warning("Verifier: answer has no sentences")
        return {
            "verification_passed": False,
            "confidence": 0.0,
            "verification_issues": [{
                "type": "empty_answer",
                "message": "Answer contains no sentences",
                "severity": "error",
            }],
            "unsupported_claims": [],
        }

    # Sentence-level coverage (Phase 7 baseline)
    cited_sentences = _count_cited_sentences(answer)
    sentence_coverage = cited_sentences / total_sentences

    mode = settings.verification_mode

    if mode == "disabled":
        logger.info("Verification disabled, auto-passing")
        return {
            "verification_passed": True,
            "confidence": sentence_coverage,
            "verification_issues": [],
            "unsupported_claims": [],
        }

    # Collect all issues
    all_issues: List[Dict] = []

    # 1. Paragraph citation checks
    paragraph_score = 1.0
    if settings.paragraph_citation_required:
        paragraph_issues = _check_paragraph_citations(answer)
        all_issues.extend(paragraph_issues)
        paragraphs = _split_paragraphs(answer)
        if paragraphs:
            paragraph_score = 1.0 - len(paragraph_issues) / len(paragraphs)
    else:
        paragraph_score = 1.0

    # 2. Citation reference validation
    ref_issues = _validate_citation_references(citations, context)
    all_issues.extend(ref_issues)
    ref_score = 1.0
    if citations:
        ref_score = 1.0 - len(ref_issues) / len(citations)

    # 3. Unsupported claims
    unsupported = _find_unsupported_claims(answer)

    # 4. Contradiction detection (strict mode only)
    if (
        mode == "strict"
        and settings.enable_contradiction_detection
        and llm is not None
    ):
        contradiction_issues = _detect_contradictions(context, llm=llm)
        all_issues.extend(contradiction_issues)

    # Weighted confidence: coverage 50%, paragraph 30%, valid refs 20%
    confidence = (
        sentence_coverage * 0.5
        + paragraph_score * 0.3
        + ref_score * 0.2
    )

    # Determine threshold
    if mode == "lenient":
        threshold = settings.min_citation_coverage / 2
    else:  # strict
        threshold = settings.min_citation_coverage

    passed = confidence >= threshold

    logger.info(
        "Verifier: confidence=%.2f (coverage=%.2f para=%.2f refs=%.2f) "
        "threshold=%.2f mode=%s passed=%s issues=%d",
        confidence,
        sentence_coverage,
        paragraph_score,
        ref_score,
        threshold,
        mode,
        passed,
        len(all_issues),
    )

    return {
        "verification_passed": passed,
        "confidence": confidence,
        "verification_issues": all_issues,
        "unsupported_claims": unsupported,
    }


def should_repair(state: AgentState) -> str:
    """Route to repair loop or completion.

    Args:
        state: Current agent state.

    Returns:
        ``"repair"`` if verification failed and repairs remain,
        ``"complete"`` otherwise.
    """
    settings = get_settings()

    if state.get("verification_passed", False):
        return "complete"

    repair_iterations = state.get("repair_iterations", 0)
    if repair_iterations < settings.max_repair_iterations:
        logger.info(
            "Routing to repair (iteration %d/%d)",
            repair_iterations + 1,
            settings.max_repair_iterations,
        )
        return "repair"

    logger.warning(
        "Max repair iterations (%d) reached, completing with current answer",
        settings.max_repair_iterations,
    )
    return "complete"
