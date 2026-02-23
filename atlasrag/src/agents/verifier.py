"""Verifier Agent Node + Router

Checks that the answer has adequate citation coverage. The router
function decides whether to repair or complete based on verification
result and remaining repair iterations.
"""

import logging
import re

from atlasrag.src.agents.state import AgentState
from atlasrag.src.config import get_settings

logger = logging.getLogger(__name__)


def _count_sentences(text: str) -> int:
    """Count sentences by splitting on sentence-ending punctuation."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def _count_cited_sentences(text: str) -> int:
    """Count sentences that contain at least one citation reference."""
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip() and re.search(r"\[\d+\]", s)])


def verifier_node(state: AgentState) -> dict:
    """Verify citation coverage of the generated answer.

    Args:
        state: Current agent state with ``answer`` and ``citations``.

    Returns:
        Partial state update with ``verification_passed`` and ``confidence``.
    """
    settings = get_settings()
    answer = state.get("answer", "")

    total_sentences = _count_sentences(answer)

    if total_sentences == 0:
        logger.warning("Verifier: answer has no sentences")
        return {"verification_passed": False, "confidence": 0.0}

    cited_sentences = _count_cited_sentences(answer)
    coverage = cited_sentences / total_sentences

    # Determine threshold based on verification mode
    mode = settings.verification_mode
    if mode == "disabled":
        logger.info("Verification disabled, auto-passing")
        return {"verification_passed": True, "confidence": coverage}
    elif mode == "lenient":
        threshold = settings.min_citation_coverage / 2
    else:  # strict
        threshold = settings.min_citation_coverage

    passed = coverage >= threshold

    logger.info(
        "Verifier: coverage=%.2f threshold=%.2f mode=%s passed=%s",
        coverage,
        threshold,
        mode,
        passed,
    )

    return {
        "verification_passed": passed,
        "confidence": coverage,
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
