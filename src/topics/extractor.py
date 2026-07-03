"""Topic extraction (from uploaded documents) and grounding classification
(for topics the user types in manually).
"""

from pydantic import ValidationError

from src.config import MAX_CHUNK_CHARS
from src.documents.chunker import chunk_text
from src.llm.base import LLMClient, LLMError
from src.llm.schemas import TopicClassificationResult, TopicExtractionResult
from src.utils.jsonutil import JsonParseError, parse_json_loose

from .prompts import (
    TOPIC_CLASSIFICATION_SYSTEM,
    TOPIC_EXTRACTION_SYSTEM,
    topic_classification_user,
    topic_extraction_user,
)


def dedupe_topics(topics: list[str]) -> list[str]:
    # Remove duplicate topics while preserving original order.
    seen = set()
    result = []
    for topic in topics:
        key = topic.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(topic.strip())
    return result


def extract_topics(doc_texts: dict[str, str], llm: LLMClient, max_topics: int = 20) -> list[str]:
    """Map-reduce topic extraction across every uploaded document.

    Returns an empty list ONLY when the LLM itself ran fine but genuinely
    found nothing extractable (e.g. all documents were too thin) -- callers
    treat that as the "document too thin" edge case. A systemic failure (bad
    API key, unknown model, safety-filter block, network error) is NOT
    swallowed into that same empty-list result -- it's raised as LLMError so
    the real cause reaches the user instead of a misleading "thin document"
    message. A single chunk failing to parse as valid JSON is tolerated
    (one odd chunk shouldn't sink the whole extraction), but if EVERY chunk
    fails to parse, that's also raised rather than silently reported as
    "thin document".
    """
    all_topics: list[str] = []
    total_chunks = 0
    parse_failures = 0

    for text in doc_texts.values():
        for chunk in chunk_text(text, max_chars=MAX_CHUNK_CHARS):
            total_chunks += 1
            # Let LLMError propagate immediately: a bad key/model/network
            # issue will fail identically on every remaining chunk, so retrying
            # it chunk-by-chunk only wastes time and hides the real error.
            raw = llm.complete(
                system=TOPIC_EXTRACTION_SYSTEM,
                user=topic_extraction_user(chunk),
                max_tokens=600,
                temperature=0.3,
            )
            try:
                result = TopicExtractionResult.model_validate(parse_json_loose(raw))
            except (JsonParseError, ValidationError):
                parse_failures += 1
                continue
            all_topics.extend(t.strip() for t in result.topics if t.strip())

    if total_chunks and parse_failures == total_chunks:
        raise LLMError(
            "The model's response could not be parsed as the expected topic JSON on any document chunk "
            "(the provider may not be returning valid JSON, or is wrapping it in unexpected text). "
            "Try a different model, then retry."
        )

    return dedupe_topics(all_topics)[:max_topics]


def _sample_doc_excerpt(doc_texts: dict[str, str], budget_chars: int = 8000) -> str:
    if not doc_texts:
        return ""
    per_doc_budget = max(500, budget_chars // len(doc_texts))
    parts = [text[:per_doc_budget] for text in doc_texts.values()]
    return "\n\n---\n\n".join(parts)


def classify_manual_topics(
    manual_topics: list[str],
    extracted_topics: list[str],
    doc_texts: dict[str, str],
    llm: LLMClient,
) -> tuple[list[str], list[str]]:
    """Split user-typed extra topics into (included, ignored) based on
    whether the source documents support them.
    whether the source documents actually support them.
    """
    extracted_lower = {t.strip().lower() for t in extracted_topics}
    excerpt = _sample_doc_excerpt(doc_texts)

    included: list[str] = []
    ignored: list[str] = []

    for topic in manual_topics:
        topic = topic.strip()
        if not topic:
            continue
        if topic.lower() in extracted_lower:
            included.append(topic)
            continue
        # A systemic LLMError (bad key/model/network) is allowed to propagate --
        # it means the check never actually ran, which is different from the
        # model running fine and concluding "not grounded".
        raw = llm.complete(
            system=TOPIC_CLASSIFICATION_SYSTEM,
            user=topic_classification_user(topic, excerpt),
            max_tokens=400,
            temperature=0.0,
        )
        try:
            result = TopicClassificationResult.model_validate(parse_json_loose(raw))
            grounded = result.grounded
        except (JsonParseError, ValidationError):
            grounded = False  # got a response, but couldn't parse it -> conservatively treat as not grounded

        (included if grounded else ignored).append(topic)

    return included, ignored
