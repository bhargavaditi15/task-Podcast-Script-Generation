"""Pydantic contracts for the JSON the LLM is asked to return.

Every prompt that expects structured JSON back (topic extraction, topic
grounding classification) validates the parsed response against one of these
models instead of ad-hoc isinstance()/dict.get() checks. A malformed or
partial response raises pydantic.ValidationError, which callers treat the
same way they already treat a JSON parse failure -- skip/fall back rather
than crash.
"""

from pydantic import BaseModel, Field


class TopicExtractionResult(BaseModel):
    # Structure expected from the topic extraction prompt.
    topics: list[str] = Field(default_factory=list)


class TopicClassificationResult(BaseModel):
    # Structure expected from the manual topic grounding prompt.
    grounded: bool
    reason: str = ""
