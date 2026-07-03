TOPIC_EXTRACTION_SYSTEM = """# task: topic_extraction
You are a podcast producer's research assistant. You read source material and
pull out concrete, specific discussion-worthy topics that are actually
supported by the text -- never generic filler like "Introduction" or
"Overview". Return 3 to 8 topics.

Respond with ONLY a JSON object of the form {"topics": ["Topic one", "Topic two", "Topic three"]}, nothing else."""


def topic_extraction_user(chunk_text: str) -> str:
    return f'DOCUMENT EXCERPT:\n{chunk_text}\n\nExtract the candidate topics as JSON: {{"topics": [...]}}.'


TOPIC_CLASSIFICATION_SYSTEM = """# task: topic_classification
You check whether a user-proposed podcast topic is actually grounded in a
set of source documents (i.e. the documents contain enough material to
meaningfully discuss it), or whether it's unrelated / not covered.

Respond with ONLY a JSON object, nothing else, e.g.:
{"grounded": true, "reason": "short reason"}"""


def topic_classification_user(topic: str, doc_excerpt: str) -> str:
    return f"TOPIC: {topic}\n\nDOCUMENT EXCERPTS:\n{doc_excerpt}\n\nIs this topic grounded in the documents above?"
