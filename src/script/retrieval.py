"""Cheap keyword-overlap retrieval: pull the paragraphs most relevant to a
topic out of the uploaded documents, so each generated section stays grounded
without shipping entire documents into every prompt.
"""

import re


def _significant_words(text: str) -> set[str]:
    # Extract useful words for scoring relevance, ignoring common stopwords.
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "into", "about", "your", "their"}
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w for w in words if w not in stopwords}


def retrieve_relevant_excerpt(topic: str, doc_texts: dict[str, str], max_chars: int = 3000) -> str:
    # Pull the most relevant paragraph text for a topic, staying within a
    # limited character budget so prompts remain efficient.
    if not doc_texts:
        return ""

    if topic is None:
        return "\n\n".join(text[: max_chars // max(1, len(doc_texts))] for text in doc_texts.values())

    topic_words = _significant_words(topic)
    scored = []
    for text in doc_texts.values():
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            para_words = _significant_words(para)
            overlap = len(topic_words & para_words)
            if overlap > 0:
                scored.append((overlap, para))

    scored.sort(key=lambda pair: -pair[0])

    if not scored:
        return "\n\n".join(text[: max_chars // max(1, len(doc_texts))] for text in doc_texts.values())

    excerpt_parts = []
    total = 0
    for _, para in scored:
        if total >= max_chars:
            break
        excerpt_parts.append(para)
        total += len(para)

    return "\n\n".join(excerpt_parts)[:max_chars]
