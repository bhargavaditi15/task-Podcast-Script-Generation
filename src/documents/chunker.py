"""Split large document text into LLM-sized chunks without cutting sentences
mid-way any more than necessary. Used for topic extraction map-reduce over
oversized documents.
"""


def chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    # Split large document text into chunks that fit within LLM prompt size
    # limits, preserving paragraph boundaries when possible.
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars:
            chunks.append(current)
            current = para
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
