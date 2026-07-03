"""Best-effort JSON extraction from LLM text output.

Models frequently wrap JSON in ```json fences or add a stray sentence before/
after it. This strips the common noise instead of failing the whole pipeline
on a formatting slip.
"""

import json
import re


class JsonParseError(Exception):
    pass


def parse_json_loose(text: str):
    # Try to extract valid JSON from model responses that may include
    # extra text or markdown fences.
    if text is None:
        raise JsonParseError("Empty response from the model.")

    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise JsonParseError(f"Could not parse JSON from model output: {text[:200]!r}")
