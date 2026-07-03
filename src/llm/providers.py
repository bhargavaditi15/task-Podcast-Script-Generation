"""One adapter per LLM provider, all implementing LLMClient.

Every adapter wraps provider SDK exceptions into LLMError with an actionable
message, and retries transient failures (timeouts / rate limits / 5xx) a
couple of times with backoff before giving up.
"""

import re
import time

from .base import LLMClient, LLMError

_RETRYABLE_HINTS = ("rate limit", "rate_limit", "timeout", "timed out", "429", "503", "502", "overloaded", "connection")
_AUTH_HINTS = ("api key", "apikey", "unauthorized", "401", "invalid_api_key", "authentication")
_MODEL_HINTS = ("model", "not found", "404", "does not exist")


def _friendly_error(provider: str, exc: Exception) -> str:
    # Normalize raw provider SDK exceptions into user-friendly messages.
    msg = str(exc).lower()
    if any(h in msg for h in _AUTH_HINTS):
        return f"{provider} rejected the API key. Double-check the key and try 'Test connection' again."
    if any(h in msg for h in _MODEL_HINTS):
        return f"{provider} could not find model '{{model}}'. Pick a different model or check the spelling."
    if any(h in msg for h in _RETRYABLE_HINTS):
        return f"{provider} is temporarily unavailable (rate limit or timeout). Please retry in a moment."
    return f"{provider} request failed: {exc}"


def _with_retry(fn, provider: str, model: str, attempts: int = 3, base_delay: float = 1.5) -> str:
    # Retry transient provider errors before failing permanently.
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we classify below
            last_exc = exc
            msg = str(exc).lower()
            is_retryable = any(h in msg for h in _RETRYABLE_HINTS)
            if is_retryable and attempt < attempts - 1:
                time.sleep(base_delay * (2**attempt))
                continue
            break
    friendly = _friendly_error(provider, last_exc).replace("{model}", model)
    raise LLMError(friendly) from last_exc


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        super().__init__(model)
        if not api_key:
            raise LLMError("OpenAI API key is required.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("The 'openai' package is not installed. Run: pip install openai") from exc
        self._client = OpenAI(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.8) -> str:
        def _call():
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

        return _with_retry(_call, "OpenAI", self.model)


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        super().__init__(model)
        if not api_key:
            raise LLMError("Anthropic API key is required.")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("The 'anthropic' package is not installed. Run: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.8) -> str:
        def _call():
            resp = self._client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")

        return _with_retry(_call, "Anthropic", self.model)


class GeminiClient(LLMClient):
    """Uses the current 'google-genai' SDK (the deprecated 'google-generativeai'
    package doesn't give clean control over 'thinking', which matters a lot
    here -- see complete() below).
    """

    def __init__(self, api_key: str, model: str):
        super().__init__(model)
        if not api_key:
            raise LLMError("Google API key is required.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError("The 'google-genai' package is not installed. Run: pip install google-genai") from exc
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.8) -> str:
        types = self._types

        def _call():
            # Gemini 2.x models "think" by default, spending part of
            # max_output_tokens on invisible internal reasoning before ever
            # emitting the visible answer. For short, non-reasoning tasks
            # like ours (extract topics / classify a topic) with a small
            # token budget, that can consume the entire budget and leave
            # nothing for the actual answer -- resulting in an empty/blank
            # response that then fails to parse as the JSON we asked for.
            # Disable thinking explicitly; some models (certain Pro variants)
            # don't allow disabling it and reject budget=0, so retry once
            # without the override if that happens.
            config = types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            try:
                resp = self._client.models.generate_content(model=self.model, contents=user, config=config)
            except Exception as exc:  # noqa: BLE001
                if "thinking" not in str(exc).lower() and "budget" not in str(exc).lower():
                    raise
                config = types.GenerateContentConfig(
                    system_instruction=system, max_output_tokens=max_tokens, temperature=temperature
                )
                resp = self._client.models.generate_content(model=self.model, contents=user, config=config)
            return resp.text or ""

        return _with_retry(_call, "Google Gemini", self.model)


class GroqClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        super().__init__(model)
        if not api_key:
            raise LLMError("Groq API key is required.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMError("The 'groq' package is not installed. Run: pip install groq") from exc
        self._client = Groq(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.8) -> str:
        def _call():
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

        return _with_retry(_call, "Groq", self.model)


class CustomOpenAICompatibleClient(LLMClient):
    """Any OpenAI-compatible chat completions endpoint: local Ollama, vLLM,
    LM Studio, self-hosted gateways, etc. API key is optional for local servers.
    """

    def __init__(self, api_key: str, model: str, base_url: str):
        super().__init__(model)
        if not base_url:
            raise LLMError("Base URL is required for a custom OpenAI-compatible endpoint.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("The 'openai' package is not installed. Run: pip install openai") from exc
        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)

    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.8) -> str:
        def _call():
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

        return _with_retry(_call, "Custom endpoint", self.model)


class MockClient(LLMClient):
    """Deterministic, offline, no-network provider.

    This provider is used for demos and local testing when no API key is
    available. It produces simplified mock responses based on prompt content.

    Used for local dry runs, generating sample_outputs/, and letting a
    reviewer without any API key exercise the whole app. It reads the
    "# task: <name>" marker line every prompts.py system prompt starts with
    to decide which canned generator to run, then works off the actual
    document text / topic list included in the user prompt so its output
    isn't meaningless boilerplate.
    """

    def __init__(self, model: str = "mock-1"):
        super().__init__(model)

    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.8) -> str:
        task_match = re.match(r"#\s*task:\s*(\w+)", system.strip())
        task = task_match.group(1) if task_match else "generic"

        if task == "topic_extraction":
            return self._mock_topics(user)
        if task == "topic_classification":
            return self._mock_classification(user)
        if task == "section_generation":
            return self._mock_section(user)
        return "OK"

    @staticmethod
    def _keywordish_phrases(text: str, limit: int = 6):
        # Naive noun-phrase-ish extraction: capitalized words / runs, then fall
        # back to the most frequent longer words. Good enough for offline demo data.
        # A real LLM would refuse to extract topics from a near-empty document,
        # so mirror that here instead of always fabricating a fallback topic --
        # otherwise the "document too thin" edge case can never be exercised
        # against the Mock provider.
        if len(text.split()) < 8:
            return []

        sentence_starter_stopwords = {
            "since", "the", "this", "that", "these", "those", "however", "meanwhile", "despite",
            "while", "some", "many", "most", "government", "companies", "critics", "analysts",
            "surveys", "researchers", "consumer", "electric",
        }
        candidates = re.findall(r"(?:[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,2})", text)
        seen, uniq = set(), []
        for c in candidates:
            key = c.lower()
            if " " not in c and key in sentence_starter_stopwords:
                continue
            if key not in seen and len(c) > 3:
                seen.add(key)
                uniq.append(c)
        if len(uniq) >= limit:
            return uniq[:limit]
        words = re.findall(r"[a-zA-Z]{5,}", text.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        for w, _ in sorted(freq.items(), key=lambda kv: -kv[1]):
            phrase = w.capitalize()
            if phrase.lower() not in seen:
                uniq.append(phrase)
                seen.add(phrase.lower())
            if len(uniq) >= limit:
                break
        return uniq[:limit] or ["General overview"]

    def _mock_topics(self, user: str) -> str:
        import json

        excerpt_match = re.search(r"DOCUMENT EXCERPT:\s*(.*?)\n\nExtract", user, re.DOTALL)
        content = excerpt_match.group(1) if excerpt_match else user
        phrases = self._keywordish_phrases(content, limit=6)
        return json.dumps({"topics": phrases})

    def _mock_classification(self, user: str) -> str:
        import json

        topic_match = re.search(r"TOPIC:\s*(.+)", user)
        doc_match = re.search(r"DOCUMENT EXCERPTS?:\s*(.+)", user, re.DOTALL)
        topic = topic_match.group(1).strip() if topic_match else "unknown"
        doc_text = doc_match.group(1) if doc_match else user
        grounded = topic.lower().split()[0] in doc_text.lower() if topic else False
        return json.dumps({"grounded": bool(grounded), "reason": "mock heuristic keyword match"})

    def _mock_section(self, user: str) -> str:
        host_match = re.search(r"HOST:\s*([^(]+)\(", user)
        guest_match = re.search(r"GUEST:\s*([^(]+)\(", user)
        host_name = host_match.group(1).strip() if host_match else "the host"
        guest_name = guest_match.group(1).strip() if guest_match else "the guest"

        if "OPENING of the episode" in user:
            return (
                f"HOST: Hey everyone, welcome back to the show, I'm {host_name}.\n"
                f"GUEST: And I'm {guest_name}, happy to be here.\n"
                f"HOST: Today we're getting into a few things pulled straight from the source material -- "
                f"should be a good one.\n"
                f"GUEST: Yeah, um, I'm looking forward to it.\n"
            )
        if "CLOSING of the episode" in user:
            return (
                f"HOST: Well, that about wraps it up for today.\n"
                f"GUEST: Yeah, this was a great conversation, thanks for having me.\n"
                f"HOST: Thanks so much, {guest_name}. And thank you all for listening -- see you next time.\n"
            )

        topic_match = re.search(r"SECTION TOPIC:\s*(.+)", user)
        topic = topic_match.group(1).strip() if topic_match else "this topic"
        return (
            f"HOST: So, um, let's get into {topic} -- what's the headline here?\n"
            f"GUEST: Yeah, great question. The short version is that {topic} matters more than people think, "
            f"and the documents really back that up.\n"
            f"HOST: Hmm, interesting. Can you unpack that a little?\n"
            f"GUEST: Sure -- basically, when you look at the source material, {topic} keeps showing up as a "
            f"recurring thread, and it connects to a lot of what we already covered.\n"
            f"HOST: Right, right. That makes sense.\n"
        )
