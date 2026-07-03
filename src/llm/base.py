"""Provider-agnostic LLM client interface.

Every provider adapter in providers.py implements this interface so the rest
of the app (topic extraction, script generation) never needs to know which
LLM is actually configured.
"""

from abc import ABC, abstractmethod


class LLMError(Exception):
    """Raised for any provider-facing failure: bad key, rate limit, timeout,
    unsupported model, network error, etc. Always carries a user-actionable
    message -- callers can show str(err) directly in the UI.
    """


class LLMClient(ABC):
    # Base interface for all provider-specific LLM adapters.
    # The rest of the application relies on this common complete() contract.
    def __init__(self, model: str):
        if not model or not model.strip():
            raise LLMError("No model selected. Pick a model before continuing.")
        self.model = model

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.8) -> str:
        """Return the model's text completion, or raise LLMError."""
        raise NotImplementedError

    def test_connection(self) -> None:
        # A simple request to validate the provider API key/model before the flow.
        reply = self.complete(
            system="You are a connection test endpoint.",
            user="Reply with exactly one word: OK",
            max_tokens=5,
            temperature=0,
        )
        if not reply or not reply.strip():
            raise LLMError("Provider returned an empty response during the connection test.")
