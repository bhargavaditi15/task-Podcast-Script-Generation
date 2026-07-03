"""Single entry point for turning a (provider, api_key, model, base_url)
selection -- whatever the Streamlit sidebar or CLI flags collected -- into a
ready-to-use LLMClient.
"""

from .base import LLMClient, LLMError
from .providers import (
    AnthropicClient,
    CustomOpenAICompatibleClient,
    GeminiClient,
    GroqClient,
    MockClient,
    OpenAIClient,
)


def get_llm_client(provider: str, api_key: str = "", model: str = "", base_url: str = "") -> LLMClient:
    # Factory that returns an LLM client for the selected provider.
    provider = (provider or "").strip()

    if provider == "OpenAI":
        return OpenAIClient(api_key=api_key, model=model)
    if provider == "Anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    if provider == "Google Gemini":
        return GeminiClient(api_key=api_key, model=model)
    if provider == "Groq":
        return GroqClient(api_key=api_key, model=model)
    if provider == "Custom (OpenAI-compatible)":
        return CustomOpenAICompatibleClient(api_key=api_key, model=model, base_url=base_url)
    if provider == "Mock (offline/dev)":
        return MockClient(model=model or "mock-1")

    raise LLMError(f"Unknown LLM provider: '{provider}'.")
