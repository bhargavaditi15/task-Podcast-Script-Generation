"""Shared constants for the podcast script generator."""

# Allowed duration choices, in minutes, for the generated podcast script.
DURATION_OPTIONS_MIN = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Speaker gender options used consistently in form validation and prompts.
GENDER_OPTIONS = ["male", "female"]

# Speaking speed range used to compute relative words-per-minute.
SPEED_MIN = 50
SPEED_MAX = 150
SPEED_DEFAULT = 100

# Words-per-minute a speaker produces at speed=SPEED_DEFAULT (natural pace).
BASE_WPM = 150
# At SPEED_MIN a speaker is slower (~70% of base), at SPEED_MAX faster (~130%).
WPM_AT_MIN_SPEED = BASE_WPM * 0.70
WPM_AT_MAX_SPEED = BASE_WPM * 1.30

# Fraction of total script devoted to opening / closing (rest is topic coverage).
OPENING_FRACTION = 0.08
CLOSING_FRACTION = 0.07

# Supported file extensions for source documents.
SUPPORTED_DOC_EXTENSIONS = [".pdf", ".doc", ".docx", ".txt"]

# Configurable LLM providers exposed in the UI/CLI/API.
LLM_PROVIDERS = ["OpenAI", "Anthropic", "Google Gemini", "Groq", "Custom (OpenAI-compatible)", "Mock (offline/dev)"]

# Predefined model recommendations for each provider.
PROVIDER_MODELS = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini"],
    "Anthropic": [
        "claude-sonnet-4-6",
        "claude-opus-4-8",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-20250514",
    ],
    "Google Gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
    "Groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "Custom (OpenAI-compatible)": ["llama3", "mistral", "custom-model"],
    "Mock (offline/dev)": ["mock-1"],
}

# Chunk size used for topic extraction so each prompt stays within reasonable
# provider input limits.
MAX_CHUNK_CHARS = 12000  # ~3k tokens per chunk sent to the LLM for extraction

# Maximum number of script modification versions retained in the UI.
MAX_MODIFICATION_HISTORY = 20
