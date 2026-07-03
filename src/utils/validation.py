"""Pre-flow input validation.

validate_setup_inputs returns a list of every problem found (not just the
first one), so the UI can show a single consolidated error message per the
task spec, instead of forcing the user through a fix-one-at-a-time loop.
"""

from src.config import DURATION_OPTIONS_MIN, GENDER_OPTIONS, SPEED_MAX, SPEED_MIN


def validate_setup_inputs(
    host_name: str,
    guest_name: str,
    host_gender: str,
    guest_gender: str,
    host_speed: int,
    guest_speed: int,
    doc_texts: dict,
    duration_minutes,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
) -> list[str]:
    # Validate every required field before the generation flow begins.
    # This returns a list of all problems so the UI can show a consolidated
    # error message instead of failing one check at a time.
    errors = []

    if not host_name or not host_name.strip():
        errors.append("Host name is required.")
    if not guest_name or not guest_name.strip():
        errors.append("Guest name is required.")
    if host_gender not in GENDER_OPTIONS:
        errors.append(f"Host gender is required (choose one of: {', '.join(GENDER_OPTIONS)}).")
    if guest_gender not in GENDER_OPTIONS:
        errors.append(f"Guest gender is required (choose one of: {', '.join(GENDER_OPTIONS)}).")
    if not (SPEED_MIN <= int(host_speed or 0) <= SPEED_MAX):
        errors.append(f"Host speaking speed must be between {SPEED_MIN} and {SPEED_MAX}.")
    if not (SPEED_MIN <= int(guest_speed or 0) <= SPEED_MAX):
        errors.append(f"Guest speaking speed must be between {SPEED_MIN} and {SPEED_MAX}.")
    if not doc_texts:
        errors.append("At least one document (PDF, DOC/DOCX, or TXT) must be uploaded successfully.")
    if duration_minutes not in DURATION_OPTIONS_MIN:
        errors.append(f"Please select a target duration ({', '.join(str(d) for d in DURATION_OPTIONS_MIN)} minutes).")
    if not llm_provider:
        errors.append("Please select an LLM provider.")
    if not llm_model:
        errors.append("Please select a model for the chosen LLM provider.")
    if llm_provider not in ("Mock (offline/dev)", "Custom (OpenAI-compatible)") and not llm_api_key:
        errors.append(f"An API key is required for {llm_provider or 'the selected provider'}.")

    return errors
