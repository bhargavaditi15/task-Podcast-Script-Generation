"""Turns (topics, duration, speaking speeds) into a deterministic word-count
budget per section. Doing this arithmetically -- instead of asking the LLM to
self-budget -- is what makes the "length reasonably matches selected
duration" evaluation criterion reliably hittable rather than a coin flip.
"""

from src.config import CLOSING_FRACTION, OPENING_FRACTION, SPEED_MAX, SPEED_MIN, WPM_AT_MAX_SPEED, WPM_AT_MIN_SPEED

from .models import SectionBudget


def speed_to_wpm(speed: int) -> float:
    # Convert the user-selected speaking speed into a relative words-per-minute value.
    speed = max(SPEED_MIN, min(SPEED_MAX, int(speed)))
    ratio = (speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)
    return WPM_AT_MIN_SPEED + ratio * (WPM_AT_MAX_SPEED - WPM_AT_MIN_SPEED)


def filler_density_description(speed: int) -> str:
    # Choose a human-style description for how much filler or pacing the speaker should use.
    if speed <= 80:
        return "frequent light fillers (um, uh, hmm) and unhurried, slightly longer sentences"
    if speed >= 120:
        return "minimal fillers, clipped and energetic sentences"
    return "occasional light fillers, natural relaxed pacing"


def build_plan(topics: list[str], duration_minutes: int, host_speed: int, guest_speed: int):
    """Returns (sections: list[SectionBudget], warnings: list[str], target_words: int)."""
    avg_wpm = (speed_to_wpm(host_speed) + speed_to_wpm(guest_speed)) / 2
    target_words = max(1, int(avg_wpm * duration_minutes))

    opening_budget = max(60, int(target_words * OPENING_FRACTION))
    closing_budget = max(50, int(target_words * CLOSING_FRACTION))
    topics_budget_total = max(100, target_words - opening_budget - closing_budget)

    warnings: list[str] = []
    effective_topics = topics or ["General discussion"]

    per_topic = topics_budget_total // len(effective_topics)
    if per_topic < 80:
        warnings.append(
            f"{len(effective_topics)} topics selected for a {duration_minutes}-minute script means each "
            f"topic gets roughly {per_topic} words. Consider fewer topics or a longer duration for deeper coverage."
        )
    per_topic = max(per_topic, 40)

    sections = [SectionBudget(kind="opening", topic=None, word_budget=opening_budget)]
    sections.extend(SectionBudget(kind="topic", topic=t, word_budget=per_topic) for t in effective_topics)
    sections.append(SectionBudget(kind="closing", topic=None, word_budget=closing_budget))

    return sections, warnings, target_words
