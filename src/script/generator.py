"""Two-phase-ish script generation: a deterministic word-budget plan (see
planner.py) drives one LLM call per section (opening / each topic / closing).
Generating section-by-section, carrying forward the previous section's tail
for continuity, is what keeps long scripts (45-60 min => thousands of words)
from hitting a single completion's output-token ceiling or trailing off.

Modification re-runs every section through the same call, passing the user's
instruction and the section's previous text, and always returns a brand new
complete ScriptResult -- never a partial patch.
"""

from src.llm.base import LLMClient
from src.script.models import ScriptResult, Section, Speaker
from src.script.planner import build_plan, filler_density_description
from src.script.prompts import SECTION_SYSTEM, section_user_prompt
from src.script.retrieval import retrieve_relevant_excerpt


def _tail_words(text: str, n_words: int = 60) -> str:
    # Keep the end of the previous section so the next section can continue naturally.
    words = text.split()
    return " ".join(words[-n_words:])


def _max_tokens_for(word_budget: int) -> int:
    # Derive a safe LLM max_tokens limit from the expected word budget.
    return max(300, min(4000, int(word_budget * 2.2) + 150))


def generate_script(
    topics: list[str],
    host: Speaker,
    guest: Speaker,
    duration_minutes: int,
    doc_texts: dict[str, str],
    llm: LLMClient,
    progress_callback=None,
) -> ScriptResult:
    budgets, warnings, target_words = build_plan(topics, duration_minutes, host.speed, guest.speed)
    result = ScriptResult(target_words=target_words, warnings=warnings)

    host_style = filler_density_description(host.speed)
    guest_style = filler_density_description(guest.speed)

    for i, budget in enumerate(budgets):
        doc_excerpt = retrieve_relevant_excerpt(budget.topic, doc_texts) if budget.kind == "topic" else ""
        previous_tail = _tail_words(result.sections[-1].text) if result.sections else ""

        user_prompt = section_user_prompt(
            position=budget.kind,
            topic=budget.topic,
            word_budget=budget.word_budget,
            host_name=host.name,
            host_gender=host.gender,
            host_style=host_style,
            guest_name=guest.name,
            guest_gender=guest.gender,
            guest_style=guest_style,
            doc_excerpt=doc_excerpt,
            previous_tail=previous_tail,
            all_topics=topics,
        )

        text = llm.complete(system=SECTION_SYSTEM, user=user_prompt, max_tokens=_max_tokens_for(budget.word_budget), temperature=0.85)
        section = Section(kind=budget.kind, topic=budget.topic, word_budget=budget.word_budget, text=text.strip())
        result.sections.append(section)

        if progress_callback:
            progress_callback(i + 1, len(budgets), section)

    return result


def modify_script(
    existing: ScriptResult,
    modification_instruction: str,
    host: Speaker,
    guest: Speaker,
    doc_texts: dict[str, str],
    llm: LLMClient,
    progress_callback=None,
) -> ScriptResult:
    all_topics = [s.topic for s in existing.sections if s.kind == "topic"]
    host_style = filler_density_description(host.speed)
    guest_style = filler_density_description(guest.speed)

    result = ScriptResult(target_words=existing.target_words, warnings=list(existing.warnings))

    for i, prev_section in enumerate(existing.sections):
        doc_excerpt = retrieve_relevant_excerpt(prev_section.topic, doc_texts) if prev_section.kind == "topic" else ""
        previous_tail = _tail_words(result.sections[-1].text) if result.sections else ""

        user_prompt = section_user_prompt(
            position=prev_section.kind,
            topic=prev_section.topic,
            word_budget=prev_section.word_budget,
            host_name=host.name,
            host_gender=host.gender,
            host_style=host_style,
            guest_name=guest.name,
            guest_gender=guest.gender,
            guest_style=guest_style,
            doc_excerpt=doc_excerpt,
            previous_tail=previous_tail,
            all_topics=all_topics,
            modification_instruction=modification_instruction,
            previous_version=prev_section.text,
        )

        text = llm.complete(system=SECTION_SYSTEM, user=user_prompt, max_tokens=_max_tokens_for(prev_section.word_budget), temperature=0.85)
        section = Section(kind=prev_section.kind, topic=prev_section.topic, word_budget=prev_section.word_budget, text=text.strip())
        result.sections.append(section)

        if progress_callback:
            progress_callback(i + 1, len(existing.sections), section)

    return result
