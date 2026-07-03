SECTION_SYSTEM = """# task: section_generation
You write natural, humanoid two-person podcast dialogue between a Host and a
Guest, in English.

Style rules:
- Light fillers ("um", "uh", "hmm") used in MODERATION -- not on every line, and only where a real speaker would pause.
- Realistic back-and-forth: short reactions, follow-up questions, occasional gentle interruptions.
- Smooth transition into and out of this section so it reads as one continuous conversation, not an isolated clip.
- When document material is provided, stay grounded in it -- do not invent facts, statistics, names, or claims that aren't supported by the excerpt.
- Keep each speaker's personality and tone consistent with their described style.
- Respect each speaker's target word count -- it's a budget, not a hard wall, but stay close to it.
- Output ONLY dialogue lines formatted exactly as "HOST: ..." or "GUEST: ..." on their own lines. No stage directions, no scene headers, no markdown, no section titles.
"""


def _speaker_block(label: str, name: str, gender: str, filler_style: str) -> str:
    return f"{label}: {name} ({gender}). Speaking style: {filler_style}."


def section_user_prompt(
    *,
    position: str,  # "opening" | "topic" | "closing"
    topic: str | None,
    word_budget: int,
    host_name: str,
    host_gender: str,
    host_style: str,
    guest_name: str,
    guest_gender: str,
    guest_style: str,
    doc_excerpt: str,
    previous_tail: str,
    all_topics: list[str],
    modification_instruction: str | None = None,
    previous_version: str | None = None,
) -> str:
    lines = [
        _speaker_block("HOST", host_name, host_gender, host_style),
        _speaker_block("GUEST", guest_name, guest_gender, guest_style),
        f"TARGET WORD COUNT FOR THIS SECTION: ~{word_budget} words",
    ]

    if position == "opening":
        lines.append("SECTION TOPIC: podcast opening / cold open and introductions")
        lines.append(
            "Write the OPENING of the episode: a warm cold-open, Host welcomes listeners, introduces the Guest "
            f"by name, and briefly previews these topics without covering them yet: {', '.join(all_topics)}."
        )
    elif position == "closing":
        lines.append("SECTION TOPIC: podcast closing")
        lines.append(
            "Write the CLOSING of the episode: Host briefly recaps the conversation at a high level, thanks the "
            "Guest by name, and signs off naturally. Do not introduce any new factual claims here."
        )
    else:
        lines.append(f"SECTION TOPIC: {topic}")
        lines.append(
            "Write this middle section of the conversation, covering the topic above through natural dialogue "
            "grounded in the document excerpt below."
        )
        lines.append(f"DOCUMENT EXCERPT:\n{doc_excerpt}")

    if previous_tail:
        lines.append(
            f"PREVIOUS SECTION ENDED WITH (continue naturally from here, don't repeat it):\n{previous_tail}"
        )

    if modification_instruction:
        lines.append(f"USER REQUESTED MODIFICATION (apply this to the section, then rewrite it in full):\n{modification_instruction}")
        if previous_version:
            lines.append(f"PREVIOUS VERSION OF THIS SECTION (for reference, rewrite -- don't just append):\n{previous_version}")

    return "\n\n".join(lines)
