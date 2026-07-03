"""Small data holders shared by the planner and generator."""

from dataclasses import dataclass, field


@dataclass
class Speaker:
    name: str
    gender: str
    speed: int


@dataclass
class SectionBudget:
    kind: str  # "opening" | "topic" | "closing"
    topic: str | None
    word_budget: int


@dataclass
class Section:
    kind: str
    topic: str | None
    word_budget: int
    text: str = ""


@dataclass
class ScriptResult:
    sections: list[Section] = field(default_factory=list)
    target_words: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text.strip() for s in self.sections if s.text.strip())

    @property
    def actual_words(self) -> int:
        return len(self.full_text.split())
