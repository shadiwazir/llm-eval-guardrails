"""Contract implemented by every judge backend; backends are injected, never switched on."""

from typing import Protocol


class Judge(Protocol):
    name: str  # recorded in traces so a run shows which backend judged it

    def safety(self, text: str) -> str:
        """Return 'safe' or 'unsafe: <reason>' for a bot answer."""

    def correctness(self, question: str, answer: str, expected: str) -> bool:
        """True if the answer conveys the same meaning as the expected one."""
