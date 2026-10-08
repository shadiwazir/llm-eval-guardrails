"""Guardrails: rule-based checks + an LLM safety judge.

Every check returns a GuardrailReport so the *decision* (allow/block) can be
logged as an eval metric — this is the link the course material is missing:
guardrails only become defensible once their decisions are measured.
"""

import re
from dataclasses import dataclass, field

from .judge import Judge

# --- Rule-based layer: deterministic, free, runs on every input/output ---

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}
BANNED_WORDS = {"bomb", "kill", "hack into"}
MAX_LEN = 1_000  # chars


@dataclass
class GuardrailReport:
    """What happened at one guardrail checkpoint."""

    passed: bool
    violations: list[str] = field(default_factory=list)  # rule hits
    judge_verdict: str = "not-run"  # LLM-judge verdict for outputs
    stage: str = ""  # "input" or "output"


def rule_based_check(text: str) -> list[str]:
    """Return the list of violated rules ([] means clean)."""
    violations = [f"pii:{name}" for name, pat in PII_PATTERNS.items() if pat.search(text)]
    low = text.lower()
    violations += [f"banned:{w}" for w in BANNED_WORDS if w in low]
    if len(text) > MAX_LEN:
        violations.append("too_long")
    return violations


def check_input(question: str) -> GuardrailReport:
    """Gate 1 — runs before the LLM call; rules only (judge not needed here)."""
    violations = rule_based_check(question)
    return GuardrailReport(passed=not violations, violations=violations, stage="input")


def check_output(answer: str, judge: Judge) -> GuardrailReport:
    """Gate 2 — runs after the LLM call; rules + the injected safety judge."""
    violations = rule_based_check(answer)
    verdict = judge.safety(answer)
    if not verdict.startswith("safe"):
        violations.append(f"judge:{verdict}")
    return GuardrailReport(passed=not violations, violations=violations, judge_verdict=verdict, stage="output")
