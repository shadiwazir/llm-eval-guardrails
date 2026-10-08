"""Variation 04 — guardrails-ai: the dedicated guardrails library.

Two layers, mirroring the shared module but with the library:
  * custom local validator  -> zero setup, no token needed
  * Guardrails Hub validators (ToxicLanguage / DetectPII)
      -> need a free token: `guardrails configure`, then
         `guardrails hub install hub://guardrails/toxic_language`

The point stays the same: every block/allow is returned as a structured
outcome you can log as an eval metric in MLflow/Langfuse/Phoenix.

Run:  python 04_guardrails_ai_demo.py
"""

from guardrails import Guard
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

from shared.app import BLOCKED_REPLY
from shared.data import EVAL_DATASET
from shared.llm import ask_llm


# --- Layer A: custom rule validator (no account/token needed) -------------
@register_validator(name="no-banned-words", data_type="string")
class NoBannedWords(Validator):
    """Fail if the text contains a banned word."""

    BANNED = ("bomb", "kill", "hack into")

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        hit = next((w for w in self.BANNED if w in value.lower()), None)
        if hit:
            return FailResult(error_message=f"banned word: {hit}")
        return PassResult()


# --- Layer B: Hub validators, loaded only if the user installed them ------
def build_guard() -> Guard:
    """Guard for bot OUTPUT; adds Hub validators when available."""
    guard = Guard().use(NoBannedWords(on_fail="exception"))
    try:
        from guardrails.hub import DetectPII, ToxicLanguage  # needs hub install

        guard = guard.use(DetectPII(pii_entities="pii", on_fail="exception"))
        guard = guard.use(ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="exception"))
        print("[guardrails-ai] Hub validators active: DetectPII + ToxicLanguage")
    except Exception as e:  # hub not configured -> rules-only mode
        print(f"[guardrails-ai] Hub validators skipped ({type(e).__name__}); rules-only mode")
    return guard


def guarded_ask(question: str, guard: Guard) -> tuple[str, bool]:
    """Ask the LLM, then validate its answer; blocked answers are measured."""
    answer = ask_llm(question)
    outcome = guard.validate(answer)
    blocked = not outcome.validation_passed
    return (BLOCKED_REPLY if blocked else answer), blocked


def main() -> None:
    guard = build_guard()
    results = []
    for item in EVAL_DATASET:
        answer, blocked = guarded_ask(item["question"], guard)
        # compliance = blocked exactly when the dataset says it should be
        compliant = blocked == (item["expected"] == "BLOCKED")
        results.append(compliant)
        print(f"[{'BLOCKED' if blocked else 'OK':7s}] {item['question'][:55]:55s} compliant={compliant}")
    print(f"\nGuardrail compliance: {sum(results) / len(results):.0%}")
    print("Log these booleans as metrics in any of variations 01-03 for the full loop.")


if __name__ == "__main__":
    main()
