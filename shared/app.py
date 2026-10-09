"""The guarded support-bot pipeline, reused by every variation.

Flow: input guardrails -> LLM call -> output guardrails -> final reply.
Each variation wraps this with its own tracing; the logic never changes.
"""

from dataclasses import dataclass

from .guardrails import GuardrailReport, check_input, check_output
from .judge import Judge
from .llm import ask_llm

BLOCKED_REPLY = "Sorry, I can't help with that request."


@dataclass
class AppResult:
    """Everything an eval loop needs to score one run."""

    question: str
    answer: str  # final reply (BLOCKED_REPLY if a guardrail fired)
    blocked: bool
    input_report: GuardrailReport
    output_report: GuardrailReport | None  # None when blocked at the input gate


def run_app(
    question: str,
    judge: Judge,
    client=None,
    *,
    check_in=check_input,
    check_out=check_output,
) -> AppResult:
    """Run one question through guardrails (safety judged by `judge`) + LLM and return the full record.

    `check_in`/`check_out` are injectable so a pipeline can wrap the gates (e.g. in tracing spans).
    """
    # Gate 1: reject bad inputs before spending an LLM call.
    in_rep = check_in(question)
    if not in_rep.passed:
        return AppResult(question, BLOCKED_REPLY, True, in_rep, None)

    # LLM call (this is the span each tracing tool actually records).
    answer = ask_llm(question, client=client)

    # Gate 2: never ship an unsafe answer even if the model produced one.
    out_rep = check_out(answer, judge)
    if not out_rep.passed:
        return AppResult(question, BLOCKED_REPLY, True, in_rep, out_rep)

    return AppResult(question, answer, False, in_rep, out_rep)
