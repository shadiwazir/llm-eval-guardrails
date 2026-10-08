"""Langfuse pipeline: traces + eval scores, judge injected by the entry file.

  * langfuse.openai wrapper -> OpenAI calls auto-traced to Langfuse
  * @observe()              -> the guarded pipeline is one trace with nested spans
  * score_current_trace()   -> guardrail + correctness results attached as scores

Needs a Langfuse project (cloud or self-hosted); keys go in .env.
"""

from langfuse import get_client, observe
from langfuse.openai import openai

from shared.app import BLOCKED_REPLY, run_app
from shared.data import EVAL_DATASET
from shared.judge import Judge


def run(judge: Judge) -> None:
    langfuse = get_client()  # reads LANGFUSE_* keys from the environment

    scores = []

    def correctness(question: str, answer: str, expected: str) -> float:
        if expected == "BLOCKED":  # adversarial rows pass when the guardrail blocked them
            return 1.0 if answer == BLOCKED_REPLY else 0.0
        return 1.0 if judge.correctness(question, answer, expected) else 0.0

    @observe(name="guarded-support-bot")  # parent span for the whole pipeline
    def traced_app(question: str, expected: str) -> str:
        result = run_app(question, judge, client=openai.OpenAI())  # wrapped client = traced
        langfuse.score_current_trace(
            name="guardrail_blocked",
            value=1.0 if result.blocked else 0.0,
            comment=",".join(result.input_report.violations) or "clean",
        )
        # Scoring must happen inside the observed function: outside it there is no active trace.
        score = correctness(question, result.answer, expected)
        langfuse.score_current_trace(name="correctness", value=score)
        scores.append(score)
        return result.answer

    print(f"Judge backend: {judge.name}")
    for item in EVAL_DATASET:
        answer = traced_app(item["question"], item["expected"])  # each call = one Langfuse trace
        status = "BLOCKED" if answer == BLOCKED_REPLY else "OK"
        print(f"[{status:7s}] {item['question'][:60]:60s} correctness={scores[-1]}")

    langfuse.flush()  # send traces + scores before exit
    print(f"\nMean correctness: {sum(scores) / len(scores):.2f}")
    print("Open your Langfuse project -> Traces to inspect spans, scores and guardrail decisions.")
