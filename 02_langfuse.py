"""Variation 02 — Langfuse: LLM tracing + evals as trace scores.

  * langfuse.openai wrapper -> OpenAI calls auto-traced to Langfuse
  * @observe()              -> the guarded pipeline is one trace with nested spans
  * score_current_trace()   -> eval + guardrail results attached as scores,
    so every trace carries its own evaluation (guardrails WITH eval)

Needs a Langfuse server: free cloud project (https://cloud.langfuse.com) or
`docker compose` self-host — see README. Keys go in .env.

Run:  python 02_langfuse.py
"""

from langfuse import get_client, observe
from langfuse.openai import openai  # drop-in replacement that auto-traces

from shared.app import run_app
from shared.data import EVAL_DATASET
from shared.llm import llm_judge

langfuse = get_client()  # reads LANGFUSE_* keys from the environment


@observe(name="guarded-support-bot")  # parent span for the whole pipeline
def traced_app(question: str) -> str:
    """Run the guarded app; guardrail verdicts become scores on the trace."""
    result = run_app(question, client=openai.OpenAI())  # wrapped client = traced

    # 0/1 score per trace: did the guardrails behave as designed?
    langfuse.score_current_trace(
        name="guardrail_blocked",
        value=1.0 if result.blocked else 0.0,
        comment=",".join(result.input_report.violations) or "clean",
    )
    return result.answer


def correctness_judge(question: str, answer: str, expected: str) -> float:
    """LLM-as-judge correctness score (1 = matches the expected answer)."""
    if expected == "BLOCKED":  # adversarial rows are scored by guardrail_blocked
        return 1.0 if answer.startswith("Sorry") else 0.0
    verdict = llm_judge(
        "Score 1 if ANSWER matches the EXPECTED meaning, else 0. Reply with just 1 or 0.\n\n"
        f"QUESTION: {question}\nANSWER: {answer}\nEXPECTED: {expected}",
        client=openai.OpenAI(),  # judge calls are traced too
    )
    return 1.0 if verdict.strip().startswith("1") else 0.0


def main() -> None:
    scores = []
    for item in EVAL_DATASET:
        answer = traced_app(item["question"])  # each call = one Langfuse trace
        score = correctness_judge(item["question"], answer, item["expected"])
        # Attach the eval score to the trace that produced the answer.
        langfuse.score_current_trace(name="correctness", value=score)
        scores.append(score)
        print(f"[{'BLOCKED' if answer.startswith('Sorry') else 'OK':7s}] {item['question'][:60]:60s} correctness={score}")

    langfuse.flush()  # make sure traces+scores are sent before exit
    print(f"\nMean correctness: {sum(scores) / len(scores):.2f}")
    print("Open your Langfuse project -> Traces to inspect spans, scores and guardrail decisions.")


if __name__ == "__main__":
    main()
