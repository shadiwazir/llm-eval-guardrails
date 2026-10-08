"""Jev judge backend — TypeSafe's System One decision model.

Jev can't generate text; it answers typed questions with probabilities —
exactly the shape of our judge tasks (safe/unsafe, correct/incorrect).
Much cheaper than an LLM judge ($0.042/1M input tokens, output free).

Setup: pip install typesafe-sdk  +  TYPESAFE_API_KEY in .env
Missing key -> warn once and fall back to the OpenAI judge.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Probability cutoff for turning Jev's answer into a boolean decision.
THRESHOLD = float(os.getenv("JEV_THRESHOLD", "0.5"))

_warned = False  # print the missing-key warning only once per process


def jev_binary(state, instructions: str, criteria: dict | None = None) -> float:
    """Return P(instructions is true) in [0, 1] via one Noul question.

    ``state`` is the content to judge: a string, or a dict whose fields the
    instructions can reference in `backticks`.
    """
    global _warned
    if not os.getenv("TYPESAFE_API_KEY"):
        if not _warned:
            print("[jev] TYPESAFE_API_KEY not set -> falling back to OpenAI judge. "
                  "Get a key at https://console.typesafe.ai/keys")
            _warned = True
        return _openai_binary(state, instructions)

    from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient  # lazy: optional dep

    with TypeSafeClient() as client:
        resp = client.system_one(
            model="jev-latest",
            state=state,
            questions={
                "verdict": Noul(
                    instructions=instructions,
                    criteria=NoulCriteria(**criteria) if criteria else None,
                )
            },
        )
    return float(resp.answers["verdict"].noul)


def _openai_binary(state, instructions: str) -> float:
    """OpenAI fallback: ask for 1/0 and parse it into a float."""
    from .llm import llm_judge

    verdict = llm_judge(
        "Judge the statement below. Reply with exactly 1 (true) or 0 (false).\n\n"
        f"CONTEXT:\n{state}\n\nSTATEMENT: {instructions}"
    )
    return 1.0 if verdict.strip().startswith("1") else 0.0
