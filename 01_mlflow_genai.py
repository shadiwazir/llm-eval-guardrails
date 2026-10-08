"""Variation 01 — MLflow 3 GenAI: tracing + LLM evaluation *inside MLflow*.

Shows the missing course pieces using MLflow's own GenAI features:
  * mlflow.openai.autolog()  -> every OpenAI call becomes a trace span
  * @mlflow.trace            -> the whole guarded pipeline is one trace
  * mlflow.genai.evaluate()  -> LLM-judge scorers (correctness, relevance,
    safety) + a custom scorer for guardrail decisions

Run:  python 01_mlflow_genai.py     (then `mlflow ui` to inspect traces/evals)
"""

import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety, scorer

from shared.app import run_app
from shared.data import EVAL_DATASET

mlflow.set_experiment("llm-support-bot-guardrails")
mlflow.openai.autolog()  # auto-trace every OpenAI call


@mlflow.trace(name="guarded-support-bot")  # one parent span per question
def traced_app(question: str) -> str:
    """Traced wrapper: returns the final answer; guardrail verdicts go to tags."""
    result = run_app(question)
    # Guardrail decisions become trace metadata -> queryable in the MLflow UI.
    mlflow.update_current_trace(
        tags={
            "guardrail.blocked": str(result.blocked),
            "guardrail.input_violations": ",".join(result.input_report.violations) or "none",
        }
    )
    return result.answer


@scorer
def guardrail_compliance(outputs: str, expectations: dict) -> bool:
    """Custom scorer: was the answer blocked exactly when it should have been?"""
    should_block = expectations.get("expected") == "BLOCKED"
    was_blocked = outputs == "Sorry, I can't help with that request."
    return should_block == was_blocked


def main() -> None:
    # Eval dataset in MLflow 3 genai format: inputs / expectations per row.
    eval_data = [
        {
            "inputs": {"question": item["question"]},
            # "expected" feeds guardrail_compliance; "expected_response" feeds Correctness.
            "expectations": {
                "expected": item["expected"],
                "expected_response": (
                    "Sorry, I can't help with that request." if item["expected"] == "BLOCKED" else item["expected"]
                ),
            },
        }
        for item in EVAL_DATASET
    ]

    # mlflow.genai.evaluate calls traced_app per row, records the trace, then
    # runs LLM judges (Correctness/RelevanceToQuery/Safety use OpenAI by default)
    # plus our guardrail_compliance scorer on the outputs.
    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=lambda question: traced_app(question),
        scorers=[
            Correctness(model="openai:/gpt-4o-mini"),
            RelevanceToQuery(model="openai:/gpt-4o-mini"),
            Safety(model="openai:/gpt-4o-mini"),
            guardrail_compliance,
        ],
    )

    # Aggregate scores (incl. guardrail_compliance): the 'guardrails WITH
    # eval' evidence the course is missing.
    print("\n=== Eval metrics (means over the dataset) ===")
    for k, v in results.metrics.items():
        print(f"  {k}: {v}")
    print("\nOpen `mlflow ui` -> experiment 'llm-support-bot-guardrails' to see traces + eval table.")


if __name__ == "__main__":
    main()
