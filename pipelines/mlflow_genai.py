"""MLflow 3 GenAI pipeline: tracing + mlflow.genai.evaluate, judge injected by the entry file.

  * mlflow.openai.autolog()  -> every OpenAI call becomes a trace span
  * @mlflow.trace            -> the whole guarded pipeline is one trace
  * mlflow.genai.evaluate()  -> runs the injected judge scorers + guardrail_compliance
"""

import mlflow
from mlflow.genai.scorers import scorer

from shared.app import BLOCKED_REPLY, run_app
from shared.data import EVAL_DATASET
from shared.judge import Judge

EXPERIMENT = "llm-support-bot-guardrails"


@scorer
def guardrail_compliance(outputs: str, expectations: dict) -> bool:
    """Was the answer blocked exactly when it should have been?"""
    should_block = expectations.get("expected") == "BLOCKED"
    return should_block == (outputs == BLOCKED_REPLY)


def run(judge: Judge, judge_scorers: list) -> None:
    """Run the eval with `judge` guarding the app and `judge_scorers` grading it."""
    mlflow.set_experiment(EXPERIMENT)
    mlflow.openai.autolog()

    @mlflow.trace(name="guarded-support-bot")  # one parent span per question
    def traced_app(question: str) -> str:
        result = run_app(question, judge)
        # Tags are the only trace evidence for non-OpenAI judges (e.g. Jev makes no spans).
        tags = {
            "guardrail.blocked": str(result.blocked),
            "guardrail.input_violations": ",".join(result.input_report.violations) or "none",
            "guardrail.judge_backend": judge.name,
        }
        if result.output_report is not None:  # output gate runs only if the input passed
            tags["guardrail.judge_verdict"] = result.output_report.judge_verdict
            tags["guardrail.output_violations"] = ",".join(result.output_report.violations) or "none"
        mlflow.update_current_trace(tags=tags)
        return result.answer

    eval_data = [
        {
            "inputs": {"question": item["question"]},
            # "expected" feeds guardrail_compliance; "expected_response" feeds correctness scorers.
            "expectations": {
                "expected": item["expected"],
                "expected_response": BLOCKED_REPLY if item["expected"] == "BLOCKED" else item["expected"],
            },
        }
        for item in EVAL_DATASET
    ]

    print(f"Judge backend: {judge.name}")
    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=lambda question: traced_app(question),
        scorers=[*judge_scorers, guardrail_compliance],
    )

    print("\n=== Eval metrics (means over the dataset) ===")
    for k, v in results.metrics.items():
        print(f"  {k}: {v}")
    print(f"\nOpen `mlflow ui` -> experiment '{EXPERIMENT}' to see traces + eval table.")
