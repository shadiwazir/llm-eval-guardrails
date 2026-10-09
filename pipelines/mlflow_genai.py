"""MLflow 3 GenAI pipeline: tracing + mlflow.genai.evaluate, judge injected by the entry file.

  * mlflow.openai.autolog()  -> every OpenAI call becomes a trace span
  * @mlflow.trace            -> the whole guarded pipeline is one trace
  * mlflow.genai.evaluate()  -> runs the injected judge scorers + guardrail_compliance
"""

import mlflow
from mlflow.entities import SpanType
from mlflow.genai.scorers import scorer

from shared.app import BLOCKED_REPLY, run_app
from shared.data import EVAL_DATASET
from shared.guardrails import check_input, check_output
from shared.judge import Judge

EXPERIMENT = "llm-support-bot-guardrails"


@scorer
def guardrail_compliance(outputs: str, expectations: dict) -> bool:
    """Was the answer blocked exactly when it should have been?"""
    should_block = expectations.get("expected") == "BLOCKED"
    return should_block == (outputs == BLOCKED_REPLY)


class _TracedJudge:
    """Records each verdict as an EVALUATOR span wrapping the judge's own (autologged) LLM span."""

    def __init__(self, judge: Judge):
        self.name = judge.name
        span = {"span_type": SpanType.EVALUATOR}
        self.safety = mlflow.trace(judge.safety, name=f"{judge.name}_safety_judge", **span)
        self.correctness = mlflow.trace(judge.correctness, name=f"{judge.name}_correctness_judge", **span)


def run(judge: Judge, judge_scorers: list) -> None:
    """Run the eval with `judge` guarding the app and `judge_scorers` grading it."""
    mlflow.set_experiment(EXPERIMENT)
    mlflow.openai.autolog()

    traced_judge = _TracedJudge(judge)
    input_gate = mlflow.trace(check_input, name="input_guardrail", span_type=SpanType.GUARDRAIL)
    output_gate = mlflow.trace(check_output, name="output_guardrail", span_type=SpanType.GUARDRAIL)

    @mlflow.trace(name="guarded-support-bot", span_type=SpanType.AGENT)  # root span: one trace per question
    def traced_app(question: str) -> str:
        result = run_app(question, traced_judge, check_in=input_gate, check_out=output_gate)
        # Tags make the verdict filterable in the UI (incl. Jev probabilities, e.g. "unsafe: jev p=0.97").
        tags = {
            "guardrail.blocked": str(result.blocked),
            "guardrail.input_violations": ",".join(result.input_report.violations) or "none",
            "guardrail.judge_backend": judge.name,
        }
        if result.output_report is not None:  # output gate runs only if the input passed
            tags["guardrail.judge_verdict"] = result.output_report.judge_verdict
            tags["guardrail.output_violations"] = ",".join(result.output_report.violations) or "none"
        # MLflow's pre-flight call runs predict_fn with tracing disabled; there is no trace to tag then.
        if mlflow.get_current_active_span() is not None:
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
