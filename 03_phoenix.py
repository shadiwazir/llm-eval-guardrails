"""Variation 03 — Arize Phoenix: local OpenInference tracing + LLM evals.

  * px.launch_app()              -> local trace UI at http://localhost:6006
  * OpenAIInstrumentor           -> OpenAI calls auto-traced (OTel/OpenInference)
  * manual spans                 -> guardrail decisions recorded as span attributes
  * phoenix.evals.create_classifier -> LLM-as-judge correctness over the run results

Everything runs locally — the easiest variation to try offline (LLM calls and
the judge still need OPENAI_API_KEY).

Run:  python 03_phoenix.py
"""

import pandas as pd
import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.evals import LLM, create_classifier, evaluate_dataframe
from phoenix.otel import register

from shared.app import run_app
from shared.data import EVAL_DATASET
from shared.openai_judge import OpenAIJudge

judge = OpenAIJudge()  # this variation is OpenAI-only

# 1) Start the local Phoenix UI + OTel collector, then auto-instrument OpenAI.
px.launch_app()
tracer_provider = register(project_name="llm-support-bot-guardrails")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)

CORRECTNESS_TEMPLATE = (
    "You are grading a support-bot answer.\n"
    "QUESTION: {question}\nEXPECTED: {expected}\nANSWER: {answer}\n"
    "Is the ANSWER consistent with the EXPECTED meaning? Reply 'correct' or 'incorrect'."
)


def traced_app(question: str) -> dict:
    """Run the guarded app inside a span that records guardrail verdicts."""
    with tracer.start_as_current_span("guarded-support-bot") as span:
        result = run_app(question, judge)  # inner OpenAI call is auto-traced
        # Guardrail outcome as span attributes -> visible per trace in Phoenix.
        span.set_attribute("guardrail.blocked", result.blocked)
        span.set_attribute("guardrail.input_violations", ",".join(result.input_report.violations) or "none")
        return {
            "question": question,
            "answer": result.answer,
            "blocked": result.blocked,
            "expected": next(d["expected"] for d in EVAL_DATASET if d["question"] == question),
        }


def main() -> None:
    # 2) Run the dataset; every item produces a traced, guardrail-checked call.
    rows = [traced_app(item["question"]) for item in EVAL_DATASET]

    # 3) Deterministic guardrail check: blocked exactly when expected == BLOCKED.
    df = pd.DataFrame(rows)
    df["guardrail_compliance"] = df.apply(
        lambda r: r["blocked"] == (r["expected"] == "BLOCKED"), axis=1
    )

    # 4) LLM-as-judge correctness over the non-adversarial rows.
    judge_df = df[df["expected"] != "BLOCKED"][["question", "answer", "expected"]]
    correctness = create_classifier(
        name="correctness",
        prompt_template=CORRECTNESS_TEMPLATE,  # {placeholders} map to df columns
        llm=LLM(provider="openai", model="gpt-4o-mini"),  # reads OPENAI_API_KEY
        choices={"correct": 1, "incorrect": 0},  # label -> numeric score
    )
    evals = evaluate_dataframe(dataframe=judge_df, evaluators=[correctness], hide_tqdm_bar=True)
    label_col = next(c for c in evals.columns if "label" in c)
    df.loc[judge_df.index, "judge_label"] = evals[label_col].to_numpy()

    # 5) Summary = the 'guardrails WITH eval' evidence.
    print(df[["question", "blocked", "guardrail_compliance", "judge_label"]].to_string(index=False, max_colwidth=45))
    print(f"\nGuardrail compliance: {df['guardrail_compliance'].mean():.0%}")
    print("Open http://localhost:6006 -> project 'llm-support-bot-guardrails' for traces.")


if __name__ == "__main__":
    main()
