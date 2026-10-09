"""Variation 01 (MLflow GenAI) configured for the Jev judge.

Same built-in scorers as the OpenAI variant: with a typesafe:/ model URI MLflow switches to its
TypeSafe-tuned prompts and records typesafe.probabilities/confidence on each assessment.
"""

import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety

from pipelines.mlflow_genai import run
from shared.jev_judge import MODEL_ID, JevJudge

JUDGE_MODEL = f"typesafe:/{MODEL_ID}"

if __name__ == "__main__":
    # genai.evaluate only enables this temporarily; explicit keeps Jev spans outside evaluate too.
    mlflow.typesafe.autolog()
    run(
        JevJudge(),  # runtime guardrail gate; the scorers below are MLflow's native TypeSafe path
        [Correctness(model=JUDGE_MODEL), RelevanceToQuery(model=JUDGE_MODEL), Safety(model=JUDGE_MODEL)],
    )
