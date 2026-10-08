"""Variation 01 (MLflow GenAI) configured for OpenAI judges: MLflow's built-in scorers."""

from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety

from pipelines.mlflow_genai import run
from shared.openai_judge import OpenAIJudge

JUDGE_MODEL = "openai:/gpt-4o-mini"

if __name__ == "__main__":
    run(
        OpenAIJudge(),
        [Correctness(model=JUDGE_MODEL), RelevanceToQuery(model=JUDGE_MODEL), Safety(model=JUDGE_MODEL)],
    )
