"""Variation 01 (MLflow GenAI) configured for the Jev judge: custom scorers, one Noul question each."""

from mlflow.genai.scorers import scorer

from pipelines.mlflow_genai import run
from shared.jev_judge import JevJudge

judge = JevJudge()


@scorer
def jev_correctness(inputs: dict, outputs: str, expectations: dict) -> bool:
    return judge.correctness(inputs["question"], outputs, expectations["expected_response"])


@scorer
def jev_relevance(inputs: dict, outputs: str) -> bool:
    return judge.relevance(inputs["question"], outputs)


@scorer
def jev_safety(outputs: str) -> bool:
    return judge.safety(outputs) == "safe"


if __name__ == "__main__":
    run(judge, [jev_correctness, jev_relevance, jev_safety])
