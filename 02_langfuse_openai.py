"""Variation 02 (Langfuse) configured for the OpenAI judge."""

from langfuse.openai import openai

from pipelines.langfuse_eval import run
from shared.openai_judge import OpenAIJudge

if __name__ == "__main__":
    run(OpenAIJudge(client=openai.OpenAI()))  # Langfuse-wrapped client: judge calls are traced
