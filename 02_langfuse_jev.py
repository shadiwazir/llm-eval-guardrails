"""Variation 02 (Langfuse) configured for the Jev judge."""

from pipelines.langfuse_eval import run
from shared.jev_judge import JevJudge

if __name__ == "__main__":
    run(JevJudge())
