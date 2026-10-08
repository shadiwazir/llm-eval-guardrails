"""Single entry point: python run.py <variation> [--judge openai|jev]

Each (variation, judge) pair maps to its own file that is already configured
for exactly one judge backend; this script only picks and runs that file.
"""

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).parent

TARGETS = {
    ("mlflow", "openai"): "01_mlflow_genai_openai.py",
    ("mlflow", "jev"): "01_mlflow_genai_jev.py",
    ("langfuse", "openai"): "02_langfuse_openai.py",
    ("langfuse", "jev"): "02_langfuse_jev.py",
    ("phoenix", "openai"): "03_phoenix.py",
    ("guardrails-ai", "openai"): "04_guardrails_ai_demo.py",
}


def main() -> None:
    variations = sorted({v for v, _ in TARGETS})
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variation", choices=variations)
    parser.add_argument("--judge", choices=["openai", "jev"], default="openai")
    args = parser.parse_args()

    target = TARGETS.get((args.variation, args.judge))
    if target is None:
        available = sorted(j for v, j in TARGETS if v == args.variation)
        sys.exit(f"'{args.variation}' has no '{args.judge}' variant (available: {', '.join(available)})")

    runpy.run_path(str(ROOT / target), run_name="__main__")


if __name__ == "__main__":
    main()
