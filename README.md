# LLM Evaluation, Tracing & Guardrails — the missing MLflow lab pieces

Companion project to `10-Experiment_Tracking_MLflow`. The course covers
classic-ML tracking (params/metrics/registry) only — this project adds the
three missing pieces: **LLM tracing**, **LLM evaluation**, and **guardrails
whose decisions are measured by evals**.

## Layout

| File | Tool | What it shows |
|---|---|---|
| `01_mlflow_genai.py` | MLflow 3 GenAI | `mlflow.genai.evaluate` + LLM scorers + `@mlflow.trace` / OpenAI autolog tracing |
| `02_langfuse.py` | Langfuse | `@observe` traces + eval & guardrail scores attached to each trace |
| `03_phoenix.py` | Arize Phoenix | Local OpenInference tracing + `create_classifier` LLM-judge evals |
| `04_guardrails_ai_demo.py` | guardrails-ai | Dedicated guardrail library (rules + Hub validators) |
| `shared/` | — | Reused by all: `app.py` (guarded pipeline), `guardrails.py` (rule + LLM-judge checks), `llm.py`, `data.py` (eval dataset incl. one adversarial item) |

## Setup

```bash
cd "my_work/llm-eval-guardrails"
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip                # new resolver = much faster
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

If one-shot install hangs (old pip backtracking on the big dependency graph),
install in stages instead: core first, then `mlflow`, `langfuse`,
`arize-phoenix`, `guardrails-ai` — each stage resolves in seconds.

Required: `OPENAI_API_KEY` (all variations call OpenAI, incl. LLM judges).

### Tool-specific notes

- **01 MLflow**: no server needed; after running, `mlflow ui` → experiment
  `llm-support-bot-guardrails` for traces + the eval table.
- **02 Langfuse**: create a free project at <https://cloud.langfuse.com> and put
  the keys in `.env`, or self-host:
  `git clone https://github.com/langfuse/langfuse && cd langfuse && docker compose up`.
- **03 Phoenix**: fully local; the UI opens at <http://localhost:6006>.
- **04 guardrails-ai**: works out of the box with the custom validator. For Hub
  validators: `guardrails configure` (free token) then
  `guardrails hub install hub://guardrails/toxic_language`
  `guardrails hub install hub://guardrails/detect_pii`.

## Judge backend: OpenAI or Jev

All judge calls (the safety gate in `shared/guardrails.py`, correctness in
`02_langfuse.py`) can run on **OpenAI** (default) or **Jev** — TypeSafe's
System One decision model, which returns typed probabilities instead of text
and is much cheaper per judgment. Select the backend in `.env`:

```
JUDGE_BACKEND=jev          # requires TYPESAFE_API_KEY
```

or per run: `python 01_mlflow_genai.py --judge jev`. Missing key -> warns once
and falls back to OpenAI. In `01`, `--judge jev` also swaps the three built-in
MLflow scorers for Jev-backed custom scorers (`jev_correctness/relevance/
safety`), and every trace is tagged with the judge verdict (incl. the Jev
probability) — the only Jev evidence in traces, since autolog only instruments
OpenAI calls.

## Run

```bash
python 01_mlflow_genai.py
python 02_langfuse.py
python 03_phoenix.py
python 04_guardrails_ai_demo.py
```

Each script runs the same guarded support-bot over the same dataset and ends
with the same evidence: **guardrail compliance** (blocked exactly when it
should be) plus **LLM-judge correctness** — i.e. guardrails *with* eval.
