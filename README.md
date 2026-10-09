# LLM Evaluation, Tracing & Guardrails — the missing MLflow lab pieces

Companion project to `10-Experiment_Tracking_MLflow`. The course covers
classic-ML tracking (params/metrics/registry) only — this project adds the
three missing pieces: **LLM tracing**, **LLM evaluation**, and **guardrails
whose decisions are measured by evals**.

## Layout

| File | Tool | What it shows |
|---|---|---|
| `run.py` | — | Single entry point: `python run.py <variation> [--judge openai\|jev]` |
| `01_mlflow_genai_{openai,jev}.py` | MLflow 3 GenAI | `mlflow.genai.evaluate` + LLM scorers + `@mlflow.trace` / OpenAI autolog tracing |
| `02_langfuse_{openai,jev}.py` | Langfuse | `@observe` traces + eval & guardrail scores attached to each trace |
| `03_phoenix.py` | Arize Phoenix | Local OpenInference tracing + `create_classifier` LLM-judge evals (OpenAI only) |
| `04_guardrails_ai_demo.py` | guardrails-ai | Dedicated guardrail library (rules + Hub validators) |
| `pipelines/` | — | Judge-agnostic tracing/eval wiring for MLflow and Langfuse; the judge is injected |
| `shared/` | — | `app.py` (guarded pipeline), `guardrails.py` (rule checks + injected safety judge), `judge.py` (Judge contract), `openai_judge.py` / `jev_judge.py` (backends), `llm.py`, `data.py` (eval dataset incl. one adversarial item) |

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

The judge (safety gate inside the app + eval scoring) is **injected**, not
switched on. Each `*_openai.py` / `*_jev.py` file builds exactly one judge and
passes it to the shared pipeline, so no file branches on the backend.

- **OpenAI**: LLM-as-judge; MLflow uses its built-in scorers.
- **Jev** (TypeSafe System One): returns typed probabilities instead of text,
  much cheaper per judgment. Needs `TYPESAFE_API_KEY` and fails fast without
  it (no silent fallback to OpenAI). In MLflow (>=3.17) the same built-in
  scorers run on `typesafe:/jev-1.13.0`, and each assessment records
  `typesafe.probabilities` / `typesafe.confidence`. The model ID is pinned so
  scores stay comparable; `jev-latest` moves when TypeSafe ships.
  `JEV_THRESHOLD` only applies to the runtime gate and Langfuse; MLflow's native
  judges use a fixed 0.5.

Every MLflow trace is a typed span tree: `AGENT` root -> `GUARDRAIL` gates ->
`EVALUATOR` judge verdict -> the raw `CHAT_MODEL`/`LLM` call (OpenAI autolog or
`mlflow.typesafe.autolog()`), and is tagged with `guardrail.judge_backend` /
`guardrail.judge_verdict` (incl. the Jev probability) for filtering.

Adding a backend = one class implementing `shared/judge.py` + one entry file
per variation + one line in `run.py`.

## Run

```bash
python run.py mlflow --judge openai     # or --judge jev (default: openai)
python run.py langfuse --judge jev
python run.py phoenix
python run.py guardrails-ai
```

Each entry file also runs directly, e.g. `python 01_mlflow_genai_jev.py`.

Each script runs the same guarded support-bot over the same dataset and ends
with the same evidence: **guardrail compliance** (blocked exactly when it
should be) plus **LLM-judge correctness** — i.e. guardrails *with* eval.
