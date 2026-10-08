"""OpenAI access shared by every variation.

Each variation swaps in its own instrumented client (Langfuse wrapper,
OpenInference auto-instrumentation, MLflow autolog) — the call sites stay
identical so the only thing that changes is *who records the trace*.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env next to the project root

MODEL = "gpt-4o-mini"  # cheap default; override with OPENAI_MODEL env var
MODEL = os.getenv("OPENAI_MODEL", MODEL)


def get_client() -> OpenAI:
    """Plain OpenAI client (requires OPENAI_API_KEY in the environment)."""
    return OpenAI()


def ask_llm(question: str, client=None) -> str:
    """The 'app' under test: one support-bot answer for one question."""
    client = client or get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise customer-support bot. Answer in 1-3 sentences."},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    
    resp_content = resp.choices[0].message.content.strip()
    
    # Debug: print the LLM response before returning it
    print("********")  # separator for readability
    print(f"LLM response for question '{question}': {resp_content}")
    
    return resp_content


def llm_judge(prompt: str, client=None) -> str:
    """One-call LLM judge; returns raw text that callers parse themselves."""
    client = client or get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()
