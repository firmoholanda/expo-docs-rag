"""LLM-as-judge for the two things string matching cannot check.

faithful: every claim in the answer is supported by the sources it was given.
correct:  the answer states the required facts, in meaning not just in wording.
"""

from __future__ import annotations

import json
import os
import re

JUDGE_SYSTEM = """You grade answers from a documentation assistant. Reply with JSON only:
{"faithful": true|false, "correct": true|false, "reason": "one short sentence"}

faithful = every claim in the answer is supported by the provided sources. Unsupported extra claims make it false.
correct = the answer clearly conveys all of the required facts. Be strict: a vague answer is not correct."""


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in judge reply: {text[:120]!r}")
    return json.loads(match.group(0))


def judge(client, question: str, answer: str, context: str, facts: list[str], model: str | None = None) -> dict:
    model = model or os.environ.get("JUDGE_MODEL") or os.environ.get("AI_MODEL") or "claude-sonnet-5-5"
    prompt = (
        f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer to grade:\n{answer}\n\n"
        f"Required facts: {', '.join(facts) if facts else '(none)'}"
    )
    message = client.messages.create(
        model=model,
        max_tokens=200,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
    data = extract_json(text)
    return {
        "faithful": bool(data.get("faithful")),
        "correct": bool(data.get("correct")),
        "reason": str(data.get("reason", ""))[:200],
    }
