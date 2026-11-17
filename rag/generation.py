"""Answer a question from retrieved chunks, with numbered citations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .retrieval import Hit

DEFAULT_MODEL = "claude-sonnet-5-5"
ABSTAIN = "I couldn't find that in the Expo Router docs."

SYSTEM = f"""You answer questions about Expo Router using only the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not add outside knowledge.
- Cite every claim with its source number in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: {ABSTAIN}
- Keep answers short and practical. Show a code snippet only when a source has one that helps.
- If sources disagree or only partly answer, say what is covered and what is not."""

CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    cited: list[int] = field(default_factory=list)  # 1-based source numbers actually cited
    invalid_citations: list[int] = field(default_factory=list)
    abstained: bool = False
    model: str = ""
    generated: bool = True  # False when no API key and only sources were returned


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        c = hit.chunk
        blocks.append(f"[{n}] {c.title} > {c.heading} ({c.url})\n{c.text}")
    return "\n\n".join(blocks)


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) citation numbers in order of first appearance."""
    valid: list[int] = []
    invalid: list[int] = []
    for raw in CITATION.findall(text):
        n = int(raw)
        bucket = valid if 1 <= n <= n_sources else invalid
        if n not in bucket:
            bucket.append(n)
    return valid, invalid


def is_abstention(text: str) -> bool:
    return text.strip().lower().startswith("i couldn't find that")


def load_env_file(path: Path | None = None) -> None:
    """Read KEY=value lines from .env without overriding variables already set."""
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def make_client():
    """Returns an Anthropic client, or None when no key is configured."""
    load_env_file()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    from anthropic import Anthropic

    return Anthropic()


def answer(question: str, hits: list[Hit], client=None, model: str | None = None) -> Answer:
    model = model or os.environ.get("AI_MODEL") or DEFAULT_MODEL
    if client is None:
        # No key: still useful. Show the best passage so the UI and retrieval evals work.
        top = hits[0].chunk if hits else None
        text = (
            f"No API key is set, so here is the closest passage instead of a written answer.\n\n"
            f"From {top.title} > {top.heading} [1]:\n\n```\n{top.text[:600]}\n```"
            if top
            else ABSTAIN
        )
        return Answer(text=text, cited=[1] if top else [], abstained=not top, model="none", generated=False)

    if not hits:
        return Answer(text=ABSTAIN, abstained=True, model=model)

    message = client.messages.create(
        model=model,
        max_tokens=700,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Sources:\n\n{build_context(hits)}\n\nQuestion: {question}",
            }
        ],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text").strip()
    valid, invalid = parse_citations(text, len(hits))
    return Answer(
        text=text,
        cited=valid,
        invalid_citations=invalid,
        abstained=is_abstention(text),
        model=model,
    )
