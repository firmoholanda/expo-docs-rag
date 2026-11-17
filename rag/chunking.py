"""Turn the .mdx docs into small, self-describing chunks.

Each chunk keeps the page title and the heading path, so a chunk such as
"Router > Reserved paths > /manifest" still makes sense on its own when it is
retrieved and shown next to an answer.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHARS = 1400
MIN_CHARS = 120
DOCS_BASE_URL = "https://docs.expo.dev"

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HEADING = re.compile(r"^(#{2,3})\s+(.*\S)\s*$")
FENCE = re.compile(r"^```")
# Lines that are pure MDX plumbing: imports, JSX-only lines and comments.
IMPORT_LINE = re.compile(r"^\s*import\s+.*from\s+['\"].*['\"];?\s*$")
JSX_ONLY = re.compile(r"^\s*</?[A-Z][A-Za-z0-9.]*[^>]*/?>\s*$")
COMMENT = re.compile(r"^\s*\{/\*.*\*/\}\s*$")


@dataclass
class Chunk:
    id: str
    source: str  # path relative to the docs folder, e.g. router/web/api-routes.mdx
    url: str
    title: str
    heading: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def search_text(self) -> str:
        """What gets embedded and indexed: context first, then the body."""
        return f"{self.title} > {self.heading}\n{self.text}"


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    match = FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, raw[match.end():]


def clean_lines(body: str) -> list[str]:
    """Drop MDX plumbing but never touch the inside of code fences."""
    out: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if IMPORT_LINE.match(line) or JSX_ONLY.match(line) or COMMENT.match(line):
            continue
        out.append(line)
    return out


def split_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split on ## and ### headings, ignoring # lines inside code fences."""
    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    in_fence = False
    h2 = ""
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            level, text = len(match.group(1)), match.group(2).strip("` ")
            if level == 2:
                h2 = text
                name = text
            else:
                name = f"{h2} > {text}" if h2 else text
            sections.append((name, []))
        else:
            sections[-1][1].append(line)
    return sections


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split on blank lines, keeping code fences whole when they fit."""
    if len(text) <= max_chars:
        return [text]
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        current.append(line)
        if not in_fence and line.strip() == "":
            blocks.append("\n".join(current).strip())
            current = []
    if current:
        blocks.append("\n".join(current).strip())

    parts: list[str] = []
    buf = ""
    for block in blocks:
        if not block:
            continue
        if buf and len(buf) + len(block) + 2 > max_chars:
            parts.append(buf)
            buf = block
        else:
            buf = f"{buf}\n\n{block}" if buf else block
    if buf:
        parts.append(buf)
    # A single huge code block is hard-cut as a last resort.
    final: list[str] = []
    for part in parts:
        while len(part) > max_chars * 2:
            final.append(part[:max_chars * 2])
            part = part[max_chars * 2:]
        final.append(part)
    return final


def url_for(source: str) -> str:
    path = source.removesuffix(".mdx")
    return f"{DOCS_BASE_URL}/{path}/"


def chunk_document(source: str, raw: str) -> list[Chunk]:
    meta, body = parse_frontmatter(raw)
    title = meta.get("title", Path(source).stem)
    chunks: list[Chunk] = []
    for heading, lines in split_sections(clean_lines(body)):
        text = "\n".join(lines).strip()
        if len(text) < MIN_CHARS:
            continue
        for part in split_long(text):
            if len(part) < MIN_CHARS:
                continue
            chunks.append(
                Chunk(
                    id=f"{source}#{len(chunks)}",
                    source=source,
                    url=url_for(source),
                    title=title,
                    heading=heading,
                    text=part,
                )
            )
    return chunks


def load_corpus(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*.mdx")):
        source = path.relative_to(docs_dir).as_posix()
        chunks.extend(chunk_document(source, path.read_text(encoding="utf-8")))
    return chunks
