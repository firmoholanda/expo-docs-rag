from rag.chunking import chunk_document, clean_lines, parse_frontmatter, url_for

DOC = """---
title: Reserved paths
description: x
---

import { FileTree } from '~/ui/components/FileTree';

Some intro text that is long enough to be kept as its own overview chunk for the page, with detail.

## `/manifest`

The dev server serves the native app manifest at this path. If you create a route at app/manifest.tsx it is replaced.

<FileTree files={['a']} />

```bash
# a comment that must not become a heading
## also not a heading
```

## Other

Another section with enough words in it to pass the minimum length so that it becomes a chunk of its own, and then a few more words to be safe.
"""


def test_frontmatter_is_parsed_and_removed():
    meta, body = parse_frontmatter(DOC)
    assert meta["title"] == "Reserved paths"
    assert not body.startswith("---")


def test_mdx_plumbing_is_dropped_but_code_is_kept():
    lines = clean_lines(parse_frontmatter(DOC)[1])
    text = "\n".join(lines)
    assert "import {" not in text
    assert "<FileTree" not in text
    assert "## also not a heading" in text


def test_chunks_carry_title_heading_and_url():
    chunks = chunk_document("router/reference/reserved-paths.mdx", DOC)
    headings = [c.heading for c in chunks]
    assert "/manifest" in headings and "Other" in headings
    manifest = next(c for c in chunks if c.heading == "/manifest")
    assert manifest.title == "Reserved paths"
    assert manifest.url == "https://docs.expo.dev/router/reference/reserved-paths/"
    assert "## also not a heading" in manifest.text  # fence content stayed in the same chunk


def test_long_sections_are_split_under_the_limit():
    body = "---\ntitle: T\n---\n\n## Big\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(30))
    chunks = chunk_document("x.mdx", body)
    assert len(chunks) > 1
    assert all(len(c.text) <= 2800 for c in chunks)


def test_url_for_top_level_page():
    assert url_for("faq.mdx") == "https://docs.expo.dev/faq/"
