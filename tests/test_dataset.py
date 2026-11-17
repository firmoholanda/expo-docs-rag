"""The eval set is only worth trusting if every expected fact really is in its source page."""

import json

import pytest

from rag.retrieval import DOCS_DIR

ROWS = [json.loads(l) for l in (DOCS_DIR.parent.parent / "evals" / "dataset.jsonl").read_text().splitlines() if l.strip()]


def test_ids_are_unique():
    assert len({r["id"] for r in ROWS}) == len(ROWS)


def test_has_unanswerable_questions():
    assert sum(1 for r in ROWS if not r["answerable"]) >= 3


@pytest.mark.parametrize("row", [r for r in ROWS if r["answerable"]], ids=lambda r: r["id"])
def test_expected_pages_exist_and_contain_the_facts(row):
    texts = []
    for source in row["sources"]:
        path = DOCS_DIR / source
        assert path.exists(), f"{source} is missing"
        texts.append(path.read_text(encoding="utf-8").lower())
    joined = "\n".join(texts)
    for fact in row["facts"]:
        assert fact.lower() in joined, f"{row['id']}: '{fact}' not found in expected sources"


@pytest.mark.parametrize("row", [r for r in ROWS if not r["answerable"]], ids=lambda r: r["id"])
def test_unanswerable_questions_have_no_expected_source(row):
    assert row["sources"] == [] and row["facts"] == []
