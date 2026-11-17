"""Plain functions that score retrieval and answers. No model calls in here."""

from __future__ import annotations

from statistics import mean


def first_relevant_rank(retrieved_sources: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk whose page is an expected source."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return rank
    return None


def retrieval_summary(ranks: list[int | None], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    summary = {"n": n}
    for k in ks:
        summary[f"hit@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
    summary["mrr"] = mean((1 / r) if r else 0.0 for r in ranks)
    return summary


def fact_coverage(answer: str, facts: list[str]) -> float:
    """Share of required facts that appear in the answer (case-insensitive)."""
    if not facts:
        return 1.0
    low = answer.lower()
    return sum(1 for f in facts if f.lower() in low) / len(facts)


def citation_report(cited: list[int], invalid: list[int], cited_sources: list[str], expected: list[str]) -> dict:
    """Did the answer cite at all, cite only real sources, and cite the right page?"""
    return {
        "has_citation": bool(cited),
        "all_valid": not invalid,
        "cites_expected_page": any(s in expected for s in cited_sources),
    }


def rate(values: list[bool | float]) -> float:
    return sum(float(v) for v in values) / len(values) if values else 0.0
