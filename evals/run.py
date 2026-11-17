"""Run the evals and write a result file the dashboard can read.

    python -m evals.run                     # retrieval only, no API key needed
    python -m evals.run --generate          # also write answers (needs ANTHROPIC_API_KEY)
    python -m evals.run --generate --judge  # also grade faithfulness and correctness with a model
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rag.generation import answer as generate_answer
from rag.generation import build_context, make_client
from rag.retrieval import MODES, Index

from .judge import judge as judge_answer
from .metrics import citation_report, fact_coverage, first_relevant_rank, rate, retrieval_summary

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
RESULTS = HERE / "results"


def load_dataset(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_retrieval(index: Index, rows: list[dict], k: int) -> dict:
    """Score every search mode on the answerable questions."""
    out: dict = {}
    answerable = [r for r in rows if r["answerable"]]
    for mode in MODES:
        ranks, per_question = [], []
        for row in answerable:
            hits = index.search(row["question"], k=k, mode=mode)
            rank = first_relevant_rank([h.chunk.source for h in hits], row["sources"])
            ranks.append(rank)
            per_question.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "rank": rank,
                    "top": [{"source": h.chunk.source, "heading": h.chunk.heading} for h in hits[:3]],
                }
            )
        out[mode] = {"summary": retrieval_summary(ranks), "questions": per_question}
    return out


def run_generation(index: Index, rows: list[dict], k: int, client, use_judge: bool) -> dict:
    per_question = []
    for row in rows:
        hits = index.search(row["question"], k=k, mode="hybrid")
        ans = generate_answer(row["question"], hits, client=client)
        item: dict = {
            "id": row["id"],
            "question": row["question"],
            "answerable": row["answerable"],
            "answer": ans.text,
            "abstained": ans.abstained,
            "cited": ans.cited,
        }
        if row["answerable"]:
            cited_sources = [hits[n - 1].chunk.source for n in ans.cited]
            item["fact_coverage"] = fact_coverage(ans.text, row["facts"])
            item["citations"] = citation_report(ans.cited, ans.invalid_citations, cited_sources, row["sources"])
            if use_judge and not ans.abstained:
                item["judge"] = judge_answer(client, row["question"], ans.text, build_context(hits), row["facts"])
        per_question.append(item)

    answerable = [q for q in per_question if q["answerable"]]
    unanswerable = [q for q in per_question if not q["answerable"]]
    summary = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "fact_coverage": rate([q["fact_coverage"] for q in answerable]),
        "answered_rate": rate([not q["abstained"] for q in answerable]),
        "citation_present": rate([q["citations"]["has_citation"] for q in answerable if not q["abstained"]]),
        "citation_valid": rate([q["citations"]["all_valid"] for q in answerable if not q["abstained"]]),
        "citation_on_expected_page": rate(
            [q["citations"]["cites_expected_page"] for q in answerable if not q["abstained"]]
        ),
        "correct_abstention": rate([q["abstained"] for q in unanswerable]),
    }
    judged = [q["judge"] for q in answerable if "judge" in q]
    if judged:
        summary["faithful"] = rate([j["faithful"] for j in judged])
        summary["correct"] = rate([j["correct"] for j in judged])
    return {"summary": summary, "questions": per_question}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="chunks passed to the model")
    parser.add_argument("--generate", action="store_true", help="also evaluate written answers")
    parser.add_argument("--judge", action="store_true", help="grade answers with a model (implies --generate)")
    parser.add_argument("--label", default="", help="short note saved with the run, e.g. 'bigger chunks'")
    args = parser.parse_args()

    rows = load_dataset()
    index = Index.load_or_build()
    result: dict = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "config": {"k": args.k, "chunks": len(index.chunks), "embed_model": "BAAI/bge-small-en-v1.5"},
        "retrieval": run_retrieval(index, rows, args.k),
        "generation": None,
    }

    if args.generate or args.judge:
        client = make_client()
        if client is None:
            raise SystemExit("Set ANTHROPIC_API_KEY to evaluate generated answers.")
        result["generation"] = run_generation(index, rows, args.k, client, use_judge=args.judge)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{stamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copyfile(path, RESULTS / "latest.json")

    print(f"{'mode':8} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'mrr':>6}")
    for mode, data in result["retrieval"].items():
        s = data["summary"]
        print(f"{mode:8} {s['hit@1']:6.2f} {s['hit@3']:6.2f} {s['hit@5']:6.2f} {s['mrr']:6.2f}")
    if result["generation"]:
        print("\ngeneration:", json.dumps(result["generation"]["summary"], indent=2))
    print(f"\nsaved {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
