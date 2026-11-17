"""Index building and search: keyword (BM25), semantic (embeddings) and hybrid.

Hybrid uses reciprocal rank fusion, which needs no score tuning: a chunk ranks
high if either method ranks it high.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk, load_corpus

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "data" / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODES = ("bm25", "dense", "hybrid")
RRF_K = 60

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase words, plus the pieces of snake_case and camelCase names."""
    tokens: list[str] = []
    for word in TOKEN.findall(text):
        low = word.lower()
        tokens.append(low)
        pieces = re.findall(r"[A-Z]?[a-z]+|\d+", word.replace("_", " "))
        if len(pieces) > 1:
            tokens.extend(p.lower() for p in pieces)
    return tokens


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion over several ranked lists of chunk indexes."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Embedder:
    """Thin wrapper so tests can swap in a fake without downloading a model."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)

    def passages(self, texts: list[str]) -> np.ndarray:
        # Small batches keep memory low on small machines.
        return np.array(list(self.model.embed(texts, batch_size=16)), dtype=np.float32)

    def query(self, text: str) -> np.ndarray:
        return np.array(list(self.model.embed([QUERY_PREFIX + text]))[0], dtype=np.float32)


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder | None = None):
        self.chunks = chunks
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.clip(norms, 1e-9, None)
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.search_text) for c in chunks])

    # -- building and loading -------------------------------------------------
    @classmethod
    def build(cls, docs_dir: Path = DOCS_DIR, out_dir: Path = INDEX_DIR, embedder: Embedder | None = None) -> "Index":
        chunks = load_corpus(docs_dir)
        embedder = embedder or Embedder()
        vectors = embedder.passages([c.search_text for c in chunks])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks]), encoding="utf-8")
        np.save(out_dir / "vectors.npy", vectors)
        return cls(chunks, vectors, embedder)

    @classmethod
    def load(cls, out_dir: Path = INDEX_DIR, embedder: Embedder | None = None) -> "Index":
        chunks = [Chunk(**c) for c in json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))]
        vectors = np.load(out_dir / "vectors.npy")
        return cls(chunks, vectors, embedder or Embedder())

    @classmethod
    def load_or_build(cls) -> "Index":
        if (INDEX_DIR / "chunks.json").exists() and (INDEX_DIR / "vectors.npy").exists():
            return cls.load()
        return cls.build()

    # -- searching --------------------------------------------------------------
    def _bm25_ranking(self, query: str, pool: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:pool] if scores[i] > 0]

    def _dense_ranking(self, query: str, pool: int) -> tuple[list[int], np.ndarray]:
        scores = self.vectors @ self.embedder.query(query)
        return [int(i) for i in np.argsort(-scores)[:pool]], scores

    def search(self, query: str, k: int = 5, mode: str = "hybrid", pool: int = 30) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "bm25":
            ranking = self._bm25_ranking(query, pool)
            raw = self.bm25.get_scores(tokenize(query))
            scored = [(i, float(raw[i])) for i in ranking]
        elif mode == "dense":
            ranking, raw = self._dense_ranking(query, pool)
            scored = [(i, float(raw[i])) for i in ranking]
        else:
            dense_rank, _ = self._dense_ranking(query, pool)
            scored = rrf([self._bm25_ranking(query, pool), dense_rank])
        # One page can fill the whole result list. Keep at most 2 chunks per page
        # so the answer can draw on more than one place.
        hits: list[Hit] = []
        per_source: dict[str, int] = {}
        for idx, score in scored:
            chunk = self.chunks[idx]
            if per_source.get(chunk.source, 0) >= 2:
                continue
            per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
            hits.append(Hit(chunk=chunk, score=score, rank=len(hits) + 1))
            if len(hits) == k:
                break
        return hits


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["build"]:
        built = Index.build()
        print(f"indexed {len(built.chunks)} chunks into {INDEX_DIR}")
    else:
        raise SystemExit("usage: python -m rag.retrieval build")
