"""Small API and static pages: a chat with citations and an evals dashboard.

    uvicorn rag.server:app --reload
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .generation import answer, make_client
from .retrieval import MODES, Index

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
RESULTS = ROOT / "evals" / "results"

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["index"] = Index.load_or_build()
    state["client"] = make_client()
    yield


app = FastAPI(title="expo docs rag", lifespan=lifespan)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = "hybrid"
    k: int = Field(default=5, ge=1, le=10)


@app.get("/api/health")
def health():
    return {"ok": True, "chunks": len(state["index"].chunks), "generation": state["client"] is not None}


@app.post("/api/ask")
def ask(body: Ask):
    if body.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {list(MODES)}")
    hits = state["index"].search(body.question, k=body.k, mode=body.mode)
    result = answer(body.question, hits, client=state["client"])
    return {
        "answer": result.text,
        "generated": result.generated,
        "abstained": result.abstained,
        "model": result.model,
        "cited": result.cited,
        "sources": [
            {
                "n": h.rank,
                "title": h.chunk.title,
                "heading": h.chunk.heading,
                "url": h.chunk.url,
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ],
    }


@app.get("/api/evals")
def evals():
    """Latest run in full, plus a one-line summary of every earlier run."""
    latest = RESULTS / "latest.json"
    if not latest.exists():
        raise HTTPException(404, "No eval runs yet. Run: python -m evals.run")
    runs = []
    for path in sorted(RESULTS.glob("2*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs.append(
            {
                "created": data["created"],
                "label": data.get("label", ""),
                "hybrid": data["retrieval"]["hybrid"]["summary"],
                "generation": (data.get("generation") or {}).get("summary"),
            }
        )
    return {"latest": json.loads(latest.read_text(encoding="utf-8")), "history": runs}


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/evals")
def evals_page():
    return FileResponse(WEB / "evals.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
