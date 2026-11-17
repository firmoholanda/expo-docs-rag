# expo docs rag

Ask questions about Expo Router and get answers that cite the exact doc passages they came from. The repo also ships an eval suite, so every change to chunking, search or prompts can be measured instead of guessed.

Python, FastAPI, local embeddings (`BAAI/bge-small-en-v1.5` through fastembed), BM25 and Claude. A small web UI shows the chat with citations and an evals dashboard.

## What's in it

- **Hybrid search.** Keyword (BM25) and semantic (embeddings) results are merged with reciprocal rank fusion. A query like `usePreventRemove` needs the keyword side; a query like "ask before leaving a screen" needs the semantic side.
- **Chunks that explain themselves.** Pages are split on headings, MDX plumbing is removed, code fences stay whole, and each chunk carries its page title and heading path.
- **Cited answers.** The model gets numbered sources and must cite them as `[1]`. If the docs don't cover the question it says so instead of guessing.
- **Works without an API key.** Search, the UI sources and the retrieval evals run locally. Only written answers and answer evals need a key.
- **Evals you can trust.** 22 answerable questions and 4 that the docs cannot answer. A test checks that every expected fact really appears in its expected page, so the eval set can't quietly drift from the corpus.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY for written answers

python -m rag.retrieval build   # optional: the built index is already in data/index (rebuilding takes ~12 min on 2 CPUs)
uvicorn rag.server:app          # chat at http://localhost:8000, dashboard at /evals
```

Export your key before starting the server (`export ANTHROPIC_API_KEY=...`) or load `.env` with your tool of choice. `AI_MODEL` picks the Claude model (default `claude-sonnet-5-5`).

## Evals

```bash
python -m evals.run                      # retrieval only, no key needed
python -m evals.run --generate           # also write answers for every question
python -m evals.run --judge              # also grade them with a model
python -m evals.run --judge --label "chunks 1400 -> 900"
pytest                                   # unit tests, including the dataset check
```

Each run is saved in `evals/results/` and shows up on the dashboard, with the history of earlier runs so a change can be compared with the one before it.

| Check | Needs a key | What it answers |
| --- | --- | --- |
| hit@1, hit@3, hit@5, MRR | no | Does a chunk from the right page come back, and how high? Scored for keyword, semantic and hybrid search. |
| Required facts present | yes | Does the answer contain the specific facts the question needs (string match, no model)? |
| Citations present, valid, on the right page | yes | Does it cite, cite only sources it was given, and cite the page that holds the answer? |
| Correct abstention | yes | On the 4 questions the docs can't answer, does it say so instead of inventing an answer? |
| Faithful, correct (judge) | yes | A model grades each answer: is every claim supported by the sources, and does it state the required facts? |

### Retrieval results

Measured on the 22 answerable questions, 753 chunks, `k = 5`:

| Search | hit@1 | hit@3 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| keyword (BM25) | 0.82 | 0.91 | 0.91 | 0.86 |
| semantic (embeddings) | 0.95 | 1.00 | 1.00 | 0.98 |
| hybrid (both, fused) | 0.91 | 1.00 | 1.00 | 0.95 |

What this says, and what it doesn't:

- Keyword search alone misses 2 questions where the user's words differ from the docs' words (for example "back button" vs `anchor`). Semantic search alone misses none in the top 3.
- Hybrid matches semantic at hit@3 but is not better at hit@1 on this set. With 22 questions one question is about 4.5 points, so I would not read this as "hybrid is worse". It is a reason to grow the question set before choosing.
- Two misses are real and unsolved: "how do I turn off the `/_sitemap` route" ranks the reserved-paths page first (it mentions the route but not how to disable it), and "what must an API route file be named" ranks a static-rendering FAQ above the API routes page.
- One dataset change was made after seeing results: question q04 also accepts `navigation.mdx`, because its "Anchor routes" section answers the question too. The first run is kept in the history as the baseline.

Answer evals (faithfulness, correctness, abstention) need an `ANTHROPIC_API_KEY`; run `python -m evals.run --judge` to fill in the dashboard.

## How it's laid out

| Path | Role |
| --- | --- |
| `rag/chunking.py` | MDX to chunks: frontmatter, headings, code fences, size limits |
| `rag/retrieval.py` | Index build/load, BM25, embeddings, reciprocal rank fusion |
| `rag/generation.py` | Prompt, numbered sources, citation parsing, abstention |
| `rag/server.py` | FastAPI: `/api/ask`, `/api/evals`, static pages |
| `web/` | Chat with citations, evals dashboard |
| `evals/dataset.jsonl` | Questions, expected pages, required facts |
| `evals/run.py`, `metrics.py`, `judge.py` | Runner, scoring functions, model judge |
| `data/docs/` | Expo Router docs (MDX), copied from the Expo repo |

## Ideas for next steps

- Add a reranker between search and generation and let the eval history show whether it pays for its latency
- Grow the question set from real user questions and mark the ones that fail
- Track cost and latency per question next to quality

## Credits

The documentation in `data/docs/` is from [expo/expo](https://github.com/expo/expo) (MIT licensed, see `data/docs/LICENSE-expo`). This project is not affiliated with Expo.
