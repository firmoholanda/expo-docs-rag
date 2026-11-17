import numpy as np

from rag.chunking import Chunk
from rag.retrieval import Index, rrf, tokenize


class FakeEmbedder:
    """Maps a few keywords to fixed directions so no model download is needed."""

    AXES = ["sitemap", "middleware", "modal", "drawer"]

    def _vec(self, text: str) -> np.ndarray:
        low = text.lower()
        v = np.array([low.count(a) for a in self.AXES], dtype=np.float32) + 1e-3
        return v

    def passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


def make_index():
    def chunk(i, source, heading, text):
        return Chunk(id=f"{source}#{i}", source=source, url="u", title=source, heading=heading, text=text)

    chunks = [
        chunk(0, "a.mdx", "Sitemap", "The sitemap lists all routes. Disable it with sitemap false in the plugin."),
        chunk(1, "b.mdx", "Middleware", "Create a +middleware.ts file. The middleware must be the default export."),
        chunk(2, "c.mdx", "Modals", "Set presentation to modal on the route to present a modal screen."),
        chunk(3, "d.mdx", "Drawer", "The drawer navigator opens from the side of the screen."),
    ]
    emb = FakeEmbedder()
    return Index(chunks, emb.passages([c.search_text for c in chunks]), emb)


def test_tokenize_splits_names():
    toks = tokenize("useLocalSearchParams and snake_case")
    assert "uselocalsearchparams" in toks and "local" in toks and "search" in toks and "snake" in toks


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 3, 1]])
    assert fused[0][0] == 2


def test_each_mode_finds_the_right_page():
    index = make_index()
    for mode in ("bm25", "dense", "hybrid"):
        hits = index.search("how do I disable the sitemap", k=2, mode=mode)
        assert hits[0].chunk.source == "a.mdx", mode


def test_results_are_capped_per_page():
    index = make_index()
    extra = [index.chunks[0].__class__(id=f"a.mdx#{i}", source="a.mdx", url="u", title="a", heading=f"H{i}",
                                       text="sitemap sitemap sitemap") for i in range(1, 5)]
    chunks = index.chunks + extra
    emb = FakeEmbedder()
    big = Index(chunks, emb.passages([c.search_text for c in chunks]), emb)
    hits = big.search("sitemap", k=5, mode="hybrid")
    assert sum(1 for h in hits if h.chunk.source == "a.mdx") <= 2


def test_bad_mode_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        make_index().search("x", mode="nope")
