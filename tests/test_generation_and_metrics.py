from types import SimpleNamespace

from evals.judge import extract_json, judge
from evals.metrics import citation_report, fact_coverage, first_relevant_rank, retrieval_summary
from rag.chunking import Chunk
from rag.generation import ABSTAIN, answer, parse_citations
from rag.retrieval import Hit


def hit(n, source="a.mdx"):
    c = Chunk(id=f"{source}#{n}", source=source, url="u", title="T", heading="H", text="body text")
    return Hit(chunk=c, score=1.0, rank=n)


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.messages = SimpleNamespace(create=self._create)
        self.last = None

    def _create(self, **kwargs):
        self.last = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


def test_parse_citations_splits_valid_and_invalid():
    assert parse_citations("Yes [1][3] and again [1], also [9].", 3) == ([1, 3], [9])


def test_answer_with_client_sends_numbered_sources():
    client = FakeClient("Set it to false [1].")
    result = answer("q?", [hit(1), hit(2)], client=client)
    assert result.cited == [1] and not result.abstained and result.generated
    prompt = client.last["messages"][0]["content"]
    assert "[1] T > H" in prompt and "[2] T > H" in prompt


def test_abstention_is_detected():
    assert answer("q?", [hit(1)], client=FakeClient(ABSTAIN)).abstained


def test_no_client_returns_closest_passage_without_pretending():
    result = answer("q?", [hit(1)], client=None)
    assert not result.generated and "No API key" in result.text


def test_retrieval_metrics():
    ranks = [first_relevant_rank(["x", "a"], ["a"]), first_relevant_rank(["a"], ["a"]), first_relevant_rank(["x"], ["a"])]
    assert ranks == [2, 1, None]
    s = retrieval_summary(ranks)
    assert s["hit@1"] == 1 / 3 and s["hit@3"] == 2 / 3 and round(s["mrr"], 3) == round((0.5 + 1 + 0) / 3, 3)


def test_fact_coverage_is_case_insensitive():
    assert fact_coverage("Use StatusError from expo-server", ["statuserror", "expo-server", "404"]) == 2 / 3
    assert fact_coverage("anything", []) == 1.0


def test_citation_report():
    r = citation_report([1], [], ["b.mdx"], ["a.mdx"])
    assert r == {"has_citation": True, "all_valid": True, "cites_expected_page": False}


def test_judge_parses_json_even_with_extra_text():
    assert extract_json('Sure! {"faithful": true, "correct": false, "reason": "x"} done')["faithful"] is True
    out = judge(FakeClient('{"faithful": true, "correct": true, "reason": "ok"}'), "q", "a", "ctx", ["f"])
    assert out == {"faithful": True, "correct": True, "reason": "ok"}
