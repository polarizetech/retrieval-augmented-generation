"""Reranking with the text model: one call, 0-10 marks, mapped onto the reranker's scale."""

from typing import Any

from research_mcp import llm_rerank
from research_mcp.ask import apply_relevance_floor


class FakeModel:
    """A text model that answers every chat_json call with one answer, or raises it."""

    name = "fake"

    def __init__(self, answer: dict[str, Any] | Exception) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str, float]] = []

    def chat_json(
        self,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        self.calls.append((task, user, timeout))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def rows(n: int) -> list[dict[str, Any]]:
    return [
        {"evidence_id": f"W{i}#p0.abcdef12", "text": f"passage {i}", "score": 0.5}
        for i in range(1, n + 1)
    ]


def test_one_call_scores_at_most_the_capped_passages() -> None:
    answer = {
        "scores": [
            {"passage": 2, "score": 9},
            {"passage": 1, "score": 12},
            {"passage": 99, "score": 5},
        ]
    }
    model = FakeModel(answer)
    given = llm_rerank.score(model, "pink noise", rows(20))
    assert len(model.calls) == 1
    task, user, timeout = model.calls[0]
    assert (task, timeout) == ("rerank", llm_rerank.SECONDS)
    assert len(given) == llm_rerank.CANDIDATES
    assert given[:3] == [10.0, 9.0, 0.0]  # clamped; a skipped passage gets 0
    assert f"[{llm_rerank.CANDIDATES}]" in user
    assert f"[{llm_rerank.CANDIDATES + 1}]" not in user


def test_marks_sort_map_and_meet_the_relevance_floor() -> None:
    found = {"retrieval": "hybrid", "n": 3, "results": rows(3), "reranker": None}
    out = llm_rerank.apply(found, [1.0, 9.0, 5.0], 5, "llm:fake")
    assert [r["evidence_id"][:2] for r in out["results"]] == ["W2", "W3", "W1"]
    assert [r["score"] for r in out["results"]] == [4.0, 0.0, -4.0]
    assert out["results"][0]["fused_score"] == 0.5
    assert out["reranker"] == "llm:fake"
    floored = apply_relevance_floor(out)
    assert floored["n"] == 2
    assert floored["below_floor"]["n"] == 1
    assert llm_rerank.apply(found, [1.0, 9.0, 5.0], 2, "llm:fake")["n"] == 2
