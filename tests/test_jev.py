"""Jev judge and client against a mock transport: no network, no key needed."""

import json

import httpx
import pytest

from policygate_rag.index.corpus import Document
from policygate_rag.judges.base import Candidate
from policygate_rag.judges.jev import JevJudge
from policygate_rag.judges.jev_client import JevClient, JevError


def transport(handler):
    return httpx.MockTransport(handler)


def ok(request):
    body = json.loads(request.content)
    assert body["model"] == "jev-1.13.0"
    assert set(body["state"]) == {"request", "document"}
    return httpx.Response(200, json={
        "model": "jev-1.13.0",
        "answers": {k: {"type": "noul", "noul": 0.8} for k in body["questions"]},
        "usage": {"input_tokens": 1000, "output_tokens": 0},
    })


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")


def cand(i):
    return Candidate(doc=Document(doc_id=f"d{i}", title="T", content="C"), fused_score=0.1)


def test_one_pair_per_request_and_cost_billed_externally():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return ok(request)

    client = JevClient(transport=transport(handler))
    verdict = JevJudge(client.nouls, workers=2).judge("dispute", [cand(1), cand(2)])
    assert verdict.scores == (0.8, 0.8) and verdict.is_probability
    assert verdict.usd == 0.0
    assert verdict.external_usd == pytest.approx(2 * 1000 * 0.042 / 1e6)
    assert len(seen) == 2 and all(len(s["questions"]) == 1 for s in seen)


def test_the_key_is_sent_as_a_bearer_token():
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        return ok(request)

    JevClient(transport=transport(handler)).nouls(
        {"request": "q", "document": {}}, {"r": {"instructions": "x"}})


def test_only_the_pinned_version_is_accepted():
    with pytest.raises(ValueError):
        JevClient(model="jev-latest", transport=transport(ok))


def test_an_answer_from_another_version_is_refused():
    def handler(request):
        r = ok(request)
        body = json.loads(r.content) | {"model": "jev-1.14.0"}
        return httpx.Response(200, json=body)

    with pytest.raises(JevError, match="answered by"):
        JevClient(transport=transport(handler)).nouls({"request": "q", "document": {}},
                                                      {"r": {"instructions": "x"}})


def test_unknown_token_usage_is_an_error_not_free():
    def handler(request):
        body = json.loads(ok(request).content)
        body["usage"] = {}
        return httpx.Response(200, json=body)

    with pytest.raises(JevError, match="cost unknown"):
        JevClient(transport=transport(handler)).nouls({"request": "q", "document": {}},
                                                      {"r": {"instructions": "x"}})


def test_no_key_no_client(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY")
    with pytest.raises(JevError, match="not set"):
        JevClient(transport=transport(ok))
