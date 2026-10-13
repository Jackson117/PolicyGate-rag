"""retrieval/build_index.py with a fake embedder: no network, no spend."""

import json

import numpy as np
import pytest

from policygate_rag.config import RagConfig
from policygate_rag.cost.ledger import BudgetLine, Ledger
from policygate_rag.retrieval.build_index import load_or_build
from policygate_rag.retrieval.embedder import Embedding


class FakeEmbedder:
    model = "text-embedding-3-large"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        rng = np.random.default_rng(0)
        return Embedding(vectors=rng.normal(size=(len(texts), 8)).tolist(),
                         tokens=10 * len(texts), usd=0.0123)


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for i, (title, body) in enumerate([("Debit card disputes", "How to file"),
                                       ("Statement credits", "When to apply"),
                                       ("Savings rates", "Current rates")]):
        (d / f"doc{i}.json").write_text(json.dumps({"id": f"doc_{i}", "title": title,
                                                    "content": body}))
    return d


def test_builds_once_books_the_cost_then_loads_from_cache(tmp_path, docs_dir):
    ledger = Ledger(tmp_path / "ledger.json")
    embedder = FakeEmbedder()
    cache = tmp_path / "cache" / "emb"
    first = load_or_build(RagConfig(), embedder=embedder, ledger=ledger,
                          docs_dir=docs_dir, cache=cache)
    assert ledger.spent(BudgetLine.STAGE1) == pytest.approx(0.0123)
    second = load_or_build(RagConfig(), docs_dir=docs_dir, cache=cache)
    assert embedder.calls == 1
    q = np.ones(8)
    assert [d for d, _ in first.dense.search(q, 3)] == [d for d, _ in second.dense.search(q, 3)]
    hits = second.search("card dispute", q, k_bm25=3, k_dense=3, rrf_k=60)
    assert hits[0].doc_id == "doc_0" or "bm25" in hits[0].ranks


def test_without_a_cache_it_refuses_to_spend_unasked(tmp_path, docs_dir):
    with pytest.raises(RuntimeError, match="spends money"):
        load_or_build(RagConfig(), docs_dir=docs_dir, cache=tmp_path / "none" / "emb")
