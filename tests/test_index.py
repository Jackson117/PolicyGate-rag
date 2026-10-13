"""Tests for index/: BM25, dense search, RRF. No network, no embeddings API.

BM25 is checked two ways: on a corpus small enough to score by hand, and on the
real 698-document knowledge base against a brute-force evaluation of the same
formula, so the inverted-index implementation cannot drift from the definition.
"""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from policygate_rag.index.bm25 import BM25Index, tokenize
from policygate_rag.index.corpus import Document, corpus_hash, load_corpus
from policygate_rag.index.dense import DenseIndex
from policygate_rag.index.fusion import HybridIndex, rrf

DOCS_DIR = (Path(__file__).resolve().parents[1] / "third_party" / "tau2-bench" / "data"
            / "tau2" / "domains" / "banking_knowledge" / "documents")


def doc(i, title, content=""):
    return Document(doc_id=f"d{i}", title=title, content=content)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DOCS_DIR)


class TestCorpus:
    def test_the_real_corpus_loads_in_id_order(self, corpus):
        assert len(corpus) == 698
        assert [d.doc_id for d in corpus] == sorted(d.doc_id for d in corpus)

    def test_the_hash_moves_with_any_text(self):
        a = [doc(1, "Card disputes", "x")]
        assert corpus_hash(a) != corpus_hash([doc(1, "Card disputes", "y")])
        assert corpus_hash(a) == corpus_hash([doc(1, "Card disputes", "x")])


class TestBM25:
    def test_tokens_split_on_punctuation_and_case(self):
        assert tokenize("Card. CARD, card-holder's 2x") == [
            "card", "card", "card", "holder", "s", "2x",
        ]

    def test_a_score_by_hand(self):
        docs = [doc(1, "card dispute"), doc(2, "card"), doc(3, "loan")]
        idx = BM25Index(docs, k1=1.5, b=0.75)
        avg = (2 + 1 + 1) / 3
        idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))  # "dispute" is in one document
        norm = 1 - 0.75 + 0.75 * 2 / avg
        expected = idf * 1 * 2.5 / (1 + 1.5 * norm)
        assert idx.scores("dispute")[0] == pytest.approx(expected)

    def test_idf_never_goes_negative(self):
        docs = [doc(i, "card") for i in range(10)] + [doc(10, "loan")]
        assert BM25Index(docs).scores("card")[0] > 0

    def test_ties_break_by_corpus_order_and_zero_scores_are_dropped(self):
        docs = [doc(1, "card"), doc(2, "card"), doc(3, "loan")]
        assert [d for d, _ in BM25Index(docs).search("card", 5)] == ["d1", "d2"]

    def test_the_real_corpus_matches_a_brute_force_evaluation(self, corpus):
        idx = BM25Index(corpus, k1=1.5, b=0.75)
        tokenized = [tokenize(d.text) for d in corpus]
        n, avg = len(corpus), sum(map(len, tokenized)) / len(corpus)
        df = Counter(t for toks in tokenized for t in set(toks))
        for query in ("dispute a debit card transaction", "statement credit", "zzzz unseen"):
            q = set(tokenize(query))
            for i in (0, 123, 697):
                tf = Counter(tokenized[i])
                brute = sum(
                    math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * 2.5
                    / (tf[t] + 1.5 * (0.25 + 0.75 * len(tokenized[i]) / avg))
                    for t in q if t in df
                )
                assert idx.scores(query)[i] == pytest.approx(brute)

    def test_a_document_is_found_by_its_own_title(self, corpus):
        # Not a quality benchmark: a smoke check that indexing and ranking are
        # wired to the right documents. Titles repeat across products, so top 5.
        idx = BM25Index(corpus)
        hits = sum(d.doc_id in [x for x, _ in idx.search(d.title, 5)] for d in corpus)
        assert hits / len(corpus) > 0.8


class TestDense:
    def _index(self):
        vectors = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=float)
        return DenseIndex(["a", "b", "c", "d"], vectors, model="m")

    def test_cosine_ranking_ignores_vector_length(self):
        assert [d for d, _ in self._index().search([0, 0, 0.1], 1)] == ["d"]
        top = self._index().search([1, 0.9, 0], 2)
        assert [d for d, _ in top] == ["c", "a"]

    def test_ties_break_by_corpus_order(self):
        assert [d for d, _ in self._index().search([1, 1, 0], 3)][1:] == ["a", "b"]

    @pytest.mark.parametrize("bad", [[0, 0, 0], [1, 0]])
    def test_bad_queries_are_refused(self, bad):
        with pytest.raises(ValueError):
            self._index().search(bad, 1)

    def test_the_cache_round_trips_and_refuses_a_mismatch(self, tmp_path):
        idx = self._index()
        idx.save(tmp_path / "emb", corpus_hash="h1")
        back = DenseIndex.load(tmp_path / "emb", model="m", corpus_hash="h1")
        got, want = back.search([1, 0.9, 0], 4), idx.search([1, 0.9, 0], 4)
        assert [d for d, _ in got] == [d for d, _ in want]
        assert [s for _, s in got] == pytest.approx([s for _, s in want])
        with pytest.raises(ValueError, match="different corpus"):
            DenseIndex.load(tmp_path / "emb", model="m", corpus_hash="h2")
        with pytest.raises(ValueError, match="built with"):
            DenseIndex.load(tmp_path / "emb", model="other", corpus_hash="h1")


class TestFusion:
    def test_rrf_by_hand(self):
        fused = rrf({"bm25": ["a", "b"], "dense": ["b", "c"]}, k=60)
        assert [f.doc_id for f in fused] == ["b", "a", "c"]
        assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
        assert fused[0].ranks == {"bm25": 2, "dense": 1}

    def test_equal_scores_break_by_best_rank_then_id(self):
        fused = rrf({"bm25": ["x", "y"], "dense": ["y", "x"]}, k=60)
        assert [f.doc_id for f in fused] == ["x", "y"]

    def test_a_list_ranking_a_document_twice_is_refused(self):
        with pytest.raises(ValueError):
            rrf({"bm25": ["a", "a"]})

    def test_hybrid_refuses_indexes_over_different_documents(self):
        bm25 = BM25Index([doc(1, "card"), doc(2, "loan")])
        dense = DenseIndex(["d1", "d3"], np.eye(2), model="m")
        with pytest.raises(ValueError):
            HybridIndex(bm25, dense)

    def test_hybrid_fuses_both_sources(self):
        bm25 = BM25Index([doc(1, "card dispute"), doc(2, "loan rates")])
        dense = DenseIndex(["d1", "d2"], np.eye(2), model="m")
        out = HybridIndex(bm25, dense).search("loan", [1, 0], k_bm25=2, k_dense=2, rrf_k=60)
        assert {f.doc_id for f in out} == {"d1", "d2"}
        assert out[0].ranks.keys() == {"bm25", "dense"} or out[1].ranks.keys() == {"bm25", "dense"}
