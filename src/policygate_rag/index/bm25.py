"""BM25 over the knowledge base, written here rather than imported.

Okapi BM25 with the Lucene form of idf, ``ln(1 + (N - df + 0.5) / (df + 0.5))``,
which is never negative. The classic form goes negative for terms in more than
half the documents, so a common word in a query would push relevant documents
down; rank_bm25 (tau2's library) patches that with an epsilon floor instead.

Tokens: lowercase runs of letters and digits. tau2 splits on whitespace, which
keeps punctuation attached ("card." and "card" are different terms there). No
stemming and no stopword list: both are tunable choices with their own
failure modes, and the study's question is not BM25 engineering.

Ties are broken by corpus order (document id), so a ranking is a function of
the query alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from policygate_rag.index.corpus import Document

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, docs: Sequence[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.doc_ids = tuple(d.doc_id for d in docs)
        self.k1, self.b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        for i, doc in enumerate(docs):
            counts = Counter(tokenize(doc.text))
            self._lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings.setdefault(term, []).append((i, tf))
        n = len(docs)
        self._avg_length = sum(self._lengths) / n if n else 0.0
        self._idf = {
            term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self._postings.items()
        }

    def scores(self, query: str) -> list[float]:
        """BM25 score of every document for ``query``, in corpus order."""
        out = [0.0] * len(self.doc_ids)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                norm = 1 - self.b + self.b * self._lengths[i] / self._avg_length
                out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top ``k`` documents with a positive score, best first."""
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda si: (-si[0], si[1]))
        return [(self.doc_ids[i], s) for s, i in scored[:k]]
