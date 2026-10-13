"""The knowledge base as the index sees it: id, title, content, and the indexed text.

Pure Python. The caller passes the documents directory; nothing here knows
where tau2 keeps its data, and nothing here reads tasks.

Indexed text is the title and the content. tau2's own bm25 indexes the content
alone; titles here are short and specific ("Debit Card Dispute Process"), and
dropping them would throw away the best keyword evidence a document has. This
is one of the ways our pipeline differs from the A1 baseline, on purpose.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    content: str

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.content}"


def load_corpus(directory: Path) -> tuple[Document, ...]:
    """Every ``*.json`` document under ``directory``, ordered by id."""
    docs = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        docs.append(Document(doc_id=raw["id"], title=raw["title"], content=raw["content"]))
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate document ids in the corpus")
    if not docs:
        raise ValueError(f"no documents under {directory}")
    return tuple(sorted(docs, key=lambda d: d.doc_id))


def corpus_hash(docs: Sequence[Document]) -> str:
    """Identity of the indexed text: changes if any id, title or content does."""
    h = hashlib.sha256()
    for d in docs:
        h.update(json.dumps([d.doc_id, d.text], ensure_ascii=False).encode())
    return h.hexdigest()
