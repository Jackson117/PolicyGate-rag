"""Build or load the hybrid index over the knowledge base.

    python -m policygate_rag.retrieval.build_index --dry-run   # estimate only
    python -m policygate_rag.retrieval.build_index             # embed once, book the cost

Document embeddings are computed once and cached under ``index_cache/``
(git-ignored), keyed by the model and the corpus hash; a later run loads the
cache and spends nothing. The build is checked against the stage-1 budget line
before any call, and its actual cost is booked to the ledger.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from policygate_rag.config import RagConfig
from policygate_rag.cost.ledger import ALLOCATIONS, BudgetLine, Entry, EntryKind, Ledger
from policygate_rag.index.bm25 import BM25Index
from policygate_rag.index.corpus import Document, corpus_hash, load_corpus
from policygate_rag.index.dense import DenseIndex
from policygate_rag.index.fusion import HybridIndex
from policygate_rag.retrieval.embedder import Embedder, LiteLLMEmbedder

ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "third_party/tau2-bench/data/tau2/domains/banking_knowledge/documents"
CACHE_DIR = ROOT / "index_cache"
LEDGER = ROOT / "runs" / "ledger.json"


def cache_path(model: str) -> Path:
    return CACHE_DIR / model.replace("/", "_")


def estimate_usd(docs: list[Document], model: str) -> tuple[int, float]:
    """Tokens by the ~4 characters/token rule, priced from litellm's table."""
    import litellm

    tokens = sum(len(d.text) for d in docs) // 4
    price = litellm.model_cost.get(model, {}).get("input_cost_per_token")
    if price is None:
        raise ValueError(f"no price for {model} in litellm's table")
    return tokens, tokens * price


def load_or_build(
    config: RagConfig,
    *,
    embedder: Embedder | None = None,
    ledger: Ledger | None = None,
    docs_dir: Path = DOCS_DIR,
    cache: Path | None = None,
) -> HybridIndex:
    """The hybrid index for ``config``. Embeds the corpus only if no valid cache exists."""
    docs = list(load_corpus(docs_dir))
    digest = corpus_hash(docs)
    cache = cache or cache_path(config.embedding_model)
    bm25 = BM25Index(docs, k1=config.bm25_k1, b=config.bm25_b)
    if cache.with_suffix(".json").exists():
        dense = DenseIndex.load(cache, model=config.embedding_model, corpus_hash=digest)
        return HybridIndex(bm25, dense)

    if embedder is None or ledger is None:
        raise RuntimeError(
            f"no embedding cache at {cache}; building one spends money and needs an "
            "embedder and the ledger (python -m policygate_rag.retrieval.build_index)"
        )
    _, estimate = estimate_usd(docs, config.embedding_model)
    line = BudgetLine.STAGE1
    if ledger.spent(line) + estimate > ALLOCATIONS[line]:
        raise RuntimeError(f"embedding (~{estimate:.4f} USD) would exceed the {line.value} line")
    result = embedder.embed([d.text for d in docs])
    ledger.record(Entry(
        line=line, kind=EntryKind.CALL, usd=result.usd, model=config.embedding_model,
        label=f"embed {len(docs)} KB documents ({result.tokens} tokens)",
    ))
    dense = DenseIndex([d.doc_id for d in docs], result.vectors, model=config.embedding_model)
    dense.save(cache, corpus_hash=digest)
    return HybridIndex(bm25, dense)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    config = RagConfig()
    docs = list(load_corpus(DOCS_DIR))
    tokens, usd = estimate_usd(docs, config.embedding_model)
    cache = cache_path(config.embedding_model)
    present = "present" if cache.with_suffix(".json").exists() else "absent"
    print(f"{len(docs)} documents, ~{tokens} tokens, ~{usd:.4f} USD with "
          f"{config.embedding_model}; cache {present}")
    if args.dry_run:
        return 0
    ledger = Ledger.load(LEDGER)
    before = ledger.spent(BudgetLine.STAGE1)
    index = load_or_build(config, embedder=LiteLLMEmbedder(config.embedding_model), ledger=ledger)
    print(f"index ready: {len(index.bm25.doc_ids)} docs, {index.dense.dimensions} dims; "
          f"stage1 spent {before:.4f} -> {ledger.spent(BudgetLine.STAGE1):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
