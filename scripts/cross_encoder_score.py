"""Score (query, document) pairs with the pinned cross-encoder, in its own environment.

    .venv-ce/bin/python scripts/cross_encoder_score.py PAIRS.jsonl OUT.jsonl

Runs outside the project environment on purpose: sentence-transformers pulls in
transformers, which forces huggingface-hub from 2.2.0 down to 1.33.0 -- a change
to the environment every arm runs in, made mid-study, for a judge that is only
benchmarked offline. Pairs come from ``python -m policygate_rag.stage1.collect
--export-pairs``; output rows match the other judges' score files, so
``eval/judge_bench.py`` reads them unchanged. Resumable: pairs already in OUT
are skipped.

Imports only the standard library and sentence-transformers. The model and
revision must match ``policygate_rag/judges/cross_encoder.py``; the environment
is pinned in ``scripts/cross_encoder.requirements.txt``.
"""

import json
import math
import sys
from pathlib import Path

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
BATCH = 64


def main(pairs_path: str, out_path: str) -> None:
    from sentence_transformers import CrossEncoder

    out = Path(out_path)
    done = set()
    if out.exists():
        done = {(r["query"], r["doc_id"]) for r in map(json.loads, out.read_text().splitlines())}
    pairs = [p for p in map(json.loads, Path(pairs_path).read_text().splitlines())
             if (p["query"], p["doc_id"]) not in done]
    encoder = CrossEncoder(MODEL, revision=REVISION)
    with out.open("a") as f:
        for start in range(0, len(pairs), BATCH):
            batch = pairs[start:start + BATCH]
            logits = encoder.predict([(p["query"], p["text"]) for p in batch],
                                     activation_fn=None).tolist()
            for p, x in zip(batch, logits, strict=True):
                f.write(json.dumps({"judge": "cross_encoder", "query": p["query"],
                                    "doc_id": p["doc_id"], "score": 1 / (1 + math.exp(-x)),
                                    "is_probability": True}) + "\n")
    print(f"scored {len(pairs)} pairs ({len(done)} already done) -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
