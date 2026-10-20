"""Product lookup: the products named in a conversation, and their documents.

The knowledge base's document ids encode category and product
(``doc_checking_accounts_light_blue_account_007``), so a dictionary of product
names comes from the ids alone; no document is read and no label is used.
A product named anywhere in the conversation -- by the customer, the agent, or
a tool result -- brings in its summary document (the lowest-numbered one) and
its best BM25 match for the customer's latest message.

Why: a top-k search returns the few documents closest to the query, but tasks
that involve several products need one document per product. In A3's traces,
55% of the product documents never retrieved belonged to a product the
conversation had named (docs/replay_a3.md).

Pure Python: no model, no network.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping

from policygate_rag.index.bm25 import BM25Index

DOC_ID = re.compile(r"^doc_(?P<product>.+)_(?P<n>\d{3})$")
CATEGORIES = (
    "business_checking_accounts", "business_credit_cards", "business_savings_accounts",
    "checking_accounts", "savings_accounts", "credit_cards", "bank_accounts",
    "buy_now_pay_later", "everyone_pay", "personal_subscriptions", "customer_support",
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_]", " ", text.lower()))


def product_names(doc_ids: Iterable[str]) -> dict[str, set[str]]:
    """Product phrase -> product keys (the id without its number). General
    sections are not products. A phrase can name several products
    ("green account" is a checking and a savings account)."""
    names: dict[str, set[str]] = defaultdict(set)
    for d in doc_ids:
        m = DOC_ID.match(d)
        if m is None or "(general)" in m["product"]:
            continue
        product = m["product"]
        rest = next((product[len(c) + 1:] for c in CATEGORIES if product.startswith(c + "_")),
                    product)
        phrase = _normalise(re.sub(r"_\((checking|savings)\)", "", rest)).strip()
        if phrase:
            names[phrase].add(product)
    return dict(names)


def mentioned_products(text: str, names: Mapping[str, set[str]]) -> set[str]:
    """Longest phrases first, each match masked so "dark green account" does
    not also count as "green account"."""
    text = f" {_normalise(text)} "
    found: set[str] = set()
    for phrase in sorted(names, key=len, reverse=True):
        pattern = re.compile(rf"(?<=[^a-z0-9]){re.escape(phrase)}(?=[^a-z0-9])")
        if pattern.search(text):
            found |= names[phrase]
            text = pattern.sub(" # ", text)
    return found


class ProductLookup:
    def __init__(self, doc_ids: Iterable[str], bm25: BM25Index) -> None:
        ids = list(doc_ids)
        self.names = product_names(ids)
        self.bm25 = bm25
        self.by_product: dict[str, list[str]] = defaultdict(list)
        for d in sorted(ids):
            m = DOC_ID.match(d)
            if m:
                self.by_product[m["product"]].append(d)

    def mentioned(self, text: str) -> set[str]:
        return mentioned_products(text, self.names)

    def documents(self, products: Iterable[str], latest_user_message: str) -> list[str]:
        """Per product, in product order: its summary, then its best BM25 match for
        the latest customer message among its other documents, if any matches."""
        products = sorted(products)
        if not products:
            return []
        scores = dict(self.bm25.search(latest_user_message, len(self.bm25.doc_ids)))
        out = []
        for product in products:
            members = self.by_product[product]
            out.append(members[0])
            rest = [d for d in members[1:] if scores.get(d, 0) > 0]
            if rest:
                out.append(max(rest, key=lambda d: (scores[d], d)))
        return out
