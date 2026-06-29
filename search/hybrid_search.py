"""
Hybrid search — combines vector similarity with structured metadata filters
on Pinecone to find the most relevant products.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dll_loader

from typing import List

from sentence_transformers import SentenceTransformer

from clients import pinecone_index
from models.schemas import ParsedQuery, ProductResult


# ── Module-level model cache (loaded once) ──────────────────────────────────
_embedding_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the cached embedding model, loading it on first call."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def hybrid_search(parsed_query: ParsedQuery, top_k: int = 5) -> List[ProductResult]:
    """
    Run a hybrid (vector + metadata filter) search against Pinecone and
    return the top-k matching products.
    """

    # ── 1. Generate query embedding ─────────────────────────────────────────
    model = _get_model()
    query_embedding = model.encode(parsed_query.semantic_query).tolist()

    # ── 2. Build metadata filter dynamically ────────────────────────────────
    conditions: list[dict] = []

    if parsed_query.max_price is not None:
        conditions.append({"price": {"$lte": parsed_query.max_price}})

    if parsed_query.min_price is not None:
        conditions.append({"price": {"$gte": parsed_query.min_price}})

    if parsed_query.required_tags:
        conditions.append({"tags": {"$in": parsed_query.required_tags}})

    if parsed_query.gender is not None:
        conditions.append({"gender": {"$eq": parsed_query.gender}})

    if len(conditions) == 1:
        metadata_filter = conditions[0]
    elif len(conditions) > 1:
        metadata_filter = {"$and": conditions}
    else:
        metadata_filter = None

    # ── 3. Query Pinecone ───────────────────────────────────────────────────
    query_kwargs = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
    }
    if metadata_filter is not None:
        query_kwargs["filter"] = metadata_filter

    results = pinecone_index.query(**query_kwargs)

    # ── 4. Parse matches into ProductResult objects ─────────────────────────
    product_results: List[ProductResult] = []

    for match in results.get("matches", []):
        meta = match.get("metadata", {})

        # Tags may already be a list (Pinecone returns list metadata natively)
        tags = meta.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split("|") if t.strip()]

        product_results.append(
            ProductResult(
                product_id=match["id"],
                name=meta.get("name", "Unknown"),
                brand=meta.get("brand", "Unknown"),
                price=float(meta.get("price", 0.0)),
                rating=float(meta.get("rating", 0.0)),
                score=float(match.get("score", 0.0)),
                description=meta.get("description", ""),
                tags=tags,
            )
        )

    # Sort by score descending (highest relevance first)
    product_results.sort(key=lambda p: p.score, reverse=True)

    return product_results
