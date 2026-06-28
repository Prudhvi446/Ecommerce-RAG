"""
Pydantic v2 schemas for the E-Commerce RAG search engine.
"""

from typing import List, Optional

from pydantic import BaseModel


# ── Query parsing result ─────────────────────────────────────────────────────

class ParsedQuery(BaseModel):
    """Structured representation of a natural-language shopping query."""

    raw_query: str
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    required_tags: List[str] = []
    preferred_brand: Optional[str] = None
    category: Optional[str] = None
    gender: Optional[str] = None
    semantic_query: str  # cleaned version of the query used for embedding


# ── Single product result ────────────────────────────────────────────────────

class ProductResult(BaseModel):
    """A single product returned from the hybrid search."""

    product_id: str
    name: str
    brand: str
    price: float
    rating: float
    score: float
    description: str
    tags: List[str]


# ── API request / response ───────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Incoming search request body."""

    query: str


class SearchResponse(BaseModel):
    """Full search response returned to the client."""

    parsed_query: ParsedQuery
    results: List[ProductResult]
    narrative: str
