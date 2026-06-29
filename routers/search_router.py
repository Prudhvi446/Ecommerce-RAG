"""
FastAPI router for the /api/v1/search endpoint.
"""

import logging

from fastapi import APIRouter, HTTPException
from pinecone.exceptions import PineconeException

from models.schemas import SearchRequest, SearchResponse
from search.query_parser import parse_query
from search.hybrid_search import hybrid_search
from synthesis.response_generator import generate_narrative

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Search"])


@router.post("/search", response_model=SearchResponse)
def search_products(request: SearchRequest) -> SearchResponse:
    """
    Accept a natural-language shopping query, parse it, run hybrid search
    against Pinecone, and return a conversational recommendation.

    NOTE: This is intentionally a synchronous ``def`` (not ``async def``)
    so that FastAPI dispatches each request to the default thread-pool.
    All downstream calls (Groq, Pinecone, SentenceTransformer) are
    blocking / CPU-bound and would freeze the async event loop otherwise.
    """

    # Validate non-empty query
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query must be a non-empty string.",
        )

    # 1. Parse the raw query into structured filters
    #    (parse_query already has its own internal fallback)
    parsed_query = parse_query(request.query)

    # 2. Run hybrid (vector + metadata) search
    try:
        results = hybrid_search(parsed_query)
    except PineconeException as exc:
        logger.error("Pinecone search failed: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=503,
            detail="Product search is temporarily unavailable. Please try again shortly.",
        )
    except Exception as exc:
        logger.error("Unexpected error during hybrid search: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=502,
            detail="An error occurred while searching for products. Please try again.",
        )

    # 3. Generate a conversational narrative
    #    (generate_narrative has its own internal fallback — it never raises)
    narrative = generate_narrative(request.query, results)

    return SearchResponse(
        parsed_query=parsed_query,
        results=results,
        narrative=narrative,
    )
