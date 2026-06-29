"""
FastAPI router for the /api/v1/search endpoint.
"""

from fastapi import APIRouter, HTTPException

from models.schemas import SearchRequest, SearchResponse
from search.query_parser import parse_query
from search.hybrid_search import hybrid_search
from synthesis.response_generator import generate_narrative

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
    parsed_query = parse_query(request.query)

    # 2. Run hybrid (vector + metadata) search
    results = hybrid_search(parsed_query)

    # 3. Generate a conversational narrative
    narrative = generate_narrative(request.query, results)

    return SearchResponse(
        parsed_query=parsed_query,
        results=results,
        narrative=narrative,
    )
