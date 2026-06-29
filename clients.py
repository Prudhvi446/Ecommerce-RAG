"""
Shared singleton clients for external services (Groq, Pinecone)
and the embedding model.

Import from here instead of creating new client instances per request.
This avoids repeated TCP/TLS handshakes and prevents socket exhaustion
under production traffic.
"""

from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from config import GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# ── Timeout configuration (seconds) ────────────────────────────────────────
# Prevents the server from hanging indefinitely if a downstream service is
# slow or unresponsive.  Adjust these values based on your SLA requirements.
GROQ_TIMEOUT = 10       # default SDK value is 120s — far too generous
PINECONE_TIMEOUT = 10   # default SDK value is 300s — far too generous

# ── Groq LLM client (reused across query_parser & response_generator) ───────
groq_client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT)

# ── Pinecone vector-DB client & index handle ────────────────────────────────
_pc = Pinecone(api_key=PINECONE_API_KEY, timeout=PINECONE_TIMEOUT)
pinecone_index = _pc.Index(PINECONE_INDEX_NAME)

# ── Embedding model (loaded eagerly at import time) ─────────────────────────
embedding_model: SentenceTransformer | None = None


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model into memory (idempotent).

    Called once during the FastAPI lifespan hook so the first user
    request does not pay the 10-30 s model-load penalty.
    """
    global embedding_model
    if embedding_model is None:
        print("[INFO] Loading embedding model (all-MiniLM-L6-v2) ...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[OK] Embedding model ready")
    return embedding_model
