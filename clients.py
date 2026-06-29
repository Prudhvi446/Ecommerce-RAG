"""
Shared singleton clients for external services (Groq, Pinecone).

Import from here instead of creating new client instances per request.
This avoids repeated TCP/TLS handshakes and prevents socket exhaustion
under production traffic.
"""

from groq import Groq
from pinecone import Pinecone

from config import GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# ── Groq LLM client (reused across query_parser & response_generator) ───────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Pinecone vector-DB client & index handle ────────────────────────────────
_pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = _pc.Index(PINECONE_INDEX_NAME)
