"""
Configuration module — loads environment variables from .env and exposes them
as module-level constants.  Raises a clear error at import time if any
required key is missing.
"""

import os
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
load_dotenv()

# ── Required environment variables ──────────────────────────────────────────

_REQUIRED_KEYS = [
    "GROQ_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_ENV",
    "PINECONE_INDEX_NAME",
]

_missing = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(_missing)}. "
        "Please add them to your .env file.  See .env.example for reference."
    )

# ── Expose as constants ─────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")          # type: ignore[assignment]
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")    # type: ignore[assignment]
PINECONE_ENV: str = os.getenv("PINECONE_ENV")            # type: ignore[assignment]
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")  # type: ignore[assignment]
