"""
E-Commerce RAG Search — FastAPI application entry point.

Start the server with:
    uvicorn main:app --reload
"""

# Register Windows Store Python DLL search paths before importing torch or sentence-transformers
import dll_loader

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.search_router import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    print("\n[INFO] E-Commerce RAG API is running!")
    print("Interactive docs -> http://127.0.0.1:8000/docs\n")
    yield


app = FastAPI(
    title="E-Commerce RAG Search",
    version="1.0.0",
    description="Conversational product discovery powered by hybrid RAG search.",
    lifespan=lifespan,
)

# ── CORS — allow all origins for easy local testing ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(search_router)


# ── Health check ────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "E-Commerce RAG API is running"}
