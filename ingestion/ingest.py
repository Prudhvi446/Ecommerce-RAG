"""
Standalone ingestion script — reads products.csv, generates embeddings with
all-MiniLM-L6-v2, and upserts vectors into a Pinecone serverless index.

Run from the project root:
    python ingestion/ingest.py
"""

import os
import sys

# Ensure project root is on sys.path so `config` and helper modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Register Windows Store Python DLL search paths before importing torch or sentence-transformers
import dll_loader

import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


def main() -> None:
    # ── 1. Load product data ────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")
    df = pd.read_csv(csv_path)
    print(f"[OK] Loaded {len(df)} products from {csv_path}")

    # ── 2. Initialise embedding model ───────────────────────────────────────
    print("[INFO] Loading embedding model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[OK] Embedding model ready")

    # ── 3. Generate embeddings ──────────────────────────────────────────────
    descriptions = df["description"].tolist()
    print("[INFO] Generating embeddings ...")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    print(f"[OK] Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    # ── 4. Connect to Pinecone ──────────────────────────────────────────────
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"[INFO] Creating Pinecone index '{PINECONE_INDEX_NAME}' ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[OK] Index '{PINECONE_INDEX_NAME}' created")
    else:
        print(f"[OK] Index '{PINECONE_INDEX_NAME}' already exists")

    index = pc.Index(PINECONE_INDEX_NAME)

    # ── 5. Build upsert vectors ─────────────────────────────────────────────
    vectors = []
    for i, row in df.iterrows():
        # Parse tags from pipe-separated string to list
        tags_list = [t.strip() for t in str(row["tags"]).split("|") if t.strip()]

        metadata = {
            "name": str(row["name"]),
            "brand": str(row["brand"]),
            "category": str(row["category"]),
            "price": float(row["price"]),
            "tags": tags_list,
            "stock_level": int(row["stock_level"]),
            "rating": float(row["rating"]),
            "is_waterproof": bool(row["is_waterproof"])
            if isinstance(row["is_waterproof"], bool)
            else str(row["is_waterproof"]).strip().lower() == "true",
            "gender": str(row["gender"]),
        }

        vectors.append(
            {
                "id": str(row["product_id"]),
                "values": embeddings[i].tolist(),
                "metadata": metadata,
            }
        )

    # ── 6. Upsert in batches ────────────────────────────────────────────────
    BATCH_SIZE = 50
    total_upserted = 0

    for start in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Upserting batches"):
        batch = vectors[start : start + BATCH_SIZE]
        index.upsert(vectors=batch)
        total_upserted += len(batch)

    print(f"\n[DONE] {total_upserted} vectors upserted into '{PINECONE_INDEX_NAME}'")


if __name__ == "__main__":
    main()
