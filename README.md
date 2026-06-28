# 🛒 E-Commerce Conversational Search & Product Discovery Engine

A Python backend that accepts natural-language shopping queries, parses them into structured filters using an LLM, runs a **hybrid metadata + vector search** on Pinecone, and returns a conversational product recommendation.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Parser (LLM) │  ← Groq / Llama 3 70B
│  Extracts filters    │
└─────────┬───────────┘
          │  ParsedQuery
          ▼
┌─────────────────────┐
│  Hybrid Search       │  ← Pinecone (vector + metadata)
│  all-MiniLM-L6-v2   │
└─────────┬───────────┘
          │  ProductResults
          ▼
┌─────────────────────┐
│  Response Generator  │  ← Groq / Llama 3 70B
│  Conversational reply│
└─────────────────────┘
```

## Tech Stack

| Component          | Technology                              |
| ------------------ | --------------------------------------- |
| API Framework      | FastAPI + Uvicorn                       |
| LLM                | Groq — `llama3-70b-8192`               |
| Embeddings         | `all-MiniLM-L6-v2` (384-dim, local)    |
| Vector Database    | Pinecone (serverless, cosine, dim=384)  |
| Schemas            | Pydantic v2                             |
| Data Processing    | pandas                                  |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

| Variable              | Where to get it                                      |
| --------------------- | ---------------------------------------------------- |
| `GROQ_API_KEY`        | [console.groq.com](https://console.groq.com)         |
| `PINECONE_API_KEY`    | [app.pinecone.io](https://app.pinecone.io)           |
| `PINECONE_ENV`        | Leave as `us-east-1` for serverless                  |
| `PINECONE_INDEX_NAME` | Leave as `ecommerce-products` or choose your own     |

### 3. Ingest product data into Pinecone

```bash
python ingestion/ingest.py
```

This reads `data/products.csv`, generates embeddings, and upserts 80 products into your Pinecone index.

### 4. Start the API server

```bash
python -m uvicorn main:app --reload
```

### 5. Open the interactive docs

Navigate to **http://127.0.0.1:8000/docs** to use the Swagger UI.

## Example Queries

Try these in the `/api/v1/search` endpoint:

### 1. Trail running shoes

```json
{
  "query": "waterproof trail running shoes under $100 for men"
}
```

### 2. Day-hike backpack

```json
{
  "query": "lightweight backpack for day hikes under $60"
}
```

### 3. Women's rain jacket

```json
{
  "query": "women's rain jacket that's breathable and under $150"
}
```

## Project Structure

```
ecommerce-rag/
├── main.py                     # FastAPI app entry point
├── config.py                   # Environment variable loader
├── .env.example                # Template for API keys
├── requirements.txt            # Pinned dependencies
├── data/
│   └── products.csv            # 80 realistic outdoor gear products
├── models/
│   └── schemas.py              # Pydantic v2 data models
├── ingestion/
│   └── ingest.py               # One-time Pinecone ingestion script
├── search/
│   ├── query_parser.py         # LLM-powered query → structured filters
│   └── hybrid_search.py        # Vector + metadata search on Pinecone
├── synthesis/
│   └── response_generator.py   # LLM-powered conversational response
└── routers/
    └── search_router.py        # FastAPI router (/api/v1/search)
```

## API Endpoints

| Method | Path              | Description                        |
| ------ | ----------------- | ---------------------------------- |
| GET    | `/`               | Health check                       |
| POST   | `/api/v1/search`  | Natural-language product search    |

## License

MIT
