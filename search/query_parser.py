"""
LLM-powered query parser — sends the raw shopping query to Groq (Llama 3 70B)
and returns a structured ParsedQuery.
"""

import json

from groq import Groq

from config import GROQ_API_KEY
from models.schemas import ParsedQuery


_SYSTEM_PROMPT = """\
You are a shopping query parser for an outdoor gear store. Your job is to extract \
structured information from the user's shopping query and return ONLY a valid JSON \
object. Do not include markdown, backticks, code fences, or any explanation — just \
raw JSON.

Return null for any field you cannot confidently determine from the query.

The JSON object MUST have exactly these keys:

{
  "raw_query": "<the original query, unchanged>",
  "max_price": <number or null>,
  "min_price": <number or null>,
  "required_tags": [<list of relevant product tags like "waterproof", "lightweight", "breathable", "trail", "grip", etc.>],
  "preferred_brand": "<brand name or null>",
  "category": "<one of: running shoes, trail shoes, hiking boots, rain jackets, trail shorts, backpacks, water bottles, headlamps — or null>",
  "gender": "<men, women, or unisex — or null>",
  "semantic_query": "<a cleaned, search-friendly version of the query>"
}
"""


def parse_query(raw_query: str) -> ParsedQuery:
    """Parse a natural-language shopping query into a structured ParsedQuery."""

    client = Groq(api_key=GROQ_API_KEY)

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw_query},
            ],
            temperature=0,
            max_tokens=500,
        )

        response_text = chat_completion.choices[0].message.content.strip()

        parsed_dict = json.loads(response_text)

        # Ensure raw_query is always preserved from the original input
        parsed_dict["raw_query"] = raw_query

        # Ensure semantic_query falls back to raw_query if missing
        if not parsed_dict.get("semantic_query"):
            parsed_dict["semantic_query"] = raw_query

        return ParsedQuery(**parsed_dict)

    except (json.JSONDecodeError, KeyError, TypeError, Exception) as exc:
        # Graceful fallback — return a minimal ParsedQuery so the pipeline
        # can still run a pure semantic search.
        print(f"[WARNING] Query parsing failed ({type(exc).__name__}: {exc}). Using fallback.")
        return ParsedQuery(
            raw_query=raw_query,
            semantic_query=raw_query,
        )
