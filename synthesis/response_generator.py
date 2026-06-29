"""
Response synthesiser — takes the raw query and search results and uses
Groq (Llama 3 70B) to generate a friendly, conversational recommendation.
"""

import logging
from typing import List

from clients import groq_client
from models.schemas import ProductResult

logger = logging.getLogger(__name__)


def generate_narrative(raw_query: str, results: List[ProductResult]) -> str:
    """
    Generate a conversational product recommendation based on search results.

    If the LLM call fails (timeout, rate-limit, outage), returns a plain-text
    fallback so the caller still receives usable search results.
    """

    if not results:
        return (
            "I couldn't find any products matching your exact criteria. "
            "Try broadening your search — for example, remove a price limit "
            "or try a different category."
        )

    # ── Build product summary ───────────────────────────────────────────────
    lines: list[str] = []
    for r in results:
        tags_str = ", ".join(r.tags) if r.tags else "N/A"
        lines.append(
            f"- {r.name} by {r.brand} — ${r.price:.2f} | "
            f"Rating: {r.rating}/5.0 | Tags: {tags_str}"
        )
    product_summary = "\n".join(lines)

    # ── Build LLM prompt ────────────────────────────────────────────────────
    system_message = (
        "You are a helpful, enthusiastic personal shopping assistant for an "
        "outdoor gear store. Be conversational, specific, and explain exactly "
        "why each product fits the customer's needs."
    )

    user_message = (
        f"The customer searched for: '{raw_query}'.\n\n"
        f"Here are the top matching products:\n{product_summary}\n\n"
        "Write a friendly 3-5 sentence response explaining why these products "
        "are great for their specific scenario. Mention specific product names "
        "and prices."
    )

    # ── Call Groq (with graceful fallback) ──────────────────────────────────
    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return chat_completion.choices[0].message.content.strip()

    except Exception as exc:
        logger.error("Narrative generation failed (%s: %s). Using fallback.", type(exc).__name__, exc)
        # Return a simple but usable plain-text summary so the response
        # still contains meaningful product information.
        fallback_lines = [f"Here are the top results for '{raw_query}':\n"]
        for r in results:
            fallback_lines.append(f"• {r.name} by {r.brand} — ${r.price:.2f} (★ {r.rating})")
        fallback_lines.append(
            "\n(Our recommendation engine is temporarily unavailable. "
            "Please check back shortly for personalized suggestions.)"
        )
        return "\n".join(fallback_lines)
