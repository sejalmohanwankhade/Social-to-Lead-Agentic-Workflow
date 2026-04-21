"""
utils/rag_pipeline.py
Local RAG (Retrieval-Augmented Generation) pipeline for AutoStream knowledge base.
Uses TF-IDF style keyword matching for local retrieval without external vector DBs.
"""

import json
import os
import re
from typing import Optional


KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "autostream_kb.json")


def load_knowledge_base() -> dict:
    """Load the AutoStream knowledge base from JSON."""
    with open(KB_PATH, "r") as f:
        return json.load(f)


def _flatten_kb_to_chunks(kb: dict) -> list[dict]:
    """
    Break the knowledge base into retrievable text chunks with metadata.
    Each chunk has: { "text": str, "topic": str, "keywords": list[str] }
    """
    chunks = []

    # Company info chunk
    company = kb["company"]
    chunks.append({
        "text": f"{company['name']} — {company['tagline']}. {company['description']}",
        "topic": "company_overview",
        "keywords": ["autostream", "about", "what", "platform", "tool", "company", "video", "editing"]
    })

    # Pricing chunks — one per plan
    for plan in kb["pricing"]["plans"]:
        features_text = ", ".join(plan["features"])
        price_text = f"${plan['price_monthly']}/month (or ${plan['price_annual']}/year)"
        chunk_text = (
            f"{plan['name']}: {price_text}. "
            f"Features: {features_text}."
        )
        if plan["limitations"]:
            chunk_text += f" Limitations: {', '.join(plan['limitations'])}."

        chunks.append({
            "text": chunk_text,
            "topic": f"pricing_{plan['name'].lower().replace(' ', '_')}",
            "keywords": [
                "price", "pricing", "plan", "cost", "how much", "pay", "subscription",
                plan["name"].lower(), "basic", "pro", "monthly", "annual", "features",
                "4k", "resolution", "videos", "captions", "unlimited"
            ]
        })

    # Policy chunks
    policies = kb["policies"]
    policy_map = {
        "refund": ("refund_policy", policies["refund_policy"],
                   ["refund", "return", "money back", "cancel", "7 days", "days"]),
        "support": ("support_policy", policies["support_policy"],
                    ["support", "help", "24/7", "contact", "customer service", "chat"]),
        "trial": ("trial_policy", policies["trial_policy"],
                  ["trial", "free", "test", "try", "7 day", "no credit card"]),
        "cancellation": ("cancellation_policy", policies["cancellation_policy"],
                         ["cancel", "cancellation", "stop", "end subscription"]),
    }
    for key, (topic, text, keywords) in policy_map.items():
        chunks.append({"text": text, "topic": topic, "keywords": keywords})

    # FAQ chunks
    for faq in kb["faqs"]:
        chunks.append({
            "text": f"Q: {faq['question']} A: {faq['answer']}",
            "topic": "faq",
            "keywords": faq["question"].lower().split()
        })

    return chunks


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Retrieve the most relevant knowledge base chunks for the given query.

    Args:
        query  : The user's question or message
        top_k  : Number of top chunks to return

    Returns:
        A formatted string of retrieved context for the LLM prompt.
    """
    kb = load_knowledge_base()
    chunks = _flatten_kb_to_chunks(kb)

    query_lower = query.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))

    scored_chunks = []
    for chunk in chunks:
        kw_set = set(chunk["keywords"])
        # Keyword overlap score
        kw_score = len(query_tokens & kw_set)

        # Substring match bonus (handles multi-word phrases)
        text_lower = chunk["text"].lower()
        substr_bonus = sum(1 for token in query_tokens if token in text_lower)

        total_score = kw_score + substr_bonus
        if total_score > 0:
            scored_chunks.append((total_score, chunk))

    # Sort descending by score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c["text"] for _, c in scored_chunks[:top_k]]

    if not top_chunks:
        return "No specific information found in the knowledge base for this query."

    context = "\n\n---\n\n".join(top_chunks)
    return context


def get_full_pricing_context() -> str:
    """Return full pricing context for when users ask broadly about plans."""
    kb = load_knowledge_base()
    plans = kb["pricing"]["plans"]
    lines = []
    for plan in plans:
        lines.append(f"**{plan['name']}** — ${plan['price_monthly']}/month")
        for f in plan["features"]:
            lines.append(f"  • {f}")
    return "\n".join(lines)
