"""
shopping_agent.py
-----------------
A tool-calling shopping assistant built with LangChain + Groq.

Flow:
  1. (optional) describe_product_image -> vision model turns a photo into search keywords
  2. search_products                   -> SQL search with price / organic / min-rating filters
  3. checkout                          -> writes the confirmed order to the `orders` table

API KEY: the only key needed is GROQ_API_KEY (used by both the text and vision
models). Put it in a `.env` file next to this script:  GROQ_API_KEY=your_key_here
Everything else (SQLite database, reviews "API") is local and needs no key.
"""

import base64
import json
import os
import re
import sqlite3
from contextlib import closing
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating, get_ratings_for_products

# Load variables from .env (GROQ_API_KEY, optional model overrides).
load_dotenv()

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store.db")

# Model names can be overridden from .env, so you don't have to edit code if
# Groq renames or retires a model.
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen/qwen3-32b")
VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

SUPPORTED_IMAGE_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}

# Words ignored when matching the free-text query against the catalogue.
# ("organic" is handled by the is_organic filter, not by text matching.)
QUERY_STOPWORDS = {"a", "an", "the", "and", "or", "with", "for", "of", "some", "organic", "non", "nonorganic"}


def _require_api_key() -> None:
    """Fail early with a readable message instead of a cryptic stack trace."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file containing "
            "GROQ_API_KEY=your_key_here (get a free key at https://console.groq.com)."
        )


# LLMs are created lazily (and cached) so that importing this module -- e.g. to
# unit-test the tools -- does not require an API key.
@lru_cache(maxsize=1)
def _text_llm() -> ChatGroq:
    _require_api_key()
    return ChatGroq(model=TEXT_MODEL, temperature=0)


@lru_cache(maxsize=1)
def _vision_llm() -> ChatGroq:
    _require_api_key()
    return ChatGroq(model=VISION_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Tools (functions the LLM is allowed to call)
# The docstring of each tool is what the LLM reads to decide when to use it.
# ---------------------------------------------------------------------------

@tool
def search_products(
    query: str,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
    min_rating: Optional[float] = None,
) -> str:
    """
    Search the product catalogue. Use a SHORT product keyword as the query
    (e.g. 'honey', 'olive oil', 'almonds') and put everything else in the filters:
      - max_price:  only products costing this much or less
      - is_organic: true for organic only, false for non-organic only, omit for both
      - min_rating: only products whose average customer rating is at least this value
    Returns a JSON array sorted by rating (best first). Each item has: id, name,
    category, price, description, is_organic, average_rating, review_count.
    """
    # Split "olive oil" into ["olive", "oil"]; every remaining word must appear
    # somewhere in the name/description/category (so word order doesn't matter).
    words = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if w not in QUERY_STOPWORDS]

    def run_query(fields: list[str]) -> list:
        """Run the search, requiring every query word to match at least one of `fields`."""
        sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
        params: list = []
        for word in words:
            sql += " AND (" + " OR ".join(f"{f} LIKE ?" for f in fields) + ")"
            params.extend([f"%{word}%"] * len(fields))
        if max_price is not None:
            sql += " AND price <= ?"
            params.append(max_price)
        if is_organic is not None:
            sql += " AND is_organic = ?"
            params.append(1 if is_organic else 0)
        with closing(sqlite3.connect(DB_PATH)) as conn:
            return conn.execute(sql, params).fetchall()

    # Match on name/category first so 'honey' doesn't return 'Organic Granola'
    # (whose description merely mentions honey). Only if that finds nothing do we
    # widen the search to descriptions (e.g. 'gluten-free', 'omega-3').
    rows = run_query(["name", "category"]) or run_query(["name", "category", "description"])

    # Fetch every rating in ONE query, then filter in Python. Doing this here
    # (instead of asking the LLM to call get_rating N times and compare numbers)
    # is faster and removes a common source of LLM arithmetic mistakes.
    ratings = {r["product_id"]: r for r in get_ratings_for_products([row[0] for row in rows])}

    products = []
    for pid, name, category, price, description, organic in rows:
        rating = ratings[pid]
        if min_rating is not None and rating["average_rating"] < min_rating:
            continue
        products.append({
            "id": pid,
            "name": name,
            "category": category,
            "price": price,
            "description": description,
            "is_organic": bool(organic),
            "average_rating": rating["average_rating"],
            "review_count": rating["review_count"],
        })

    products.sort(key=lambda p: (-p["average_rating"], p["price"]))
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for ONE product ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    return json.dumps(get_product_rating(product_id))


@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves the order to the database and
    returns a confirmation message with the order ID, product name and price.
    Only call this after the user has explicitly confirmed the purchase.
    """
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            row = conn.execute("SELECT name, price FROM products WHERE id = ?", (product_id,)).fetchone()
            if row is None:
                return f"Error: product with ID {product_id} not found."

            name, price = row
            cursor = conn.execute(
                "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
                (product_id, name, price),
            )
            conn.commit()
            order_id = cursor.lastrowid
    except sqlite3.OperationalError as exc:
        # Most likely the tables are missing because setup_db.py was never run.
        return f"Error: could not save the order ({exc}). Has setup_db.py been run?"

    return (
        f"Order #{order_id} confirmed! '{name}' has been ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product photo and return its attributes as a JSON object with:
    product_type, search_query, is_organic, description.
    Use this when the user uploads an image. The result feeds directly into search_products.
    """
    if not os.path.isfile(image_path):
        return json.dumps({"error": f"Image file not found: {image_path}"})

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext not in SUPPORTED_IMAGE_TYPES:
        return json.dumps({"error": f"Unsupported image type '.{ext}'. Use jpg, png or webp."})

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # Multimodal message: one image part + one text instruction part.
    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:{SUPPORTED_IMAGE_TYPES[ext]};base64,{image_b64}"}},
        {"type": "text", "text": (
            "Look at this product image and extract its key attributes. "
            "Return ONLY a JSON object with these fields:\n"
            "- product_type: what kind of product it is (e.g. honey, olive oil, almonds)\n"
            "- search_query: one or two keywords to search for it (e.g. 'honey', 'olive oil')\n"
            "- is_organic: true if the label says organic, false if not, null if unclear\n"
            "- description: one sentence describing the product"
        )},
    ])

    raw = extract_text(_vision_llm().invoke([message]))

    # Vision models often wrap JSON in ```json fences or add chatter, so pull
    # out the first {...} block and validate it before handing it to the agent.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.dumps(json.loads(match.group(0)))
        except json.JSONDecodeError:
            pass
    return json.dumps({"error": "Could not read the image", "raw_response": raw[:300]})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text(message) -> str:
    """
    Get plain text out of a LangChain message and remove any <think>...</think>
    reasoning that Qwen3 sometimes includes in its output.
    """
    content = message.content
    if isinstance(content, list):  # newer LangChain versions can return content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


SYSTEM_PROMPT = (
    "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
    "IMAGE SEARCH - when the user provides an image path:\n"
    "1. Call describe_product_image with the path.\n"
    "2. Use the returned search_query and is_organic to call search_products.\n"
    "3. Continue with the BROWSING flow from step 3. If the tool returns an error, "
    "   tell the user briefly and ask them to describe the product in words.\n\n"
    "BROWSING - when the user describes what they want to buy:\n"
    "1. Call search_products ONCE with a short keyword as query and put every constraint "
    "   in the filters: max_price, is_organic (true when they ask for organic) and min_rating.\n"
    "2. Ratings are already included in the results, so do NOT call get_rating for each item.\n"
    "3. If nothing matches, say so and suggest relaxing one filter.\n"
    "4. Present matching products as a numbered list. Use this exact format "
    "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
    "   #<number>. <name> (ID:<product_id>) - $<price> - <average_rating> stars - <organic or non-organic>\n\n"
    "   Put a blank line between entries. Always include (ID:X) so you can reference it later.\n"
    "5. End by asking: 'Would you like to order one? Just say yes or give me the number.'\n"
    "6. Do NOT call checkout at this stage.\n\n"
    "ORDERING - when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
    "'order number 2', 'the first one', 'get me #3'):\n"
    "1. Find the (ID:X) of the chosen product in your previous message "
    "   (if only one product was listed and the user says 'yes', use that one).\n"
    "2. If it is unclear which product they mean, ask instead of guessing.\n"
    "3. Call checkout with that product_id, then confirm the order in plain text.\n\n"
    "Never place an order unless the user explicitly confirms. "
    "Never guess a product_id - always take it from the (ID:X) in your own previous message."
)


@lru_cache(maxsize=1)
def get_agent():
    """Build the agent once and reuse it (creating it needs the API key)."""
    return create_agent(
        model=_text_llm(),
        tools=[search_products, get_rating, checkout, describe_product_image],
        system_prompt=SYSTEM_PROMPT,
    )


def ask(messages: list[dict]) -> str:
    """
    Send the chat history ([{"role": "user"|"assistant", "content": str}, ...])
    to the agent and return the final reply as clean text.
    """
    result = get_agent().invoke({"messages": messages})
    # Strip backticks: the UI renders markdown and we asked for plain text.
    return extract_text(result["messages"][-1]).replace("`", "")


if __name__ == "__main__":
    # Smoke test: python shopping_agent.py
    print(ask([{"role": "user", "content": "I want to buy organic honey with 4.5+ rating and less than $20 price."}]))
