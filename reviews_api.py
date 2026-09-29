"""
reviews_api.py
--------------
A tiny "reviews service" that reads the `reviews` table in store.db and returns
aggregated rating data. It is a local SQLite query, so NO API key is needed.

Public functions:
    get_product_rating(product_id)          -> rating info for one product
    get_ratings_for_products(product_ids)   -> rating info for many products (one query)
"""

import os
import sqlite3
from contextlib import closing

# The database lives next to this file, regardless of where the script is run from.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store.db")


def _empty_rating(product_id: int) -> dict:
    """Default result for a product that has no reviews yet."""
    return {"product_id": product_id, "average_rating": 0.0, "review_count": 0}


def get_product_rating(product_id: int) -> dict:
    """Return the average rating (rounded to 2 dp) and review count for one product."""
    # closing() guarantees the connection is released even if the query raises.
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT AVG(rating), COUNT(*) FROM reviews WHERE product_id = ?",
            (product_id,),
        ).fetchone()

    # AVG() returns NULL (None) when there are no rows, so guard against that.
    if not row or row[0] is None:
        return _empty_rating(product_id)
    return {
        "product_id": product_id,
        "average_rating": round(row[0], 2),
        "review_count": row[1],
    }


def get_ratings_for_products(product_ids: list[int]) -> list[dict]:
    """
    Return ratings for many products using a single GROUP BY query
    (much cheaper than calling get_product_rating in a loop).
    The output order matches the input order; unreviewed products get 0.0 / 0.
    """
    if not product_ids:
        return []

    # Build "?,?,?" so the IDs are passed as bound parameters (no SQL injection).
    placeholders = ",".join("?" * len(product_ids))
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            f"""
            SELECT product_id, AVG(rating), COUNT(*)
            FROM reviews
            WHERE product_id IN ({placeholders})
            GROUP BY product_id
            """,
            product_ids,
        ).fetchall()

    by_id = {pid: (round(avg, 2), cnt) for pid, avg, cnt in rows}
    results = []
    for pid in product_ids:
        avg, cnt = by_id.get(pid, (0.0, 0))
        results.append({"product_id": pid, "average_rating": avg, "review_count": cnt})
    return results


if __name__ == "__main__":
    # Quick manual check: python reviews_api.py
    single = get_product_rating(3)
    print("Single product rating:")
    print(f"  Product {single['product_id']}: {single['average_rating']} stars "
          f"({single['review_count']} reviews)")

    print("\nBatch ratings:")
    for r in get_ratings_for_products([1, 2, 4, 5, 6]):
        print(f"  Product {r['product_id']}: {r['average_rating']} stars "
              f"({r['review_count']} reviews)")
