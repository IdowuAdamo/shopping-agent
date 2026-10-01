"""Tools the agent can call. All numbers are computed here, in plain Python.

The LLM decides WHICH tool to call and explains the result in words.
It never does the arithmetic itself.
"""
import json
from pathlib import Path
from statistics import median, quantiles

DATA_PATH = Path(__file__).parent / "data" / "headphones.json"

MIN_COMPARABLES = 10   # fewer similar products than this -> refuse to judge the price
BELOW_TYPICAL = 0.8    # price under 80% of the median  -> "below_typical"
ABOVE_TYPICAL = 1.25   # price over 125% of the median  -> "above_typical"
PRIOR_VOTES = 20       # how many reviews it takes before a rating is trusted (see weighted_rating)
MAX_RESULTS = 10
SEGMENTS = ("earbuds", "headphones")

_catalog = None


def get_catalog() -> list[dict]:
    """Load the product data once and reuse it."""
    global _catalog
    if _catalog is None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _catalog = json.load(f)
    return _catalog


def claimed_discount_pct(product: dict):
    """The discount the seller advertises, computed from old price vs price."""
    old = product["old_price"]
    if old and old > product["price"]:
        return round(100 * (old - product["price"]) / old)
    return None


def weighted_rating(rating, votes, mean_rating, prior_votes=PRIOR_VOTES):
    """Pull ratings with few reviews toward the catalog average.

    A 5.0 from 2 reviews should not beat a 4.6 from 900 reviews.
    """
    if rating is None:
        return None
    v = votes or 0
    return (v / (v + prior_votes)) * rating + (prior_votes / (v + prior_votes)) * mean_rating


def _mean_rating(catalog):
    ratings = [p["rating"] for p in catalog if p["rating"] is not None]
    return sum(ratings) / len(ratings) if ratings else 0.0


def _rank_key(product, mean_rating):
    score = weighted_rating(product["rating"], product["num_reviews"], mean_rating)
    # Products with no rating at all go last
    return (score is not None, score if score is not None else 0.0, product["num_reviews"] or 0)


def _summary(product, mean_rating):
    score = weighted_rating(product["rating"], product["num_reviews"], mean_rating)
    return {
        "id": product["id"],
        "name": product["name"],
        "price": product["price"],
        "old_price": product["old_price"],
        "claimed_discount_pct": claimed_discount_pct(product),
        "rating": product["rating"],
        "num_reviews": product["num_reviews"],
        "rank_score": round(score, 2) if score is not None else None,
        "seller": product["seller"],
        "segment": product["segment"],
    }


def search_products(max_price, keyword=None, segment=None, min_rating=None, limit=5, catalog=None):
    """Find products at or under max_price (Naira), best-rated first."""
    if catalog is None:
        catalog = get_catalog()
    if isinstance(max_price, bool) or not isinstance(max_price, (int, float)) or max_price <= 0:
        return {"error": "max_price must be a positive number in Naira"}
    if segment is not None and segment not in SEGMENTS:
        return {"error": f"segment must be one of {list(SEGMENTS)}"}

    limit = max(1, min(int(limit), MAX_RESULTS))
    mean_rating = _mean_rating(catalog)

    # The budget is enforced here, in code - not just requested in the prompt.
    matches = [p for p in catalog if p["price"] <= max_price]
    if segment:
        matches = [p for p in matches if p["segment"] == segment]
    if keyword:
        matches = [p for p in matches if keyword.lower() in p["name"].lower()]
    if min_rating is not None:
        matches = [p for p in matches if p["rating"] is not None and p["rating"] >= min_rating]

    matches.sort(key=lambda p: _rank_key(p, mean_rating), reverse=True)
    result = {
        "max_price_applied": max_price,
        "total_matches": len(matches),
        "results": [_summary(p, mean_rating) for p in matches[:limit]],
    }
    if not matches:
        result["note"] = "No products matched. Suggest raising the budget or relaxing filters."
    return result


def price_check(product_id, catalog=None):
    """Compare one product's price with similar products in the dataset."""
    if catalog is None:
        catalog = get_catalog()
    product = next((p for p in catalog if p["id"] == product_id), None)
    if product is None:
        return {"error": f"No product with id '{product_id}'"}

    mean_rating = _mean_rating(catalog)
    group = [p for p in catalog if p["segment"] == product["segment"]]
    result = {"product": _summary(product, mean_rating), "comparison_group": product["segment"]}

    if len(group) < MIN_COMPARABLES:
        result["price_position"] = "insufficient_data"
        result["note"] = (
            f"Only {len(group)} comparable products; need at least {MIN_COMPARABLES} "
            "to judge whether the price is fair."
        )
        return result

    prices = [p["price"] for p in group]
    med = median(prices)
    q1, _, q3 = quantiles(prices, n=4, method="inclusive")
    ratio = product["price"] / med

    if ratio < BELOW_TYPICAL:
        position = "below_typical"
    elif ratio > ABOVE_TYPICAL:
        position = "above_typical"
    else:
        position = "typical"

    result["price_position"] = position
    result["comparison"] = {
        "comparable_products": len(group),
        "median_price": round(med),
        "typical_range_25th_to_75th": [round(q1), round(q3)],
        "price_vs_median": round(ratio, 2),
        "cheaper_than_pct_of_comparables": round(100 * sum(1 for x in prices if x > product["price"]) / len(prices)),
        "old_price_vs_median": round(product["old_price"] / med, 2) if product["old_price"] else None,
    }
    result["note"] = "Comparison uses a fixed snapshot of Jumia listings, not live prices."
    return result


# ---- What the LLM is told about the tools -------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search wireless headphones and earbuds by budget. Returns up to `limit` products "
                "at or under max_price (Naira), best-rated first, plus the total number of matches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {"type": "number", "description": "Maximum price in Naira"},
                    "segment": {"type": "string", "enum": list(SEGMENTS), "description": "Optional product type"},
                    "keyword": {"type": "string", "description": "Optional word that must appear in the product name, e.g. a brand"},
                    "min_rating": {"type": "number", "description": "Optional minimum star rating (1-5)"},
                    "limit": {"type": "integer", "description": "How many products to return (default 5, max 10)"},
                },
                "required": ["max_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "price_check",
            "description": (
                "Check whether one product's price is typical, below typical or above typical "
                "compared with similar products. Needs the product id from search_products."
            ),
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string", "description": "The id of the product"}},
                "required": ["product_id"],
            },
        },
    },
]

TOOL_FUNCTIONS = {"search_products": search_products, "price_check": price_check}


def call_tool(name: str, arguments: dict) -> dict:
    """Run a tool by name. Errors become data the LLM can read, not crashes."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return func(**arguments)
    except (TypeError, ValueError) as e:
        return {"error": f"Invalid arguments for {name}: {e}"}