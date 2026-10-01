"""Try the tools directly, with no LLM involved."""
import json

from tools import search_products, price_check


def show(title, data):
    print(f"\n=== {title} ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))


found = search_products(max_price=15000, limit=3)
show("search_products(max_price=15000, limit=3)", found)

if found["results"]:
    show("price_check on the top result", price_check(found["results"][0]["id"]))

show("A budget nobody can meet", search_products(max_price=500))
show("An invalid budget", search_products(max_price=-1))