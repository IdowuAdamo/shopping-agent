import json

from tools import TOOLS, call_tool

MODEL = "gpt-4o-mini"
MAX_STEPS = 5        # stop the loop even if the model keeps calling tools
MAX_HISTORY = 10     # how many past messages we send back to the model

SYSTEM_PROMPT = """You are a shopping assistant for wireless headphones and earbuds sold on Jumia Nigeria.

Rules you must follow:
1. Only talk about products and prices that came from your tools. Never invent products, prices, ratings or reviews.
2. Prices are in Nigerian Naira. Write them like ₦12,500.
3. The user's budget is a hard limit. Remember it for the whole conversation and always pass it to search_products.
4. When you list products, show each product's id in brackets, like [id: 3fa2b1c9], so you can refer back to it later.
5. To judge whether a price is fair, call price_check and explain its result in plain words. Never do your own arithmetic; quote the numbers the tool gives you.
6. If price_check says "insufficient_data", or a product has no rating or few reviews, say so honestly instead of guessing.
7. A rating based on very few reviews is weak evidence. Say so when it applies.
8. The data is a snapshot of Jumia listings, so prices may differ from the live site. Mention this briefly when you give a price verdict.
9. If the request is unclear (for example no budget), ask one short clarifying question.
Keep answers short and easy to scan."""


def run_agent(client, history: list[dict], user_message: str) -> str:
    """Send the conversation to the model, run any tools it asks for, and return the final answer."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-MAX_HISTORY:]
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message

        # No tool requested: this is the final answer
        if not message.tool_calls:
            return message.content

        # Record the model's tool request, then answer each one
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            print(f"[tool] {call.function.name}({arguments})")
            result = call_tool(call.function.name, arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)}
            )

    return "Sorry, I couldn't finish that request. Could you try rephrasing it?"