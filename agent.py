import json
import ast
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI


products_df = pd.read_csv("product_inventory.csv")
orders_df = pd.read_csv("orders.csv")

valid_tags = sorted(set(
    tag.strip().lower()
    for tags_str in products_df["tags"]
    for tag in str(tags_str).split(",")
))

with open("policy.txt", "r") as f:
    policy_text = f.read()


def parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ('true', '1', 'yes')
    return bool(val)

products_df['is_sale'] = products_df['is_sale'].map({'True': True, 'False': False})
products_df['is_clearance'] = products_df['is_clearance'].map({'True': True, 'False': False})


def slim_product(product: dict) -> dict:
    """Return only fields relevant for reasoning."""
    return {
        "product_id": product.get("product_id"),
        "title": product.get("title"),
        "vendor": product.get("vendor"),
        "price": product.get("price"),
        "tags": product.get("tags"),
        "is_sale": product.get("is_sale"),
        "is_clearance": product.get("is_clearance"),
        "bestseller_score": product.get("bestseller_score"),
    }

def slim_order(order: dict) -> dict:
    return {
        "order_id": order.get("order_id"),
        "order_date": order.get("order_date"),
        "product_id": order.get("product_id"),
        "size": order.get("size"),
        "price_paid": order.get("price_paid"),
    }

def parse_stock_per_size(stock_str):
    """Convert stock_per_size string like "{'8': 18, '4': 13}" to dict."""
    try:
        
        return ast.literal_eval(stock_str)
    except:
        
        try:
            return json.loads(stock_str.replace("'", "\""))
        except:
            return {}

def search_products(filters):
    
    if "filters" in filters:
        filters = filters["filters"]

    df = products_df.copy()

    
    size = str(filters.get("size")) if filters.get("size") is not None else None
    
    
    sort_by = filters.get("sort_by", "bestseller")
    if sort_by == "price_asc":
        df = df.sort_values("price", ascending=True)
    elif sort_by == "price_desc":
        df = df.sort_values("price", ascending=False)
    else:
        df = df.sort_values("bestseller_score", ascending=False)
    
    if size:
        def has_size_in_stock(row):
            sizes = str(row["sizes_available"]).split("|")
            if size not in sizes:
                return False
            stock_dict = parse_stock_per_size(row["stock_per_size"])
            return stock_dict.get(size, 0) > 0
        df = df[df.apply(has_size_in_stock, axis=1)]

    
    max_price = filters.get("max_price")
    if max_price:
        df = df[df["price"] <= float(max_price)]

    
    tags = filters.get("tags", []) or []
    tags = [t.lower().strip() for t in (filters.get("tags") or []) if t.lower().strip() in valid_tags]
    
    if tags:
        cleaned_tags = []
        stop_words = {'gown', 'dress', 'style', 'a', 'an', 'the', 'and'}
        for tag in tags:
            parts = tag.strip().lower().split()
            cleaned_tags.extend([p for p in parts if p not in stop_words])

        def has_all_tags(row):
            row_tags = [t.strip().lower() for t in str(row["tags"]).split(",")]
            return all(tag in row_tags for tag in cleaned_tags)
        df = df[df.apply(has_all_tags, axis=1)]

    
    if df.empty:
        return {"message": "No products found matching your criteria.", "results": []}

   
    sale_only = filters.get("sale_only", False)
    if sale_only:
        df["_sale_priority"] = ((df["is_sale"] == True) | (df["is_clearance"] == True)).astype(int)
        df = df.sort_values(["_sale_priority", "bestseller_score"], ascending=[False, False])
        df = df.drop(columns=["_sale_priority"])
    else:
        df = df.sort_values("bestseller_score", ascending=False)

    
    results = df.head(5).to_dict(orient="records")
    for r in results:
        for k, v in r.items():
            if isinstance(v, pd.Timestamp):
                r[k] = str(v)
    return results

def get_product(product_id):
    row = products_df[products_df["product_id"] == product_id]
    if row.empty:
        return {"error": f"Product {product_id} not found"}
    prod = row.iloc[0].to_dict()
    for k, v in prod.items():
        if isinstance(v, pd.Timestamp):
            prod[k] = str(v)
    return prod

def get_order(order_id):
    row = orders_df[orders_df["order_id"] == order_id]
    if row.empty:
        return {"error": f"Order {order_id} not found"}
    order = row.iloc[0].to_dict()
    for k, v in order.items():
        if isinstance(v, pd.Timestamp):
            order[k] = str(v)
    return order

def evaluate_return(order_id):
    """Return a dict with a 'decision' text that the LLM should repeat verbatim."""
    order = get_order(order_id)
    if "error" in order:
        return {"error": order["error"]}

    product = get_product(order["product_id"])
    if "error" in product:
        return {"error": product["error"]}

    order_date = datetime.strptime(str(order["order_date"]), "%Y-%m-%d")
    days_since = (datetime.now() - order_date).days

    
    is_clearance = bool(product.get("is_clearance"))
    is_sale = bool(product.get("is_sale"))
    if is_clearance:
        item_type = "clearance"
    elif is_sale:
        item_type = "sale"
    else:
        item_type = "normal"

    vendor = product.get("vendor", "")
    # vendor exceptions
    aurelia_exchange = (vendor == "Aurelia Couture")
    nocturnal_21 = (vendor == "Nocturne")

    # Build decision
    if item_type == "clearance":
        decision = "NO. Clearance items are final sale and cannot be returned or exchanged."
    elif item_type == "sale":
        if days_since > 7:
            decision = f"NO. Sale items must be returned within 7 days. It has been {days_since} days."
        else:
            credit = "Store credit only."
            if aurelia_exchange:
                credit = "Aurelia Couture: exchange only, no refund."
            decision = f"YES. Returnable within 7 days. {credit}"
    else:  # normal
        window = 21 if nocturnal_21 else 14
        if days_since > window:
            decision = f"NO. Normal items must be returned within {window} days. It has been {days_since} days."
        else:
            note = ""
            if aurelia_exchange:
                note = " Aurelia Couture: exchange only, no refund."
            else:
                note = " Full refund."
            decision = f"YES. Within {window}-day return window.{note}"

    return {
        "order": slim_order(order),
        "product": slim_product(product),
        "days_since_order": days_since,
        "item_type": item_type,
        "decision": decision
    }
    


tools = [
    {
    "type": "function",
    "function": {
        "name": "search_products",
        "description": "Search for products. Extract filters directly from what the user said.",
        "parameters": {
            "type": "object",
            "properties": {
                "size": {
                    "type": "string",
                    "description": "Clothing size if mentioned. E.g. '8', '10', '14'"
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price in dollars if mentioned. E.g. 300"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Style tags ONLY if explicitly in user message. Valid tags: {valid_tags}. "
                                    f"Default to empty array if user mentioned no style words."
                },
                "sale_only": {
                    "type": "boolean",
                    "description": "Set true if user asks for sale OR clearance items."
                },
                "sort_by": {
                "type": "string", 
                "description": "How to sort results. Options: 'price_asc' for cheapest first, 'price_desc' for most expensive, 'bestseller' for most popular. Default is bestseller.",
                "enum": ["price_asc", "price_desc", "bestseller"]
            }
            },
            "required": []
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Fetch full details of a single product by its product_id (e.g. 'P0001').",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Fetch order details by order_id (e.g. 'O0001').",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "evaluate_return",
            "description": "Returns all data needed PLUS a 'decision' text that must be read to the customer verbatim. Do not modify or reinterpret it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    }
]


SYSTEM_PROMPT = f"""You are a concise retail assistant with two roles: Personal Shopper and Customer Support.
You have access to functions to search products, get product details, get order details, and gather data for return evaluation.

BEFORE calling any tool, check if the request makes sense for our store:
- We ONLY sell fashion gowns and dresses from brands like Lumiere, Aurelia Couture, Nocturne, Eden Atelier, Silk Avenue, Velour House.
- If user asks about shoes, jewelry, electronics, or any non-dress item: respond immediately with 'We only carry fashion gowns and dresses. We do not carry [item].' Do NOT call any tool.
- If asked about return policy for product types not in our policy: say 'Our policy does not cover that product type.' Do NOT call any tool.
- If user asks to return an order but provides NO order ID: ask 'Could you please provide your order ID?' Do NOT call any tool.

STRICT RULES:
1. NEVER invent or assume product details or return policies. Always use the provided functions.
2. If a product or order ID does not exist, clearly state that. Do not guess.
3. NEVER apply rules from memory — always use data returned by the functions.

CRITICAL TOOL USAGE RULES:
- "cheapest dress" → tags=[], sort_by="price_asc". NEVER add tags for this.
- "clearance items" → tags=[], sale_only=true. NEVER add tags for this.
- "modest evening gown" → tags=["modest", "evening"]. Only what user said.
- NEVER use tags from previous search results to make a new search.
- NEVER guess tags. If user did not say a style word, tags must be empty.

WHEN RECOMMENDING PRODUCTS:
- Only recommend products whose tags EXACTLY match what the user asked for.
- If no product matches all constraints, say "No exact match found for [tags]. Here are the closest available options: ..." Never present a partial match as a full match.
- Always state: price, sale status, size availability, and which tags matched.

WHEN EVALUATING RETURNS:
1. Call evaluate_return.
2. Read the returned 'decision' text aloud to the user. Do not modify it.
3. If the decision is YES, mention any conditions (e.g., store credit only).
4. If the decision is NO, briefly explain the reason directly from the 'decision' text.
Store Return Policy:
{policy_text}
"""


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" 
)


def run_agent(user_message):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    for _ in range(5):
        response = client.chat.completions.create(
            model="qwen2.5:3b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=500,
            stream=True,        
        )

        
        full_content = ""
        tool_calls_data = []
        finish_reason = None

        for chunk in response:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                print(delta.content, end="", flush=True)  # print as it arrives
                full_content += delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx >= len(tool_calls_data):
                        tool_calls_data.append({
                            "id": "", "name": "", "arguments": ""
                        })
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function.name:
                        tool_calls_data[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls_data[idx]["arguments"] += tc.function.arguments


        if finish_reason == "tool_calls" and tool_calls_data:
            messages.append({
                "role": "assistant",
                "content": full_content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]}
                    }
                    for tc in tool_calls_data
                ]
            })

            for tc in tool_calls_data:
                func_name = tc["name"]
                result = {"error": f"Unknown function {func_name}"}

                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    args = {}

                if func_name == "search_products":
                    result = search_products(args)
                    
                elif func_name == "get_product":
                    result = get_product(args.get("product_id", ""))

                elif func_name == "get_order":
                    order_id = args.get("order_id", "")
                    try:
                        order_id = int(order_id)
                    except (ValueError, TypeError):
                        pass
                    result = get_order(order_id)

                elif func_name == "evaluate_return":
                    order_id = args.get("order_id", "")
                    try:
                        order_id = int(order_id)
                    except (ValueError, TypeError):
                        pass
                    result = evaluate_return(order_id)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, default=str)
                })

        else:
            print()  
            return full_content

    return "I'm sorry, I couldn't complete the request within the allowed steps."

if __name__ == "__main__":
    print(" Retail AI Assistant ")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        print("\nAssistant: ", end="")
        result = run_agent(user_input)
        print()