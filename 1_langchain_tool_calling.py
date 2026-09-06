from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

MODEL = "qwen3.5:2b"
MAX_ITERATIONS=10

# ---------- Tools ----------

@tool
def get_product_price(product_name: str) -> float:
    """
    Get the price of the given product.

    Args:
        product_name (str): The name of the product.

    Returns:
        float: The price of the product.
    """
    print(f"    >> Executing get_product_price(product='{product_name}')")
    product_map = {
        "television": 1050.00,
        "laptop": 1500.00,
        "mobile": 800.00,
    }
    return product_map.get(product_name, 0.0)

@tool
def apply_discount(product_price: float, discount_tier: str) -> float:
    """
    Apply a discount to the price of the given product.

    Args:
        product_price (float): The original price of the product.
        discount_tier (str): The discount tier to apply (e.g., "silver", "gold", "platinum").

    Returns:
        float: The discounted price of the product.
    """
    print(f"    >> Executing apply_discount(product_price='{product_price}', discount_tier='{discount_tier  }')")
    discount_map = {
        "silver": 5,
        "gold": 10,
        "platinum": 15,
    }
    discount = discount_map.get(discount_tier, 0)
    return product_price * (1 - discount / 100)

tools = [get_product_price, apply_discount]
tools_map = {tool.name: tool for tool in tools}

llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0)
llm_with_tools = llm.bind_tools(tools)

messages: list[AIMessage | HumanMessage | SystemMessage | ToolMessage] = [
    SystemMessage(content=(
        "You are an AI agent that can provide product prices and apply discounts based on the given tools."
        "STRICT RULES - You must follow these rules strictly:"
        "1. Always use the provided tools to get product prices and apply discounts."
        "2. Never make assumptions about product prices or discounts."
        "3. Respond only with the output of the tools, do not provide additional commentary."
    ))
]

@traceable(name="Product Price Agent")
def run_agent(question: str):
    print(f"\nQuestion: {question}\n")
    print("=" * 50)

    messages.append(HumanMessage(content=question))
    
    for i in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {i} ---")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls if hasattr(ai_message, "tool_calls") else None

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool selected]: {tool_name} with args: {tool_args}")

        tool_func = tools_map.get(tool_name)

        if tool_func is None:
            raise RuntimeError(f"Tool function for {tool_name} not found.")
        
        tool_result = tool_func.invoke(tool_args)
        print(f"[Tool result]: {tool_result}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))

    print("\nReached maximum iterations without a final answer.")
    return None


if __name__ == "__main__":
    print(run_agent("What is the price of a laptop after applying a gold discount?"))