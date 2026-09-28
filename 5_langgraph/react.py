from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


@tool
def tripple(num: float) -> float:
    """
    Triples the given number.

    Args:
        num (float): The number to be tripled.

    Returns:
        float: The tripled value of the input number.
    """
    return float(num) * 3


tools = [
    TavilySearch(max_results=1),
    tripple,
]

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0).bind_tools(tools)
