from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)


def run_query(search_queries: list[str]):
    """Run a search query using the TavilySearch tool and return the results."""
    return tavily_tool.batch(
        [{"query": search_query} for search_query in search_queries]
    )


execute_tools = ToolNode(
    tools=[
        StructuredTool.from_function(run_query, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_query, name=ReviseAnswer.__name__),
    ]
)
