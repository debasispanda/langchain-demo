from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools

SYSTEM_MESSAGE = """You are a helpful assistant that can use available tools to answer user queries."""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning process using the provided state.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)
