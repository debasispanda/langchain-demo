from typing import Literal

from chains import first_responder, revisor
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from tool_executor import execute_tools

load_dotenv()

MAX_ITERATIONS = 2


def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState):
    """Revise the existing response using tool response."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    iterations = sum(isinstance(message, ToolMessage) for message in state["messages"])

    if iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)

builder.add_node("draft", draft_node)
builder.add_node("revise", revise_node)
builder.add_node("execute_tools", execute_tools)

builder.set_entry_point("draft")

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

graph = builder.compile()

print(graph.get_graph().draw_mermaid())


res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)

last_message = res["messages"][-1]

if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
