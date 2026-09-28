from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    last_message = state.get("messages", [])[LAST]
    if not last_message.tool_calls:
        return END
    return ACT


graph = StateGraph(MessagesState)

graph.add_node(AGENT_REASON, run_agent_reasoning)
graph.set_entry_point(AGENT_REASON)
graph.add_node(ACT, tool_node)

graph.add_conditional_edges(AGENT_REASON, should_continue, {ACT: ACT, END: END})
graph.add_edge(ACT, AGENT_REASON)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="./graph.png")

if __name__ == "__main__":
    print("Running langgraph demo")
    response = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the temperature in Tokyo?. List it and then tripple it."
                )
            ]
        }
    )
    print(response.get("messages", [])[LAST].content)
