from typing import Annotated, TypedDict

from chains import generate_chain, reflection_chain
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def reflection_node(state: MessageGraph):
    return {"messages": reflection_chain.invoke({"messages": state["messages"]})}


def generation_node(state: MessageGraph):
    res = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": HumanMessage(content=res.content)}


builder = StateGraph(state_schema=MessageGraph)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, {REFLECT: REFLECT, END: END})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

# print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)]}

    result = graph.invoke(inputs)
    print(result["messages"][-1].content)

# Output:
"""
Here’s a sharper tweet-ready version:

**@LangChainAI’s new Tool Calling feature is seriously underrated.**

It finally makes building agents with function calling across different models incredibly easy.

I also made a video breaking down their latest blog post.

If you want, I can make it:
- more technical
- more viral/hype
- shorter and punchier
"""
