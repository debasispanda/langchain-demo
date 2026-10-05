import asyncio

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini")

stdio_server_params = StdioServerParameters(
    command="python",
    args=[
        "/Users/debasispanda/Workspace/Learnings/AI/langchain-demo/9_mcp/servers/math_server.py"
    ],
)


async def main():
    print("MCP Example ------->")
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("Session initialized.")
            tools = await load_mcp_tools(session)
            print("Tools loaded.")
            agent = create_agent(llm, tools)

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 54 + 2 * 3?")]}
            )
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
