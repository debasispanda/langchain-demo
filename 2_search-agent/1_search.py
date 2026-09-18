from dotenv import load_dotenv
from langchain.agents import create_agent
# from langchain.tools import tool
from langchain_tavily import TavilySearch

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient

load_dotenv()

# tavily_client = TavilyClient()

# @tool
# def search(query: str) -> dict:
#     """
#     Search for the given query using internet.

#     Args:
#         query (str): The search query.

#     Returns:
#         dict: The search result.
#     """
#     print(f"Searching for query: {query}")
#     return tavily_client.search(query)

llm = ChatOpenAI(model="gpt-5.4-mini")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(
    tools=tools,
    model=llm,
)

query = """
    Search the recent senior fullstack jobs posted on linkedin for hyderabad, india area. 
    Required skills are React, NodeJS, Python. Provide the links of jobs under 50 applications.
"""

def main():
    print("Hello from langchain-demo!")
    response = agent.invoke({ "messages": [HumanMessage(content=query)] })
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
