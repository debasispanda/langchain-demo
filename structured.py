from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """Schema for search result source"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response"""
    answer: str = Field(description="The answer to the search query")
    sources: list[Source] = Field(description="The list of sources for the answer")

model = ChatOpenAI(model="gpt-5.4-mini")
tools = [TavilySearch()]

agent = create_agent(
    model=model,
    tools=tools,
    response_format=AgentResponse
) 

query = """
    Search the langchain developer jobs posted on linkedin for hyderabad, india area. 
    Include only the active jobs with less than 50 applications.
"""

def run():
    print("Running structured script")

    response = agent.invoke({ "messages": [HumanMessage(content=query)]})
    structured_response = response["structured_response"]

    print(structured_response.answer)
    for source in structured_response.sources:
        print(source.url)


if __name__ == "__main__":
    run()


"""
Running structured script
I couldn’t verify any LinkedIn jobs that satisfy all of these filters at once: LangChain developer, Hyderabad/Greater Hyderabad area, active, and fewer than 50 applications.

What I did find on LinkedIn in/around Hyderabad were relevant LangChain/GenAI roles, but the accessible search snippets did not confirm the application count under 50, and some were already marked no longer accepting applications or over 200 applicants.

Relevant LinkedIn results found:
- TECHNICAL LEAD - Python-Langchain — Hyderabad
- AI Engineer — Hyderabad
- AI Engineering - Development Lead — Hyderabad
- Senior AI/ML Engineer_102655 — Hyderabad, but marked no longer accepting applications
- AI Developer Trainee — Hyderabad, but over 200 applicants
- Several posts mentioning LangChain/LangGraph in Hyderabad, but not verifiable as active job listings with <50 applications

If you want, I can do a tighter pass and return only listings that explicitly show “Be among the first 25 applicants” or a similar low-application indicator on LinkedIn.
https://in.linkedin.com/jobs/view/technical-lead-python-langchain-at-happiest-minds-technologies-4458728659
"""