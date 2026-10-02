from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import (JsonOutputToolsParser,
                                           PydanticToolsParser)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini")

json_parser = JsonOutputToolsParser(return_id=True)
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])


SYSTEM_INSTRUCTION = """You are a expert researcher.

current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize the improvement.
3. Recommend search queries to research and improve your answer.
"""

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTION),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.now().isoformat())


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed answer in ~250 words."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

REVISE_INSTRUCTION = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=REVISE_INSTRUCTION
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC/autonomous SOC problem domain, list startups that do that and raised capital."
    )

    chain = first_responder | pydantic_parser
    result = chain.invoke({"messages": [human_message]})
    print(result)
