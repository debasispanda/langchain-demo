from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for user's tweet."
            "Always provide detailed recommendations including tweet structure, content quality, request length, virality, style, and engagement tips etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a techie twitter influencer assistant tasked with writing execellent tweeter posts."
            "Generate best twitter post possible for the user's request"
            "If user provides critique, respond with an improved version of your preview attempt.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-5.4-mini")

generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
