from operator import itemgetter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

print("Initializing...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
vector_store = PineconeVectorStore(
    embedding=embeddings, index_name=os.environ["PINECONE_INDEX"]
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
"""Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

def format_docs(docs: list[Document]):
    """Formats a list of documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

# ============================================================================
# IMPLEMENTATION 0: Raw invocation without RAG)
# ============================================================================
def retrieve_without_rag(query: str):
    """Retrieves documents from the vector store without using RAG."""
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    return result_raw.content

# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.
    """
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Retrieval Without LCEL (Simple Function-Based Approach)")
    print("=" * 70)
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content

# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print('Retrieving...')
    query = "What pinecone in machine learning?"
    # # =======================================
    # # Option 0: Raw invocation without RAG
    # # =======================================
    # answer_without_rag = retrieve_without_rag(query)
    # print("\nAnswer:")
    # print(answer_without_rag)

    # # =======================================
    # # Option 1: Use implementation WITHOUT LCEL
    # # =======================================
    # answer_without_lcel = retrieval_chain_without_lcel(query)
    # print("\nAnswer:")
    # print(answer_without_lcel)

    # =======================================
    # Option 2: Use implementation WITH LCEL
    # =======================================
    retrieval_chain_with_lcel = create_retrieval_chain_with_lcel()
    answer_with_lcel = retrieval_chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(answer_with_lcel)
