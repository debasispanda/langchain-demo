from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import os
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    print("Loading....")
    loader = UnstructuredLoader(file_path="/Users/debasispanda/Workspace/Learnings/AI/langchain-demo/4_rag/vector-db.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("Splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(f"Number of chunks created: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.environ["PINECONE_INDEX"]
    )
    print("finish")


