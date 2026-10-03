import logging

from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING)

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    res = app.invoke(input={"question": "what is jev ai?"})
    print(res["generation"])
