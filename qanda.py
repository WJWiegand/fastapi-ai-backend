import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from get_embedding import get_embedding

ChromaPath = "chroma_db"
db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
retriever = db.as_retriever()
# Setup LLM
llm = OllamaLLM(model="mistral")

# Setup chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

PROMPT_TEMPLATE = """
You are an expert in absolutely every subject - Often the best teacher to exist. Answer the question based only on the following context:

{context}

If the context does not contain enough information to answer the question, respond with "The context does not provide enough information."

---

Question: {question}
Answer:
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str) -> dict:
    """
    Query the Chroma database using a RetrievalQA chain.
    """
    response = qa_chain.invoke({"query": query_text})
    answer = response.get("result", "No answer found.")
    sources = response.get("source_documents", [])
    source_list = [doc.metadata.get("id", "Unknown source") for doc in sources]

    return {
        "answer": answer,
        "sources": source_list
    }

if __name__ == "__main__":
    main()