import argparse
import os
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# Paths and configurations
ChromaPath = "chroma_db"

# Initialize Groq Embeddings
def get_embedding():
    """
    Returns an embedding function using Groq's ChatGroqEmbeddings.
    """
    return OpenAIEmbeddings(
        model="nomic-embed-text",  # Replace with the correct Groq embedding model if needed
        api_key=os.getenv("GROQ_API_KEY")
    )

# Initialize Chroma with Groq embeddings
db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
retriever = db.as_retriever()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # Replace with the correct Groq model if needed
    api_key=os.getenv("GROQ_API_KEY")
)

# Setup RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Prompt template for the QA system
PROMPT_TEMPLATE = """
You are an expert in absolutely every subject - Often the best teacher to exist. Answer the question based only on the following context:

{context}

If the context does not contain enough information to answer the question, respond with "The context does not provide enough information."

---

Question: {question}
Answer:
"""

def query_rag(query_text: str) -> dict:
    """
    Query the Chroma database using a RetrievalQA chain.
    """
    # Format the prompt with the query
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    qa_chain.prompt = prompt

    # Invoke the QA chain
    response = qa_chain.invoke({"query": query_text})
    answer = response.get("result", "No answer found.")
    sources = response.get("source_documents", [])
    source_list = [doc.metadata.get("id", "Unknown source") for doc in sources]

    return {
        "answer": answer,
        "sources": source_list
    }

def main():
    """
    Command-line interface for querying the QA system.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Query the QA system
    result = query_rag(query_text)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])

if __name__ == "__main__":
    main()