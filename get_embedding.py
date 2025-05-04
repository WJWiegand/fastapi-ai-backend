import os
from langchain_groq import ChatGroq

def get_embedding():
    """
    Returns an embedding function using Groq's ChatGroq.
    """
    return ChatGroq(
        model_name="nomic-embed-text",  # Replace with the correct Groq embedding model if needed
        api_key=os.getenv("GROQ_API_KEY")
    )