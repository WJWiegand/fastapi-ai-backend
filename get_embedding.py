import os
from langchain_groq import GroqEmbeddings

def get_embedding():
    """
    Returns an embedding function using Groq's OpenAI-compatible embedding API.
    """
    return GroqEmbeddings(
        model="nomic-embed-text",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.getenv("GROQ_API_KEY")
    )
