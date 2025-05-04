import os
from langchain_groq import ChatGroqEmbeddings

def get_embedding():
    """
    Returns an embedding function using Groq's ChatGroqEmbeddings.
    """
    return ChatGroqEmbeddings(
        model="nomic-embed-text",  # Replace with the correct Groq embedding model if needed
        api_key=os.getenv("GROQ_API_KEY")  # Ensure the GROQ_API_KEY environment variable is set
    )