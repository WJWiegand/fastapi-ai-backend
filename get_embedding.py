from langchain_ollama import OllamaEmbeddings

def get_embedding():
    """
    Returns an embedding function using OllamaEmbeddings.
    """
    return OllamaEmbeddings(model="nomic-embed-text")