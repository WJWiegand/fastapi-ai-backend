import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embedding import get_embedding
from langchain.prompts import ChatPromptTemplate

ChromaPath = "chroma_db"
db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
retriever = db.as_retriever()

def generate_flashcards(context: str) -> list:
    """
    Generate flashcards in a question-and-answer format based on the given context.
    """
    llm = OllamaLLM(model="mistral")
    prompt = ChatPromptTemplate.from_template("""
    Given the following context, generate flashcards in a question-and-answer format.
    Each flashcard should focus on a key concept or topic from the context.

    CONTEXT:
    {context}

    For each flashcard, generate:
    - A question that tests understanding of the key concept.
    - A concise and accurate answer to the question.

    Return the flashcards as a JSON array, where each flashcard is an object with "question" and "answer" fields.
    """)

    formatted_prompt = prompt.format(context=context)
    response = llm.invoke(formatted_prompt)

    try:
        flashcards = json.loads(response)
    except json.JSONDecodeError:
        print("Error: Failed to parse flashcards. Please check the LLM response.")
        flashcards = []

    return flashcards


def save_flashcards(flashcards: list, output_path="flashcards_output.json"):
    """
    Save the generated flashcards to a JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(flashcards, f, indent=4)
    print(f"Flashcards saved to {output_path}")


def main():
    print("🔍 Loading documents from Chroma...")
    db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
    raw_chunks = db.get()["documents"]
    ids = db.get()["ids"]
    metadatas = db.get()["metadatas"]

    # Reconstruct Document objects with metadata
    chunks = [
        Document(page_content=text, metadata={"id": ids[i], **metadatas[i]})
        for i, text in enumerate(raw_chunks)
    ]



if __name__ == "__main__":
    main()