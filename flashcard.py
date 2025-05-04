import json
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from get_embedding import get_embedding
from langchain.prompts import ChatPromptTemplate

ChromaPath = "chroma_db"
db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
retriever = db.as_retriever()

# Initialize the Groq LLM
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # or "mistral-7b-8k"
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_flashcards(context: str) -> list:
    """
    Generate flashcards in a question-and-answer format based on the given context.
    """
    prompt = ChatPromptTemplate.from_template("""
    You are an expert teacher. Based on the following context, generate flashcards in a question-and-answer format.
    Each flashcard should focus on a key concept or topic from the context.

    CONTEXT:
    {context}

    For each flashcard, generate:
    - A question that tests understanding of the key concept.
    - A concise and accurate answer to the question.

    Return the flashcards as a JSON array, where each flashcard is an object with "question" and "answer" fields.
    """)

    # Format the prompt with the provided context
    formatted_prompt = prompt.format(context=context)

    # Invoke the Groq LLM to generate flashcards
    response = llm.invoke({"query": formatted_prompt})

    try:
        # Parse the response as JSON
        flashcards = json.loads(response["result"])
    except (json.JSONDecodeError, KeyError):
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
    raw_chunks = db.get()["documents"]
    ids = db.get()["ids"]
    metadatas = db.get()["metadatas"]

    # Reconstruct Document objects with metadata
    chunks = [
        Document(page_content=text, metadata={"id": ids[i], **metadatas[i]})
        for i, text in enumerate(raw_chunks)
    ]

    # Combine all chunks into a single context
    combined_context = " ".join([chunk.page_content for chunk in chunks])

    # Generate flashcards
    print("✨ Generating flashcards...")
    flashcards = generate_flashcards(combined_context)

    # Save flashcards to a file
    save_flashcards(flashcards)


if __name__ == "__main__":
    main()