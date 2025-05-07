import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq, OpenAIEmbeddings
from langchain.schema.document import Document
from collections import defaultdict

# Paths and configurations
ChromaPath = "chroma_db"
DataPath = "data"
ALLOW_RESET = True

# Initialize Groq LLM and Embeddings
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # Replace with the correct Groq model if needed
    api_key=os.getenv("GROQ_API_KEY")
)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the Chroma database")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    run_chunking(reset=args.reset)

def run_chunking(reset=False):
    """
    Process the uploaded file and chunk it into smaller pieces.
    """
    try:
        if reset:
            print("✨ Clearing Database")
            clear_database(allow_reset=ALLOW_RESET)

        documents = load_documents()
        chunks = split_documents(documents)
        return add_to_chroma(chunks)
    except Exception as e:
        print(f"❌ Error during chunking: {e}")
        raise ValueError(f"Error during chunking: {e}")

def load_documents():
    documents = []
    abs_path = os.path.abspath(DataPath)
    print("🛠️ DataPath resolved to:", abs_path)

    if not os.path.isdir(abs_path):
        raise ValueError(f"❌ DataPath {abs_path} is not a valid directory.")

    for filename in os.listdir(abs_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(abs_path, filename)
            print(f"📄 Loading file: {full_path}")
            loader = PyPDFLoader(full_path)
            docs = loader.load()
            documents.extend(docs)

    print(f"✅ Loaded {len(documents)} documents.")
    return documents

def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    """
    Add chunks to the Chroma database.
    """
    db = Chroma(
        persist_directory=ChromaPath,
        embedding_function=get_embedding()  # Ensure embedding function is passed
    )
    ChunkWithID = calculate_chunk_ids(chunks)

    existing_Items = db.get(include=[])
    existing_Ids = set(existing_Items["ids"])
    print(f"🔍 Existing IDs in DB: {len(existing_Ids)}")

    new_chunks = []
    for chunk in ChunkWithID:
        if chunk.metadata["id"] not in existing_Ids:
            chunk.metadata["doc_name"] = os.path.basename(chunk.metadata["source"])
            new_chunks.append(chunk)

    if new_chunks:
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
        print(f"✅ Added {len(new_chunks)} new chunks to Chroma.")

    # No need to call db.persist() as changes are automatically saved
    return new_chunks

def calculate_chunk_ids(chunks):
    """
    Assign unique IDs to each chunk based on the document source and page number.
    """
    id_counts = defaultdict(int)
    chunks_with_ids = []

    for chunk in chunks:
        source = os.path.basename(chunk.metadata.get("source", "unknown"))
        base_id = os.path.splitext(source)[0]
        id_counts[base_id] += 1
        chunk.metadata["id"] = f"{base_id}-{id_counts[base_id]}"
        chunks_with_ids.append(chunk)

    return chunks_with_ids

def clear_database(allow_reset=True):
    if not allow_reset:
        raise ValueError("Reset is disabled by config")

    try:
        db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
        db.delete(where={"id": {"$ne": True}})  # 🧼 Clear all stored documents safely
        print("🧹 Cleared all documents from Chroma database.")
    except Exception as e:
        raise ValueError(f"Failed to clear database: {e}")

def clear_local_files():
    """
    Clear local files in the DataPath directory.
    """
    try:
        for filename in os.listdir(DataPath):
            file_path = os.path.join(DataPath, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"🗑️ Removed file: {file_path}")
            else:
                print(f"❌ {file_path} is not a file.")
    except Exception as e:
        raise ValueError(f"Failed to clear local files: {e}")
    
if __name__ == "__main__":
    main()