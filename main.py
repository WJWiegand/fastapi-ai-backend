from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os

from prep import extract_main_topic, extract_keywords_with_descriptions, youtube_suggestions, save_extracted_info
from testknowledge import generate_questions_from_context, save_questions_to_file,fetch_and_generate_questions
from flashcard import generate_flashcards , save_flashcards
from get_embedding import GroqEmbeddings
from load_chunk import clear_local_files, run_chunking , DataPath, clear_database
from qanda import query_rag

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from get_embedding import get_embedding
from langchain_chroma import Chroma
from langchain_core.documents import Document

ChromaPath = "chroma_db"
DataPath = "data"

# Initialize Groq LLM and Embeddings
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # Replace with the correct Groq model if needed
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Chroma with Groq embeddings
db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
retriever = db.as_retriever()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://your-angular-app.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def clear_db_on_start():
    if os.getenv("RESET_ON_STARTUP", "false").lower() == "true":
        try:
            os.makedirs(DataPath, exist_ok=True)
            print(f"📂 Ensured DataPath exists: {DataPath}")

            print("🗑️ Clearing local files...")
            clear_local_files()

            print("🧹 Clearing Chroma database...")
            clear_database()

            print("✅ Startup cleanup completed successfully.")
        except Exception as e:
            print(f"❌ Error during startup cleanup: {e}")
    else:
        print("🚀 Skipping database and file cleanup on startup.")

@app.post("/prep")
async def prep_existing_data():
    try:
        if not os.path.exists(ChromaPath):
            raise HTTPException(status_code=400, detail="Chroma database does not exist.")

        raw_chunks = db.get()["documents"]
        ids = db.get()["ids"]
        metadatas = db.get()["metadatas"]

        chunks = [
            Document(page_content=text, metadata={"id": ids[i], **metadatas[i]})
            for i, text in enumerate(raw_chunks)
        ]

        combined_text = " ".join(chunk.page_content for chunk in chunks)
        main_topic = extract_main_topic(combined_text)

        relevant_chunks = [
            chunk for chunk in chunks
            if main_topic.lower() in chunk.page_content.lower()
        ]

        relevant_text = " ".join(chunk.page_content for chunk in relevant_chunks)
        keywords_with_descriptions = extract_keywords_with_descriptions(
            relevant_text, main_topic, "Description of the key concepts or topics that are essential to understanding the material"
        )

        youtube_queries = youtube_suggestions({"combined_document": list(keywords_with_descriptions.keys())})

        save_extracted_info(
            keywords_by_doc={"combined_document": keywords_with_descriptions},
            youtube_queries=youtube_queries
        )

        return {
            "message": "Prep completed successfully.",
            "main_topic": main_topic,
            "keywords_with_descriptions": keywords_with_descriptions,
            "youtube_suggestions": youtube_queries
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/test")
async def test_knowledge():
    try:
        if not os.path.exists(ChromaPath):
            raise HTTPException(status_code=400, detail="Chroma database does not exist.")

        raw_chunks = db.get()["documents"]
        ids = db.get()["ids"]
        metadatas = db.get()["metadatas"]

        chunks = [
            Document(page_content=text, metadata={"id": ids[i], **metadatas[i]})
            for i, text in enumerate(raw_chunks)
        ]

        combined_text = " ".join(chunk.page_content for chunk in chunks)
        main_topic = extract_main_topic(combined_text)

        questions = fetch_and_generate_questions()

        if not questions:
            return {
                "message": "No questions were generated.",
                "chunks_count": len(chunks),
                "questions": []
            }

        save_questions_to_file(questions)

        return {
            "message": "Test knowledge generated successfully.",
            "chunks_count": len(chunks),
            "questions": questions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/flashcards")
async def generate_flashcards_endpoint():
    try:
        if not os.path.exists(ChromaPath):
            raise HTTPException(status_code=400, detail="Chroma database does not exist.")

        raw_chunks = db.get()["documents"]
        ids = db.get()["ids"]
        metadatas = db.get()["metadatas"]

        chunks = [
            Document(page_content=text, metadata={"id": ids[i], **metadatas[i]})
            for i, text in enumerate(raw_chunks)
        ]

        combined_text = " ".join(chunk.page_content for chunk in chunks)
        flashcards = generate_flashcards(combined_text)

        if flashcards:
            save_flashcards(flashcards)

        return {
            "message": "Flashcards generated successfully.",
            "chunks_count": len(chunks),
            "flashcards": flashcards
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs(DataPath, exist_ok=True)
        clear_local_files()

        file_path = Path(DataPath) / file.filename
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        if not file_path.exists() or file_path.stat().st_size == 0:
            raise ValueError(f"File {file.filename} was not saved correctly.")

        clear_database()

        new_chunks = run_chunking(reset=True)

        return {
            "message": f"File '{file.filename}' processed successfully.",
            "new_chunks_added": len(new_chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/qanda")
async def ask_question(user_question: str = Query(...)):
    try:
        result = query_rag(user_question)
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as e:
        return {"detail": f"An error occurred: {str(e)}"}