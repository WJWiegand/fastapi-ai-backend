from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os

from prep import extract_main_topic, extract_keywords_with_descriptions, youtube_suggestions, save_extracted_info
from testknowledge import generate_questions_from_context, save_questions_to_file,fetch_and_generate_questions
from flashcard import generate_flashcards , save_flashcards
from get_embedding import get_embedding
from load_chunk import clear_local_files, run_chunking , DataPath, clear_database
from qanda import query_rag

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from get_embedding import get_embedding
from langchain_chroma import Chroma
from langchain_core.documents import Document


ChromaPath = "chroma_db"
llm = OllamaLLM(model="llama3")  # Or the model you prefer
db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
retriever = db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://your-angular-app.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

@app.on_event("startup")
def clear_db_on_start():
    if os.getenv("RESET_ON_STARTUP", "false").lower() == "true":
        try:
            os.makedirs(DataPath, exist_ok=True)
            print(f"📂 Ensured DataPath exists: {DataPath}")

            print("🗑️ Clearing local files...")
            clear_local_files()

            print("🧹 Clearing Chroma database...")
            clear_database(allow_reset=True)

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

        db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
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
    """
    Endpoint to generate multiple-choice and open-ended questions based on the existing chunks in the Chroma database.
    """
    try:
        # Check if the Chroma database exists
        if not os.path.exists(ChromaPath):
            raise HTTPException(status_code=400, detail="Chroma database does not exist.")

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

        # Combine all chunks into a single text
        print("📚 Combining all chunks into a single text...")
        combined_text = " ".join(chunk.page_content for chunk in chunks)

        # Extract the main topic of the document
        print("✨ Extracting main topic...")
        main_topic = extract_main_topic(combined_text)
        print(f"Main Topic: {main_topic}")

        # Fetch and generate questions
        print("📝 Generating questions...")
        questions = fetch_and_generate_questions()

        if not questions:
            print("⚠️ No questions were generated.")
            return {
                "message": "No questions were generated.",
                "chunks_count": len(chunks),
                "questions": []
            }

        # Save the generated questions
        print("💾 Saving your questions...")
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
    """
    Endpoint to generate flashcards based on the chunks within the Chroma database.
    """
    try:
        # Check if the Chroma database exists
        if not os.path.exists(ChromaPath):
            raise HTTPException(status_code=400, detail="Chroma database does not exist.")

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

        # Combine all chunks into a single text
        print("📚 Combining all chunks into a single text...")
        combined_text = " ".join(chunk.page_content for chunk in chunks)

        # Generate flashcards
        print("✨ Generating flashcards...")
        flashcards = generate_flashcards(combined_text)

        if flashcards:
            print("Generated Flashcards:")
            for i, flashcard in enumerate(flashcards, 1):
                print(f"\nFlashcard {i}:")
                print(f"Q: {flashcard['question']}")
                print(f"A: {flashcard['answer']}")

            # Save flashcards to a file
            save_flashcards(flashcards)
        else:
            print("No flashcards generated.")

        return {
            "message": "Flashcards generated successfully.",
            "chunks_count": len(chunks),
            "flashcards": flashcards
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file and process it using the run_chunking function.
    """
    try:
     

        # Ensure the data directory exists
        os.makedirs(DataPath, exist_ok=True)
        clear_local_files()  # Clear old files if needed
        # Save the uploaded file to the data directory
        file_path = Path(DataPath) / file.filename
        print(f"📂 Saving file to: {file_path}")
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # 🌟 Double-check: wait for file to be fully saved
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise ValueError(f"File {file.filename} was not saved correctly.")

        print(f"📂 File saved successfully: {file.filename}")

        print("📥 Clearing old data and preparing for upload...")
        clear_database(allow_reset=True)
  # Ensure this works without restrictions



        # Run the chunking process
        print(f"📂 Processing file: {file.filename}")
        new_chunks = run_chunking(reset=True)  # Pass reset=True if needed

        print(f"✅ File processed successfully. New chunks added: {len(new_chunks)}")
        return {
            "message": f"File '{file.filename}' processed successfully.",
            "new_chunks_added": len(new_chunks),
        }

    except Exception as e:
        print(f"❌ Error during upload: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

@app.post("/qanda")
async def ask_question(user_question: str = Query(...)):
    try:
        print(f"🔍 User question: {user_question}")
        result = query_rag(user_question)
        print(f"🔍 Retrieved answer: {result['answer']}")
        print(f"🔍 Retrieved sources: {result['sources']}")
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as e:
        print(f"❌ Error during QandA: {e}")
        return {"detail": f"An error occurred: {str(e)}"}