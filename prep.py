import os
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from get_embedding import get_embedding
from googleapiclient.discovery import build

# Initialize Chroma DB
ChromaPath = "chroma_db"
db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
retriever = db.as_retriever()
raw_chunks = db.get()["documents"]
ids = db.get()["ids"]
metadatas = db.get()["metadatas"]
chunks = [
    Document(page_content=text, metadata={"id": ids[i], **metadatas[i]})
    for i, text in enumerate(raw_chunks)
]

# Load Groq LLM
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # or "mistral-7b-8k"
    api_key=os.getenv("GROQ_API_KEY")
)

# Extract the main topic of the document
def extract_main_topic(text: str) -> str:
    prompt = PromptTemplate(
        template="""
        Given the following text, identify the main topic or theme of the document in one sentence.

        TEXT:
        {text}

        MAIN TOPIC:
        """,
        input_variables=["text"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"text": text})
    return result['text'].strip()

# Extract keywords from the document
def extract_keywords(text: str, main_topic: str, description: str) -> list:
    prompt = PromptTemplate(
        template="""
        Given the following text and the main topic, extract the most important key concepts or topics that are directly relevant to the main topic.

        TEXT:
        {text}

        MAIN TOPIC:
        {main_topic}

        Description of the key concepts or topics that are essential to understanding the material:
        {description}

        Return them as a comma-separated list, prioritizing relevance and uniqueness.
        """,
        input_variables=["text", "main_topic", "description"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"text": text, "main_topic": main_topic, "description": description})['text']
    keywords = [kw.strip() for kw in response.split(",")]
    return filter_keywords(keywords, main_topic)

# Filter out filler keywords
def filter_keywords(keywords: list, main_topic: str) -> list:
    stop_words = {"the", "and", "of", "in", "to", "for", "with", "on", "by", "an", "or"}
    filtered_keywords = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower not in stop_words and keyword_lower not in filtered_keywords:
            if main_topic.lower() in keyword_lower or keyword_lower in main_topic.lower():
                filtered_keywords.append(keyword)
    return filtered_keywords

# Score relevance of a chunk to the main topic
def score_chunk_relevance(chunk: str, main_topic: str) -> float:
    prompt = PromptTemplate(
        template="""
        Given the following chunk of text and the main topic, rate the relevance of the chunk to the main topic on a scale of 0 to 1.

        CHUNK:
        {chunk}

        MAIN TOPIC:
        {main_topic}

        RELEVANCE SCORE:
        """,
        input_variables=["chunk", "main_topic"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    score = chain.invoke({"chunk": chunk, "main_topic": main_topic})['text']
    try:
        return float(score.strip())
    except ValueError:
        return 0.0

# Extract keywords and provide descriptions
def extract_keywords_with_descriptions(text: str, main_topic: str, description: str) -> dict:
    prompt = PromptTemplate(
        template="""
        Given the following text and the main topic, extract the most important key concepts or topics that are directly relevant to the main topic.

        TEXT:
        {text}

        MAIN TOPIC:
        {main_topic}

        {description}

        For each key concept or topic, provide a brief description explaining its relevance to the main topic.

        Return the results as a JSON object where each key is a concept, and the value is its description.
        """,
        input_variables=["text", "main_topic", "description"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"text": text, "main_topic": main_topic, "description": description})['text']
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}

# Get YouTube suggestions
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def youtube_suggestions(keywords_by_doc: dict) -> dict:
    if not YOUTUBE_API_KEY:
        raise ValueError("Missing YouTube API Key")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    queries = {}

    for doc_id, keywords in keywords_by_doc.items():
        search_query = " ".join(keywords)
        request = youtube.search().list(
            q=search_query,
            part="snippet",
            type="video",
            maxResults=5,
            videoDuration="any"
        )
        response = request.execute()
        video_links = [
            f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            for item in response.get("items", [])
        ]
        queries[doc_id] = video_links
    return queries

# Save final output
def save_extracted_info(keywords_by_doc, youtube_queries, output_path="prep_output.json"):
    combined = {
        doc_id: {
            "keywords": keywords_by_doc[doc_id],
            "youtube_search": youtube_queries[doc_id]
        } for doc_id in keywords_by_doc
    }
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=4)
