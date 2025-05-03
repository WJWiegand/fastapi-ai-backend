import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embedding import get_embedding
from googleapiclient.discovery import build

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


# Need to extract the main topic of the document so there is an accurate description later on
def extract_main_topic(text: str) -> str:
    llm = Ollama(model="mistral")
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
    main_topic = chain.run({"text": text})
    return main_topic.strip()


# Need to extract keywords from the document
def extract_keywords(text: str, main_topic: str, Description: str) -> list:
    llm = Ollama(model="mistral")
    prompt = PromptTemplate(
        template="""
        Given the following text and the main topic, extract the most important key concepts or topics that are directly relevant to the main topic.

        TEXT:
        {text}

        MAIN TOPIC:
        {main_topic}

        Description of the key concepts or topics that are essential to understanding the material:
        {Description}

        Return them as a comma-separated list, prioritizing relevance and uniqueness.
        """,
        input_variables=["text", "main_topic", "Description"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"text": text, "main_topic": main_topic, "Description": Description})
    keywords = [kw.strip() for kw in response.split(",")]
    return filter_keywords(keywords, main_topic) 

# Score the chunk relevance to the main topic se we know what the main topic is
def score_chunk_relevance(chunk: str, main_topic: str) -> float:
    llm = Ollama(model="mistral")
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
    relevance_score = chain.run({"chunk": chunk, "main_topic": main_topic})
    relevant_chunks = [
    chunk for chunk in chunks
    if score_chunk_relevance(chunk.page_content, main_topic) >= 0.7
]
    return float(relevance_score.strip())




# Need to remove all the fill in words
def filter_keywords(keywords: list, main_topic: str) -> list:
    stop_words = {"the", "and", "of", "in", "to", "for", "with", "on", "by", "an", "or"}
    filtered_keywords = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower not in stop_words and keyword_lower not in filtered_keywords:
            if main_topic.lower() in keyword_lower or keyword_lower in main_topic.lower():
                filtered_keywords.append(keyword)
    return filtered_keywords

# Need to retieve more informaiton about the keywords
def extract_keywords_with_descriptions(text: str, main_topic: str, description: str) -> dict:
    llm = Ollama(model="mistral")
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
    response = chain.run({"text": text, "main_topic": main_topic, "description": description})
    keywords_with_descriptions = json.loads(response)
    return keywords_with_descriptions


# Make Youtube suggetions based on the keywords
YOUTUBE_API_KEY = "AIzaSyCOPEDia9hYvb1pzKxGQubgoF-rkB6GWIA"

def youtube_suggestions(keywords_by_doc: dict) -> dict:
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

        # Extract video links from the response
        video_links = [
            f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            for item in response.get("items", [])
        ]
        queries[doc_id] = video_links
    return queries

# Save the extracted information to a JSON file so we can see what we have
def save_extracted_info(keywords_by_doc, youtube_queries, output_path="prep_output.json"):
    combined = {
        doc_id: {
            "keywords": keywords_by_doc[doc_id],
            "youtube_search": youtube_queries[doc_id]
        } for doc_id in keywords_by_doc
    }
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=4)


