import argparse
import random
import os
import json
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from get_embedding import get_embedding

# Initialize Chroma database and retriever
ChromaPath = "chroma_db"
db = Chroma(persist_directory=ChromaPath, embedding_function=get_embedding())
retriever = db.as_retriever()

# Initialize LLM
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # or "mistral-7b-8k"
    api_key=os.getenv("GROQ_API_KEY")
)
def generate_questions_from_context(context: str, main_topic: str):
    """
    Generate a question (multiple-choice or open-ended) based on the provided context and main topic.
    """
    question_type = random.choice(["multiple_choice", "open_ended"])

    if question_type == "multiple_choice":
        prompt = f"""
        You are an expert assistant. Generate a question based only on the following context:

        {context}

        The question must be directly related to the topic "{main_topic}".
        Return the question in JSON format like this:
        {{
            "question": "Your question here",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],  # For multiple-choice
            "correct_answer": "Option 1"  # For multiple-choice
        }}
        """
    else:
        prompt = f"""
        You are an expert assistant. Generate an open-ended question based only on the following context:

        {context}

        The question must be directly related to the topic "{main_topic}".
        Return the question in JSON format like this:
        {{
            "question": "Your question here",
            "correct_answer": "Your answer here"  # For open-ended
        }}
        """

    print(f"Generated Prompt:\n{prompt}")  # Debugging log

    try:
        response = llm(prompt)
        print(f"Raw LLM Response: {response}")
        response_data = json.loads(response)
        if question_type == "multiple_choice":
            return {
                "type": "multiple_choice",
                "question": response_data.get("question", "No question generated."),
                "options": response_data.get("options", ["Option 1", "Option 2", "Option 3", "Option 4"]),
                "correct_answer": response_data.get("correct_answer", "Option 1")
            }
        else:
            return {
                "type": "open_ended",
                "question": response_data.get("question", "No question generated."),
                "correct_answer": response_data.get("correct_answer", "No answer provided.")
            }
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON.")
        return None
    

    
def filter_irrelevant_questions(questions):
    """
    Filter out questions that are not relevant to the topic.
    """
    relevant_questions = []
    for question in questions:
        prompt = f"""
        Question:
        {question['question']}

        Respond with "yes" if it is relevant, otherwise respond with "no".
        """
        response = llm(prompt).strip().lower()
        if response == "yes":
            relevant_questions.append(question)
    return relevant_questions

def fetch_and_generate_questions():
    """
    Fetch data from the Chroma database and generate questions.
    """
    try:
        documents = retriever.get_relevant_documents("")
        if not documents:
            print("No documents found in the Chroma database.")
            return []

        print(f"Found {len(documents)} documents in the Chroma database.")
        all_questions = []
        for doc in documents:
            context = doc.page_content
            main_topic = "Main Topic"  # Replace with actual logic to extract the main topic
            print(f"Generating questions for document: {doc.metadata.get('id', 'Unknown ID')}")
            questions = [generate_questions_from_context(context, main_topic) for _ in range(5)]
            questions = [q for q in questions if q is not None]  # Filter out None values
            all_questions.extend(questions)

        print(f"Generated {len(all_questions)} questions in total.")
        return all_questions
    except Exception as e:
        print(f"Error fetching or generating questions: {e}")
        return []
    

def save_questions_to_file(questions, filename="questions.json"):
    """
    Save the generated questions to a JSON file.
    """
    print(f"Questions to save: {questions}")  # Debugging log
    file_path = os.path.join(os.path.dirname(__file__), filename)
    with open(file_path, "w") as file:
        json.dump(questions, file, indent=4)
    print(f"Questions saved to {file_path}")

def main():
    """
    Main function to generate, filter, and save questions.
    """
    print("Fetching data, generating questions, and filtering for relevance...")
    questions = fetch_and_generate_questions()
    if not questions:
        print("No questions were generated. File not saved.")
        return
    save_questions_to_file(questions)

if __name__ == "__main__":
    main()