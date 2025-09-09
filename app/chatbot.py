import faiss
import numpy as np
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import settings
from app.utils.tools import configure_tools

genai.configure(api_key=settings.GEMINI_API_KEY)

# Initialize Gemini LLM model and embedding model
model = genai.GenerativeModel("gemini-2.5-flash")
embedding_model = "models/embedding-001"

# FAISS index for document chunks
dimension = 768
index = faiss.IndexFlatL2(dimension)
doc_chunks = []
sessions = {}

# Example movie dataset
movies = pd.DataFrame({
    "title": [
        "Inception", "Interstellar", "The Dark Knight", "Shutter Island",
        "The Matrix", "Memento", "Fight Club", "The Prestige", "Tenet", "Avatar"
    ],
    "genres": [
        "sci-fi thriller", "sci-fi space drama", "action superhero", "psychological thriller",
        "sci-fi action", "psychological thriller", "drama thriller", "mystery drama", "sci-fi thriller", "sci-fi adventure"
    ]
})

# Build TF-IDF matrix
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])


tool_descriptions = {
    "document_search": "Find detailed answers based on document content uploaded by the user for compliance or legal inquiries.",
    "movie_recommendation": "Recommend top movies based on genre similarity and user preferences.",
    "general_qa": "Answer general knowledge or conversational questions from the user."
}

vectorizer = TfidfVectorizer(stop_words="english")
desc_matrix = vectorizer.fit_transform(tool_descriptions.values())


def select_tool_based_on_semantics(user_query: str) -> str:
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, desc_matrix).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    threshold = 0.4  # Tunable threshold

    if best_score < threshold:
        return "general_qa"
    else:
        return list(tool_descriptions.keys())[best_idx]


def embed_text(text: str) -> np.ndarray:
    result = genai.embed_content(model=embedding_model, content=text)
    return np.array(result["embedding"], dtype="float32")


def add_document_to_index(doc_text: str, chunk_size: int = 500):
    global doc_chunks
    words = doc_text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    embeddings = [embed_text(chunk) for chunk in chunks]
    index.add(np.vstack(embeddings))
    doc_chunks.extend(chunks)


def retrieve_relevant_chunks(query: str, top_k: int = 3):
    q_emb = embed_text(query).reshape(1, -1)
    distances, indices = index.search(q_emb, top_k)
    return [doc_chunks[i] for i in indices[0] if i < len(doc_chunks)]


def start_session(session_id: str, doc_text: str):
    sessions[session_id] = {"doc_text": doc_text, "history": []}
    add_document_to_index(doc_text)


def chat_with_doc(session_id: str, user_message: str) -> str:
    session = sessions.get(session_id)
    if not session:
        raise Exception("Session not found")

    retrieved_chunks = retrieve_relevant_chunks(user_message, top_k=3)

    history_str = "\n".join(h["role"] + ": " + h["content"] for h in session["history"])

    prompt = (
        "You are a compliance assistant. Answer based on relevant document chunks and conversation history.\n\n"
        f"Relevant document chunks:\n{chr(10).join(retrieved_chunks)}\n\n"
        f"Conversation so far:\n{history_str}\n\n"
        f"User: {user_message}"
    )

    response = model.generate_content(prompt)
    answer = response.text

    session["history"].append({"role": "user", "content": user_message})
    session["history"].append({"role": "assistant", "content": answer})

    return answer


def chat_with_ai(user_message: str) -> str:
    retrieved_chunks = []
    if doc_chunks:
        retrieved_chunks = retrieve_relevant_chunks(user_message, top_k=2)

    prompt = (
        "You are a helpful AI assistant. Use knowledge base + relevant docs if available.\n\n"
        f"Relevant document chunks:\n{chr(10).join(retrieved_chunks)}\n\n"
        f"User question: {user_message}"
    )

    response = model.generate_content(prompt)
    return response.text


# Configure tools
from app.utils import tools
configure_tools(chat_with_doc, model, tfidf, tfidf_matrix, movies)


def agent_call(tool_name: str, query: str, session_id: str = None) -> str:
    if tool_name == "document_search":
        return tools.document_search_tool(query, session_id)
    elif tool_name == "movie_recommendation":
        return tools.movie_recommendation_tool(query)
    elif tool_name == "general_qa":
        return tools.general_qa_tool(query)
    else:
        return "Tool not found"
