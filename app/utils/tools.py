from sklearn.metrics.pairwise import cosine_similarity

# Tools are initialized later to avoid circular import issues
document_search_tool = None
movie_recommendation_tool = None
general_qa_tool = None

def configure_tools(chat_with_doc_func, model, tfidf, tfidf_matrix, movies):
    global document_search_tool, movie_recommendation_tool, general_qa_tool

    document_search_tool = lambda query, session_id: chat_with_doc_func(session_id, query)

    movie_recommendation_tool = lambda query: (
        f"Recommended movies: {', '.join(movies.iloc[cosine_similarity(tfidf.transform([query]), tfidf_matrix)
            .flatten()
            .argsort()[-3:][::-1]]['title'].tolist())}"
    )

    general_qa_tool = lambda query: model.generate_content(f"You are a helpful AI assistant. Answer: {query}").text
