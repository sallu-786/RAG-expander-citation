
from embeddings import get_file, get_text_chunks, create_embeddings
import streamlit as st

# when file is uploaded by user, create new vector data for that file
def create_new_vector_db(file):
    with st.spinner("Creating vector data"):
        text = get_file(file)
        text_chunks = get_text_chunks(text)
        vectordb = create_embeddings(text_chunks)
    return vectordb,text_chunks

def handle_file_upload(file):
    if file:
        vectordb,text_chunks = create_new_vector_db(file)
        st.write("Vector data created successfully.")
        return vectordb,text_chunks

    else:                             
        pass

# Rerank the results based on weighted result of bm25_search and vector similarity
def normalize_scores(results):  #make sure the vector and bm25_search score are between range 0 to 1
        scores = [score for doc, score in results]
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return [1] * len(scores)  # Avoid division by zero if all scores are the same
        return [(score - min_score) / (max_score - min_score) for score in scores] 

def add_scores(results, weight, normalized_scores, merged_scores):
    for rank, ((doc, _), normalized_score) in enumerate(zip(results, normalized_scores)):
        if isinstance(doc, dict):
            doc_id = (doc['metadata']['page'], doc['content'])
        else:
            doc_id = (doc.metadata['page'], doc.page_content)

        if doc_id in merged_scores:
            merged_scores[doc_id] += weight * normalized_score / (rank + 1)
        else:
            merged_scores[doc_id] = weight * normalized_score / (rank + 1)

def rrf(bm25_results, vector_results, k=3):
    merged_scores = {}

    bm25_normalized_scores = normalize_scores(bm25_results)
    vector_normalized_scores = normalize_scores(vector_results)

    add_scores(bm25_results, weight=0.25, normalized_scores=bm25_normalized_scores, merged_scores=merged_scores)
    add_scores(vector_results, weight=0.75, normalized_scores=vector_normalized_scores, merged_scores=merged_scores)

    sorted_docs = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
    final_results = [{"content": content, "metadata": {"page_number": page_number}} for (page_number, content), score in sorted_docs[:k]]
    return final_results



