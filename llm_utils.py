import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load text-generation model
llm = pipeline("text-generation", model="gpt2")

def read_and_chunk_files(folder_path="docs"):
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                sentences = text.split('\n')
                for i in range(0, len(sentences), 2):
                    chunk = " ".join(sentences[i:i+2])
                    chunks.append(chunk)
    return chunks

def create_vector_index(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def search_query(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        results.append({'chunk': chunks[idx], 'distance': dist})  # Ensure this is a dictionary
    return results

from transformers import pipeline

# Function to generate an answer based on relevant chunks
def generate_answer_from_chunks(retrieved_chunks, query):
    # Extract relevant context chunks based on the question
    relevant_chunks = []

    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            context_text = chunk['chunk']
            # Exclude chunks that don't seem to directly answer the query (e.g., reviews)
            if 'Q:' in context_text and 'A:' in context_text:
                relevant_chunks.append(context_text)
            elif "return" in query.lower() and "return" in context_text.lower():
                relevant_chunks.append(context_text)  # Keep only return-related chunks
            elif "warranty" in query.lower() and "warranty" in context_text.lower():
                relevant_chunks.append(context_text)  # Keep warranty-related chunks
        else:
            relevant_chunks.append(chunk)

    # Combine only relevant chunks for the answer
    context = "\n".join(relevant_chunks)

    # If no context found, return a generic response
    if not context:
        return "Sorry, I couldn't find a clear answer. Please try again later."

    # Adjusting the prompt to ensure the answer focuses on the query
    prompt = f"""
You are a helpful assistant. Based on the provided context, answer the user's question concisely and clearly. Do not repeat the question. If the context does not contain sufficient information, reply with 'Sorry, I couldn't find a clear answer. Please try again later.'

Context:
{context}

Question: {query}
Answer: """

    # Call the model with the updated max_new_tokens
    response = llm(
        prompt,
        max_new_tokens=50,  # Limit output length
        do_sample=False,  # Use deterministic output
        temperature=0.7,
        top_p=0.85,
        top_k=50,
        pad_token_id=50256
    )[0]['generated_text']

    # Clean the response by extracting the answer part
    answer = response.split("Answer:")[-1].strip()

    # Check for empty or irrelevant responses
    if not answer or "Sorry" in answer or answer.lower().startswith("i don't know"):
        return "Sorry, I couldn't find a clear answer. Please try again later."

    return answer
