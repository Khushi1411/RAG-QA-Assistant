from tools import calculator_tool, define_tool
from vector_index import read_and_chunk_files, create_vector_index, search_query
from llm_utils import generate_answer_from_chunks

# Global cache for storing retrieved chunks (for Streamlit display)
retrieved_chunks_cache = []

def agent_router(query):
    global retrieved_chunks_cache
    print(f"\n[Agent] Received query: {query}")

    # Route to calculator if query contains 'calculate'
    if "calculate" in query.lower():
        print("[Agent] Detected keyword: 'calculate'")
        print("[Agent] Routing to calculator tool...")
        retrieved_chunks_cache = []  # Clear cache since RAG isn't used
        return calculator_tool(query), "Calculator Tool"

    # Route to dictionary if query contains 'define'
    elif "define" in query.lower():
        print("[Agent] Detected keyword: 'define'")
        print("[Agent] Routing to dictionary tool...")
        retrieved_chunks_cache = []  # Clear cache since RAG isn't used
        return define_tool(query), "Dictionary Tool"

    # Default: use RAG pipeline
    else:
        print("[Agent] No special keyword found. Using RAG pipeline...")

        # Step 1: Read and chunk documents
        chunks = read_and_chunk_files("docs")

        # Step 2: Create vector index
        index, _ = create_vector_index(chunks)

        # Step 3: Retrieve relevant chunks
        retrieved_chunks = search_query(query, index, chunks, top_k=3)
        retrieved_chunks_cache = [chunk['chunk'] for chunk in retrieved_chunks]  # Store for Streamlit

        print("\n[Agent] Retrieved Chunks:")
        for i, result in enumerate(retrieved_chunks):
            print(f"Chunk {i+1}: {result}")

        # Step 4: Generate answer
        answer = generate_answer_from_chunks(retrieved_chunks_cache, query)
        return answer, "RAG Pipeline"

# Function to return retrieved chunks for Streamlit UI
def get_retrieved_chunks():
    return retrieved_chunks_cache

# CLI interface if run directly
if __name__ == "__main__":
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            print("[Agent] Exiting...")
            break
        answer, tool = agent_router(query)
        print(f"[Agent] Tool Used: {tool}")
        print(f"[Agent] Final Answer: {answer}")




