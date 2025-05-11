from tools import calculator_tool, define_tool
from vector_index import read_and_chunk_files, create_vector_index, search_query
from llm_utils import generate_answer_from_chunks

# Store retrieved chunks for display
retrieved_chunks_cache = []

def agent_router(query):
    global retrieved_chunks_cache
    print(f"\n[Agent] Received query: {query}")

    if "calculate" in query.lower():
        print("[Agent] Detected keyword: 'calculate'")
        print("[Agent] Routing to calculator tool...")
        retrieved_chunks_cache = []  
        return calculator_tool(query), "Calculator Tool"

    elif "define" in query.lower():
        print("[Agent] Detected keyword: 'define'")
        print("[Agent] Routing to dictionary tool...")
        retrieved_chunks_cache = []  
        return define_tool(query), "Dictionary Tool"
    
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

# Function to return retrieved chunks for Streamlit 
def get_retrieved_chunks():
    return retrieved_chunks_cache

if __name__ == "__main__":
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            print("[Agent] Exiting...")
            break
        answer, tool = agent_router(query)
        print(f"[Agent] Tool Used: {tool}")
        print(f"[Agent] Final Answer: {answer}")




