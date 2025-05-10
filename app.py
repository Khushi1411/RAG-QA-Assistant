import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for torch+Streamlit error

import streamlit as st
from agent import agent_router, get_retrieved_chunks

st.set_page_config(page_title="RAG Assistant", layout="centered")

st.title("RAG-Powered Multi-Agent Q&A Assistant")
st.markdown("Ask a question about the product or policy.")

# User input
query = st.text_input("Enter your question:")

if query:
    # Run the agent router
    with st.spinner("Processing..."):
        answer, decision = agent_router(query)
        retrieved_chunks = get_retrieved_chunks()

    # Show which tool/agent was used
    st.markdown("### Agent Decision")
    st.markdown(f"**Tool Used:** {decision}")

    # Show retrieved chunks (if RAG was used)
    st.markdown("### Retrieved Context")
    if retrieved_chunks:
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1}:** {chunk}")
    else:
        st.markdown("_No relevant chunks found._")

    # Show final answer
    st.markdown("### Final Answer")
    st.success(answer)


