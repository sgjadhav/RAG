
import streamlit as st
from document_loader import load_documents
from vector_store import create_vector_store
from rag_pipeline import build_rag_chain

st.title("Finance RAG Assistant")

# User input for URLs
urls_input = st.text_area(
    "Enter URLs (one per line)",
    height=150
)

# Process URLs
if st.button("Process URLs"):

    urls = [url.strip() for url in urls_input.split("\n") if url.strip()]

    if len(urls) == 0:
        st.warning("Please enter at least one URL")

    else:

        with st.spinner("Loading documents..."):
            docs = load_documents(urls)

        with st.spinner("Creating vector database..."):
            vectorstore = create_vector_store(docs)

        with st.spinner("Building RAG pipeline..."):
            rag_chain = build_rag_chain(vectorstore)

        st.session_state.rag_chain = rag_chain

        st.success("RAG system ready! Ask questions below.")

# Question input
question = st.text_input("Ask a question about the URLs")

if question:

    if "rag_chain" not in st.session_state:
        st.warning("Please process URLs first")

    else:

        with st.spinner("Generating answer..."):
            answer = st.session_state.rag_chain.invoke(question)

        st.write("### Answer")
        st.write(answer)
