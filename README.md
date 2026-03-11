# Web RAG Assistant

A Retrieval-Augmented Generation (RAG) system that extracts information from user-provided URLs and answers questions using vector search and Large Language Models.

This project demonstrates how to build a modular RAG pipeline with document ingestion, embeddings, vector databases, and LLM-powered question answering.

---

## Features

- Extracts content dynamically from user-provided URLs
- Splits documents into semantic chunks
- Generates embeddings using Sentence Transformers
- Stores vectors in a FAISS vector database
- Retrieves relevant context using similarity search
- Generates answers using Google's Gemini LLM
- Interactive Streamlit UI for querying the knowledge base
- Modular architecture for easy extension

---

## Architecture

User Input (URLs)  
↓  
Document Loader  
↓  
Text Chunking  
↓  
Embedding Generation  
↓  
FAISS Vector Database  
↓  
Retriever (Similarity Search)  
↓  
Gemini LLM  
↓  
Generated Answer

---

## Project Structure
