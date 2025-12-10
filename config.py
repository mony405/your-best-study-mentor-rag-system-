# config.py
import os

class Config:
    # --- Paths ---
    DATA_PATH = "data"
    CHROMA_PATH = "chroma_db"
    
    # --- Model Settings ---
    # You can swap these out easily in the future
    MODEL_NAME = "gpt-4o-mini"
    EMBEDDING_MODEL_NAME = "text-embedding-3-small"
    TEMPERATURE = 0.3
    
    # --- RAG Settings ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    RETRIEVER_K = 5  # Number of docs to retrieve for QA
    SUMMARY_K = 7    # Number of docs to retrieve for Summary (needs more context)