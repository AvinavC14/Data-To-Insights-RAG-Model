import os
import shutil
import gc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def get_embeddings():
    """Initialize FREE HuggingFace embeddings (no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def init_chroma(persist_directory=None, embedding_fn=None):
    """Initialize Chroma vector store."""
    if persist_directory is None:
        persist_directory = PERSIST_DIR
    if embedding_fn is None:
        embedding_fn = get_embeddings()
    
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_fn
    )
    return vectordb

def clear_vectorstore(persist_directory=None):
    if persist_directory is None:
        persist_directory = PERSIST_DIR

    # Try removing
    try:
        shutil.rmtree(persist_directory)
    except Exception:
        pass  # ignore any Windows permission errors

    # Always recreate same stable directory
    os.makedirs(persist_directory, exist_ok=True)
    return persist_directory
