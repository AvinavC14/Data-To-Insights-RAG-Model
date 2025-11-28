import os
import logging
import gc
from langchain_core.documents import Document
from langchain_chroma import Chroma
from backend.vectorstore import get_embeddings, clear_vectorstore
from backend.utils import df_to_passages

logger = logging.getLogger(__name__)

def ingest_dataframe(df, persist_dir=None, clear_existing=True):
    """
    Convert DataFrame to documents and store in Chroma vector database.
    
    Args:
        df: pandas DataFrame to ingest
        persist_dir: Directory to persist the vector store
        clear_existing: Whether to clear existing data before ingesting
    
    Returns:
        Dictionary with ingestion statistics
    """
    try:
        # Force garbage collection before we start
        gc.collect()
        
        # Get embeddings
        embedding_fn = get_embeddings()
        persist = persist_dir if persist_dir else os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        
        # Clear existing data if requested
        if clear_existing:
            persist = clear_vectorstore(persist)
        
        # Convert DataFrame to passages
        passages = df_to_passages(df)
        
        # Create documents
        docs = [
            Document(
                page_content=p["text"],
                metadata=p["metadata"]
            ) for p in passages
        ]
        
        logger.info(f"Created {len(docs)} documents from DataFrame")
        
        # Create vector store from documents
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_fn,
            persist_directory=persist
        )
        
        # Force cleanup
        del vectordb
        gc.collect()
        
        logger.info(f"Successfully ingested {len(docs)} documents")
        
        return {
            "success": True,
            "documents_ingested": len(docs),
            "rows_processed": len(df)
        }
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }