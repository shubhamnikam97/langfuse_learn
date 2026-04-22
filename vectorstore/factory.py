"""
Factory for creating vector store instances based on configuration.
Keeps the rest of the system decoupled from concrete implementations.
"""

from functools import lru_cache
from core.config import get_settings
from vectorstore.base import BaseVectorStore
from vectorstore.chroma_store import ChromaVectorStore

# FAISS will be added later when implemented
# from rag_app.vectorstore.faiss_store import FaissVectorStore

@lru_cache()
def get_vector_store() -> BaseVectorStore:
    """
    Returns a singleton vector store instance base on config.
    Uses LRU cache to ensure only one instance is created. (singleton-like)
    while still allowing easy testing/mocking by clearing the cache if needed.
    """

    settings = get_settings()

    if settings.vector_db == "chroma":
        return ChromaVectorStore()
        
    
    # elif settings.vector_db == "faiss":
    #     return FaissVectorStore()

    raise ValueError(f"Unsupported vector_db: {settings.vector_db}")

def reset_vector_store_cache() -> None:
    """
    Clears the cached vector store instance.
    Useful for tests or when reloading configuration.
    """
    get_vector_store.cache_clear()