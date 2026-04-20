"""
Abstract base interface for vector stores.
This allows switching between Chroma, FAISS, or any future DB
without changing business logic.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    All vector databases (Chroma, FAISS, etc.) must follow this interface.
    """

    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to vector store.
        
        Args:
            texts: List of document chunks
            embeddings: Corresponding embeddings
            metadataa: Optional metadata for each document
            ids: Optional unique IDs
            
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search.

        Args:
            query_embedding: Embedding of query
            k: Number of results
            filter: Optional metadata filter

        Returns:
            List of results with text + metadata + score
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """
        Persist the vector store to disk (if applicable).
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load the vector store from disk (if applicable).
        """
        pass
