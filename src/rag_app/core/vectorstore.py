"""Vector store management for RAG."""

from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

from ..config.settings import settings


class VectorStoreManager:
    """Manages vector store operations for document embeddings."""

    def __init__(self):
        self.store_type = settings.vector_store_type
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=settings.embedding_model
        )
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document]) -> Any:
        """Create and populate vector store with documents."""
        if self.store_type == "chroma":
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=settings.chroma_persist_directory
            )
        elif self.store_type == "faiss":
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

        return self.vectorstore

    def load_vectorstore(self) -> Any:
        """Load existing vector store."""
        if self.store_type == "chroma":
            self.vectorstore = Chroma(
                persist_directory=settings.chroma_persist_directory,
                embedding_function=self.embeddings
            )
        elif self.store_type == "faiss":
            # FAISS stores need to be loaded from disk
            # This is a simplified example
            raise NotImplementedError("FAISS loading not implemented")
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")

        return self.vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        self.vectorstore.add_documents(documents)

        if self.store_type == "chroma":
            self.vectorstore.persist()

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        return self.vectorstore.similarity_search(query, k=k)

    def save_vectorstore(self) -> None:
        """Save vector store to disk."""
        if self.store_type == "chroma":
            self.vectorstore.persist()
        elif self.store_type == "faiss":
            # Save FAISS index
            self.vectorstore.save_local(settings.data_directory + "/faiss_index")