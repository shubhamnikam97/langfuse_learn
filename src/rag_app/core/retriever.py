"""Retrieval component for RAG."""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

from .vectorstore import VectorStoreManager
from ..config.settings import settings


class Retriever:
    """Handles document retrieval for RAG queries."""

    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vectorstore_manager = vectorstore_manager
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vectorstore_manager.vectorstore.as_retriever()
        )

    def retrieve(self, query: str, k: int = 5, use_compression: bool = False) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if use_compression:
            return self.compression_retriever.get_relevant_documents(query, k=k)
        else:
            return self.vectorstore_manager.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Retrieve documents with similarity scores."""
        if not hasattr(self.vectorstore_manager.vectorstore, 'similarity_search_with_score'):
            raise NotImplementedError("Similarity search with scores not supported for this vector store")

        return self.vectorstore_manager.vectorstore.similarity_search_with_score(query, k=k)

    def filter_by_metadata(self, documents: List[Document], metadata_filters: Dict[str, Any]) -> List[Document]:
        """Filter documents by metadata."""
        filtered_docs = []
        for doc in documents:
            if all(doc.metadata.get(key) == value for key, value in metadata_filters.items()):
                filtered_docs.append(doc)
        return filtered_docs