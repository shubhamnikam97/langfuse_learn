"""
Chroma Vector Store implementation.
Implements BaseVectorStore using ChromaDB with local persistence.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
# from core.config import Settings as ChromaSettings

from core.config import get_settings
from vectorstore.base import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    def __init__(self):
        self.settings = get_settings()

        self.client = chromadb.Client(
            ChromaSettings(
                persist_directory=self.settings.vector_store_path,
                anonymized_telemetry=False,
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=self.settings.collection_name
        )

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        if not texts:
            return
        
        # generate IDs if not provided
        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            output.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "score": dist,
                }
            )

        return output
    
    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def persist(self) -> None:
        """
        Persist changes to disk.
        """
        # self.client.persist()
        pass # Chroma auto-persists in newer versions

    def load(self) -> None:
        """
        Chroma auto-loads from disk on initialization.
        This method is kept for interface consistency.
        """
        pass


# Optional singleton instance
# chroma_store = ChromaVectorStore()