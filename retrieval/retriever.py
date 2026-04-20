"""
Retriever module for fetching relevant documents from vector store.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from embedding.client import EmbeddingClient
from vectorstore.factory import get_vector_store
from core.config import get_settings

class Retriever:
    """
    Handles retrieval of relevant document for single query.
    """
    def __init__(self):
        self.embedding_client = EmbeddingClient()
        self.vector_store = get_vector_store()
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    # =========================
    # Reranker 
    # =========================
    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using LLM.
        """

        if not docs:
            return docs

        # Limit docs to avoid huge prompts
        docs = docs[:20]

        doc_texts = "\n\n".join(
            [f"{i+1}. {doc.get('text', '')}" for i, doc in enumerate(docs)]
        )

        prompt = f"""
        You are a ranking assistant.

        Given a query and a list of documents, rank the most relevant ones.

        Query:
        {query}

        Documents:
        {doc_texts}

        Return the top {top_k} document numbers in order (comma separated).
        Example: 3,1,5
        """

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            content = response.choices[0].message.content.strip()

            indices = [int(i.strip()) - 1 for i in content.split(",")]

            reranked = [docs[i] for i in indices if 0 <= i < len(docs)]

            return reranked[:top_k]

        except Exception:
            # fallback to original ranking
            return docs[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query string
            top_k: Number of results (overrides config)
            filter: Optional metadata filter

        Returns:
            List of relevant documents with metadata and scores
        """

        if not query:
            return[]
        
        k = top_k or self.settings.top_k

        # Step 1: Convert query to embedding
        query_embedding = self.embedding_client.embed_query(query)

        candidate_k = k * 3  # fetch more for reranking

        # Step 2: Search vector DB
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=candidate_k,
            filter=filter if filter else None,
        )

        # Optional: apply score threshold filtering
        if self.settings.score_threshold is not None:
            results = [
                r for r in results
                if r.get("score") is not None and r["score"] <= self.settings.score_threshold
            ]

        results = self.rerank(query, results, top_k=k)

        return results
    
# Optional retrieval
retriever = Retriever()