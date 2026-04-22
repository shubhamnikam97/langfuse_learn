"""
OpenAI Embedding Client
Handles embedding generation with batching and retry logic.
"""

from typing import List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings

class EmbeddingClient:
    """
    Wrapper around openai embedding API.
    Supports batching and retry logic.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_embedding_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding for list of texts.
        
        Args:
            texts: List of input strings
            
        Returns:
            List of embedding vectors"""
        
        if not texts:
            return []
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, query: str)-> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Input query string
            
        Returns:
            Embedding vector
        """
        return self.embed_texts([query])[0]
    

# Optional singleton instance
embedding_client = EmbeddingClient()

        