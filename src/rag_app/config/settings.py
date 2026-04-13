"""Configuration settings for the RAG application."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # OpenAI Settings
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"

    # Vector Store Settings
    vector_store_type: str = "chroma"  # chroma, faiss, pinecone
    chroma_persist_directory: str = "./data/chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Langfuse Settings
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Data Settings
    data_directory: str = "./data"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()