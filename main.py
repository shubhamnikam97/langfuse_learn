"""Main entry point for the RAG application."""

import uvicorn
from src.rag_app.config.settings import settings


def main():
    """Run the RAG API server."""
    print("Starting RAG API server...")

    uvicorn.run(
        "src.rag_app.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


if __name__ == "__main__":
    main()
