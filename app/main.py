"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from core.config import get_settings

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Application",
        version="1.0.0",
        description="Production-grade RAG system with OpenAI and Chroma/FAISS",
        debug=settings.debug,
    )

    # CORS (adjust origins in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api")

    @app.get("/")
    def root():
        return {"message": "RAG app is running!"}
    
    return app

app = create_app()
