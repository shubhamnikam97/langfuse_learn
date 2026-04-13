"""FastAPI application for the RAG service."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from ..config.settings import settings
from ..core.vectorstore import VectorStoreManager
from ..core.retriever import Retriever
from ..core.generator import Generator
from ..data.ingestion import DataIngestion
from ..utils.logging import get_logger, setup_langfuse_logging

logger = get_logger(__name__)

# Initialize components
vectorstore_manager = VectorStoreManager()
retriever = Retriever(vectorstore_manager)
generator = Generator(retriever)
data_ingestion = DataIngestion()

# Setup Langfuse logging
setup_langfuse_logging()

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    use_compression: Optional[bool] = False
    top_k: Optional[int] = 5


class IngestRequest(BaseModel):
    data_source: str


class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]
    query: str
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the vector store on startup."""
    try:
        # Try to load existing vector store
        vectorstore_manager.load_vectorstore()
        logger.info("Loaded existing vector store")
    except:
        logger.info("No existing vector store found - will create on first ingestion")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = generator.generate_answer(
            query=request.query,
            use_compression=request.use_compression
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """Ingest documents into the vector store."""
    try:
        # Load and preprocess documents
        documents = data_ingestion.load_and_preprocess(request.data_source)

        # Create or update vector store
        if vectorstore_manager.vectorstore is None:
            vectorstore_manager.create_vectorstore(documents)
        else:
            vectorstore_manager.add_documents(documents)

        return {
            "message": f"Successfully ingested {len(documents)} document chunks",
            "source": request.data_source
        }
    except Exception as e:
        logger.error(f"Ingest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List available documents (simplified)."""
    # This is a placeholder - in production you'd track ingested documents
    return {"documents": []}