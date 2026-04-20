"""
FastAPI routes for RAG application.
Provides endpoints for querying and ingestion.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from rag.pipeline import RAGPipeline
from ingestion.pipeline import IngestionPipeline
import traceback
import os
import shutil
import uuid
router = APIRouter()

# Initialize pipelines (could later move to dependency injection)
rag_pipeline = RAGPipeline()
ingestion_pipeline = IngestionPipeline()

# =========================
# Request Schemas
# =========================

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None

class IngestionRequest(BaseModel):
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None

# =========================
# Routes
# =========================

@router.post("/query")
def query_rag(request: QueryRequest):
    """
    Query the Rag system
    """
    try:
        result = rag_pipeline.run(
            query=request.query,
            top_k=request.top_k,
            filter=request.filter,
        )
        return result
    except Exception as e:
        # raise HTTPException(status_code=500, detail=str(e))
        traceback.print_exc()   # 👈 prints full error in terminal
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
def ingest_data(request: IngestionRequest):
    """
    Ingest documents into vector store
    """
    try:
        # ingestion_pipeline.run(
        #     documents=request.documents,
        #     metadatas=request.metadatas,
        # )
        ingestion_pipeline.ingest_documents(
            documents=request.documents,
            metadatas=request.metadatas,
        )
        return {'status': 'success', 'message':'Documents ingested successfully'}
    except Exception as e:
        # raise HTTPException(status_code=500, detail=str(e))
        traceback.print_exc()   # 👈 prints full error in terminal
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload files and ingest into vector store.
    """

    if not files:
        return {"status": "error", "message": "No files uploaded"}

    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    saved_paths = []

    try:
        # =========================
        # Save uploaded files
        # =========================
        for file in files:
            file_ext = os.path.splitext(file.filename)[1]
            unique_name = f"{uuid.uuid4()}{file_ext}"

            file_path = os.path.join(upload_dir, unique_name)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_paths.append(file_path)

        # =========================
        # Ingest files
        # =========================
        ingestion_pipeline.ingest_files(saved_paths)

        return {
            "status": "success",
            "files_processed": len(saved_paths),
            "files": saved_paths,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
    
    finally:
        # =========================
        #  CLEANUP FILES HERE
        # =========================
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)