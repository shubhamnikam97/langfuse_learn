"""
Ingestion pipeline for processing documents into the vector store.
Flow: raw documents -> chunking -> embeddings -> vector store
"""

from typing import List, Dict, Any, Optional
import uuid
import time
import traceback
# from chromadb.app import settings
# from langfuse import get_client, Langfuse
from core.langfuse_client import langfuse

from ingestion.processors.chunker import TextChunker
from embedding.client import EmbeddingClient
from vectorstore.factory import get_vector_store
from core.config import get_settings
from ingestion.loaders.loader_factory import get_loader

class IngestionPipeline:
    """
    Orchestrates document ingestion into the vector store.
    """

    def __init__(self):
        self.chunker = TextChunker()
        self.embedding_client = EmbeddingClient()
        self.vector_store = get_vector_store()
        self.settings = get_settings()
        
        self.langfuse = langfuse
        # # self.langfuse = get_client() if getattr(self.settings, "langfuse_enabled", False) else None
        # Langfuse(
        #     public_key=self.settings.langfuse_public_key,
        #     secret_key=self.settings.langfuse_secret_key,
        #     host=self.settings.langfuse_host 
        # )
        # # self.langfuse = get_client() if getattr(self.settings, "langfuse_enabled", False) else None
        # self.langfuse = (
        #     get_client()
        #     if self.settings.langfuse_public_key and self.settings.langfuse_secret_key
        #     else None
        # )

    # =========================
    # DOCUMENT INGESTION
    # =========================
    def ingest_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Ingest documents into vector store.

        Args:
            documents: List of raw document texts
            metadatas
        """
        if not documents:
            return
        
        start_time = time.time()

        try:
            # =========================
            # ROOT SPAN
            # =========================
            if self.langfuse:
                with self.langfuse.start_as_current_observation(
                    name="ingestion_pipeline",
                    as_type="span",
                    input={"num_documents": len(documents)},
                ) as root_span:
                    self._process_documents(documents, metadatas, root_span, start_time)
                
                self.langfuse.flush()
            else:
                self._process_documents(documents, metadatas, None, start_time)

        except Exception as e:
            traceback.print_exc()
            raise e
    
    # =========================
    # INTERNAL PROCESSING
    # =========================
    def _process_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]],
        root_span,
        start_time: float
    ):
        all_chunks = []
        all_metadatas = []

        # -------------------------
        # CHUNKING
        # -------------------------
        if self.langfuse:
            with self.langfuse.start_as_current_observation(
                name="chunking",
                as_type="span",
            ) as span:
                
                for i, doc in enumerate(documents):
                    chunks = self.chunker.split_text(doc)
                    # print("chunks:", chunks)
                    if not chunks:
                        continue

                    for chunk in chunks:
                        all_chunks.append(chunk)

                        if metadatas and i < len(metadatas):
                            all_metadatas.append(metadatas[i])
                        else:
                            all_metadatas.append({})
            self.langfuse.flush()
        else:
            for i, doc in enumerate(documents):
                chunks = self.chunker.split_text(doc)
                if not chunks:
                    continue

                for chunk in chunks:
                    all_chunks.append(chunk)

                    if metadatas and i < len(metadatas):
                        all_metadatas.append(metadatas[i])
                    else:
                        all_metadatas.append({})

        if not all_chunks:
            return

        # -------------------------
        # EMBEDDING
        # -------------------------
        if self.langfuse:
            with self.langfuse.start_as_current_observation(
                name="embedding",
                as_type="span",
            ) as span:

                embeddings = self.embedding_client.embed_texts(all_chunks)

                span.update(
                    output={
                        "num_embeddings": len(embeddings),
                        "embedding_dim": len(embeddings[0]) if embeddings else 0,
                    }
                )
            self.langfuse.flush()
        else:
            embeddings = self.embedding_client.embed_texts(all_chunks)

        # -------------------------
        # STORAGE
        # -------------------------
        if self.langfuse:
            with self.langfuse.start_as_current_observation(
                name="vector_store",
                as_type="span",
            ) as span:

                ids = [str(uuid.uuid4()) for _ in all_chunks]

                self.vector_store.add_documents(
                    texts=all_chunks,
                    embeddings=embeddings,
                    metadatas=all_metadatas,
                    ids=ids
                )

                span.update(output={"num_vectors": len(all_chunks)})
        else:
            ids = [str(uuid.uuid4()) for _ in all_chunks]

            self.vector_store.add_documents(
                texts=all_chunks,
                embeddings=embeddings,
                metadatas=all_metadatas,
                ids=ids
            )

        # -------------------------
        # PERSIST
        # -------------------------
        try:
            self.vector_store.persist()
        except Exception:
            pass

        latency = round(time.time() - start_time, 3)

        if root_span:
            root_span.update(
                output={
                    "status": "success",
                    "num_chunks": len(all_chunks),
                    "latency": latency,
                }
            )




    # =========================
    # FILE INGESTION
    # =========================
    def ingest_files(
        self,
        file_paths: List[str],
    ) -> None:
        """
        Ingest files using loaders.
        """
        if not file_paths:
            return
        
        start_time = time.time()

        all_documents = []
        all_metadatas = []

        for path in file_paths:
            try:
                loader = get_loader(path)
                docs = loader.load(path)

                if not docs:
                    continue

                for doc in docs:
                    all_documents.append(doc)
                    all_metadatas.append({"source": path})

            except Exception as e:
                continue
            
        self.ingest_documents(
            documents=all_documents,
            metadatas=all_metadatas,
        )

        latency = round(time.time() - start_time, 3)
            

# Optional Instance
ingestion_pipeline = IngestionPipeline()
