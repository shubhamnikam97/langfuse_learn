"""
Ingestion pipeline for processing documents into the vector store.
Flow: raw documents -> chunking -> embeddings -> vector store
"""

from typing import List, Dict, Any, Optional
import uuid
import time
import traceback
from langfuse import observe

from ingestion.processors.chunker import TextChunker
from embedding.client import EmbeddingClient
from vectorstore.factory import get_vector_store
from core.langfuse_client import langfuse_client
from ingestion.loaders.loader_factory import get_loader

class IngestionPipeline:
    """
    Orchestrates document ingestion into the vector store.
    """

    def __init__(self):
        self.chunker = TextChunker()
        self.embedding_client = EmbeddingClient()
        self.vector_store = get_vector_store()

    @observe(name="ingest_documents")
    def ingest_documents(
        self,
        # documents: List[Dict[str, Any]],
        documents: List[str],
        # metadata: Optional[List[Dict[str, Any]]] = None
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
        
        # Create langfuse trace
        # trace = None
        # if langfuse_client.enabled:
        #     trace = langfuse_client.trace(
        #         name="ingestion_pipeline",
        #         metadata={
        #             "num_documents": len(documents),
        #         }
        #     )

        try:
            all_chunks = []
            all_metadatas = []

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

            # Log chunking
            # if trace:
            #     trace.span(
            #         name="chunking",
            #         input={"documents": len(documents)},
            #         output={"num_chunks": len(all_chunks)},
            #     )

            if not all_chunks:
                return


            # Generate embeddings
            embeddings = self.embedding_client.embed_texts(all_chunks)

            # Log embeddings
            # if trace:
            #     trace.span(
            #         name="embedding",
            #         input={"num_chunks": len(all_chunks)},
            #         output={"embedding_dim": len(embeddings[0]) if embeddings else 0},
            #     )

            # Generate unique IDs
            ids = [str(uuid.uuid4()) for _ in all_chunks]

            # store in vector DB
            self.vector_store.add_documents(
                texts=all_chunks,
                embeddings=embeddings,
                metadatas=all_metadatas,
                ids=ids
            )

            # Log storage
            # if trace:
            #     trace.span(
            #         name="vector_store",
            #         output={
            #             "num_vectors": len(all_chunks),
            #             # "collection": "chroma"
            #         },
            #     )

            try:
                # Persist changes
                self.vector_store.persist()
            except Exception:
                pass

            latency = round(time.time() - start_time, 3)

            # Final trace update
            # if trace:
            #     trace.update(
            #         output={
            #             "status": "success",
            #             "total_chunks": len(all_chunks),
            #             "latency": latency,
            #         }
            #     )

        except Exception as e:
            traceback.print_exc()

            # if trace:
            #     trace.update(output={"error": str(e)})
            raise e
    
    @observe(name="ingest_files")
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

        # # Langfuse Trace
        # if langfuse_client.enabled:
        #     trace = langfuse_client.trace(
        #         name="file_ingestion_pipeline",
        #         metadata={
        #             "num_files": len(file_paths),
        #             # "files": file_paths,
        #         }
        #     )

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
                # if trace:
                #     trace.span(
                #         name="loader_error",
                #         input={"file": path},
                #         output={"error": str(e)},
                #     )
                continue
        
        # if trace:
        #     trace.span(
        #         name="loading",
        #         input={"file_paths": file_paths},
        #         output={"num_documents": len(all_documents)},
        #     )
            
        self.ingest_documents(
            documents=all_documents,
            metadatas=all_metadatas,
        )

        latency = round(time.time() - start_time, 3)

        # if trace:
        #     trace.update(
        #         output={
        #             "status": "success",
        #             "total_documents": len(all_documents),
        #             "latency": latency,
        #         }
        #     )
            

# Optional Instance
ingestion_pipeline = IngestionPipeline()
