"""
Ingestion pipeline for processing documents into the vector store.
Flow: raw documents -> chunking -> embeddings -> vector store
"""

from typing import List, Dict, Any, Optional
import uuid
import time
import traceback
from core.langfuse_client import langfuse

from ingestion.processors.chunker import TextChunker
from embedding.client import EmbeddingClient
from vectorstore.factory import get_vector_store
from core.config import get_settings
from ingestion.loaders.loader_factory import get_loader
import base64
import os

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

        ids = [str(uuid.uuid4()) for _ in all_chunks]

        if self.langfuse:
            with self.langfuse.start_as_current_observation(
                name="vector_store",
                as_type="span",
            ) as span:

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
    # IMAGE HANDLING
    # =========================
    def save_image_locally(self, image_bytes: bytes) -> str:
        folder = "data/images"
        os.makedirs(folder, exist_ok=True)

        file_name = f"{uuid.uuid4()}.png"
        path = os.path.join(folder, file_name)

        with open(path, "wb") as f:
            f.write(image_bytes)

        return path

    def describe_image(self, image_bytes: bytes) -> str:
        """
        Convert image → semantic text using LLM
        """

        start_time = time.time()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        if self.langfuse:
            with self.langfuse.start_as_current_observation(
                name="image_captioning",
                as_type="generation",
                model=self.settings.openai_llm_model,
            ) as gen:
                response = self.embedding_client.client.chat.completions.create(
                    model=self.settings.openai_llm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in detail for retrieval."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300,
                )
                content = response.choices[0].message.content

                gen.update(
                    output=content,
                    metadata={
                        "latency": round(time.time() - start_time, 3)
                    }
                )

            return content
        else:
            response = self.embedding_client.client.chat.completions.create(
                model=self.settings.openai_llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image for retrieval."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
            )

            return response.choices[0].message.content


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

        try:
            if self.langfuse:
                with self.langfuse.start_as_current_observation(
                    name="file_ingestion",
                    as_type="span",
                    input={"num_files": len(file_paths)}
                ) as span:

                    self._process_files(file_paths, all_documents, all_metadatas)

                    span.update(output={"num_docs": len(all_documents)})

                self.langfuse.flush()
            else:
                self._process_files(file_paths, all_documents, all_metadatas)

        except Exception:
            traceback.print_exc()

        self.ingest_documents(all_documents, all_metadatas)

        # for path in file_paths:
        #     try:
        #         loader = get_loader(path)
        #         items = loader.load(path)

        #         if not items:
        #             continue

        #         for item in items:
        #             if item["type"] in ["text", "table"]:
        #                 all_documents.append(item["content"])
        #                 all_metadatas.append({
        #                     "source": path,
        #                     "type": item["type"],
        #                     "page": item.get("page")
        #                 })

        #             elif item["type"] == "image":
        #                 image_path = self.save_image_locally(item["content"])
        #                 caption = self.describe_image(item["content"])

        #                 all_documents.append(caption)
        #                 all_metadatas.append({
        #                     "source": path,
        #                     "type": "image",
        #                     "image_path": image_path,
        #                     "page": item.get("page")
        #                 })

        #     except Exception as e:
        #         continue
            
        # self.ingest_documents(
        #     documents=all_documents,
        #     metadatas=all_metadatas,
        # )

        # latency = round(time.time() - start_time, 3)

    def _process_files(self, file_paths, all_documents, all_metadatas):

        for path in file_paths:
            try:
                loader = get_loader(path)
                items = loader.load(path)

                for item in items:

                    if item["type"] in ["text", "table"]:
                        all_documents.append(item["content"])
                        all_metadatas.append({
                            "source": path,
                            "type": item["type"],
                            "page": item.get("page")
                        })

                    elif item["type"] == "image":
                        image_path = self.save_image_locally(item["content"])
                        caption = self.describe_image(item["content"])

                        all_documents.append(caption)
                        all_metadatas.append({
                            "source": path,
                            "type": "image",
                            "image_path": image_path,
                            "page": item.get("page")
                        })

            except Exception as e:
                print(f"[Ingestion Error] {path}: {e}")
                continue
            

# Optional Instance
ingestion_pipeline = IngestionPipeline()
