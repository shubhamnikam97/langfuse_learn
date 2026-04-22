"""
End-to-end RAG pipeline orchestration.
Connects Retriever and ResponseGenerator to produce final answers.
"""

from typing import Dict, Any, Optional
import time
import traceback
from retrieval.retriever import Retriever
from llm.generator import ResponseGenerator
from core.config import get_settings
# from core.langfuse_client import langfuse_client
# from langfuse import get_client, Langfuse
from core.langfuse_client import langfuse

class RAGPipeline:
    """
    Orchestrates the full RAG flow:
    Query -> Retrieval -> Generation -> Response
    """

    def __init__(self):
        self.retriever = Retriever()
        self.generator = ResponseGenerator()
        self.settings = get_settings()

        # Initialize Langfuse client
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

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline.

        Args:
            query: User query
            top_k: Optional override for number of retrieved docs
            filter: Optional metadata filter

        Returns:
            Dict containing answer, context, docs, and metadata
        """
        
        if not query:
            return {"answer": "", "docs": [], "context": ""}
        
        start_time = time.time()

        try:
            
            if self.langfuse:
                with self.langfuse.start_as_current_observation(
                    name="rag_pipeline",
                    as_type="span",
                    input={
                        "query": query,
                        "top_k": top_k,
                        "filter": filter,
                    },
                ) as root_span:
                    result = self._execute_pipeline(query, start_time, top_k, filter)

                    root_span.update(
                        output={
                            "answer": result["answer"],
                            "latency": result["latency"],
                        }
                    )

                    self.langfuse.flush()

                    return result

            else:
                # Run without Langfuse
                return self._execute_pipeline(query, start_time, top_k, filter)
        
        except Exception as e:
            traceback.print_exc()
            raise e
    
    def _execute_pipeline(
            self, 
            query: str,
            start_time: float,
            top_k: Optional[int] = None,
            filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Internal method to execute the pipeline with Langfuse tracing.
        """
        # Step 1: Retrieve relevant documents
        # -------------------------
        # RETRIEVAL
        # -------------------------
        retrieval_start = time.time()

        if self.langfuse:
            with self.langfuse.start_as_current_observation(
                name="retrieval",
                as_type="span",
                input={"query": query},
            ) as span:

                docs = self.retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    filter=filter,
                )

                span.update(output={"num_docs": len(docs)})
            self.langfuse.flush()

        else:
            docs = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filter=filter,
            )

        retrieval_time = round(time.time() - retrieval_start, 3)

        # -------------------------
        # GENERATION
        # -------------------------
        generation_start = time.time()

        if self.langfuse:
            with self.langfuse.start_as_current_observation(
                name="generation",
                as_type="span",
                input={"query": query, "num_docs": len(docs)},
            ) as span:

                generation = self.generator.generate(
                    query=query,
                    docs=docs,
                )

                span.update(
                    output={
                        "answer": generation.get("answer"),
                    }
                )

            self.langfuse.flush()
        else:
            generation = self.generator.generate(
                query=query,
                docs=docs,
            )

        generation_time = round(time.time() - generation_start, 3)

        # -------------------------
        # FINAL RESULT
        # -------------------------
        end_time = time.time()

        return {
            "query": query,
            "answer": generation.get("answer"),
            "docs": docs,
            "context": generation.get("context"),
            "usage": generation.get("usage"),
            "latency": round(end_time - start_time, 3),
            "timings": {
                "retrieval": retrieval_time,
                "generation": generation_time,
            },
        }
        


# Optional singleton
rag_pipeline = RAGPipeline()