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
from core.langfuse_client import langfuse_client
from langfuse import observe


class RAGPipeline:
    """
    Orchestrates the full RAG flow:
    Query -> Retrieval -> Generation -> Response
    """

    def __init__(self):
        self.retriever = Retriever()
        self.generator = ResponseGenerator()
        self.settings = get_settings()

    @observe(name="rag_pipeline")
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

        # Create langfuse trace
        # trace = None
        # if langfuse_client.enabled:
        #     trace = langfuse_client.trace(
        #         name="rag_pipeline",
        #         metadata={
        #             "query": query,
        #             "top_k": top_k,
        #             "filter": filter,
        #         },
        #     )

        try:
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()

            docs = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filter=filter,
            )

            retrieval_time = round(time.time() - retrieval_start, 3)

            # Log retrieval span
            # if trace:
            #     trace.span(
            #         name="retrieval",
            #         input={"query": query},
            #         output={
            #             "num_docs": len(docs),
            #             "latency": retrieval_time,
            #         },
            #     )

            # Step 2: Generate response using LLM
            generation_start = time.time()

            generation = self.generator.generate(
                query=query,
                docs=docs,
            )

            generation_time = round(time.time() - generation_start, 3)

            # # Log generation span
            # if trace:
            #     trace.span(
            #         name="generation",
            #         input={
            #             "query": query,
            #             "num_docs": len(docs),
            #         },
            #         output={
            #             "answer_preview": (generation.get("answer") or "")[:200],
            #             "latency": generation_time,
            #         },
            #     )


            end_time = time.time()

            result = {
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

            # # Final trace update (optional but good practice)
            # if trace:
            #     trace.update(
            #         output={
            #             "answer": result["answer"][:300] if result["answer"] else "",
            #             "total_latency": result["latency"],
            #             "num_docs": len(docs),
            #         }
            #     )

            return result
        
        except Exception as e:
            # =========================
            # Error Handling + Logging
            # =========================
            traceback.print_exc()

            # if trace:
            #     trace.update(
            #         output={
            #             "error": str(e),
            #         }
            #     )

            raise e




# Optional singleton
rag_pipeline = RAGPipeline()