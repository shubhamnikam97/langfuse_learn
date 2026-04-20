"""
LLM response generator for RAG.
Builds prompt from retrieved context and calls OpenAI Chat Completions.
"""

from typing import List, Dict, Any, Optional
import traceback
import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.langfuse_client import langfuse_client
from langfuse import observe

class ResponseGenerator:
    """
    Generates answers using OpenAI given a query and retrieved context.
    """
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_llm_model

    def build_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        Build a single context string from retrieved documents.

        Args:
            docs: Retrieved documents with 'text' and optional 'metadata'

        Returns:
            Concatenated context string
        """
        if not docs:
            return ""
        
        parts = []
        for i, d in enumerate(docs, 1):
            text = d.get('text', '')
            meta = d.get('metadata') or {}
            source = meta.get("source") or "unknown"
            parts.append(f"[Chunk {i} | source={source}]\n{text}")
        return "\n\n".join(parts)

    def build_messages(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Construct chat messages for the LLM.
        """
        system = system_prompt or (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, say you don't know. Be concise and cite chunk numbers when useful."
        )

        user = (
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer:"
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    @observe(name="llm_generation")
    def generate(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer using the LLM.

        Returns a dict with answer text and optional usage info.
        """
        context = self.build_context(docs)
        messages = self.build_messages(query, context, system_prompt)

        start_time = time.time()

        # Create Langfuse trace
        # trace = None
        # if langfuse_client.enabled():
        #     trace = langfuse_client.trace(
        #         name="llm_generation",
        #         metadata={
        #             "model": self.model,
        #             "temperature": temperature,
        #             "max_tokens": max_tokens,
        #         },
        #     )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content if response.choices else ""

            usage = None
            try:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            except Exception:
                usage = None

            latency = round(time.time() - start_time, 3)

            # Log LLM span
            # if trace:
            #     # trace.span(
            #     #     name="llm_call",
            #     #     input={
            #     #         "query": query,
            #     #         "messages": messages,   # actual prompt structure
            #     #         "num_docs": len(docs),
            #     #     },
            #     #     output={
            #     #         "answer": content,
            #     #         "usage": usage,
            #     #     },
            #     #     metadata={
            #     #         "model": self.model,
            #     #         "temperature": temperature,
            #     #         "max_tokens": max_tokens,
            #     #     },
            #     # )

            #     trace.span(
            #         name="openai_call",
            #         input={
            #             "query": query,
            #             "num_docs": len(docs),
            #             "prompt_preview": messages[-1]["content"][:300],  # ✅ trimmed
            #         },
            #         output={
            #             "answer_preview": content[:300],  # ✅ trimmed
            #             "usage": usage,
            #             "latency": latency,
            #         },
            #         metadata={
            #             "model": self.model,
            #             "temperature": temperature,
            #         },
            #     )

                # Optional final update
                # trace.update(
                #     output={
                #         "answer": content,
                #         "total_tokens": usage["total_tokens"] if usage else None,
                #     }
                # )
                # trace.update(
                #     output={
                #         "answer": content[:500],  # ✅ avoid huge payload
                #         "total_tokens": usage["total_tokens"] if usage else None,
                #         "latency": latency,
                #     }
                # )

            return {
                "answer": content,
                "context": context,
                "usage": usage,
            }
        
        except Exception as e:
            traceback.print_exc()

            # Error tracking in Langfuse
            # if trace:
            #     trace.update(
            #         output={
            #             "error": str(e),
            #         }
            #     )

            raise e

# Optional singleton
response_generator = ResponseGenerator()