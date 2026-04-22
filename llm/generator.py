"""
LLM response generator for RAG.
Builds prompt from retrieved context and calls OpenAI Chat Completions.
"""

from typing import List, Dict, Any, Optional
import traceback
import time
import json

from openai import OpenAI
from streamlit import context
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.langfuse_client import langfuse

class ResponseGenerator:
    """
    Generates answers using OpenAI given a query and retrieved context.
    """
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_llm_model

        self.langfuse = langfuse

    # -------------------------
    # CONTEXT
    # -------------------------
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

    # -------------------------
    # PROMPT
    # -------------------------
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
    
    # -------------------------
    # SIMPLE HALLUCINATION CHECK
    # -------------------------
    def evaluate_response(
            self,
            answer: str,
            context: str,
        ) -> Dict[str, Any]:

        if not answer or not context:
            return {"hallucination_score": 1.0, "verdict": "no_context"}
        
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())

        overlap = answer_words.intersection(context_words)
        score = len(overlap) / max(len(answer_words), 1)

        return {
            "hallucination_score": round(score, 3),
            "verdict": "grounded" if score > 0.5 else "possible_hallucination",
        }

    
    def evaluate_context_relevance(
        self,
        query: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Evaluate if retrieved context is relevant to the query.
        """

        prompt = f"""
        You are an evaluator.

        Query: {query}
        Context: {context}

        Is the context relevant for answering the query?

        Return JSON:
        {{
            "score": 0 to 1,
            "reason": "short explanation"
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"score": 0, "reason": "parse_error"}
        
    def evaluate_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Check if answer is fully supported by context.
        """

        prompt = f"""
        You are an evaluator.

        Context: {context}
        Answer: {answer}

        Is the answer fully supported by the context (no hallucination)?

        Return JSON:
        {{
            "score": 0 to 1,
            "reason": "short explanation"
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        # import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"score": 0, "reason": "parse_error"}
    
    def judge_response_llm(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Use the LLM itself to judge if the answer is supported by the context.
        """

        prompt = f"""
        You are an evaluator.
        Question: {query}
        context: {context}
        Answer: {answer}
        Evaluate:
        1. Is the answer grounded in the context?
        2. Is it correct?

        Return JSON:
        {{
        "score": 0 to 1,
        "reason": "short explanation"
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content
        try:
            result = json.loads(content)
            return result
        except:
            return {"score": 0, "reason": "parse_error"}


    # -------------------------
    # GENERATE
    # -------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
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

        try:
            # =========================
            # LANGFUSE GENERATION
            # =========================
            if self.langfuse:
                with self.langfuse.start_as_current_observation(
                    name="llm_generation",
                    as_type="generation",
                    model=self.model,
                    input={
                        "query": query,
                        "messages": messages,
                        "num_docs": len(docs)
                    },
                ) as gen:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    content = response.choices[0].message.content if response.choices else ""


                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }


                    latency = round(time.time() - start_time, 3)

                    # =========================
                    # EVALUATION
                    # =========================
                    evaluation = self.evaluate_response(content, context)
                    # LLM-as-a-Judge
                    judge_eval = self.judge_response_llm(query, content, context)
                    # Context Relevance
                    context_eval = self.evaluate_context_relevance(query, context)
                    # Faithfulness
                    faithfulness_eval = self.evaluate_faithfulness(content, context)

                    # Log everything
                    gen.update(
                        output=content,
                        usage=usage,
                        metadata={
                            "latency": latency,
                            "evaluation": evaluation,
                            "judge_eval": judge_eval,
                            "context_eval": context_eval,
                            "faithfulness_eval": faithfulness_eval,
                        },
                    )
                    
                    if judge_eval:
                        score = float(judge_eval.get("score", 0))
                        
                        # self.langfuse.create_score(
                        #     name="groudnedness",
                        #     value=score,
                        #     comment=judge_eval.get("reason", "")
                        # )
                        
                        gen.score(
                            name="groudnedness",
                            value=score,
                            comment=judge_eval.get("reason", "")
                        )

                        # Context Relevance
                        gen.score(
                            name="context_relevance",
                            value=float(context_eval.get("score", 0)),
                            comment=context_eval.get("reason", ""),
                        )

                        # Faithfulness
                        gen.score(
                            name="faithfulness",
                            value=float(faithfulness_eval.get("score", 0)),
                            comment=faithfulness_eval.get("reason", ""),
                        )

                        gen.score(
                            name="hallucination",
                            value=1-score,
                        )

                        gen.score(
                            name="answer_length",
                            value=len(content)
                        )

                        gen.end()

                print("llm_generation")  
                self.langfuse.flush()
            else:
                # Without Langfuse
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                content = response.choices[0].message.content if response.choices else ""

                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }

                latency = round(time.time() - start_time, 3)
                evaluation = self.evaluate_response(content, context)
                print("not llm_generation")
            return {
                "answer": content,
                "context": context,
                "usage": usage,
                "evaluation": evaluation,
            }
        
        except Exception as e:
            traceback.print_exc()
            raise e
    
    

# Optional singleton
response_generator = ResponseGenerator()