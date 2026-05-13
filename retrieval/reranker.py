from typing import List, Dict, Any
from openai import OpenAI
from core.config import get_settings


class Reranker:

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        for doc in docs:
            prompt = f"""
            Query: {query}
            Document: {doc.get('text')}

            Score relevance from 0 to 1.
            Return ONLY number.
            """

            response = self.client.chat.completions.create(
                model=self.settings.openai_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            try:
                score = float(response.choices[0].message.content.strip())
            except:
                score = 0.0

            doc["rerank_score"] = score

        return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)